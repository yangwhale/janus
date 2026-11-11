"""被测对 golden 的 loss 偏移归因：把「差多少」拆成「偏了多少」和「抖了多少」，再看是哪个逐步量先漂。

输入只是逐步原值（各运行的 metrics.jsonl），全部从原值算，不手写。输出一份 JSON：

1. ``segments``：分段带符号均值。被测对每份 golden、golden 两两（噪声参照），看偏移何时出现、是否继续长。
2. ``tail``：后段的逐步分布——均值、标准差、被测更低的步数占比、一阶自相关。
   用来区分「中心挪了」（偏）和「整体平移不抖」：两条轨迹只要不逐位相同，逐步差就一定会抖。
3. ``lag_check``：同步号对比与错一步对比的平均绝对差（排除统计时步号错位）。
4. ``other_fields``：其余逐步量（grad_norm、param_norm、路由偏置统计……）各段的相对差，被测对噪声参照。
5. ``param_norm_abs``：param_norm 的绝对差随步数的轨迹。

相对差 d_t = (被测_t − golden_t) / |golden_t|，与 ``loss_series`` 一致。分段、后段、取样步默认按 1000 步的运行设定，
步数不同时用参数改。用法见 ``janus offsets --help``；完整的使用例子是
models/hunyuan3/alignments/01-v5p-acceptance/cases/01-loss-offset-param-norm.md。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

SEGMENTS = [(2, 101), (101, 301), (301, 601), (601, 1001)]
TAIL = (301, 1001)
NORM_STEPS = [1, 2, 10, 50, 100, 200, 300, 500, 700, 1000]
LAG_WINDOW = (2, 1001)
SKIP_FIELDS = {"step", "lr", "wd", "skipped_iter", "step_time"}  # 不是数值轨迹的字段


def load(path) -> dict[int, dict]:
    """读 metrics.jsonl：{步号: 整行}。"""
    rows = (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    return {int(r["step"]): r for r in rows}


def rel(x, y, field, steps, lag=0):
    """被测 x 对参照 y 在给定步上的相对差数组；lag ≠ 0 时 x 的第 s 步对 y 的第 s − lag 步。
    参照为 0 的步跳过（如某层专家负载最小值 router.load_min 在某步为 0），不让整次分析因除零中断。"""
    out = []
    for s in steps:
        if s in x and s - lag in y and field in x[s] and field in y[s - lag] and y[s - lag][field] != 0:
            out.append((x[s][field] - y[s - lag][field]) / abs(y[s - lag][field]))
    return np.array(out)


def attribute(
    test: tuple[str, dict],
    goldens: dict[str, dict],
    segments=SEGMENTS,
    tail=TAIL,
    norm_steps=NORM_STEPS,
    lag_window=LAG_WINDOW,
) -> dict:
    """偏移归因的全部计算。test = (名, load() 的结果)，goldens = {名: load() 的结果}。返回可 JSON 化的 dict。"""
    tname, T = test
    G = goldens
    pairs = list(itertools.combinations(G, 2))
    res = {"test": tname, "golden": list(G)}
    combos = [(f"{tname}-{g}", T, G[g]) for g in G] + [(f"{b}-{c}", G[b], G[c]) for b, c in pairs]

    seg = {}
    for lab, x, y in combos:
        seg[lab] = [float(rel(x, y, "lm_loss", range(lo, hi)).mean()) for lo, hi in segments]
    res["segments"] = {"bounds": [tuple(s) for s in segments], "signed_mean": seg}

    tl = {}
    for lab, x, y in combos:
        d = rel(x, y, "lm_loss", range(*tail))
        tl[lab] = {
            "mean": float(d.mean()),
            "std": float(d.std()),
            "neg_frac": float((d < 0).mean()),
            "autocorr1": float(np.corrcoef(d[:-1], d[1:])[0, 1]),
        }
    res["tail"] = {"window": tuple(tail), "stats": tl}

    g0 = next(iter(G))
    res["lag_check"] = {
        str(lag): float(np.abs(rel(T, G[g0], "lm_loss", range(*lag_window), lag)).mean()) for lag in (0, 1, -1)
    }

    fields = sorted({k for r in T.values() for k in r} & {k for r in G[g0].values() for k in r} - SKIP_FIELDS)
    probe = 10 if 10 in T else (min(T) if T else None)  # 用第 10 步（没有就用第一步）判断字段是不是数
    other = {}
    early = [(1, 2), (2, segments[0][1]), tuple(tail)]
    for f in fields:
        if not isinstance(T.get(probe, {}).get(f), (int, float)):
            continue
        other[f] = {
            "test": [float(rel(T, G[g0], f, range(lo, hi)).mean()) for lo, hi in early],
            "noise": [[float(rel(G[b], G[c], f, range(lo, hi)).mean()) for lo, hi in early] for b, c in pairs],
        }
    res["other_fields"] = {"segments": early, "rel_mean": other}

    res["param_norm_abs"] = {
        str(s): {
            "golden": G[g0][s]["param_norm"],
            "test_minus_golden": T[s]["param_norm"] - G[g0][s]["param_norm"],
            "noise": [G[b][s]["param_norm"] - G[c][s]["param_norm"] for b, c in pairs],
        }
        for s in norm_steps
        if s in T and s in G[g0] and "param_norm" in T[s]
    }
    return res


def _window(s: str) -> tuple[int, int]:
    a, b = s.split(":")
    return int(a), int(b)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="janus offsets", description="被测对 golden 的 loss 偏移归因（不判 pass / fail）。"
    )
    ap.add_argument("--test", required=True, help="名=metrics.jsonl")
    ap.add_argument(
        "--golden",
        nargs="+",
        action="extend",
        required=True,
        help="名=metrics.jsonl …（可以一次给几个，也可以重复写）；第一份当对照的基准",
    )
    ap.add_argument("--json", type=Path, help="把结果另写到这个文件（同时总会打印到标准输出）")
    ap.add_argument("--segments", nargs="+", type=_window, default=SEGMENTS, help="分段 a:b …（默认按 1000 步分四段）")
    ap.add_argument("--tail", type=_window, default=TAIL, help="后段窗口 a:b（默认 301:1001）")
    ap.add_argument("--lag-window", type=_window, default=LAG_WINDOW, help="错位自检的窗口（默认 2:1001）")
    ap.add_argument("--norm-steps", nargs="+", type=int, default=NORM_STEPS, help="param_norm 轨迹的取样步")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    tname, tpath = a.test.split("=", 1)
    goldens = {n: load(p) for n, p in (g.split("=", 1) for g in a.golden)}
    res = attribute((tname, load(tpath)), goldens, a.segments, a.tail, a.norm_steps, a.lag_window)
    text = json.dumps(res, ensure_ascii=False, indent=1)
    print(text)
    if a.json:
        a.json.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
