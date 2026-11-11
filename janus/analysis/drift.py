"""多步漂移分析：L4 上发现被测与 golden 有持续偏差以后，往下拆「谁在漂、漂了多少、是不是噪声」。不判 pass / fail。

四个子命令（``janus drift <子命令> --help`` 看参数），输入都是逐步原值（metrics.jsonl）：

- ``pairs``：几组运行之间两两比，各逐步量在后段的相对差范围、第 N 步 param_norm 的绝对差范围。
  组内两两（``组A-组A``）当噪声，跨组（``组A-组B``）当被测对参照。
- ``trend``：被测对每份 golden、golden 两两之间，逐步相对差的带符号均值（± 标准误）、MARE、最大单步、
  分段 95 分位包络、累积均值、100 步滑动均值；另比路由偏置统计 ``router.bias_{mean,std}.<层>``。
  判读：包络走平 = 不累积；累积 / 滑动均值一直往一边走 = 系统偏差。
- ``groups``：分组参数范数。``group_sq.<组>[.<层>]`` 是该组 FP32 主权重的平方和（抓数时开 ``--group-norms``），
  把 param_norm 的差拆成各组贡献：‖p‖_test − ‖p‖_ref ≈ Σ_组 (sq_test − sq_ref) / (2‖p‖_ref)；
  ``expert_sq.<层>`` 是每个专家的平方和，另看多涨的部分是不是集中在少数专家。
- ``rows``：输出层（与 embedding）逐行漂移。``output_row_sq`` / ``output_row_u`` / ``embedding_row_sq`` 逐行比，
  按每个词在数据里出现的次数分桶（``label_counts.npz``）。

相对差 d_t = (A_t − B_t) / |B_t|，与 ``loss_series``、``janus offsets`` 一致。完整的使用例子：
models/hunyuan3/alignments/01-v5p-acceptance/cases/01-loss-offset-param-norm.md。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 共用


def load_metrics(path) -> dict[int, dict]:
    """读 metrics.jsonl：{步号: 整行}。"""
    rows = (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    return {int(r["step"]): r for r in rows}


def _steps(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


# ---------------------------------------------------------------- pairs

PAIR_FIELDS = ["lm_loss", "param_norm", "grad_norm"]
PAIR_TAIL = (301, 1001)
PAIR_NORM_STEP = 1000


def _tail_rel(x, y, f, tail):
    d = [(x[s][f] - y[s][f]) / abs(y[s][f]) for s in range(*tail) if s in x and s in y and y[s][f] != 0]
    return float(np.mean(d))


def pairs(series: dict[str, dict], groups: dict[str, list[str]], specs: list[str], fields=PAIR_FIELDS,
          tail=PAIR_TAIL, norm_step=PAIR_NORM_STEP) -> dict:
    """series = {运行名: load_metrics()}，groups = {组: [运行名]}，specs = ["<标签>:<组A>-<组B>", …]。"""
    out = {"tail": tail, "norm_step": norm_step, "comparisons": {}}
    for spec in specs:
        label, _, ab = spec.rpartition(":")
        ga, gb = ab.split("-", 1)
        prs = list(itertools.combinations(groups[ga], 2)) if ga == gb else [(x, y) for x in groups[ga] for y in groups[gb]]
        res = {"pairs": [f"{x}-{y}" for x, y in prs], "tail_rel": {}, "param_norm_abs": None}
        for f in fields:
            v = [_tail_rel(series[x], series[y], f, tail) for x, y in prs]
            res["tail_rel"][f] = {"min": min(v), "max": max(v), "mean": float(np.mean(v)), "per_pair": v}
        v = [series[x][norm_step]["param_norm"] - series[y][norm_step]["param_norm"] for x, y in prs]
        res["param_norm_abs"] = {"min": min(v), "max": max(v), "per_pair": v}
        out["comparisons"][label] = res
    return out


def main_pairs(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="janus drift pairs", description="几组运行之间两两比的后段相对差与 param_norm 绝对差。")
    ap.add_argument("--series", nargs="+", required=True, help="名=metrics.jsonl …")
    ap.add_argument("--groups", nargs="+", required=True, help="组=名,名,… …")
    ap.add_argument("--pairs", nargs="+", required=True, help="「标签:组A-组B」…；组A 与组B 相同时取组内两两")
    ap.add_argument("--fields", nargs="+", default=PAIR_FIELDS, help="要比的逐步量（默认 lm_loss param_norm grad_norm）")
    ap.add_argument("--tail", default=f"{PAIR_TAIL[0]}:{PAIR_TAIL[1]}", help="后段窗口 a:b（默认 301:1001）")
    ap.add_argument("--norm-step", type=int, default=PAIR_NORM_STEP, help="比 param_norm 绝对差的那一步（默认 1000）")
    ap.add_argument("--json", type=Path, help="另写到这个文件")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    series = {n: load_metrics(p) for n, p in (x.split("=", 1) for x in a.series)}
    groups = {g: v.split(",") for g, v in (x.split("=", 1) for x in a.groups)}
    tail = tuple(int(x) for x in a.tail.split(":"))
    out = pairs(series, groups, a.pairs, a.fields, tail, a.norm_step)
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    print(txt)
    if a.json:
        a.json.write_text(txt + "\n")
    return 0


# ---------------------------------------------------------------- trend

TREND_FIELDS = ["lm_loss", "grad_norm", "param_norm"]
TREND_SEGS = ((1, 10), (11, 100), (101, 300), (301, 600), (601, 1000))


def _summ(x, n):
    ax = np.abs(x)
    w = x[1:]
    out = {"step1": float(x[0]), "signed_mean": float(w.mean()), "se": float(w.std(ddof=1) / np.sqrt(len(w))),
           "mare": float(ax[1:].mean()), "max": float(w[np.argmax(np.abs(w))]), "max_at": int(np.argmax(np.abs(w)) + 2)}
    out["env95"] = {f"{a}-{b}": float(np.percentile(ax[a - 1:b], 95)) for a, b in TREND_SEGS if b <= n}
    cum = np.cumsum(x) / np.arange(1, n + 1)
    out["cum"] = {s: float(cum[s - 1]) for s in (10, 100, 300, 600, 1000) if s <= n}
    if n >= 100:
        rw = np.convolve(x, np.ones(100) / 100, mode="valid")
        out["roll100_at"] = {s: float(rw[s - 100]) for s in (100, 300, 600, 1000) if s <= n}
    return out


def trend(test: dict, goldens: dict[str, dict], steps: int = 1000, fields=TREND_FIELDS) -> dict:
    """test = load_metrics()，goldens = {名: load_metrics()}，比第 1..steps 步。"""
    R = {"test": test} | goldens
    gs = list(goldens)
    S = np.arange(1, steps + 1)

    def d(x, y, f):
        return np.array([(R[x][s][f] - R[y][s][f]) / abs(R[y][s][f]) for s in S])

    res = {f: {"test": {g: _summ(d("test", g, f), steps) for g in gs},
               "golden_pairs": {f"{x}~{y}": _summ(d(x, y, f), steps) for x, y in itertools.combinations(gs, 2)}}
           for f in fields}
    bias = {}
    for k in sorted(k for k in R[gs[0]][1] if k.startswith("router.bias_")):
        if k not in R["test"][1]:
            continue
        bias[k] = {"test_max_abs": [max(abs(R["test"][s][k] - R[g][s][k]) for s in S) for g in gs],
                   "pairs_max_abs": [max(abs(R[x][s][k] - R[y][s][k]) for s in S) for x, y in itertools.combinations(gs, 2)],
                   "test_last": [R["test"][steps][k] - R[g][steps][k] for g in gs],
                   "pairs_last": [R[x][steps][k] - R[y][steps][k] for x, y in itertools.combinations(gs, 2)],
                   "value_last": R[gs[0]][steps][k]}
    res["router_bias"] = bias
    res["last"] = {f: {"test": R["test"][steps][f], gs[0]: R[gs[0]][steps][f]} for f in fields}
    return res


def main_trend(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="janus drift trend", description="被测对 golden 的多步趋势统计（包络、累积均值、滑动均值）。")
    ap.add_argument("--test", required=True, type=Path, help="被测的运行目录（读其中的 metrics.jsonl）")
    ap.add_argument("--golden", nargs="+", required=True, type=Path, help="golden 的运行目录 …（目录名当运行名）")
    ap.add_argument("--out", required=True, type=Path, help="输出 JSON")
    ap.add_argument("--steps", type=int, default=1000, help="比第 1..N 步（默认 1000）")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    goldens = {g.name: load_metrics(g / "metrics.jsonl") for g in a.golden}
    res = trend(load_metrics(a.test / "metrics.jsonl"), goldens, a.steps)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for f in TREND_FIELDS:
        for side in ("test", "golden_pairs"):
            for k, v in res[f][side].items():
                print(f"{f} {side} {k}: 均值 {v['signed_mean']:+.2e}±{v['se']:.1e}  MARE {v['mare']:.2e}  "
                      f"最大 {v['max']:+.2e}@{v['max_at']}")
    return 0


# ---------------------------------------------------------------- groups


def _contrib(T, R, s):
    pn = R[s]["param_norm"]
    keys = sorted(k for k in R[s] if k.startswith("group_sq.") and k in T[s])
    return {k: (T[s][k] - R[s][k]) / (2 * pn) for k in keys}, pn


def groups(T: dict, R: dict, P: list[dict], steps: list[int]) -> dict:
    """T、R = load_metrics()（被测、参照），P = 另几份参照（各自对 R 比，当噪声）。返回 {步: {...}}。"""
    out = {}
    for s in [s for s in steps if s in T and s in R]:
        c, pn = _contrib(T, R, s)
        cp = [_contrib(X, R, s)[0] for X in P if s in X]
        tot = T[s]["param_norm"] - R[s]["param_norm"]
        row = {"param_norm_diff": tot, "sum_of_contrib": sum(c.values()), "groups": {}}
        for k in sorted(c, key=lambda k: -abs(c[k])):
            row["groups"][k] = {"contrib": c[k], "pairs": [x.get(k) for x in cp], "rel_sq": (T[s][k] - R[s][k]) / R[s][k]}
        ex = {}
        for k in sorted(x for x in R[s] if x.startswith("expert_sq.")):
            d = np.asarray(T[s][k]) - np.asarray(R[s][k])
            srt = np.sort(d)[::-1]
            ex[k] = {"sum_diff": float(d.sum()), "pos_frac": float((d > 0).mean()),
                     "top8_share": float(srt[:8].sum() / d.sum()) if d.sum() != 0 else None,
                     "pairs_sum_diff": [float((np.asarray(X[s][k]) - np.asarray(R[s][k])).sum()) for X in P if s in X]}
        row["experts"] = ex
        out[s] = row
    return out


def main_groups(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="janus drift groups", description="分组参数范数漂移：param_norm 多涨的部分来自哪一组。")
    ap.add_argument("--test", required=True, help="被测的 metrics.jsonl（带 group_sq.*）")
    ap.add_argument("--ref", required=True, help="参照的 metrics.jsonl")
    ap.add_argument("--pair", action="append", default=[], help="另一份参照：和 --ref 比，当噪声（可重复）")
    ap.add_argument("--steps", default="100,300,1000", help="看哪几步，逗号分隔（默认 100,300,1000）")
    ap.add_argument("--out", type=Path, help="另写 JSON")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    rows = groups(load_metrics(a.test), load_metrics(a.ref), [load_metrics(x) for x in a.pair], _steps(a.steps))
    out = {"test": a.test, "ref": a.ref, "pairs": a.pair, "steps": rows}
    for s, row in rows.items():
        tot = row["param_norm_diff"]
        print(f"== 第 {s} 步：param_norm 差 {tot:+.3e}，各组贡献之和 {row['sum_of_contrib']:+.3e}")
        for k, v in list(row["groups"].items())[:12]:
            print(f"  {k:28s} {v['contrib']:+.3e}  参照对 " + " ".join(f"{x:+.1e}" for x in v["pairs"] if x is not None)
                  + f"  （该组平方和相对差 {v['rel_sq']:+.2e}）")
        for k, v in row["experts"].items():
            print(f"  {k}: 平方和差 {v['sum_diff']:+.3e}，变大的专家占 {v['pos_frac']:.0%}，前 8 个占 "
                  f"{v['top8_share'] if v['top8_share'] is None else round(v['top8_share'], 3)}；参照对 "
                  + " ".join(f"{x:+.2e}" for x in v["pairs_sum_diff"]))
    if a.out:
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- rows

ROW_BUCKETS = [(0, 0, "没出现过"), (1, 10, "1–10 次"), (11, 100, "11–100"), (101, 1000, "101–1000"), (1001, 10 ** 4, "1001–1e4"),
               (10 ** 4 + 1, 10 ** 12, ">1e4")]


def rows(T: dict, R: dict, P: dict | None, cnt: np.ndarray, steps: list[int]) -> dict:
    """T、R、P = load_metrics()（被测、参照、可选的另一份参照），cnt = 每个词出现的次数。返回 {步: {字段: [分桶行]}}。"""
    out = {}
    for s in steps:
        if not all("output_row_sq" in X.get(s, {}) for X in (T, R)):
            continue
        res = {}
        for field in ("output_row_sq", "embedding_row_sq", "output_row_u"):
            t, r = np.asarray(T[s][field]), np.asarray(R[s][field])
            q = np.asarray(P[s][field]) if P and s in P and field in P[s] else None
            rws = []
            for lo, hi, name in ROW_BUCKETS:
                m = (cnt >= lo) & (cnt <= hi)
                if not m.any():
                    continue
                if field == "output_row_u":
                    row = {"bucket": name, "rows": int(m.sum()), "test_mean": float(t[m].mean()), "ref_mean": float(r[m].mean()),
                           "pair_mean": float(q[m].mean()) if q is not None else None}
                else:
                    row = {"bucket": name, "rows": int(m.sum()), "sum_diff": float((t - r)[m].sum()),
                           "pair_sum_diff": float((q - r)[m].sum()) if q is not None else None,
                           "rel": float((t - r)[m].sum() / r[m].sum())}
                rws.append(row)
            res[field] = rws
        out[s] = res
    return out


def main_rows(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="janus drift rows", description="输出层与 embedding 逐行漂移，按词频分桶。")
    ap.add_argument("--test", required=True, help="被测的 metrics.jsonl（带 output_row_sq 等）")
    ap.add_argument("--ref", required=True, help="参照的 metrics.jsonl")
    ap.add_argument("--pair", default=None, help="另一份参照（当噪声）")
    ap.add_argument("--counts", required=True, help="label_counts.npz：每个词在 golden 前 N 步 labels 里出现的次数")
    ap.add_argument("--key", default="c300", help="npz 里用哪一份计数（默认 c300 = 前 300 步）")
    ap.add_argument("--steps", default="100,200,300", help="看哪几步，逗号分隔（默认 100,200,300）")
    ap.add_argument("--out", type=Path, help="另写 JSON")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    P = load_metrics(a.pair) if a.pair else None
    out = rows(load_metrics(a.test), load_metrics(a.ref), P, np.load(a.counts)[a.key], _steps(a.steps))
    for s, res in out.items():
        print(f"== 第 {s} 步")
        for field, rws in res.items():
            print(f"  {field}")
            for row in rws:
                print("    " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    if a.out:
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- 命令行

SUB = {"pairs": main_pairs, "trend": main_trend, "groups": main_groups, "rows": main_rows}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in SUB:
        print("用法：janus drift {pairs|trend|groups|rows} …（各加 --help 看参数）\n\n" + __doc__.split("\n\n")[1],
              file=sys.stdout if argv[:1] in (["-h"], ["--help"]) else sys.stderr)
        return 0 if argv[:1] in (["-h"], ["--help"]) else 2
    return SUB[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
