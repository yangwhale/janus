"""mini 后段飘移的单因素交叉：每格几份 TPU 1000 步，对 GPU 16 份逐步均值的主 loss 偏差，与对照格做两样本比较。

每份运行算三个量（对 GPU 均值的逐步相对差）：第 51–100 步均值（前段鼓包）、第 551–1000 步均值（后段）、
第 551–1000 步 25 步分块对步数的斜率（每 100 步）。每格报均值、范围、「几份里几份尾部还在」
（尾部在：第 751–1000 步 ≥ 以往 70 份 TPU 运行的最小值 +3.0e-6，`mini-main-loss-drift.json` 的 variants）。

两样本比较（每格对对照格）：Welch t（均值差 ÷ 合并标准误，自由度按 Welch–Satterthwaite）与置换检验的双侧 p
（两组合起来、所有分组方式穷举；组小时就是精确值）。份数少时检验力弱，p 大不等于「没区别」。

    python mini_factorial_cells.py --root <运行目录根> --control 名字=run1,run2,… --cell 名字=run1,… [...] --out <json>

`--control gpu-loo`：对照换成 GPU 16 份的留一法（每份 GPU 对其余 15 份均值），用来判 GPU 一侧的变体（GPU 对 GPU）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np
from mini_loss_isolation import BLOCK, STEPS, blocks, late_slope, series

TAIL_THRESHOLD = 3.0e-6  # 以往 70 份 TPU 运行第 751–1000 步的最小值
GPU = [f"g3-{c}" for c in "abcdefghijklmnop"]


def _summ(rel: np.ndarray) -> dict:
    return {
        "lm_51_100": float(rel[50:100].mean()),
        "lm_551_1000": float(rel[550:].mean()),
        "lm_751_1000": float(rel[750:].mean()),
        "late_slope_per100": late_slope(blocks(rel)),
    }


def gpu_loo(root: str, field: str = "lm_loss") -> dict:
    g = np.array([series(root, x, field) for x in GPU])
    out = {}
    for i, name in enumerate(GPU):
        rest = np.delete(g, i, 0).mean(0)
        out[name] = _summ((g[i] - rest) / rest)
    return out


def per_run(root: str, run: str, gm: np.ndarray, field: str = "lm_loss") -> dict:
    rel = (series(root, run, field) - gm) / gm
    return {
        "lm_51_100": float(rel[50:100].mean()),
        "lm_551_1000": float(rel[550:].mean()),
        "lm_751_1000": float(rel[750:].mean()),
        "late_slope_per100": late_slope(blocks(rel)),
    }


def welch(a: list[float], b: list[float]) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    t = (a.mean() - b.mean()) / se
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return {"diff": float(a.mean() - b.mean()), "se": se, "t": float(t), "df": float(df)}


def permutation_p(a: list[float], b: list[float]) -> float:
    allv = np.asarray(a + b)
    obs = abs(np.mean(a) - np.mean(b))
    n, hits, total = len(a), 0, 0
    for idx in itertools.combinations(range(len(allv)), n):
        m = np.zeros(len(allv), bool)
        m[list(idx)] = True
        total += 1
        hits += abs(allv[m].mean() - allv[~m].mean()) >= obs - 1e-15
    return hits / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--control", required=True, help="名字=逗号分隔的运行")
    ap.add_argument("--cell", nargs="+", required=True, help="名字=逗号分隔的运行")
    ap.add_argument(
        "--field",
        default="lm_loss",
        help="逐步记录里的字段（默认主 loss；MTP 用 mtp_loss_mt）。「尾部在」的阈值只对主 loss 有意义",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gm = np.array([series(a.root, x, a.field) for x in GPU]).mean(0)
    loo = a.control == "gpu-loo"
    cells = dict(s.split("=", 1) for s in ([] if loo else [a.control]) + a.cell)
    cname = "gpu-loo" if loo else a.control.split("=", 1)[0]
    out = {
        "what": "单因素交叉：每格 TPU 1000 步对 GPU 16 份均值的主 loss 偏差，对对照格做两样本比较",
        "tail_threshold_751_1000": TAIL_THRESHOLD,
        "block": BLOCK,
        "steps": STEPS,
        "field": a.field,
        "control": cname,
        "cells": {},
    }
    vals = {}
    if loo:
        vals[cname] = gpu_loo(a.root, a.field)
        lv = list(vals[cname].values())
        out["cells"][cname] = {
            "runs": vals[cname],
            "n": len(lv),
            "n_tail": int(sum(v["lm_751_1000"] >= TAIL_THRESHOLD for v in lv)),
        }
        for k in ("lm_51_100", "lm_551_1000", "lm_751_1000", "late_slope_per100"):
            v = [x[k] for x in lv]
            out["cells"][cname][k] = {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v))}
    for name, runs in cells.items():
        rs = runs.split(",")
        pr = {r: per_run(a.root, r, gm, a.field) for r in rs}
        vals[name] = pr
        cell = {"runs": pr}
        for k in ("lm_51_100", "lm_551_1000", "lm_751_1000", "late_slope_per100"):
            v = [pr[r][k] for r in rs]
            cell[k] = {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v))}
        cell["n_tail"] = int(sum(pr[r]["lm_751_1000"] >= TAIL_THRESHOLD for r in rs))
        cell["n"] = len(rs)
        out["cells"][name] = cell
    for name in out["cells"]:
        if name == cname:
            continue
        cmp = {}
        for k in ("lm_51_100", "lm_551_1000", "late_slope_per100"):
            x = [v[k] for v in vals[name].values()]
            y = [v[k] for v in vals[cname].values()]
            cmp[k] = {**welch(x, y), "perm_p": permutation_p(x, y)}
        out["cells"][name]["vs_control"] = cmp
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for name, c in out["cells"].items():
        line = (
            f"{name:10s} n={c['n']} 尾部在 {c['n_tail']}/{c['n']} | 51-100 {c['lm_51_100']['mean']:+.2e}"
            f" | 551-1000 {c['lm_551_1000']['mean']:+.2e} [{c['lm_551_1000']['min']:+.1e}, {c['lm_551_1000']['max']:+.1e}]"
            f" | 斜率 {c['late_slope_per100']['mean']:+.2e}"
        )
        if "vs_control" in c:
            v = c["vs_control"]["lm_551_1000"]
            e = c["vs_control"]["lm_51_100"]
            line += (
                f" || 551-1000 差 {v['diff']:+.2e} t {v['t']:+.2f} p_perm {v['perm_p']:.3f}"
                f" | 51-100 差 {e['diff']:+.2e} t {e['t']:+.2f} p_perm {e['perm_p']:.3f}"
            )
        print(line)


if __name__ == "__main__":
    main()
