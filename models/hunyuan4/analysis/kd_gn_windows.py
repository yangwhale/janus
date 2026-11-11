"""L4 两项（索引器 KD、grad_norm）按窗口拆：TPU 对 8 份 GPU 的带符号均值 vs GPU 两两 28 对的同一量。

判断偏移是第 1 步就有（常量）、某段累积，还是来回摆。带符号均值与正式判定同一口径：逐步相对差
dₜ = (TPUₜ − GPUₜ) / |分母|，分母一律是 GPU（golden；2026-11-10 前 grad_norm 用两边均值）；窗口内取平均。
每个窗口：TPU 对 8 份 GPU 各算一个（中位数 / 最小 / 最大），N = 同窗口 GPU 28 对 |带符号均值| 的最大值，
r = |TPU 中位数| ÷ N；另给 TPU 对 8 份均值、GPU 每份对其余 7 份均值（留一）的同一量，作不依赖配对的旁证。

    python models/hunyuan4/analysis/kd_gn_windows.py --root <运行根目录> --tpu t1zw-a \\
        --gpu g3-a,g3-b,g3-c,g3-d,g3-e,g3-f,g3-g,g3-h --steps 1000 --win 50 --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics as st

import numpy as np

FIELDS = {"indexer_loss": "golden", "grad_norm": "golden"}


def load(root, run, field, steps):
    d = {}
    for line in open(os.path.join(root, run, "metrics.jsonl")):
        r = json.loads(line)
        d[r["step"]] = r[field]
    return np.array([d[s] for s in range(1, steps + 1)], dtype=np.float64)


def rel(a, b, den):
    return (a - b) / (np.abs(b) if den == "golden" else np.abs(a + b) / 2)


def windows(t, g, den, bounds):
    pairs = list(itertools.combinations(sorted(g), 2))
    gm = np.mean([g[x] for x in g], 0)
    rows = []
    for a, b in bounds:
        sl = slice(a - 1, b)
        tv = [float(rel(t, g[x], den)[sl].mean()) for x in sorted(g)]
        gv = [abs(float(rel(g[y], g[x], den)[sl].mean())) for x, y in pairs]
        loo = [float(rel(g[x], np.mean([g[y] for y in g if y != x], 0), den)[sl].mean()) for x in sorted(g)]
        n = max(gv)
        rows.append({
            "steps": f"{a}-{b}", "tpu_vs_each": dict(zip(sorted(g), tv)),
            "tpu_median": st.median(tv), "tpu_min": min(tv), "tpu_max": max(tv),
            "N_gpu28_max": n, "gpu28_median": st.median(gv),
            "r_median": abs(st.median(tv)) / n, "r_max": max(abs(v) for v in tv) / n,
            "tpu_vs_gpu_mean": float(rel(t, gm, den)[sl].mean()), "gpu_loo_min": min(loo), "gpu_loo_max": max(loo),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--tpu", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--win", type=int, default=50)
    p.add_argument("--extra", default="1-1,1-100,1-700,701-1000,1-850,851-1000,101-1000", help="另算的窗口")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    gpus = a.gpu.split(",")
    out = {"what": __doc__.split("\n")[0], "tpu": a.tpu, "gpu": gpus, "steps": a.steps, "fields": {}}
    for f, den in FIELDS.items():
        t = load(a.root, a.tpu, f, a.steps)
        g = {x: load(a.root, x, f, a.steps) for x in gpus}
        win = [(s, min(s + a.win - 1, a.steps)) for s in range(1, a.steps + 1, a.win)]
        extra = [tuple(int(v) for v in w.split("-")) for w in a.extra.split(",") if w]
        full = windows(t, g, den, [(1, a.steps)])[0]
        out["fields"][f] = {
            "denominator": den, "full": full, "windows": windows(t, g, den, win), "extra": windows(t, g, den, extra),
            # 各段对全程（对第一份 GPU）带符号均值的贡献：段内 dₜ 之和 ÷ 总步数
            "contrib_to_full_vs_first_gpu": {
                f"{s}-{e}": float(rel(t, g[gpus[0]], den)[s - 1:e].sum() / a.steps) for s, e in extra},
        }
        print(f, json.dumps({k: (round(v, 8) if isinstance(v, float) else v) for k, v in full.items()
                             if k != "tpu_vs_each"}))
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
