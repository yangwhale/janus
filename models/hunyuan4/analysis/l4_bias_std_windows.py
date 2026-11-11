"""L4 路由偏置标准差（router.bias_std.<层>）拆开看：逐层、逐窗口的带符号均值，TPU 对 golden vs GPU 两两 28 对。

判定口径与正式检查（router_stats）相同：每个统计的逐步相对差 dₜ = (TPUₜ − GPUₜ) ÷ |GPUₜ|（分母 golden），窗口内取平均
（janus.metrics.series_stats）。正式判定取 7 个统计（6 个 MoE 层 + MTP）里最差的那个，对 N（28 对里同一量的最大值，每对 (A, B) 以 B 为分母、同 janus 的噪声对写法；
每对也取 7 个统计的最差）。这里另外：
- 逐层：每层自己的 TPU |带符号均值| 与每层自己的 N（28 对在这一层的最大值）；
- 逐窗口：每 --win 步一个窗口，TPU 与 N 同窗口同层算；
- 旁证：TPU 对 8 份 GPU 各算一次（中位数 / 最大值），GPU 每份对其余 7 份均值的留一。
目的：看超线是某一层、某一段（像 grad_norm 那样集中在尖峰段的轨迹差），还是从头就有的常量偏移。

    python models/hunyuan4/analysis/l4_bias_std_windows.py --root <运行根> --tpu t1zg-a --gpu g3-a,…,g3-h --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from janus import metrics as M  # noqa: E402


def load(root, run, steps):
    out = {}
    for line in open(os.path.join(root, run, "metrics.jsonl")):
        r = json.loads(line)
        if r["step"] > steps:
            continue
        for k, v in r.items():
            if k.startswith("router.bias_std."):
                out.setdefault(k, {})[r["step"]] = v
    return out


def sm(t, g, a, b):
    """窗口 [a, b]（步号，含两端）的带符号均值；步号平移成第一步 = 0，与 window_stats 相同。"""
    tt = {s - 1: v for s, v in t.items()}
    gg = {s - 1: v for s, v in g.items()}
    return M.series_stats(tt, gg, (a - 1, b), "golden").signed_mean


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--tpu", required=True)
    p.add_argument("--gpu", required=True, help="逗号分隔，第一个是 golden")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--win", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    gl = a.gpu.split(",")
    T = load(a.root, a.tpu, a.steps)
    G = {g: load(a.root, g, a.steps) for g in gl}
    stats = sorted(G[gl[0]])
    pairs = list(itertools.combinations(gl, 2))
    bounds = [(1, a.steps)] + [(s, min(s + a.win - 1, a.steps)) for s in range(1, a.steps + 1, a.win)]
    rows = []
    for lo, hi in bounds:
        per = {}
        for k in stats:
            t = sm(T[k], G[gl[0]][k], lo, hi)
            n = max(abs(sm(G[x][k], G[y][k], lo, hi)) for x, y in pairs)
            each = [sm(T[k], G[g][k], lo, hi) for g in gl]
            loo = []
            for g in gl:
                mean = {s: st.mean(G[h][k][s] for h in gl if h != g) for s in G[g][k]}
                loo.append(sm(G[g][k], mean, lo, hi))
            per[k] = {"tpu_vs_golden": t, "N_layer": n, "r_layer": abs(t) / n if n else None,
                      "tpu_vs_each_median": st.median(each), "tpu_vs_each_max_abs": max(abs(x) for x in each),
                      "gpu_loo_max_abs": max(abs(x) for x in loo)}
        # 正式口径：TPU 取 7 个统计最差，N 取每对 7 个统计最差后的 28 对最大
        tw = max(abs(per[k]["tpu_vs_golden"]) for k in stats)
        nw = max(max(abs(sm(G[x][k], G[y][k], lo, hi)) for k in stats) for x, y in pairs)
        worst_k = max(stats, key=lambda k: abs(per[k]["tpu_vs_golden"]))
        rows.append({"steps": f"{lo}-{hi}", "tpu_worst_abs": tw, "tpu_worst_stat": worst_k, "N_formal": nw, "r_formal": tw / nw,
                     "layers": per})
    full = rows[0]
    over = [k for k, v in full["layers"].items() if v["r_layer"] and v["r_layer"] > 1]
    res = {"what": "L4 路由偏置标准差逐层、逐窗口的带符号均值（TPU 对 golden vs GPU 两两 28 对）", "tpu": a.tpu, "gpu": gl,
           "steps": a.steps, "win": a.win, "denominator": "golden", "full": full, "windows": rows[1:],
           "layers_over_own_N_full": over}
    open(a.out, "w").write(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"全程：TPU 最差 {full['tpu_worst_stat']} {full['tpu_worst_abs']:.3g}，N {full['N_formal']:.3g}，r {full['r_formal']:.2f}")
    for k, v in full["layers"].items():
        print(f"  {k}: TPU {v['tpu_vs_golden']:+.3g}  N_层 {v['N_layer']:.3g}  r_层 {v['r_layer']:.2f}  对 8 份中位 {v['tpu_vs_each_median']:+.3g}  留一最大 {v['gpu_loo_max_abs']:.3g}")
    for r in rows[1:]:
        print(f"  {r['steps']:>9}: TPU 最差 {r['tpu_worst_stat'][16:]} {r['tpu_worst_abs']:.3g}  N {r['N_formal']:.3g}  r {r['r_formal']:.2f}")


if __name__ == "__main__":
    main()
