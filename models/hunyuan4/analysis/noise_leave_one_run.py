#!/usr/bin/env python3
"""留一份法：每份 GPU 重跑 x（除基线 g3-a）当被测对 g3-a 比，噪声上限用不含 x 的其余运行两两（含 g3-a），
r = |x 的判定量| ÷ 上限；对每条曲线取各份 x 的最大 r。判定量同 janus loss_series（逐步相对差，分母一律取 golden；2026-11-10 前 grad_norm 取两份均值）。"""

import itertools
import json
import sys

# 用法：python noise_leave_one_run.py <运行根目录> <份数 n（g3-a 起按字母取）> <窗口起> <窗口止+1> <输出 json>
root, n, w0, w1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
runs = [f"g3-{c}" for c in "abcdefghijklmnop"[:n]]
M = {r: {d["step"]: d for d in map(json.loads, open(f"{root}/{r}/metrics.jsonl"))} for r in runs}
fields = {
    "lm_loss": "golden",
    "mtp_loss_mt": "golden",
    "indexer_loss": "golden",
    "grad_norm": "golden",
    "param_norm_fp64": "golden",
}


def stats(g, t, f, den):
    d = []
    for s in range(w0, w1):
        a, b = M[g][s][f], M[t][s][f]
        d.append((b - a) / (a if den == "golden" else (a + b) / 2))
    return {"signed_mean": abs(sum(d) / len(d)), "mare": sum(map(abs, d)) / len(d), "max_step": max(map(abs, d))}


out = {}
for f, den in fields.items():
    P = {(u, v): stats(u, v, f, den) for u, v in itertools.combinations(runs, 2)}
    worst = {}
    for x in runs[1:]:
        ref = [s for (u, v), s in P.items() if x not in (u, v)]
        up = {q: max(s[q] for s in ref) for q in ("signed_mean", "mare", "max_step")}
        st = stats("g3-a", x, f, den)
        r = {q: st[q] / up[q] for q in up}
        worst[x] = max(r.values())
    xm = max(worst, key=worst.get)
    out[f] = {"max_r": worst[xm], "at": xm, "per_run": worst}
print(json.dumps({k: (round(v["max_r"], 3), v["at"]) for k, v in out.items()}))
json.dump(
    {"runs": n, "window": [w0, w1 - 1], "definition": __doc__, "fields": out},
    open(sys.argv[5], "w"),
    indent=1,
    ensure_ascii=False,
)
