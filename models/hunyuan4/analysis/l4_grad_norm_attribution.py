"""L4 grad_norm 定责证据包：把现行比较（t1zg-a）的 grad_norm 超线拆成窗口，逐段对上同状态单步对 CPU FP64 的结果。

三样输入：
1. 逐步记录：TPU t1zg-a（grad_norm 与 t1zw-a 逐位相同，本脚本核一遍）、GPU g2-a…h（前 300 步窗口）、g3-a…h（1000 步）；
2. 同状态单步（results/t1-kd-gn-same-state.json）：第 1、8、30、100 步，TPU、GPU 从同一个参数状态各算一步，对同一状态的
   CPU FP64 真值；
3. 1000 步分窗（results/t1zw-kd-gn-windows.json，t1zw-a 与 t1zg-a 的 grad_norm 逐位相同，直接引用）。
判定口径同正式检查：逐步相对差 dₜ = (TPUₜ − GPUₜ) ÷ GPUₜ（分母 GPU；2026-11-10 前取两边均值），窗口内带符号均值与 MARE；
N = 同窗口 GPU 两两 28 对的最大值。

  python l4_grad_norm_attribution.py --root <运行根> --results-dir models/hunyuan4/alignments/01-mini/results --out <json>
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics as stt
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from janus import metrics as M  # noqa: E402


def load(root, run, field="grad_norm"):
    d = {}
    for line in open(os.path.join(root, run, "metrics.jsonl")):
        x = json.loads(line)
        d[x["step"]] = x[field]
    return d


def st(t, g, a, b):
    s = M.series_stats({k - 1: v for k, v in t.items()}, {k - 1: v for k, v in g.items()}, (a - 1, b), "golden")
    return s.signed_mean, s.mare


def rel(a, b):
    return (a - b) / abs(b)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--tpu", default="t1zg-a")
    p.add_argument("--tpu-prev", default="t1zw-a")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    T = load(a.root, a.tpu)
    P = load(a.root, a.tpu_prev)
    same_bits = all(T[s] == P[s] for s in T)
    G2 = {f"g2-{c}": load(a.root, f"g2-{c}") for c in "abcdefgh"}
    pairs = list(itertools.combinations(sorted(G2), 2))
    win = []
    for lo, hi in [(1, 300), (1, 10), (11, 50), (51, 100), (101, 300)]:
        t = st(T, G2["g2-a"], lo, hi)
        n = [st(G2[x], G2[y], lo, hi) for x, y in pairs]
        nsm, nm = max(abs(v[0]) for v in n), max(v[1] for v in n)
        each = [st(T, G2[g], lo, hi)[0] for g in sorted(G2)]
        win.append({"steps": f"{lo}-{hi}", "tpu_signed_mean": t[0], "N_signed_mean": nsm, "r_signed_mean": abs(t[0]) / nsm,
                    "tpu_mare": t[1], "N_mare": nm, "r_mare": t[1] / nm, "tpu_vs_8gpu_signed_mean_median": stt.median(each)})
    ss = json.loads((Path(a.results_dir) / "t1-kd-gn-same-state.json").read_text())["summary"]["grad_norm"]
    steps = []
    for s, v in ss.items():
        s = int(s)
        r = [rel(T[s], G2[g][s]) for g in sorted(G2)]
        gp = [abs(rel(G2[x][s], G2[y][s])) for x, y in pairs]
        steps.append({"step": s, "same_state_gpu_vs_fp64": v["gpu_rel"], "same_state_tpu_vs_fp64": v["tpu_rel"], "same_state_R": v["R"],
                      "same_state_tpu_minus_gpu": v["tpu_minus_gpu_rel"], "run_tpu_vs_g2a": r[0], "run_tpu_vs_8gpu_median": stt.median(r),
                      "run_gpu_pairs_max_abs": max(gp)})
    w1000 = json.loads((Path(a.results_dir) / "t1zw-kd-gn-windows.json").read_text())["fields"]["grad_norm"]
    res = {
        "what": "L4 grad_norm 定责证据包（现行比较 t1zg-a；结论 + 数据）",
        "tpu": a.tpu, "grad_norm_bitwise_equal_to": {a.tpu_prev: same_bits},
        "conclusion": [
            "第 1–10 步：TPU 比 GPU 低（前 300 步窗口带符号均值 −6.5e-4 量级），同状态单步第 1、8 步 GPU 对 FP64 偏高 +6.55e-4、+8.96e-4，TPU 只偏 +1.32e-4、+2.06e-4（R 0.20、0.23）；"
            "运行里第 1、8 步 TPU 对 8 份 GPU 的差（中位数）与同状态单步的 TPU − GPU 几乎相同：差在 GPU 一侧。",
            "GPU 的这类偏差是同一状态下各份共有的（同状态单步 GPU 对 FP64 的偏差远大于同一步 GPU 两两的散布），所以 GPU 重跑的尺子量不到它；TPU 离 FP64 更近反而显得超线。",
            "第 11–100 步：差改为正，与同状态第 30 步 GPU 对 FP64 偏低（−5.44e-4，TPU −2.49e-4，R 0.46）同号；第 100 步两边对 FP64 差不多（R 1.29，GPU +1.25e-4、TPU +1.62e-4）。第 101–300 步过线以内。",
            "1000 步的负带符号均值来自第 851–1000 步 grad_norm 尖峰段（见 t1zw-kd-gn-windows.json）：尖峰段的逐步差是两条轨迹的差，不是逐步算错；同状态单步四次里 TPU 三次更近 FP64、一次相当。",
            "结论：grad_norm 的超线不是 TPU 算错。前 10 步的差由 GPU 对 FP64 的偏差定责到 GPU 一侧，其余是轨迹差；没有测量或语义差（口径一致：205 个张量、裁剪前、路由偏置不计入）。",
        ],
        "windows_300": win,
        "same_state_vs_run": steps,
        "windows_1000_ref": {"file": "t1zw-kd-gn-windows.json", "full": w1000["full"],
                             "spike_segment": [w for w in w1000["windows"] if w["steps"] in ("851-900", "901-950", "951-1000")]},
        "sources": {"same_state": "t1-kd-gn-same-state.json", "windows_1000": "t1zw-kd-gn-windows.json",
                    "formal": ["t1zga-l4.results.json:L4.grad_norm", "t1zga300-l4.results.json:L4.grad_norm"]},
    }
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for w in win:
        print(w["steps"], f"sm {w['tpu_signed_mean']:+.2e} N {w['N_signed_mean']:.2e} r {w['r_signed_mean']:.2f} | mare {w['tpu_mare']:.2e} N {w['N_mare']:.2e} r {w['r_mare']:.2f}")
    for s in steps:
        print(s)
    print("bitwise", same_bits)


if __name__ == "__main__":
    main()
