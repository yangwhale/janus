"""低阈值 QK-Clip（τ 0.1）第 1 步逐层逐头最大 logit 对 CPU FP64，用现行配置的 TPU 运行重测（原测量 tw-qk-a 在打开 M31、M33 之前）。

第 1 步在任何更新之前，最大 logit 只取决于初始权重与第 1 步数据，与阈值无关。FP64 是 step1_ref.py --forward-only --log-max-logit
在同一 16 行上算的每层 8 个头（数值取自 results/t1-l1-kd-qkclip-vs-fp64.json 的 max_logit_step1[*].fp64，与配置无关）。
每层 rel_L2 = ‖TPU 的 8 个头 − FP64‖ ÷ ‖FP64‖；R = rel_L2(TPU) ÷ GPU 两份 rel_L2 的中位数（同原判定），≤ 1.2 过，全部层取最差。
另记（只报）：旧测量 tw-qk-a 的同一量；30 步逐步逐头被裁集合的差（TPU 对 GPU-a，GPU-b 对 GPU-a）；30 步 lm_loss 的 MARE。

  python l1_qkclip_maxlogit.py --tpu mz-qk-a=<metrics.jsonl> --old tw-qk-a=<metrics.jsonl> --gpu g6-qk-a=<…> --gpu g6-qk-b=<…> \
      --fp64-from models/hunyuan4/alignments/01-mini/results/t1-l1-kd-qkclip-vs-fp64.json --out <结果 json>
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np


def load(p):
    return {json.loads(x)["step"]: json.loads(x) for x in open(p)}


def rl(x, f):
    return float(np.linalg.norm(np.asarray(x, np.float64) - f) / np.linalg.norm(f))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tpu", required=True)
    p.add_argument("--old", default=None)
    p.add_argument("--gpu", action="append", required=True)
    p.add_argument("--fp64-from", required=True)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--stage", default="t1-qkclip-maxlogit", help="结果文件的 stage 名（换代时另起，不覆盖旧结果）")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tn, tf = a.tpu.split("=")
    T = load(tf)
    O = load(a.old.split("=")[1]) if a.old else None
    G = {x.split("=")[0]: load(x.split("=")[1]) for x in a.gpu}
    gn = sorted(G)
    ref = json.loads(Path(a.fp64_from).read_text())["max_logit_step1"]
    rows = []
    for r in ref:
        L, f = r["layer"], np.array(r["fp64"], np.float64)
        k = f"qk.max_logit.{L}"
        g = [rl(G[x][1][k], f) for x in gn]
        t = rl(T[1][k], f)
        row = {"layer": L, "fp64": f.tolist(), "tpu": T[1][k], "rel_l2_tpu": t, "rel_l2_gpu": dict(zip(gn, g)),
               "gpu_median": st.median(g), "R": t / st.median(g), "gpu_pair_rel_l2": rl(G[gn[0]][1][k], np.array(G[gn[1]][1][k]))}
        if O:
            row["old_rel_l2_tpu"] = rl(O[1][k], f)
            row["old_R"] = row["old_rel_l2_tpu"] / st.median(g)
        rows.append(row)
    w = max(rows, key=lambda x: x["R"])
    # 30 步逐步逐头被裁集合（只报）
    layers = [r["layer"] for r in rows]
    steps = sorted(set(T) & set(G[gn[0]]))

    def clipped(run, s, L):
        return {i for i, v in enumerate(run[s][f"qk.max_logit.{L}"]) if v > a.tau}

    diff_t = sum(len(clipped(T, s, L) ^ clipped(G[gn[0]], s, L)) for s in steps for L in layers)
    diff_g = sum(len(clipped(G[gn[1]], s, L) ^ clipped(G[gn[0]], s, L)) for s in steps for L in layers)

    def mare(x, y):
        return float(np.mean([abs(x[s]["lm_loss"] - y[s]["lm_loss"]) / abs(y[s]["lm_loss"]) for s in steps]))

    status = "pass" if w["R"] <= 1.2 else "fail"
    chk = {"id": "L1.qkclip.max_logit", "type": "op_replay", "level": "L1", "status": status,
           "threshold": {"kind": "numeric", "values": {"r_ratio": 1.2}, "source": "PROTOCOL.md 第 5 节：对 FP64 参照（CPU 上算，与 TPU 不同源）的 R 比 ≤ 1.2 过"},
           "metrics": {"r_ratio": w["R"], "rel_l2_gpu_vs_ref": w["gpu_median"], "rel_l2_tpu_vs_ref": w["rel_l2_tpu"], "worst_layer": w["layer"],
                       "layer3": {k: v for k, v in rows[layers.index("3")].items() if k in ("rel_l2_tpu", "gpu_median", "R", "old_rel_l2_tpu", "old_R")}},
           "message": f"每层 8 个头的最大 logit 对 CPU FP64：最差 {w['layer']} 层 GPU {w['gpu_median']:.3g}、TPU {w['rel_l2_tpu']:.3g} → R {w['R']:.3f}；"
                      f"第 3 层 GPU {rows[layers.index('3')]['gpu_median']:.3g}、TPU {rows[layers.index('3')]['rel_l2_tpu']:.3g} → R {rows[layers.index('3')]['R']:.3f}",
           "details_inline": {"rows": rows}}
    info = {"id": "L1.qkclip.clipped_heads_30", "type": "op_replay", "level": "L1", "status": "inconclusive", "threshold": {},
            "metrics": {"steps": len(steps), "set_diff_tpu_vs_gpu_a": diff_t, "set_diff_gpu_b_vs_gpu_a": diff_g,
                        "lm_loss_mare_tpu_vs_gpu_a": mare(T, G[gn[0]]), "lm_loss_mare_gpu_b_vs_gpu_a": mare(G[gn[1]], G[gn[0]])},
            "message": f"只报：{len(steps)} 步逐步逐层逐头被裁集合的差 TPU 对 {gn[0]} {diff_t}、{gn[1]} 对 {gn[0]} {diff_g}；"
                       f"lm_loss MARE TPU {mare(T, G[gn[0]]):.3g}、GPU {mare(G[gn[1]], G[gn[0]]):.3g}"}
    res = {"schema_version": 1, "stage": a.stage, "model": "hunyuan4-mini",
           "inputs": {"tpu": tn, "old_tpu": a.old.split("=")[0] if a.old else None, "gpu": gn, "tau": a.tau, "fp64": "t1-l1-kd-qkclip-vs-fp64.json"},
           "checks": [chk, info]}
    res["summary"] = {k: sum(c["status"] == k for c in res["checks"]) for k in ("pass", "fail", "inconclusive", "error")}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for c in res["checks"]:
        print(c["status"], c["message"])


if __name__ == "__main__":
    main()
