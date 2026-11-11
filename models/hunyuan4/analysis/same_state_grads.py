"""同一份权重上 TPU 与 GPU 的逐张量梯度：两边差多少、超出 GPU 重跑噪声几倍，可选再对 CPU FP64 参照。

输入：
- `--tpu <运行目录>`：TPU `--diag-step1` 的运行目录（diag_step1.json + diag_grads.safetensors）；按 FP64 参照或 GPU 的
  梯度范数自动判断要不要乘全局有效 token 数（gradient_accumulation_steps=1 时诊断梯度小这个倍数，同 step1_base2_ab_vs_fp64.py）；
- `--gpu 名字=<export_hf_grads.py 的 safetensors>`（两份以上）；
- `--ref <step1_ref.py 的输出目录>`（可选）。

逐张量（HF 名字，两边都有的；路由专家 TPU 没存，不在内）：
- over_N = rel(TPU, 最近的一份 GPU) ÷ GPU 两两最大差（同一分母 ‖GPU 均值‖）；
- 给了 FP64 时另算 rel(TPU, FP64)、中位数 rel(GPU, FP64) 与 R。
按模块、按层汇总，另列 over_N 最大的若干个张量与各组梯度平方和。

    python same_state_grads.py --tpu <b2d-off151> --gpu r150-a=<…> r150-b=<…> [--ref <s151-fp64>] --out <json>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open
from step1_base2_ab_vs_fp64 import layer_of
from step1_vs_fp64 import module_of


def group_of(h: str) -> str:
    """与 train_hy4.grad_group 相同的分组。"""
    if h.startswith("model.mtp_layers."):
        return "mtp"
    if h in ("model.embed_tokens.weight", "lm_head.weight"):
        return "embed_head"
    if ".mlp.experts." in h:
        return "experts"
    if ".hc_attn_layer." in h or ".hc_mlp_layer." in h or h.startswith("model.hc_head."):
        return "hc"
    if ".indexer." in h:
        return "indexer"
    return "rest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpu", required=True)
    ap.add_argument("--gpu", nargs="+", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    diag = json.load(open(os.path.join(a.tpu, "diag_step1.json")))
    tf = safe_open(os.path.join(a.tpu, "diag_grads.safetensors"), "pt")
    gf = {k: safe_open(v, "pt") for k, v in (x.split("=", 1) for x in a.gpu)}
    ref = safe_open(os.path.join(a.ref, "grads_hf.safetensors"), "pt") if a.ref else None
    refsum = json.load(open(os.path.join(a.ref, "summary.json"))) if a.ref else None
    # 缩放：TPU 诊断梯度范数 × N 是否接近 GPU 的总范数（GPU 文件里的平方和）
    names = sorted(set(tf.keys()).intersection(*[set(f.keys()) for f in gf.values()]))
    t_sq = sum(diag["per_tensor"][h]["sq"] for h in diag["per_tensor"])
    g_sq_common = sum(float(next(iter(gf.values())).get_tensor(h).double().square().sum()) for h in names)
    t_sq_common = sum(diag["per_tensor"][h]["sq"] for h in names)
    scale = 1.0
    for cand in (refsum["global_tokens"] if refsum else None, 65453.0):
        if cand and abs(t_sq_common * cand * cand / g_sq_common - 1) < 0.2:
            scale = cand
            break
    rows = []
    for h in names:
        t = tf.get_tensor(h).to(torch.float64) * scale
        gs = {k: f.get_tensor(h).to(torch.float64) for k, f in gf.items()}
        gm = sum(gs.values()) / len(gs)
        den = float(gm.norm())
        if den == 0:
            continue
        gk = list(gs)
        noise = max(float((gs[i] - gs[j]).norm()) / den for x, i in enumerate(gk) for j in gk[x + 1 :])
        near = min(float((t - g).norm()) / den for g in gs.values())
        r = {
            "tensor": h,
            "module": module_of(h),
            "layer": layer_of(h),
            "group": group_of(h),
            "numel": int(t.numel()),
            "gpu_norm": den,
            "tpu_over_gpu_norm": float(t.norm()) / den,
            "tpu_vs_gpu_min": near,
            "gpu_noise_max": noise,
            "over_N": near / noise if noise > 0 else float("inf"),
        }
        if ref is not None and h in ref.keys():
            f64 = ref.get_tensor(h).to(torch.float64)
            dr = float(f64.norm())
            if dr > 0 and tuple(f64.shape) == tuple(t.shape):
                ge = [float((g - f64).norm()) / dr for g in gs.values()]
                r.update(
                    {
                        "tpu_vs_ref": float((t - f64).norm()) / dr,
                        "gpu_vs_ref": float(np.median(ge)),
                        "R": float((t - f64).norm()) / dr / float(np.median(ge)) if np.median(ge) > 0 else float("nan"),
                        "tpu_over_ref_norm": float(t.norm()) / dr,
                        "gpu_over_ref_norm": float(np.median([float(g.norm()) / dr for g in gs.values()])),
                    }
                )
        rows.append(r)

    def summ(rs):
        o = {
            "n": len(rs),
            "over_N_median": float(np.median([r["over_N"] for r in rs])),
            "n_over_N_gt_2": sum(r["over_N"] > 2 for r in rs),
            "tpu_vs_gpu_median": float(np.median([r["tpu_vs_gpu_min"] for r in rs])),
            "gpu_noise_median": float(np.median([r["gpu_noise_max"] for r in rs])),
        }
        rr = [r for r in rs if "R" in r]
        if rr:
            o.update(
                {
                    "tpu_vs_ref_median": float(np.median([r["tpu_vs_ref"] for r in rr])),
                    "gpu_vs_ref_median": float(np.median([r["gpu_vs_ref"] for r in rr])),
                    "R_median": float(np.median([r["R"] for r in rr])),
                    "n_R_gt_1.2": sum(r["R"] > 1.2 for r in rr),
                }
            )
        return o

    sq = {}
    for h in names:
        g = group_of(h)
        sq.setdefault(g, {"tpu": 0.0, **{k: 0.0 for k in gf}})
        sq[g]["tpu"] += diag["per_tensor"][h]["sq"] * scale * scale
        for k, f in gf.items():
            sq[g][k] += float(f.get_tensor(h).double().square().sum())
    out = {
        "what": "同一份权重上 TPU 与 GPU 的逐张量梯度（路由专家除外），可选对 CPU FP64",
        "tpu_run": os.path.basename(os.path.normpath(a.tpu)),
        "gpu_runs": list(gf),
        "tpu_scale": scale,
        "tpu_grad_norm_all_scaled": float(np.sqrt(t_sq)) * scale,
        "reference": {k: refsum.get(k) for k in ("lm_loss", "grad_norm", "global_tokens", "audit")} if refsum else None,
        "group_sq_common_tensors": sq,
        "all": summ(rows),
        "by_module": {m: summ([r for r in rows if r["module"] == m]) for m in sorted({r["module"] for r in rows})},
        "by_layer": {m: summ([r for r in rows if r["layer"] == m]) for m in sorted({r["layer"] for r in rows})},
        "largest_over_N": sorted(rows, key=lambda r: -r["over_N"])[:20],
        "rows": rows,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(
        json.dumps({k: out[k] for k in ("tpu_scale", "all", "group_sq_common_tensors")}, ensure_ascii=False, indent=1)
    )
    for m, v in out["by_module"].items():
        print(m, json.dumps(v, ensure_ascii=False))
    for r in out["largest_over_N"][:12]:
        print(f"{r['over_N']:.2f} tpu/gpu norm {r['tpu_over_gpu_norm']:.4f} {r['tensor']}")


if __name__ == "__main__":
    main()
