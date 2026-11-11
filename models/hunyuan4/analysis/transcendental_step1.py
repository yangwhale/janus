"""MaxText 开关 transcendental_accuracy_highest（FP32 exp / log / sigmoid / silu 用 accuracy=HIGHEST）对 T1 第 1 步的影响。

两件事，各自「开关关」与「开关开」两组并列（每组：被测 = T1 配置的第 1 步诊断，参照 = 同一程序 dtype float32、matmul_precision
highest 的第 1 步诊断；开关开的一组参照也开开关，免得参照自己带着近似 exp / log）：
1. 第 0 层索引器 KD 第 0 行（GPU rank 0 micro 0 = TPU 第 0 行）：TPU 图内算出的 KL 行均值对「同一批图内输入（debug_sow_indexer 记下的
   教师 q / k、索引器分数与掩码）在 CPU 上 FP64 重算」的相对差，以及 TPU log_softmax 反推的 log-sum-exp 对 FP64 的带符号平均 ULP
   （results/t1-layer0-kd-attribution.json 的口径：开关关时 −1.155e-3、−0.95 格）；
2. 第 1 步全部参数梯度：R = rel_L2(TPU, 参照) ÷ rel_L2(GPU, 参照)（GPU 取 g2s-a / b / c 三份的中位数，同 t1v-step1-grad-vs-ref.json），
   重点看 iHC 的 hc_base / hc_scale 与第 3 层 FFN 一侧的 hc_pre.hc_base。

  python models/hunyuan4/analysis/transcendental_step1.py --off <t1vdiag> --off-ref <t1diag18b> --off-kd <t1diag15> \\
      --on <tw-tadiag> --on-ref <tw-taref> --gpu-grads <g2sgrad 目录> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

S, H, DH = 4096, 8, 256


def kd_check(run: str) -> dict:
    """KD 第 0 行：TPU 图内 KL 行均值 vs FP64 重算（同一批图内输入）；log-sum-exp 的带符号平均 ULP。"""
    import torch
    from safetensors.numpy import load_file

    T = load_file(f"{run}/diag_kd_inputs.safetensors")
    k0 = [k for k in T if "dense_layers" in k]

    def get(n):
        c = [k for k in k0 if n in k]
        assert len(c) == 1, (n, c)
        return T[c[0]]

    q = torch.from_numpy(get("debug_kd_query").reshape(S, H, DH).astype(np.float64)) * 16
    k = torch.from_numpy(get("debug_kd_key").reshape(S, H, DH).astype(np.float64))
    s = torch.from_numpy(get("debug_kd_indexer_score").reshape(S, S).astype(np.float64))
    m = torch.from_numpy(get("debug_kd_indexer_mask").reshape(S, S) > -1e20) & torch.ones(S, S, dtype=torch.bool).tril()
    tl = torch.from_numpy(get("debug_kd_log_indexer").reshape(S, S).astype(np.float64))
    tr = torch.from_numpy(get("debug_kd_kl_rows").reshape(S).astype(np.float64))
    att = torch.zeros(S, S, dtype=torch.float64)
    for h in range(H):
        sc = (q[:, h] @ k[:, h].T) * 0.0625
        att += torch.nan_to_num(torch.softmax(sc.masked_fill(~m, float("-inf")), -1), nan=0.0)
    p = att / att.sum(-1, keepdim=True).clamp_min(1e-10)
    logq = torch.log_softmax(s.masked_fill(~m, float("-inf")), -1)
    kl = torch.where(m & (p > 0), p * (torch.log(p.clamp_min(1e-10)) - logq), torch.zeros_like(p)).sum(-1)
    lse64 = torch.logsumexp(s.masked_fill(~m, float("-inf")), -1)
    idx = m.float().argmax(-1)
    r = torch.arange(S)
    lse_t = s[r, idx] - tl[r, idx]
    ulp = torch.from_numpy(np.spacing(lse64.numpy().astype(np.float32)).astype(np.float64))
    e = (lse_t - lse64) / ulp
    return {"run": Path(run).name, "kd_row0_tpu_in_graph": float(tr.mean()), "kd_row0_fp64_same_inputs": float(kl.mean()),
            "kd_row0_rel": float(tr.mean() / kl.mean() - 1), "lse_ulp_signed_mean": float(e.mean()),
            "lse_frac_rows_positive": float((e > 0).float().mean())}


def grads_check(tpu: str, ref: str, gpu_dir: str) -> dict:
    from safetensors.numpy import load_file

    t = load_file(f"{tpu}/diag_grads.safetensors")
    rf = load_file(f"{ref}/diag_grads.safetensors")
    gpu = {r: load_file(f"{gpu_dir}/{r}.safetensors") for r in ("g2s-a", "g2s-b", "g2s-c")}

    def rel(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

    rows = []
    for h, g in gpu["g2s-a"].items():
        if h not in t or h not in rf:
            continue
        tt, rr = t[h], rf[h]
        if tt.shape != g.shape:
            tt, rr = tt[: g.shape[0]], rr[: g.shape[0]]
        if tt.shape != g.shape:
            continue
        gref = float(np.median([rel(gpu[x][h], rr) for x in gpu]))
        rows.append({"tensor": h, "tpu_vs_ref": rel(tt, rr), "gpu_vs_ref": gref, "R": rel(tt, rr) / gref if gref else None})
    R = [r["R"] for r in rows if r["R"] is not None]
    hc = {r["tensor"]: r["R"] for r in rows if (".hc_base" in r["tensor"] or ".hc_scale" in r["tensor"] or "hc_head" in r["tensor"])}
    focus = "model.layers.3.hc_mlp_layer.hc_pre.hc_base"
    return {"tpu": Path(tpu).name, "ref": Path(ref).name, "n_tensors": len(rows), "R_median": float(np.median(R)),
            "R_max": max(R), "n_R_gt_1.2": sum(x > 1.2 for x in R),
            "R_gt_1.2": sorted([(r["tensor"], r["R"]) for r in rows if r["R"] and r["R"] > 1.2], key=lambda x: -x[1]),
            "focus_layer3_ffn_hc_base": next((r for r in rows if r["tensor"] == focus), None), "hc_gate_params_R": hc}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--off", "--off-ref", "--off-kd", "--on", "--on-ref", "--gpu-grads", "--out"):
        p.add_argument(k, required=True)
    a = p.parse_args(argv)
    out = {
        "what": "transcendental_accuracy_highest（MaxText e8dee18f，默认关）对 T1 第 1 步的影响：开关关 / 开两组，各自对同设置的 FP32 参照",
        "kd_row0_layer0": {"off": kd_check(a.off_kd), "on": kd_check(a.on)},
        "step1_grads": {"off": grads_check(a.off, a.off_ref, a.gpu_grads), "on": grads_check(a.on, a.on_ref, a.gpu_grads)},
        "rule": "KD：TPU 图内对 FP64 的相对差应与 GPU 记下的值同一量级（+1.9e-4）；梯度：R ≤ 1.2（PROTOCOL.md 第 5 节）",
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    s = out["step1_grads"]
    print(json.dumps({"kd_rel": {k: v["kd_row0_rel"] for k, v in out["kd_row0_layer0"].items()},
                      "lse_ulp": {k: v["lse_ulp_signed_mean"] for k, v in out["kd_row0_layer0"].items()},
                      "grads": {k: {x: v[x] for x in ("R_median", "R_max", "n_R_gt_1.2")} for k, v in s.items()},
                      "focus": {k: (v["focus_layer3_ffn_hc_base"] or {}).get("R") for k, v in s.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
