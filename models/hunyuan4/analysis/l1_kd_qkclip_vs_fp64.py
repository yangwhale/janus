"""两项 L1 诊断改对 CPU FP64 参照（原来对 TPU 同一图的 FP32 + HIGHEST，被测一侧产出）。

1. 索引器 KD 的输入（第 1 步、第一行 = GPU rank 0 第 0 个 microbatch = TPU 第 0 行）：教师 q、教师 k（注意力的 q / k，
   RoPE 之后、缩放之前；MaxText 抓的 q 已乘缩放，除回去）与索引分数（ReLU(q·k) 按头加权求和，只比因果部分）。第 0 层三方都有；第 1、5 层与 MTP 层只有
   TPU 与 FP64。GPU 一侧是 KD 诊断抓数（`--diag-kd-dump`，台账 G16）：教师 q / k 是 GPU 算出的 BF16；索引分数由 GPU 抓到的
   索引器 q / k / w 在 FP64 里按同一公式重算（GPU 自己的打分是真 FP32，`indexer_fp32_scores`）。
2. 低阈值 QK-Clip 第 1 步的逐层逐头最大 logit（缩放后的 q·k，只在选中且因果的位置上、不含 sink，16 行取最大）。
   第 1 步是更新之前，最大 logit 只取决于初始权重与第 1 步数据，与阈值无关；FP64 由 `step1_ref.py --forward-only
   --log-max-logit` 在同一 16 行上算。

R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64)。

    python models/hunyuan4/analysis/l1_kd_qkclip_vs_fp64.py --fp64 <step1_ref.py --forward-only 的目录> \\
        --gpu-kd <kd_layer0_step1_rank0_micro0.pt> --tpu-kd t1zdiag2=<diag_kd_inputs.safetensors> ... \\
        --qk <results/t1-qk-clip-tpu.json> --gpu-qk-metrics g6-qk-a=<metrics.jsonl> g6-qk-b=<metrics.jsonl> --out <json>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

TPU_PREFIX = {  # hy4-mini 的 MaxText 抓数前缀（层名 → 前缀）
    "0": "decoder/dense_layers",
    "1": "decoder/moe_layers/layers_0",
    "5": "decoder/moe_layers_remainder/layers_0",
    "mtp0": "mtp_block/mtp_layer_1/mtp_1_transformer_layer",
}


def rel(a, b):
    a, b = a.to(torch.float64), b.to(torch.float64)
    return float((a - b).norm() / b.norm())


def index_scores(q, k, w):
    """[S, 1, H, D], [S, 1, D], [S, 1, H] → [S, S]（Megatron-Core _compute_index_scores 的公式，FP64）。"""
    s = torch.einsum("shd,td->sht", q[:, 0].double(), k[:, 0].double())
    return (torch.relu(s) * w[:, 0].double()[..., None]).sum(1)


def causal(x):
    S = x.shape[-1]
    return x[torch.ones(S, S, dtype=torch.bool).tril()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp64", required=True)
    p.add_argument("--gpu-kd", required=True)
    p.add_argument("--tpu-kd", nargs="+", required=True, help="名字=diag_kd_inputs.safetensors")
    p.add_argument("--qk", required=True, help="results/t1-qk-clip-tpu.json（TPU 与 GPU-a 的逐头值）")
    p.add_argument(
        "--gpu-qk-metrics", nargs="+", default=[], help="名字=GPU 低阈值 QK-Clip 运行的 metrics.jsonl（第 1 步）"
    )
    p.add_argument("--old-qk-ref", default="tw-qk-ref")
    p.add_argument("--note", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ch = safe_open(os.path.join(a.fp64, "choices.safetensors"), "pt")
    summ = json.load(open(os.path.join(a.fp64, "summary.json")))
    out = {
        "what": "L1 诊断两项对 CPU FP64 参照：索引器 KD 的输入（第 1 步第一行）、低阈值 QK-Clip 第 1 步的逐层逐头最大 logit",
        "reference": {
            "kind": "CPU FP64，GPU golden 同一套 Megatron-Core 代码（frameworks/megatron/step1_ref.py --forward-only）",
            "rows": summ["rows"],
            "code": summ["code"],
            "audit": summ.get("audit"),
        },
        "kd_inputs": {},
        "max_logit_step1": [],
    }
    # ---- KD 输入
    g = torch.load(a.gpu_kd, weights_only=False)
    f64 = {}
    for L in TPU_PREFIX:
        if f"r0.dsa.{L}.query" not in ch.keys():
            continue
        f64[L] = {
            "teacher_q": ch.get_tensor(f"r0.dsa.{L}.query")[:, 0],
            "teacher_k": ch.get_tensor(f"r0.dsa.{L}.key")[:, 0],
            "indexer_score": causal(
                index_scores(
                    ch.get_tensor(f"r0.dsa.{L}.q"), ch.get_tensor(f"r0.dsa.{L}.k"), ch.get_tensor(f"r0.dsa.{L}.w")
                )
            ),
        }
    gpu0 = {
        "teacher_q": g["query"][:, 0],
        "teacher_k": g["key"][:, 0],
        "indexer_score": causal(index_scores(g["indexer_q"], g["indexer_k"], g["indexer_w"])),
    }
    for L, ref in f64.items():
        ent = {}
        if L == "0":
            ent["gpu_g1p-kdd"] = {k: rel(gpu0[k], ref[k]) for k in ref}
        for spec in a.tpu_kd:
            name, path = spec.split("=", 1)
            t = safe_open(path, "pt")
            pre = TPU_PREFIX[L] + "/self_attention/debug_kd_"
            S = ref["teacher_q"].shape[0]
            # MaxText 抓的教师 q 已乘 softmax 缩放（1/√256 = 0.0625），除回去再比
            tq = t.get_tensor(pre + "query/0").reshape(S, -1, ref["teacher_q"].shape[-1]) / g["softmax_scale"]
            tk = t.get_tensor(pre + "key/0").reshape(S, -1, ref["teacher_k"].shape[-1])
            ts = causal(t.get_tensor(pre + "indexer_score/0").reshape(S, S))
            ent[name] = {
                "teacher_q": rel(tq, ref["teacher_q"]),
                "teacher_k": rel(tk, ref["teacher_k"]),
                "indexer_score": rel(ts, ref["indexer_score"]),
            }
            if L == "0":
                ent[name]["R_vs_gpu"] = {
                    k: ent[name][k] / ent["gpu_g1p-kdd"][k] for k in ("teacher_q", "teacher_k", "indexer_score")
                }
        out["kd_inputs"][L] = ent
    out["kd_inputs_metric"] = "rel_L2 对 FP64（第一行；索引分数只比因果部分）；R = TPU ÷ GPU（只有第 0 层有 GPU）"
    # ---- QK-Clip 最大 logit
    qk = json.load(open(a.qk))
    extra = {}
    for spec in a.gpu_qk_metrics:
        name, path = spec.split("=", 1)
        m = json.loads(open(path).readline())
        assert m["step"] == 1
        extra[name] = {k[len("qk.max_logit.") :]: v for k, v in m.items() if k.startswith("qk.max_logit.")}
    for row in qk["max_logit_step1"]:
        L = row["layer"]
        ref = torch.tensor(summ["max_logit"][L], dtype=torch.float64)
        ent = {
            "layer": L,
            "fp64": ref.tolist(),
            "tpu": rel(torch.tensor(row["tpu_max_logit"]), ref),
            "gpu_a_from_result": rel(torch.tensor(row["gpu_max_logit"]), ref),
            "old_ref_vs_fp64": rel(torch.tensor(row["ref_max_logit"]), ref),
            "old_R": row["R"],
        }
        for name, vals in extra.items():
            ent[name] = rel(torch.tensor(vals[L]), ref)
        gpus = [v for k, v in ent.items() if k in extra] or [ent["gpu_a_from_result"]]
        ent["gpu_median"] = float(np.median(gpus))
        ent["R"] = ent["tpu"] / ent["gpu_median"]
        out["max_logit_step1"].append(ent)
    out["max_logit_metric"] = (
        "每层 8 个头的向量，rel_L2 对 FP64；R = TPU ÷ GPU（GPU 取各份的中位数）；old_R 是对 TPU FP32 参照（tw-qk-ref）的旧值"
    )
    out["max_logit_worst"] = max(out["max_logit_step1"], key=lambda e: e["R"])
    out["note"] = a.note
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out["kd_inputs"], ensure_ascii=False))
    print(json.dumps([(e["layer"], round(e["R"], 3), round(e["old_R"], 3)) for e in out["max_logit_step1"]]))


if __name__ == "__main__":
    main()
