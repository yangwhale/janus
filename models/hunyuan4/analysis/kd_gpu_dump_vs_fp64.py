"""GPU 实抓的索引器 KD 输入对 CPU FP64 参照：同一行（第 1 步第 0 行）逐个输入换成 GPU 的那份，看 KD 移多少。

为什么：06-rw3 第 1 步索引器 KD 对 FP64，GPU 第 0 层 +1.53e-3、第 1 层 +2.20e-3。`kd_bf16_vs_fp64.py` 用 CPU 上的
BF16 前向（GPU 写法）复现并拆分；这里用 GPU 真实抓到的输入（`train_hy4mg.py --diag-kd-dump`：rank 0 第 0 个
microbatch = 全局批第 0 行，每个产出 KD 的层存教师 query / key、索引器 q / k / w、DSA top-k）直接核同一套拆分。

FP64 一侧：`step1_ref.py` 的 `choices.safetensors`（`--detail-rows 0`，r0.dsa.<层>.*）。
KL 用 `kd_bf16_vs_fp64.teacher / student / kl_sum`（FP64 算术，FP64 的 top-k；与原版函数核对见输出的 ref_orig_fn）；
`gpu` 一项是 GPU 的输入 + GPU 的 top-k + FP32 原版函数（即 GPU 记下的那个数的同一算法）。

用法（Megatron-Core 源码在 PYTHONPATH）：
    python kd_gpu_dump_vs_fp64.py --gpu-dir <含 kd_layer*_step1_rank0_micro0.pt 的目录> \
        --choices <choices.safetensors> --out kd_gpu_dump_vs_fp64.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "frameworks", "megatron"))

from kd_bf16_vs_fp64 import ROPE, kl_sum, mix, rel_l2, student, teacher, topk_overlap  # noqa: E402
from kd_input_rounding import kd_one  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-dir", required=True)
    p.add_argument("--choices", required=True)
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--layers", default="0,1,mtp0")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    from safetensors import safe_open
    from step1_ref import relax_mask_dtype

    relax_mask_dtype(torch.float64)
    f64 = torch.float64
    f = safe_open(a.choices, "pt")
    out = {"gpu_dir": a.gpu_dir, "choices": a.choices, "row": a.row, "layers": {}}
    for L in a.layers.split(","):
        g = torch.load(os.path.join(a.gpu_dir, f"kd_layer{L}_step1_rank0_micro0.pt"), weights_only=False)
        g16 = {
            "query": g["query"],
            "key": g["key"],
            "q": g["indexer_q"],
            "k": g["indexer_k"],
            "w": g["indexer_w"],
            "topk": g["topk"],
        }
        pre = f"r{a.row}.dsa.{L}."
        x64 = {n: f.get_tensor(pre + n) for n in ("query", "key", "q", "k", "w", "topk")}
        gd = {k: v.to(f64) if v.is_floating_point() else v.long() for k, v in g16.items()}
        r16 = {k: v.to(torch.bfloat16).to(f64) if v.is_floating_point() else v for k, v in x64.items()}
        topk = x64["topk"].long()
        lay = {
            "gpu_dtypes": {k: str(v.dtype) for k, v in g16.items()},
            "gpu_softmax_scale": g["softmax_scale"],
            "rel_l2_gpu": {k: rel_l2(gd[k], x64[k]) for k in ("query", "key", "q", "k", "w")},
            "rel_l2_round": {k: rel_l2(r16[k], x64[k]) for k in ("query", "key", "q", "k", "w")},
            "topk_overlap": topk_overlap(topk, gd["topk"]),
        }
        teach = {
            "T_ref": (x64["query"], x64["key"]),
            "T_gpu": (gd["query"], gd["key"]),
            "T_key_gpu": (x64["query"], gd["key"]),
            "T_query_gpu": (gd["query"], x64["key"]),
            "T_key_round": (x64["query"], r16["key"]),
        }
        studs = {
            "S_ref": (x64["q"], x64["k"], x64["w"]),
            "S_gpu": (gd["q"], gd["k"], gd["w"]),
            "S_q_gpu": (gd["q"], x64["k"], x64["w"]),
            "S_k_gpu": (x64["q"], gd["k"], x64["w"]),
            "S_w_gpu": (x64["q"], x64["k"], gd["w"]),
            "S_qrope_gpu": (mix(x64["q"], gd["q"], slice(-ROPE, None)), x64["k"], x64["w"]),
            "S_qnope_gpu": (mix(x64["q"], gd["q"], slice(0, -ROPE)), x64["k"], x64["w"]),
            "S_krope_gpu": (x64["q"], mix(x64["k"], gd["k"], slice(-ROPE, None)), x64["w"]),
            "S_knope_gpu": (x64["q"], mix(x64["k"], gd["k"], slice(0, -ROPE)), x64["w"]),
            "S_qk_round": (r16["q"], r16["k"], x64["w"]),
        }
        T = {n: teacher(q, k, topk, f64, a.chunk) for n, (q, k) in teach.items()}
        S = {n: student(q, k, w, topk, f64, a.chunk) for n, (q, k, w) in studs.items()}
        kd = {"ref": kl_sum(T["T_ref"], S["S_ref"], topk)}
        for n in T:
            if n != "T_ref":
                kd[n] = kl_sum(T[n], S["S_ref"], topk)
        for n in S:
            if n != "S_ref":
                kd[n] = kl_sum(T["T_ref"], S[n], topk)
        kd["T_gpu+S_gpu"] = kl_sum(T["T_gpu"], S["S_gpu"], topk)
        five = ("query", "key", "q", "k", "w")
        kd["gpu"], _ = kd_one({k: g16[k].float() for k in five}, g16["topk"].long(), torch.float32, a.chunk)
        kd["ref_orig_fn"], _ = kd_one({k: x64[k] for k in five}, topk, f64, a.chunk)
        lay["kd"] = kd
        lay["rel_vs_ref"] = {n: v / kd["ref"] - 1 for n, v in kd.items() if n != "ref"}
        out["layers"][L] = lay
        print(L, json.dumps({n: f"{v:+.3e}" for n, v in lay["rel_vs_ref"].items()}), flush=True)
        print(L, "rel_l2_gpu", json.dumps({k: f"{v:.2e}" for k, v in lay["rel_l2_gpu"].items()}), flush=True)
    with open(a.out, "w") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
