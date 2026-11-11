"""CPU 上的 BF16 前向（GPU 写法，`kd_bf16_vs_fp64.py` 的第 2 份模型）与 GPU 实抓的 KD 输入差多少。

为什么：`kd_bf16_vs_fp64.py` 在 CPU BF16 前向上拆分 KD 偏差并定位到索引器 RoPE 的 BF16 写法；这里核这份 CPU 前向的
KD 输入（第 1 步第 0 行，各产出层的教师 query / key、索引器 q / k / w）与 GPU 抓到的（`train_hy4mg.py --diag-kd-dump`）
有多近：两者之差远小于各自对 FP64 的差，就说明 CPU 这份前向的舍入基本就是 GPU 的那些舍入，拆分结论可以搬到 GPU。

用法（CPU，Megatron-Core 源码在 PYTHONPATH；内存约 70 GiB）：
    python kd_cpu16_vs_gpu_dump.py --hf <HF 目录> --tokens tokens.npy --gpu-dir <kd_layer*.pt 目录> \
        --choices <FP64 choices.safetensors> --out kd_cpu16_vs_gpu_dump.json
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

from kd_bf16_vs_fp64 import collect, cores_of, rel_l2, tap  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--gpu-dir", required=True)
    p.add_argument("--choices", required=True)
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    import numpy as np
    import step1_ref as S

    S.cpu_megatron_setup()
    import hy4_mcore as H
    import hy4_weights as W
    from megatron.core.transformer.module import Float16Module
    from safetensors import safe_open

    hf, _ = H.complete_hf_config(json.load(open(os.path.join(a.hf, "config.json"))))
    L, mtp = a.layers, True
    coeff = H.indexer_loss_coeff_per_layer(0.1, hf["indexer_types"][:L], 1)
    kw = H.hy4_config_kwargs(hf, num_layers=L)
    kw["moe_token_dispatcher_type"] = "allgather"
    extra = dict(
        params_dtype=torch.bfloat16,
        bf16=True,
        perform_initialization=False,
        transformer_impl="local",
        moe_grouped_gemm=False,
        use_cpu_initialization=True,
        hy4_train_indexer=True,
        dsa_indexer_loss_coeff=coeff,
        dsa_indexer_use_sparse_loss=True,
        calculate_per_token_loss=True,
        hy4_attn_scores_fp32=True,
        hy4_head_chunk=1024,
        cross_entropy_loss_fusion=False,
        gradient_accumulation_fusion=False,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=float(hf.get("mtp_loss_factor", 0.1)),
    )
    cfg = H.Hy4Config(**{**kw, **extra})
    m16 = Float16Module(cfg, H.build_model(cfg, use_te=False, vocab_size=hf["vocab_size"], max_seq=1 << 20, mtp=mtp))
    W.load_hf_into_mcore(m16.module, cfg, a.hf)
    m16.train()
    cores = cores_of(m16.module, mtp)
    st = {}
    tap(cores, st)
    toks = np.load(a.tokens, mmap_mode="r")
    row = np.asarray(toks[a.row, : a.seq + 1]).astype(np.int64)
    ids, labels = torch.from_numpy(row[:-1])[None], torch.from_numpy(row[1:])[None]
    lm = (ids != int(hf["eos_token_id"])).float()
    with torch.no_grad():
        m16(ids, torch.arange(a.seq).unsqueeze(0), attention_mask=None, labels=labels, loss_mask=lm)
    x16 = collect(cores, st)
    f = safe_open(a.choices, "pt")
    gk = {"query": "query", "key": "key", "q": "indexer_q", "k": "indexer_k", "w": "indexer_w"}
    out = {"row": a.row, "layers": {}}
    for name in x16:
        g = torch.load(os.path.join(a.gpu_dir, f"kd_layer{name}_step1_rank0_micro0.pt"), weights_only=False)
        lay = {}
        for n, k in gk.items():
            c, gg = x16[name][n], g[k]
            ref = f.get_tensor(f"r{a.row}.dsa.{name}.{n}")
            lay[n] = {
                "cpu16_vs_gpu": rel_l2(c, gg),
                "gpu_vs_fp64": rel_l2(gg, ref),
                "cpu16_vs_fp64": rel_l2(c, ref),
                "bitwise_equal_frac": float((c.to(gg.dtype) == gg).double().mean()),
            }
        out["layers"][name] = lay
        print(name, json.dumps({n: {k: f"{v:.2e}" for k, v in d.items()} for n, d in lay.items()}), flush=True)
    with open(a.out, "w") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
