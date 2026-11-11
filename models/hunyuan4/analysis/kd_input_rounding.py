"""索引器 KD 对输入舍入的灵敏度：CPU FP64 参照抓到的 KD 输入上，逐个输入舍成 BF16（或整段改用 FP32 算），看 KD 移多少。

为什么：真实宽度三层（06-rw3）第 1 步索引器 KD 对 CPU FP64，GPU 逐层 +1.5e-3（第 0 层）、+2.2e-3（第 1 层），
mini 上只有 +2.3e-5。这里先在 KD 这一段里定位：输入（教师的 query / key、索引器的 q / k / w）在 GPU 上是 BF16
（w 是 FP32），KD 本身用 FP32 算。只在 KD 这一段加差，看能复现多少；复现不了的部分在上游（激活一路 BF16 的累积误差）。

输入：`step1_ref.py` 的 `choices.safetensors`（`--detail-rows` 的行：r<行>.dsa.<层>.{q,k,w,query,key,topk}，FP64）。
KD 用 Megatron-Core 原版 `compute_dsa_indexer_loss` / `_compute_index_scores`（与 `hy4_mcore.Hy4DSAttention` 同一调用），
按查询行分块（per-token 求和，分块精确）。

用法（Megatron-Core 源码在 PYTHONPATH）：
    python kd_input_rounding.py --choices <choices.safetensors> --row 0 --layers 0,1,mtp0 --out kd_input_rounding.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frameworks", "megatron"))
from step1_ref import ForceDtype, relax_mask_dtype  # noqa: E402

COEFF = 0.1 / 3  # 逐层系数 = 0.1 ÷ 产出层数（D3：第 0、1 层 + MTP）
SOFTMAX_SCALE = 256**-0.5  # 注意力 head_dim 256（教师 q·k 的缩放；与 hy4_mcore 一致，由 key 的最后一维核对）

# 变体：每个输入怎么处理（f64 = 原样，bf16 = 舍成 BF16 再转回，f32 = 转 FP32）与 KD 用什么精度算
INPUTS = ("query", "key", "q", "k", "w")
VARIANTS = {
    "fp64": ({}, torch.float64),
    "fp32_arith": ({n: "f32" for n in INPUTS}, torch.float32),
    "gpu_local": ({"query": "bf16", "key": "bf16", "q": "bf16", "k": "bf16", "w": "f32"}, torch.float32),
    "teacher_bf16": ({"query": "bf16", "key": "bf16"}, torch.float64),
    "query_bf16": ({"query": "bf16"}, torch.float64),
    "key_bf16": ({"key": "bf16"}, torch.float64),
    "student_bf16": ({"q": "bf16", "k": "bf16"}, torch.float64),
    "q_bf16": ({"q": "bf16"}, torch.float64),
    "k_bf16": ({"k": "bf16"}, torch.float64),
    "w_bf16": ({"w": "bf16"}, torch.float64),
}


class _TP:
    @staticmethod
    def size():
        return 1


class _PG:
    tp = _TP()


def prep(t: torch.Tensor, how: str | None) -> torch.Tensor:
    if how == "bf16":
        return t.to(torch.bfloat16).to(t.dtype)
    if how == "f32":
        return t.to(torch.float32)
    return t


def kd_one(inp: dict, topk: torch.Tensor, arith: torch.dtype, chunk: int) -> tuple[float, torch.Tensor]:
    """KD（已乘逐层系数，per-token 求和）与逐查询行的 KL（未乘系数）。"""
    from megatron.core.transformer.experimental_attention_variant import dsa_indexer_loss as dl
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        _compute_index_scores,
        compute_dsa_indexer_loss,
    )

    # FP64 时同 step1_ref：原版里写死的 `.float()` 等在 ATen 一层改写成 FP64；FP32 时原样（GPU 的写法）
    ctx = ForceDtype(torch.float64) if arith == torch.float64 else contextlib.nullcontext()
    with ctx:
        return _kd(inp, topk, arith, chunk, dl, _compute_index_scores, compute_dsa_indexer_loss)


def _kd(inp, topk, arith, chunk, dl, _compute_index_scores, compute_dsa_indexer_loss):
    q, k, w, query, key = (inp[n].to(arith) for n in ("q", "k", "w", "query", "key"))
    sq, sk = query.shape[0], key.shape[0]
    causal = torch.full((sq, sk), float("-inf"), dtype=arith).triu(diagonal=1 + sk - sq)
    total = 0.0
    rows = []
    orig = dl.indexer_loss_from_target
    cap = {}

    def spy(target, logp, loss_coeff, **kw):  # 顺手记逐行 KL
        cap["row"] = dl.indexer_kl_per_row(target, logp)
        return orig(target, logp, loss_coeff, **kw)

    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_mod

    dsa_mod.dsa_indexer_loss.indexer_loss_from_target = spy
    try:
        for s0 in range(0, sq, chunk):
            s1 = min(sq, s0 + chunk)
            idx = _compute_index_scores(q[s0:s1], w[s0:s1], k, use_relu=True)
            loss = compute_dsa_indexer_loss(
                idx,
                topk[:, s0:s1].long(),
                query[s0:s1],
                key,
                SOFTMAX_SCALE,
                COEFF,
                True,
                _PG(),
                mask=causal[s0:s1],
                calculate_per_token_loss=True,
            )
            total += float(loss.double())
            rows.append(cap["row"].double().reshape(-1))
    finally:
        dsa_mod.dsa_indexer_loss.indexer_loss_from_target = orig
    return total, torch.cat(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--choices", required=True)
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--layers", default="0,1,mtp0")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    relax_mask_dtype(torch.float64)
    from safetensors import safe_open

    out = {"choices": a.choices, "row": a.row, "coeff": COEFF, "softmax_scale": SOFTMAX_SCALE, "layers": {}}
    f = safe_open(a.choices, "pt")
    for L in a.layers.split(","):
        pre = f"r{a.row}.dsa.{L}."
        base = {n: f.get_tensor(pre + n) for n in INPUTS}
        assert base["key"].shape[-1] == 256, base["key"].shape
        topk = f.get_tensor(pre + "topk")
        res = {}
        ref_rows = None
        for vn in a.variants.split(","):
            how, arith = VARIANTS[vn]
            inp = {n: prep(base[n], how.get(n)) for n in INPUTS}
            t0 = time.time()
            kd, rows = kd_one(inp, topk, arith, a.chunk)
            if vn == "fp64":
                ref_rows = rows
            r = {"kd": kd, "seconds": round(time.time() - t0, 1)}
            if ref_rows is not None and vn != "fp64":
                r["rel_vs_fp64"] = kd / res["fp64"]["kd"] - 1
                d = rows - ref_rows
                r["row_kl_diff_mean"] = float(d.mean())
                r["row_kl_diff_frac_pos"] = float((d > 0).double().mean())
            res[vn] = r
            print(L, vn, json.dumps(r), flush=True)
        res["fp64"]["row_kl_mean"] = float(ref_rows.mean())
        out["layers"][L] = res
    with open(a.out, "w") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
