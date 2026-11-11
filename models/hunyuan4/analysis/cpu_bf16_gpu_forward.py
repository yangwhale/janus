"""CPU 上按 GPU 写法跑一个全局批：Megatron-Core local 层，BF16 参数 + Float16Module 留 FP32 的参数（输出头、iHC、hc_head、
sink、索引器 k_norm、路由偏置），代码里写死的 `.float()` / `.to(bfloat16)` 照原样（同 `kd_bf16_vs_fp64.py` 的 BF16 一支）。
矩阵乘的累加顺序、TE 的融合算子与真 GPU 不同（推断：舍入位置相同、逐位不同），所以只当「GPU 写法的 BF16 前向」的近似，
真 GPU 能用时以真 GPU 为准。

每行打印 MICRO（与 step1_ref.py 同格式：row、loss_sum、n_tokens、mtp_mean），`mini_main_loss_same_state.py` 汇总。
- `--backward`：同时反向（MTP 经 MTPLossAutoScaler、缩放 1），全部行累加后除以有效 token 总数，存 final 块、输出头、
  词嵌入、MTP enorm / hnorm / eh_proj 与第 6 层的梯度（Megatron 名字）到 `grads_s<步>.safetensors`；
- `--final32` / `--hc-round-only`：主干末尾的 hc_head 输出与末层 RMSNorm 留 FP32（或只保留 hc_head 输出那一次 BF16 舍入），
  MTP 的 hnorm 输入照旧舍 BF16（诊断：看末尾这几次舍入对 hc_head 梯度的影响）。

    python cpu_bf16_gpu_forward.py --hf <HF 目录> --tokens tokens.npy --step N --out <目录> [--backward] [--final32]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

MEG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frameworks", "megatron")
sys.path.insert(0, MEG)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--rows", default="", help="只跑这些行（全局批内的序号 0–15）")
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--threads", type=int, default=40)
    p.add_argument("--out", required=True)
    p.add_argument("--final32", action="store_true", help="主干 hc_head 输出与末层 RMSNorm 留 FP32（不舍 BF16）")
    p.add_argument("--backward", action="store_true", help="同时反向，全部行累加后除以有效 token 总数，存 final 块与几组参数的梯度")
    p.add_argument("--hc-round-only", action="store_true", help="只保留 hc_head 输出的 BF16 舍入，末层 RMSNorm 在 FP32 里算、输出不舍")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    import step1_ref as S

    S.cpu_megatron_setup()
    import hy4_mcore as H
    import hy4_weights as W
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper

    if a.final32 or a.hc_round_only:
        import torch.nn.functional as F
        def fwd(self, x):
            s_, b_, _ = x.shape
            flat = x.float()
            mixes = F.linear(flat, self.hc_fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
            pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.hc_eps
            out = (pre.unsqueeze(-1) * x.view(s_, b_, self.n, self.H)).sum(dim=2)
            if a.hc_round_only:
                out = out.to(x.dtype).float()
            w = self.norm.weight.float()
            eps = float(getattr(self.norm, "eps", self.norm_eps))
            return out * torch.rsqrt(out.square().mean(-1, keepdim=True) + eps) * w
        H.Hy4FinalNorm.forward = fwd
        print("PATCH final32" if a.final32 else "PATCH hc_round_only", flush=True)
    hf = json.load(open(os.path.join(a.hf, "config.json")))
    L = hf["num_hidden_layers"]
    torch.set_default_dtype(torch.float32)
    kw = H.hy4_config_kwargs(hf, num_layers=L)
    kw["moe_token_dispatcher_type"] = "allgather"
    coeff = H.indexer_loss_coeff_per_layer(0.1, hf["indexer_types"][:L], 1)
    extra = dict(params_dtype=torch.bfloat16, bf16=True, perform_initialization=False, transformer_impl="local",
                 moe_grouped_gemm=False, use_cpu_initialization=True, hy4_train_indexer=True,
                 dsa_indexer_loss_coeff=coeff, dsa_indexer_use_sparse_loss=True, calculate_per_token_loss=True,
                 hy4_attn_scores_fp32=True, hy4_head_chunk=1024, cross_entropy_loss_fusion=False,
                 gradient_accumulation_fusion=False, mtp_num_layers=1,
                 mtp_loss_scaling_factor=float(hf.get("mtp_loss_factor", 0.1)))
    cfg16 = H.Hy4Config(**{**kw, **extra})
    m16 = Float16Module(cfg16, H.build_model(cfg16, use_te=False, vocab_size=hf["vocab_size"], max_seq=1 << 20, mtp=True))
    W.load_hf_into_mcore(m16.module, cfg16, a.hf)
    if a.final32 or a.hc_round_only:
        for n_, mod in m16.named_modules():
            if n_.endswith("hnorm"):
                of = mod.forward
                mod.forward = lambda x, _of=of: _of(x.to(torch.bfloat16))
                print("PATCH hnorm input -> bf16", n_, flush=True)
    m16.train()
    dts = {}
    for _n, q in m16.named_parameters():
        dts[str(q.dtype)] = dts.get(str(q.dtype), 0) + q.numel()
    print("BUILT_BF16", json.dumps(dts), flush=True)
    toks = np.load(a.tokens, mmap_mode="r")
    eod = int(hf["eos_token_id"])
    pos = torch.arange(a.seq).unsqueeze(0)
    base = (a.step - 1) * 16
    rows = [int(x) for x in a.rows.split(",")] if a.rows else list(range(16))
    os.makedirs(a.out, exist_ok=True)
    ntok_total = [0.0]
    for i in rows:
        r = base + i
        row = np.asarray(toks[r, : a.seq + 1]).astype(np.int64)
        ids = torch.from_numpy(row[:-1])[None]
        labels = torch.from_numpy(row[1:])[None]
        lm = (ids != eod).float()
        MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
        DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
        MTPLossLoggingHelper.clean_metrics_in_tracker()
        t0 = time.time()
        with torch.set_grad_enabled(a.backward):
            o = m16(ids, pos, attention_mask=None, labels=labels, loss_mask=lm)
            st = (o.float() * lm).sum()
            s = float(st)
        if a.backward:
            st.backward()
            ntok_total[0] += float(lm.sum())
        rec = {"row": r, "loss_sum": s, "n_tokens": float(lm.sum()), "fwd_s": round(time.time() - t0, 1),
               "mtp_mean": float(MTPLossLoggingHelper.tracker["loss_values"][0])}
        MTPLossLoggingHelper.clean_metrics_in_tracker()
        print("MICRO " + json.dumps(rec), flush=True)
    if a.backward:
        from safetensors.torch import save_file
        keep = {}
        for n_, q in m16.named_parameters():
            if q.grad is None:
                continue
            g = q.grad if getattr(q, "main_grad", None) is None else q.main_grad
            if any(k in n_ for k in ("final_layernorm", "output_layer", "word_embeddings", "mtp.layers.0.hnorm",
                                     "mtp.layers.0.enorm", "mtp.layers.0.eh_proj", "decoder.layers.6.")):
                keep[n_.replace("module.", "", 1)] = (g.float() / ntok_total[0]).contiguous()
        save_file(keep, os.path.join(a.out, f"grads_s{a.step}.safetensors"))
        print("SAVED", len(keep), "ntok", ntok_total[0], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
