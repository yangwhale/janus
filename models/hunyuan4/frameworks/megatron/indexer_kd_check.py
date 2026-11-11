"""混元 4 的索引器 KD loss：Megatron-Core 原样的实现（接在 Hy4DSAttention 里）对 MaxText 的定义。

MaxText（maxtext 混元 4 集成分支）的定义：
- 每个 full 索引器层与 MTP 层：`attention_mla.py::calculate_indexer_loss`；shared 层不产出；
- `train.py` 对各层取平均再乘 `indexer_loss_scaling_factor`。
Megatron 一侧每层乘逐层系数（MaxText 系数 ÷ 产出层数，见 `hy4_mcore.indexer_loss_coeff_per_layer`）后各自加进目标。

做法：同一次前向里每个产出层的输入（x、q_resid、主干 query / key、top-k）记下来，用同一个索引器重算 q / k / w，
之后按 MaxText 的公式在 FP64 里算 loss（逐行照抄，包括 EPS 1e-8 的位置），对索引器参数求梯度；
和模型反向里索引器参数的梯度比（索引器的输入是 detach 的、top-k 不可导，所以这些参数的梯度只来自 KD loss）。

    python indexer_kd_check.py --hf <截层 HF> --layers 4 --mtp --seq 4096 --tokens tokens.npy --coeff 0.1 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

MAXTEXT_EPS = 1e-8  # maxtext/utils/globals.py


def maxtext_indexer_loss(
    index_score, query, key, causal, topk, scale, sparse, scaling_factor, dtype=torch.float64, mcore_eps=False
):
    """照抄 MaxText calculate_indexer_loss（非 absorbed、多头 key、不分块），FP64。

    index_score [b, sq, sk]（不含掩码；这里加上因果掩码，同 MaxText 的 Indexer 返回值），
    query / key [s, b, h, d]（Megatron 布局；MaxText 的 query 已乘 softmax_scale），topk [b, sq, k]。
    """
    from hy4_mcore import causal_sparse_mask

    neg = torch.finfo(dtype).min / 4  # MaxText 的 DEFAULT_MASK_VALUE 是大负数；softmax 后为 0
    b, sq, sk = index_score.shape
    zero = torch.zeros((), dtype=dtype, device=causal.device)
    attn_mask = torch.where(torch.isinf(causal), zero + neg, zero)[None]  # [1, sq, sk]
    index_score = index_score.to(dtype) + attn_mask
    sel = causal_sparse_mask(torch.where(topk < sk, topk, torch.full_like(topk, -1)).long(), sq, sk)  # [b, sq, sk]
    indexer_mask = torch.where(sel, zero, zero + neg) + attn_mask  # generate_mask + 再加一次 attention_mask
    if sparse:
        index_score = index_score + indexer_mask
    indexer_probs = torch.softmax(index_score, dim=-1)
    q = query.to(dtype).permute(1, 0, 2, 3) * scale  # [b, t, h, d]
    k = key.to(dtype).permute(1, 0, 2, 3)  # [b, s, h, d]
    scores = torch.einsum("bthd,bshd->bhts", q, k)
    scores = scores + (indexer_mask[:, None] if sparse else attn_mask[:, None])
    attention_probs = torch.softmax(scores, dim=-1).sum(1)  # [b, t, s]
    attention_probs = attention_probs / (attention_probs.sum(-1, keepdim=True) + MAXTEXT_EPS)
    if mcore_eps:  # 只换 EPS 的写法：Megatron-Core 用 log(clamp(p, 1e-10)) 与精确的 log_softmax（dsa_indexer_loss.py）
        log_q = torch.log_softmax(index_score, dim=-1)
        kl = attention_probs * (torch.log(attention_probs.clamp_min(1e-10)) - log_q)
        kl = torch.where(attention_probs > 0, kl, torch.zeros_like(kl))
    else:
        kl = attention_probs * (torch.log(attention_probs + MAXTEXT_EPS) - torch.log(indexer_probs + MAXTEXT_EPS))
    return kl.sum(-1).mean() * scaling_factor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--mtp", action="store_true")
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--tokens", required=True)
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--coeff", type=float, default=0.1, help="MaxText 的 indexer_loss_scaling_factor")
    p.add_argument(
        "--dense", action="store_true", help="dense KD（默认 sparse，同 hy4-preview 的 indexer_sparse_training）"
    )
    p.add_argument("--neg-sum", action="store_true", help="负对照：参照按层求和（不除层数），应当报 FAIL")
    p.add_argument("--te", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--tol-loss",
        type=float,
        default=1e-4,
        help="loss 相对差；KL 是相近对数之差的和，FP32 里相对误差放大（同一 MaxText 公式 FP32 对 FP64 实测 1e-5–5e-5）",
    )
    p.add_argument("--tol-grad", type=float, default=1e-4)
    p.add_argument("--json")
    a = p.parse_args()

    import hy4_mcore as H
    import numpy as np
    from equiv_hf import build_mcore, rel_l2

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(a.device)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    torch.distributed.init_process_group("nccl" if dev.type == "cuda" else "gloo", rank=0, world_size=1)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    else:  # CPU 自检（hy4-tiny）：同 equiv_hf.py 的 CPU 分支
        torch.cuda.current_device = lambda: "cpu"
        import megatron.core.transformer.multi_latent_attention as _mla
        import megatron.core.transformer.multi_token_prediction as _mtp

        if _mla.QuantizedTensor is None:
            _mla.QuantizedTensor = ()
        _mtp.is_inference_column_parallel_linear = lambda m: False
    from megatron.core import parallel_state as ps

    ps.initialize_model_parallel(1, 1)

    toks = np.load(a.tokens, mmap_mode="r")
    seq = torch.from_numpy(np.asarray(toks[a.row, : a.seq + 1]).astype(np.int64))[None]
    ids, labels = seq[:, :-1].to(dev), seq[:, 1:].to(dev)

    t0 = time.time()
    hf_types = json.load(open(os.path.join(a.hf, "config.json")))["indexer_types"][: a.layers]
    per_layer = H.indexer_loss_coeff_per_layer(a.coeff, hf_types, 1 if a.mtp else 0)
    cfg, model = build_mcore(
        a.hf,
        a.layers,
        dev,
        a.te,
        mtp=a.mtp,
        hy4_train_indexer=True,
        dsa_indexer_loss_coeff=per_layer,
        dsa_indexer_use_sparse_loss=not a.dense,
        calculate_per_token_loss=False,
    )
    cores = [(f"decoder.{i}", layer.self_attention.core_attention) for i, layer in enumerate(model.decoder.layers)]
    if a.mtp:
        cores.append(("mtp.0", model.mtp.layers[0].mtp_model_layer.self_attention.core_attention))
    producers = [(n, c) for n, c in cores if c.is_full]
    for _, c in cores:
        c.capture = True
    model.train()
    pos = torch.arange(a.seq, device=dev)[None]
    out = model(ids, pos, attention_mask=None, labels=labels) if a.mtp else model(ids, pos, attention_mask=None)
    if a.mtp:
        main_loss = out.float().mean()  # MTP 的梯度由 MTPLossAutoScaler 挂在主干上；KD 由 DSAIndexerLossAutoScaler 挂上
    else:
        main_loss = torch.nn.functional.cross_entropy(out.flatten(0, 1).float(), labels.flatten())
    model.zero_grad(set_to_none=True)
    main_loss.backward()
    res = {
        "layers": a.layers,
        "mtp": a.mtp,
        "seq": a.seq,
        "coeff_maxtext": a.coeff,
        "coeff_per_layer": per_layer,
        "n_producers": len(producers),
        "producers": [n for n, _ in producers],
        "sparse": not a.dense,
        "neg_sum": a.neg_sum,
        "build_and_step_seconds": round(time.time() - t0, 1),
        "layers_detail": [],
    }

    ok = True
    for name, c in producers:
        x, qr, query, key = c.last_kd_inputs
        sq = query.shape[0]
        causal = torch.full((sq, sq), float("-inf"), device=dev).triu(1)
        params = [prm for prm in c.indexer.parameters() if prm.requires_grad]
        q, k, w = c.indexer.forward_before_topk(x, qr, None)
        # 索引分数（MaxText 的写法：relu(q·k) 按 w 加权求和），FP64
        sc = torch.relu(torch.einsum("sbhd,tbd->bsht", q.double(), k.double()))
        index_score = (sc * w.double().permute(1, 0, 2)[..., None]).sum(2)  # [b, sq, sk]
        ref = maxtext_indexer_loss(
            index_score,
            query,
            key,
            causal,
            c.last_topk,
            c.softmax_scale,
            not a.dense,
            a.coeff if a.neg_sum else a.coeff / len(producers),
        )
        ref_grads = torch.autograd.grad(ref, params)
        got = float(c.last_indexer_loss)
        # 舍入尺子：同一个 MaxText 公式在 FP32 里算，看 FP32 本身让这个 KL 偏多少（KL 是相近对数之差的和，
        # 值小时相对误差被放大）；Megatron-Core 的函数内部一律转 FP32，没有 FP64 版可比
        with torch.no_grad():
            m32 = float(
                maxtext_indexer_loss(
                    index_score.detach().float(),
                    query,
                    key,
                    causal,
                    c.last_topk,
                    c.softmax_scale,
                    not a.dense,
                    a.coeff if a.neg_sum else a.coeff / len(producers),
                    dtype=torch.float32,
                )
            )
        # 同一份输入、只把 EPS 换成 Megatron-Core 的写法（FP64）：它的梯度对模型梯度是「EPS 写法以外」的差
        q2, k2, w2 = c.indexer.forward_before_topk(x, qr, None)  # 新建一张图（TE 的线性层不能对同一张图反向两次）
        sc2 = torch.relu(torch.einsum("sbhd,tbd->bsht", q2.double(), k2.double()))
        ref_eps = maxtext_indexer_loss(
            (sc2 * w2.double().permute(1, 0, 2)[..., None]).sum(2),
            query,
            key,
            causal,
            c.last_topk,
            c.softmax_scale,
            not a.dense,
            a.coeff if a.neg_sum else a.coeff / len(producers),
            mcore_eps=True,
        )
        m_eps = float(ref_eps)
        eps_grads = torch.autograd.grad(ref_eps, params)
        rel = lambda x, y: abs(x - y) / max(abs(y), 1e-300)  # noqa: E731
        d = {
            "layer": name,
            "loss_mcore": got,
            "loss_maxtext_fp32": m32,
            "loss_maxtext_fp64": float(ref),
            "loss_rel_diff": rel(got, float(ref)),
            "rounding_rel_diff_fp32": rel(m32, float(ref)),
            "eps_writing_rel_diff_fp64": rel(m_eps, float(ref)),
            "grads": [],
        }
        named = [(n, q_) for n, q_ in c.indexer.named_parameters() if q_.requires_grad]
        for (pn, prm), rg, eg in zip(named, ref_grads, eps_grads):
            mg = getattr(prm, "main_grad", None)
            mg = prm.grad if mg is None else mg
            d["grads"].append(
                {
                    "param": pn,
                    "rel_l2": None if mg is None else rel_l2(mg, rg),
                    "rel_l2_vs_mcore_eps_writing": None if mg is None else rel_l2(mg, eg),
                    "eps_writing_grad_rel_l2": rel_l2(eg, rg),
                    "norm_ref": float(rg.double().norm()),
                }
            )
        d["grad_rel_l2_max"] = max((g["rel_l2"] for g in d["grads"] if g["rel_l2"] is not None), default=None)
        d["grad_rel_l2_max_vs_mcore_eps_writing"] = max(
            (g["rel_l2_vs_mcore_eps_writing"] for g in d["grads"] if g["rel_l2_vs_mcore_eps_writing"] is not None),
            default=None,
        )
        d["eps_writing_grad_rel_l2_max"] = max(g["eps_writing_grad_rel_l2"] for g in d["grads"])
        d["ok"] = bool(
            d["loss_rel_diff"] <= a.tol_loss
            and all(g["rel_l2"] is not None for g in d["grads"])
            and d["grad_rel_l2_max"] <= a.tol_grad
        )
        ok &= d["ok"]
        res["layers_detail"].append(d)
        print("KD", json.dumps({k_: v for k_, v in d.items() if k_ != "grads"}), flush=True)
    # 非产出层（shared）的索引器不存在；产出层以外不该有 KD
    res["shared_layers_without_loss"] = [n for n, c in cores if not c.is_full and c.last_indexer_loss is None]
    res["pass"] = bool(ok)
    print("PASS" if ok else "FAIL", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
