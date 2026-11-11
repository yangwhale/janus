"""混元 4 Megatron 模型对 HF 参考实现的等价性检查（FP32，截层，真实权重，单进程）。

同一批 token 两边各前向一次（可选再反向一次），比：每层输出的 n 路流、每个 full 层 DSA 选中的位置、logits、
（--backward）loss 与逐参数梯度。HF 一侧是 transformers `hy_v4` v5.22.1 原样代码 + 项目定下的两处读法（交错 RoPE、
MLA latent norm eps 1e-5 / 索引器 k_norm eps 1e-6），与 HF golden 一侧（frameworks/hf/hy4_model.py）相同。

用法（CPU；Megatron-Core 0.20.0 源码在 PYTHONPATH 里）：
    python equiv_hf.py --hf <截好的 4 层 HF 目录> --layers 1 --seq 4096 --tokens <tokens.npy> --json out.json [--backward]
GPU：加 --device cuda [--te]（--te 用 Transformer Engine 的层，与训练相同）；--hf-device 可以把 HF 放到另一张卡。

判据（FP32）：每层输出、logits 的相对 L2 误差 ≤ --tol（默认 1e-5）；DSA 选中集合逐 query 相同；
（--backward）每个参数梯度的相对 L2 ≤ --tol-grad（默认 1e-4）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RENAMES = [  # transformers v5.22.1 conversion_mapping.py 的 "hy_v4"
    (r"\.hc_pre\.hc_", ".hc_"),
    (r"\.learnable_sink_param", ".sinks"),
    (r"\.linear_gate", ".gate_proj"),
    (r"\.hc_attn_layer\.hc_fn", ".attn_hc.fn"),
    (r"\.hc_attn_layer\.hc_base", ".attn_hc.base"),
    (r"\.hc_attn_layer\.hc_scale", ".attn_hc.scale"),
    (r"\.hc_mlp_layer\.hc_fn", ".ffn_hc.fn"),
    (r"\.hc_mlp_layer\.hc_base", ".ffn_hc.base"),
    (r"\.hc_mlp_layer\.hc_scale", ".ffn_hc.scale"),
    (r"\.hc_head_fn", ".hc_fn"),
    (r"\.hc_head_base", ".hc_base"),
    (r"\.hc_head_scale", ".hc_scale"),
]


def ckpt_to_hf(n: str) -> str:
    for a, b in RENAMES:
        n = re.sub(a, b, n)
    return n


def interleaved_rope(q, k, cos, sin, unsqueeze_dim=1):
    """交错（GPT-J）RoPE，同 HF golden 一侧 hy4_model.interleaved_rope（vLLM is_neox_style=False）。"""
    half = cos.shape[-1] // 2
    c = cos[..., :half].unsqueeze(unsqueeze_dim)
    s = sin[..., :half].unsqueeze(unsqueeze_dim)

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)

    return rot(q), rot(k)


def build_hf(hf_dir, n_layers, device, patched=True):
    from transformers.models.hy_v4 import modeling_hy_v4 as M

    if patched:
        M.apply_rotary_pos_emb = interleaved_rope
    cfg = M.HYV4Config.from_pretrained(hf_dir)
    cfg.num_hidden_layers = n_layers
    for k in ("indexer_types", "layer_types", "mlp_layer_types"):
        setattr(cfg, k, list(getattr(cfg, k))[:n_layers])
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    with torch.device("meta"):
        model = M.HYV4ForCausalLM(cfg)
    from hy4_weights import Safetensors

    st = Safetensors(hf_dir)
    sd = {}
    keep = re.compile(r"^model\.layers\.(\d+)\.")
    for name in st.names():
        mm = keep.match(name)
        if name.startswith("model.mtp_layers.") or (mm and int(mm.group(1)) >= n_layers):
            continue
        sd[ckpt_to_hf(name)] = st.get(name).to(torch.float32)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if not m.endswith("inv_freq")]
    assert not missing and not unexpected, (missing, unexpected)
    # 非持久 buffer（RoPE 的 inv_freq）在 meta 上，重建
    rot = model.model.rotary_emb
    inv, _ = rot.compute_default_rope_parameters(cfg)
    rot.inv_freq = inv
    rot.original_inv_freq = inv.clone()
    model = model.to(device=device, dtype=torch.float32)
    if patched:
        for layer in model.model.layers:
            at = layer.self_attn
            at.q_a_layernorm.variance_epsilon = cfg.rms_norm_eps
            at.kv_a_layernorm.variance_epsilon = cfg.rms_norm_eps
            if at.indexer is not None:
                at.indexer.k_norm.eps = 1e-6
    return cfg, model


class HFMTP(torch.nn.Module):
    """HF 一侧的 MTP：同 HF golden 一侧 hy4_model.MTPLayer（按 vLLM 8b55f919 的前向）。"""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        from transformers.models.hy_v4 import modeling_hy_v4 as M

        H, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.enorm = M.HYV4RMSNorm(H, eps=eps)
        self.hnorm = M.HYV4RMSNorm(H, eps=eps)
        self.eh_proj = torch.nn.Linear(2 * H, H, bias=False)
        self.input_layernorm = M.HYV4RMSNorm(H, eps=eps)
        self.self_attn = M.HYV4Attention(cfg, layer_idx)
        self.post_attention_layernorm = M.HYV4RMSNorm(H, eps=eps)
        self.mlp = M.HYV4MoE(cfg)
        self.final_layernorm = M.HYV4RMSNorm(H, eps=eps)

    def forward(self, emb, h, mask, pe, pos):
        x = self.eh_proj(torch.cat([self.enorm(emb), self.hnorm(h)], dim=-1))
        a, _, self.topk = self.self_attn(
            hidden_states=self.input_layernorm(x),
            attention_mask=mask,
            position_embeddings=pe,
            position_ids=pos,
            prev_topk_indices=None,
        )
        x = x + a
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.final_layernorm(x)


def build_hf_mtp(hf_dir, cfg, device):
    import copy

    from hy4_weights import Safetensors

    c = copy.deepcopy(cfg)
    L = cfg.num_hidden_layers
    c.indexer_types = list(cfg.indexer_types) + ["full"]
    c.mlp_layer_types = list(cfg.mlp_layer_types) + ["sparse"]
    with torch.device("meta"):
        mtp = HFMTP(c, L)
    st = Safetensors(hf_dir)
    pre = "model.mtp_layers.0."
    sd = {ckpt_to_hf(n)[len(pre) :]: st.get(n).to(torch.float32) for n in st.names() if n.startswith(pre)}
    mtp.load_state_dict(sd, strict=True, assign=True)
    mtp = mtp.to(device=device, dtype=torch.float32)
    mtp.self_attn.q_a_layernorm.variance_epsilon = cfg.rms_norm_eps
    mtp.self_attn.kv_a_layernorm.variance_epsilon = cfg.rms_norm_eps
    mtp.self_attn.indexer.k_norm.eps = 1e-6
    return mtp


def build_mcore(hf_dir, n_layers, device, use_te, mtp=False, **overrides):
    import hy4_mcore as H
    from hy4_weights import load_hf_into_mcore

    hf, _ = H.complete_hf_config(json.load(open(os.path.join(hf_dir, "config.json"))))
    kw = H.hy4_config_kwargs(hf, num_layers=n_layers)
    extra = dict(params_dtype=torch.float32, bf16=False, perform_initialization=False)
    if use_te:
        extra.update(transformer_impl="transformer_engine", moe_grouped_gemm=True)
    else:
        extra.update(transformer_impl="local", moe_grouped_gemm=False, use_cpu_initialization=True)
        kw["moe_token_dispatcher_type"] = (
            "allgather"  # alltoall 分发器建 CUDA stream，CPU 上用 allgather（EP=1 时两者同一结果）
        )
    if mtp:
        extra.update(mtp_num_layers=1, mtp_loss_scaling_factor=float(hf.get("mtp_loss_factor", 0.1)))
    cfg = H.Hy4Config(**{**kw, **extra, **overrides})
    model = H.build_model(cfg, use_te=use_te, vocab_size=hf["vocab_size"], max_seq=1 << 20, mtp=mtp)
    model = model.to(device)
    load_hf_into_mcore(model, cfg, hf_dir)
    return cfg, model


def rel_l2(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def topk_mask(topk, sq):
    from hy4_mcore import causal_sparse_mask

    t = topk.long()
    return causal_sparse_mask(torch.where(t < sq, t, torch.full_like(t, -1)), sq, sq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--tokens", required=True, help="tokens.npy（行 = 4097 个 token，前 seq 个当输入，后移一位当标签）")
    p.add_argument("--rows", default="0", help="逗号分隔的行号，一行一条序列")
    p.add_argument("--device", default="cpu")
    p.add_argument("--hf-device", default=None)
    p.add_argument("--te", action="store_true")
    p.add_argument("--backward", action="store_true")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--tol-grad", type=float, default=1e-4)
    p.add_argument("--tie-gap", type=float, default=1e-5, help="top-k 翻转算平局的 FP64 相对分差上限")
    p.add_argument("--json", default=None)
    p.add_argument("--hf-unpatched", action="store_true", help="负对照：HF 原样读法（NeoX RoPE、原样 eps），应判不一致")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument(
        "--replay-topk",
        action="store_true",
        help="把 HF 每个 full 层（含 MTP）选出的 top-k 灌给 Megatron，隔离掉 DSA 平局翻转，只比连续部分",
    )
    p.add_argument(
        "--mtp", action="store_true", help="带 MTP（一层，自带 full 索引器、无 iHC），另比 MTP 的 logits、top-k、loss"
    )
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev, hdev = torch.device(a.device), torch.device(a.hf_device or a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    torch.distributed.init_process_group("nccl" if dev.type == "cuda" else "gloo", rank=0, world_size=1)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    else:
        torch.cuda.current_device = lambda: "cpu"  # Megatron-Core 的 RoPE 在构造时取当前设备；CPU 检查用
        import megatron.core.transformer.multi_latent_attention as _mla

        if _mla.QuantizedTensor is None:  # 没装 TE 时它是 None，MLA 前向里的 isinstance 会报错
            _mla.QuantizedTensor = ()
        import megatron.core.transformer.multi_token_prediction as _mtp

        _mtp.is_inference_column_parallel_linear = lambda m: False  # 同上：没装 TE 时推理层类是 None
    from megatron.core import parallel_state as ps

    ps.initialize_model_parallel(1, 1)
    import numpy as np

    toks = np.load(a.tokens, mmap_mode="r")
    rows = [int(x) for x in a.rows.split(",")]
    seq = torch.from_numpy(np.stack([toks[r, : a.seq + 1] for r in rows]).astype(np.int64))
    ids, labels = seq[:, :-1], seq[:, 1:]
    res = {"layers": a.layers, "seq": a.seq, "rows": rows, "device": a.device, "te": a.te, "backward": a.backward}
    t0 = time.time()
    hcfg, hm = build_hf(a.hf, a.layers, hdev, patched=not a.hf_unpatched)
    res["hf_patched"] = not a.hf_unpatched
    mcfg, mm = build_mcore(a.hf, a.layers, dev, a.te, mtp=a.mtp)
    hmtp = build_hf_mtp(a.hf, hcfg, hdev) if a.mtp else None
    res["mtp"] = a.mtp
    res["build_seconds"] = round(time.time() - t0, 1)
    if not a.backward:
        hm.requires_grad_(False)
        mm.requires_grad_(False)
        if hmtp is not None:
            hmtp.requires_grad_(False)
    hm.eval()
    mm.eval()
    head_out = []
    if a.mtp:
        # 交叉熵会就地改 logits，先拷一份
        mm.output_layer.register_forward_hook(lambda mod, inp, out: head_out.append(out[0].detach().clone()) and None)
        mm.mtp.layers[0].mtp_model_layer.self_attention.core_attention.capture = True

    # ---- 抓每层输出与 top-k
    h_out, h_topk, m_out = {}, {}, {}

    def keep_hf(i):
        def hook(mod, inp, out):
            h_out[i], h_topk[i] = out[0].detach(), out[1]

        return hook

    def keep_m(i):
        def hook(mod, inp, out):
            m_out[i] = out[0].detach()

        return hook

    h_idx_in = {}

    def keep_idx(i):
        def hook(mod, args, kwargs):
            h_idx_in[i] = (args[0].detach(), args[1].detach())

        return hook

    for i, layer in enumerate(hm.model.layers):
        layer.register_forward_hook(keep_hf(i))
        if layer.self_attn.indexer is not None:
            layer.self_attn.indexer.register_forward_pre_hook(keep_idx(i), with_kwargs=True)
    for i, layer in enumerate(mm.decoder.layers):
        layer.register_forward_hook(keep_m(i))
        layer.self_attention.core_attention.capture = True

    if hmtp is not None:
        hmtp.self_attn.indexer.register_forward_pre_hook(keep_idx("mtp"), with_kwargs=True)
    res["replay_topk"] = a.replay_topk
    ctx = torch.enable_grad() if a.backward else torch.no_grad()
    with ctx:
        t0 = time.time()
        if a.mtp:
            from transformers.masking_utils import create_causal_mask

            hids = ids.to(hdev)
            hpos = torch.arange(a.seq, device=hdev).unsqueeze(0)
            hh_last = hm.model(input_ids=hids).last_hidden_state
            hl = hm.lm_head(hh_last)
            emb = hm.model.embed_tokens(hids)
            hmask = create_causal_mask(
                config=hcfg,
                inputs_embeds=emb,
                attention_mask=None,
                past_key_values=None,
                position_ids=hpos,
                allow_is_causal_skip=False,
            )
            hpe = hm.model.rotary_emb(emb, position_ids=hpos)
            hml = hm.lm_head(hmtp(torch.roll(emb, -1, dims=1), hh_last, hmask, hpe, hpos))
        else:
            hl = hm(input_ids=ids.to(hdev)).logits
        res["hf_forward_seconds"] = round(time.time() - t0, 1)
        if a.replay_topk:
            for i, layer in enumerate(mm.decoder.layers):
                at_i = layer.self_attention.core_attention
                if at_i.is_full:
                    at_i.inject_topk = h_topk[i]
            if hmtp is not None:
                mm.mtp.layers[0].mtp_model_layer.self_attention.core_attention.inject_topk = hmtp.topk
        t0 = time.time()
        pos = torch.arange(a.seq, device=dev).unsqueeze(0).expand(ids.shape[0], -1)
        if a.mtp:  # MTP 要 labels 才跑；输出头被调两次：先 MTP、后主干（[s, b, V]）
            m_tok_loss = mm(ids.to(dev), pos, attention_mask=None, labels=labels.to(dev))
            mml, ml = (t.transpose(0, 1) for t in head_out[-2:])
        else:
            ml = mm(ids.to(dev), pos, attention_mask=None)
        res["mcore_forward_seconds"] = round(time.time() - t0, 1)
    assert ml.dtype == torch.float32 and hl.dtype == torch.float32, (ml.dtype, hl.dtype)

    n, H = hcfg.hc_mult, hcfg.hidden_size
    per_layer = []
    ok = True  # 数值项（层输出、logits、梯度）
    topk_all_ok = True  # 离散项：每个 full 层（含 MTP）的翻转都是平局
    for i in range(a.layers):
        hh = h_out[i]  # [B, S, n, H]
        mo = m_out[i].view(a.seq, ids.shape[0], n, H).transpose(0, 1).to(hh.device)
        r = {"layer": i, "rel_l2": rel_l2(mo, hh), "max_abs": float((mo.double() - hh.double()).abs().max())}
        at = mm.decoder.layers[i].self_attention.core_attention
        r["indexer"] = "full" if at.is_full else f"shared(from layer {at.source_layer - 1})"  # 0 起的层号
        hm_mask = topk_mask(h_topk[i].to(dev), a.seq)
        mm_mask = topk_mask(at.last_topk, a.seq)
        diff_rows = (hm_mask != mm_mask).any(-1)  # [B, S]
        r["topk_rows_differ"] = int(diff_rows.sum())
        r["topk_rows_differ_pos_ge_topk"] = int(diff_rows[:, mcfg.dsa_indexer_topk :].sum())
        r["topk_elems_differ"] = int((hm_mask != mm_mask).sum())
        r["topk_selected_per_row_max"] = int(mm_mask.sum(-1).max())
        if at.is_full:
            r.update(indexer_flip_report(hm.model.layers[i].self_attn.indexer, h_idx_in[i], h_topk[i], at, a.seq, hcfg))
        if not at.is_full:
            src = at.source_layer - 1
            r["shared_equals_source"] = bool(
                torch.equal(at.last_topk, mm.decoder.layers[src].self_attention.core_attention.last_topk)
            )
        # top-k：逐 query 相同；不同的 query 必须都是平局（FP64 下第 k / k+1 名的相对分差 ≤ --tie-gap）
        gap = r.get("topk_model_flip_rel_gap_max")
        r["topk_identical"] = r["topk_rows_differ"] == 0
        if at.is_full:
            # 定责三种情况算过：1) 没翻转；2) 翻转都是平局（FP64 分差 ≤ tie_gap）；3) 上游已有翻转（本层输入因此偏离
            # FP32 量级，模型里的分差不再能当平局判据），且隔离回放（HF 的输入喂 Megatron 的索引器）的翻转都是平局
            r["topk_attribution"] = topk_attribution(
                r["topk_rows_differ"],
                gap,
                r.get("topk_isolated_rows_differ", 0),
                r.get("topk_isolated_flip_rel_gap_max"),
                any(not q["topk_identical"] for q in per_layer),
                a.tie_gap,
            )
            r["topk_ok"] = r["topk_attribution"] != "unattributed"
        else:  # shared 层：top-k 必须就是源层那一份；源层的翻转已在源层定责
            r["topk_ok"] = r["shared_equals_source"] and per_layer[at.source_layer - 1]["topk_ok"]
        ok &= r["rel_l2"] <= a.tol
        topk_all_ok &= r["topk_ok"] and r.get("shared_equals_source", True)
        per_layer.append(r)
        print("LAYER", json.dumps(r), flush=True)
    res["per_layer"] = per_layer
    lg = {
        "rel_l2": rel_l2(ml.to(hl.device), hl),
        "max_abs": float((ml.to(hl.device).double() - hl.double()).abs().max()),
    }
    res["logits"] = lg
    print("LOGITS", json.dumps(lg), flush=True)
    ok &= lg["rel_l2"] <= a.tol

    if a.mtp:
        mlab = torch.roll(labels, -1, dims=1)[:, :-1]
        h_mtp_loss = F.cross_entropy(hml[:, :-1].flatten(0, 1).float(), mlab.flatten().to(hdev))
        m_mtp_loss = F.cross_entropy(mml[:, :-1].flatten(0, 1).float(), mlab.flatten().to(dev))
        at = mm.mtp.layers[0].mtp_model_layer.self_attention.core_attention
        hmask_k, mmask_k = topk_mask(hmtp.topk.to(dev), a.seq), topk_mask(at.last_topk, a.seq)
        mres = {
            "logits_rel_l2": rel_l2(mml[:, :-1].to(hdev), hml[:, :-1]),
            "loss_hf": float(h_mtp_loss),
            "loss_mcore": float(m_mtp_loss),
            "topk_rows_differ": int((hmask_k != mmask_k).any(-1).sum()),
        }
        if not a.replay_topk:
            mres.update(indexer_flip_report(hmtp.self_attn.indexer, h_idx_in["mtp"], hmtp.topk, at, a.seq, hcfg))
        res["mtp_check"] = mres
        print("MTP", json.dumps(mres), flush=True)
        ok &= mres["logits_rel_l2"] <= a.tol
        if not a.replay_topk:  # 定责同主干的 full 层（含「上游已翻转 + 隔离回放全是平局」算继承）
            mres["topk_attribution"] = topk_attribution(
                mres["topk_rows_differ"],
                mres.get("topk_model_flip_rel_gap_max"),
                mres.get("topk_isolated_rows_differ", 0),
                mres.get("topk_isolated_flip_rel_gap_max"),
                any(not q["topk_identical"] for q in per_layer),
                a.tie_gap,
            )
            mres["topk_ok"] = mres["topk_attribution"] != "unattributed"
            topk_all_ok &= mres["topk_ok"]

    if a.backward:
        hloss = F.cross_entropy(hl.flatten(0, 1).float(), labels.flatten().to(hdev))
        mloss = F.cross_entropy(ml.flatten(0, 1).float(), labels.flatten().to(dev))
        if a.mtp:  # 总 loss = 主 + 因子 × MTP；Megatron 的 MTP 梯度由 MTPLossAutoScaler 挂在主干反向上
            hloss = hloss + mcfg.mtp_loss_scaling_factor * h_mtp_loss
            mloss = m_tok_loss.float().mean()  # 反向用它（MTP 梯度自动挂上）；报告时补上 MTP 项，与 HF 同口径
            m_report = float(mloss) + mcfg.mtp_loss_scaling_factor * float(m_mtp_loss)
        else:
            m_report = float(mloss)
        res["loss"] = {"hf": float(hloss), "mcore": m_report, "abs_diff": abs(float(hloss) - m_report)}
        print("LOSS", json.dumps(res["loss"]), flush=True)
        t0 = time.time()
        hloss.backward()
        mloss.backward()
        res["backward_seconds"] = round(time.time() - t0, 1)
        res["grads"] = grad_compare(hm, mm, mcfg, hmtp)
        worst = max(res["grads"], key=lambda g: g["rel_l2"] if g["rel_l2"] is not None else -1)
        res["grad_worst"] = worst
        print("GRAD_WORST", json.dumps(worst), flush=True)
        bad = [g for g in res["grads"] if g["rel_l2"] is not None and g["rel_l2"] > a.tol_grad]
        frozen = [g for g in res["grads"] if g["rel_l2"] is None]
        res["grads_over_tol"] = len(bad)
        res["grads_none_both"] = len(frozen)
        ok &= not bad and all(g["both_none"] for g in frozen)
    # 判定：有翻转时数值项会被翻转带偏，数值以 --replay-topk 那次为准；这次只要求翻转都是平局
    any_flip = any(not r["topk_identical"] for r in per_layer) or (a.mtp and res["mtp_check"]["topk_rows_differ"] > 0)
    res["numeric_ok"], res["topk_all_ok"], res["any_flip"] = bool(ok), bool(topk_all_ok), bool(any_flip)
    res["pass"] = bool(topk_all_ok and (ok or (any_flip and not a.replay_topk)))
    print("PASS" if ok else "FAIL", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


def topk_attribution(rows_differ, gap, iso_rows, iso_gap, upstream_flip, tie_gap):
    """full 层（含 MTP）top-k 翻转的定责，返回原因：

    identical：没有翻转；tie：模型里的翻转都是平局（FP64 第 k / k+1 名相对分差 ≤ tie_gap）；
    inherited_flip：上游已有翻转、本层输入因此偏离，而隔离回放（HF 的输入喂 Megatron 的索引器）的翻转都是平局；
    input_drift：上游没有翻转，但隔离回放的翻转都是平局，说明两边索引器在同一输入上只差平局，
    模型里多出来的翻转来自上游连续的 FP32 误差（这些误差本身由层输出与 logits 的数值检查把关）；
    unattributed：以上都不是。
    """
    if rows_differ == 0:
        return "identical"
    if gap is not None and gap <= tie_gap:
        return "tie"
    iso_tie = iso_rows == 0 or (iso_gap is not None and iso_gap <= tie_gap)
    if iso_tie:
        return "inherited_flip" if upstream_flip else "input_drift"
    return "unattributed"


@torch.no_grad()
def indexer_flip_report(hidx, inputs, h_topk, at, sq, hcfg):
    """DSA 选择翻转的定责：

    - 隔离回放：把 HF 索引器的输入（同一层、同一份 x 与 q_resid）原样喂给 Megatron 的索引器，比两边 top-k；
      排除上游几层的 FP32 误差，只剩两边打分写法（求和顺序）的差。
    - FP64 参照：用 HF 的输入与权重在 FP64 里重算索引分数，对每个有差的 query 记第 k 名与第 k+1 名的分差
      （相对第 k 名的分数）；两边打分误差都在 FP32 量级，分差在这个量级之内的翻转是平局，不是写法不同。
    """
    from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive

    hs, qres = inputs  # [B, S, H], [B, S, q_lora]
    dev = at.sinks.device
    x, qr = hs.transpose(0, 1).to(dev), qres.transpose(0, 1).to(dev)
    q, k, w = at.indexer.forward_before_topk(x, qr, None)
    causal = torch.full((sq, sq), float("-inf"), device=dev).triu(1)
    _, iso = fused_qk_topk_naive(q, k, w, at.topk, mask=causal, use_relu=True)
    h_mask, iso_mask, m_mask = topk_mask(h_topk.to(dev), sq), topk_mask(iso, sq), topk_mask(at.last_topk, sq)
    # FP64 参照分数（HF 的写法，交错 RoPE）
    D, R, nh = hidx.head_dim, hidx.qk_rope_head_dim, hidx.n_heads
    hd, qd = hs.double(), qres.double()
    qq = (qd @ hidx.wq_b.weight.double().t()).view(*qd.shape[:2], nh, D)
    kk = F.layer_norm(
        hd @ hidx.wk.weight.double().t(), (D,), hidx.k_norm.weight.double(), hidx.k_norm.bias.double(), 1e-6
    )
    inv = 1.0 / (
        hcfg.rope_parameters["rope_theta"] ** (torch.arange(0, R, 2, dtype=torch.float64, device=hd.device) / R)
    )
    ang = torch.arange(sq, dtype=torch.float64, device=hd.device)[:, None] * inv[None]
    c, s_ = ang.cos(), ang.sin()

    def rope(t, unsq):
        cc, ss = (c.unsqueeze(unsq), s_.unsqueeze(unsq))
        x1, x2 = t[..., 0::2], t[..., 1::2]
        return torch.stack((x1 * cc - x2 * ss, x1 * ss + x2 * cc), -1).flatten(-2)

    qq = torch.cat([qq[..., : D - R], rope(qq[..., D - R :], 1)], -1)
    kk = torch.cat([kk[..., : D - R], rope(kk[..., D - R :], 0)], -1)
    wts = (hd @ hidx.weights_proj.weight.double().t()) * nh**-0.5 * D**-0.5
    out = {}
    for tag, other in (("model", m_mask), ("isolated", iso_mask)):
        rows = (h_mask != other).any(-1).nonzero().tolist()
        out[f"topk_{tag}_rows_differ"] = len(rows)
        gaps = []
        for b, srow in rows:
            sc = torch.relu(torch.einsum("hd,td->ht", qq[b, srow], kk[b, : srow + 1]))
            sc = (wts[b, srow][:, None] * sc).sum(0)
            top = torch.sort(sc, descending=True).values
            kth = at.topk
            gaps.append(float((top[kth - 1] - top[kth]) / top[kth - 1].abs().clamp_min(1e-300)))
        out[f"topk_{tag}_flip_rel_gap_max"] = max(gaps) if gaps else None
    return out


def grad_compare(hm, mm, mcfg, hmtp=None):
    """逐参数比梯度。名字对照沿用 hy4_weights 的表（checkpoint 名 → HF 模块名）。"""
    from hy4_weights import ATTN, GLOBAL, INDEXER, expert_param_name, layer_plan, mtp_names

    hp = dict(hm.named_parameters())
    if hmtp is not None:  # HF 一侧的 MTP 模块名前面补上 checkpoint 的前缀，好按同一张表找
        hp.update({"model.mtp_layers.0." + k: v for k, v in hmtp.named_parameters()})
    mp = dict(mm.named_parameters())
    out = []

    def cmp(mname, hgrad_fn, label):
        m = mp[mname]
        hg = hgrad_fn()
        mg = m.grad
        if mg is None and hg is not None and not hg.any():
            mg = torch.zeros_like(hg)  # 没有 token 的专家：一边没有梯度、另一边是全 0，等价
        if hg is None and mg is not None and not mg.any():
            hg = torch.zeros_like(mg)
        if mg is None or hg is None:
            out.append({"param": mname, "hf": label, "rel_l2": None, "both_none": mg is None and hg is None})
            return
        if not hg.any() and not mg.any():
            out.append({"param": mname, "hf": label, "rel_l2": 0.0, "norm": 0.0})
            return
        out.append(
            {"param": mname, "hf": label, "rel_l2": rel_l2(mg.to(hg.device), hg), "norm": float(hg.double().norm())}
        )

    def hgrad(name):
        g = hp[name].grad
        return None if g is None else g

    for m, h in GLOBAL:
        n = ckpt_to_hf(h)
        cmp(m, lambda n=n: hgrad(n), n)
    del ATTN, INDEXER
    grouped = any(".experts.linear_fc1.weight" in k for k in mp)
    for i in range(mcfg.num_layers):
        pre_m, pre_h = f"decoder.layers.{i}.", f"model.layers.{i}."
        for m, h, tf in layer_plan(mcfg, i):
            if m.endswith("expert_bias"):
                continue
            if tf == "cat":
                names = [ckpt_to_hf(pre_h + x) for x in h]
                cmp(
                    pre_m + m,
                    lambda names=names: (
                        None if hp[names[0]].grad is None else torch.cat([hp[x].grad for x in names], 0)
                    ),
                    "+".join(names),
                )
            else:
                n = ckpt_to_hf(pre_h + h)
                cmp(pre_m + m, lambda n=n: hgrad(n), n)
        if mcfg.moe_layer_freq[i]:
            gu, dn = hp[pre_h + "mlp.experts.gate_up_proj"], hp[pre_h + "mlp.experts.down_proj"]
            for e in range(mcfg.num_moe_experts):
                cmp(
                    expert_param_name(pre_m, "linear_fc1", e, grouped),
                    lambda e=e: None if gu.grad is None else gu.grad[e],
                    f"gate_up_proj[{e}]",
                )
                cmp(
                    expert_param_name(pre_m, "linear_fc2", e, grouped),
                    lambda e=e: None if dn.grad is None else dn.grad[e],
                    f"down_proj[{e}]",
                )
    if hmtp is not None:
        for m, h, tf in mtp_names(mcfg):
            if m.endswith("expert_bias"):
                continue
            if tf == "experts":
                gu, dn = hp[h + "mlp.experts.gate_up_proj"], hp[h + "mlp.experts.down_proj"]
                for e in range(mcfg.num_moe_experts):
                    cmp(
                        expert_param_name(m, "linear_fc1", e, grouped),
                        lambda e=e, gu=gu: None if gu.grad is None else gu.grad[e],
                        f"mtp.gate_up_proj[{e}]",
                    )
                    cmp(
                        expert_param_name(m, "linear_fc2", e, grouped),
                        lambda e=e, dn=dn: None if dn.grad is None else dn.grad[e],
                        f"mtp.down_proj[{e}]",
                    )
            elif tf == "cat":
                names = [ckpt_to_hf(x) for x in h]
                cmp(
                    m,
                    lambda names=names: (
                        None if hp[names[0]].grad is None else torch.cat([hp[x].grad for x in names], 0)
                    ),
                    "+".join(names),
                )
            else:
                n = ckpt_to_hf(h)
                cmp(m, lambda n=n: hgrad(n), n)
    return out


if __name__ == "__main__":
    main()
