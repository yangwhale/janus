"""阶段 A 单算子回放的 GPU 一侧参照模型（analysis/fp64_ops.py 用它在 FP64 里重算单个算子）：
HF transformers `hy_v4`（v5.22.1）原样实现 + 三处有出处的改动。

1. RoPE 交错读法、两个 eps 互换（MLA latent norm 1e-5、索引器 k_norm 1e-6）：与 vLLM 8b55f919 / SGLang b8d86350
   读 checkpoint 的方式一致。交错 RoPE 在运行时按交错布局做，
   不重排权重；q·k 与 TPU 一侧「转换时去交错 + 拼接式 RoPE」数学等价。
2. MTP：HF 不建 MTP（加载时丢 `mtp_layers.*`）。按 vLLM 8b55f919 `vllm/models/hy_v4/nvidia/mtp.py` 的前向补上：
   enorm(emb) ‖ hnorm(h) → eh_proj → 不带 iHC 的解码层（自带 full 索引器）→ final_layernorm → 共用输出头；
   h 取主干 `norm(hc_head(·))` 之后。训练错位按 DeepSeek-V3 / MaxText：位置 i 用 h_i 与 emb(t_{i+1})，预测 t_{i+2}
   （推断：模型发布方没公开 MTP 训练写法）。
3. 输出头 dgrad 分段累加（`ChunkedHead`）：词表每 `chunk` 一段，段内在输出头本身的精度里算，段结果与段间累加留 FP32。
   依据是混元 3 在 B200 上对 FP64 的实测（janus models/hunyuan3/ALIGNMENT-REPORT.md「根子在 GPU 一侧」）。

精度：HF `_keep_in_fp32_modules_strict` 里的模块保持 FP32（输出头、iHC、hc_head、sinks、索引器 weights_proj / k_norm）。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask
from transformers.models.hy_v4 import modeling_hy_v4 as M

RENAMES = [
    (".hc_pre.hc_", ".hc_"),
    (".learnable_sink_param", ".sinks"),
    (".linear_gate", ".gate_proj"),
    (".hc_attn_layer.hc_fn", ".attn_hc.fn"),
    (".hc_attn_layer.hc_base", ".attn_hc.base"),
    (".hc_attn_layer.hc_scale", ".attn_hc.scale"),
    (".hc_mlp_layer.hc_fn", ".ffn_hc.fn"),
    (".hc_mlp_layer.hc_base", ".ffn_hc.base"),
    (".hc_mlp_layer.hc_scale", ".ffn_hc.scale"),
    (".hc_head_fn", ".hc_fn"),
    (".hc_head_base", ".hc_base"),
    (".hc_head_scale", ".hc_scale"),
]  # transformers v5.22.1 conversion_mapping.py 的 "hy_v4"


def interleaved_rope(q, k, cos, sin, unsqueeze_dim=1):
    """交错（GPT-J）RoPE：第 2i、2i+1 维成对、用第 i 个频率旋转，输出仍是交错布局（vLLM is_neox_style=False）。"""
    half = cos.shape[-1] // 2
    c = cos[..., :half].unsqueeze(unsqueeze_dim)
    s = sin[..., :half].unsqueeze(unsqueeze_dim)

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)

    return rot(q), rot(k)


def apply_patches():
    M.apply_rotary_pos_emb = interleaved_rope


class _ChunkedHeadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, chunk):
        ctx.save_for_backward(x, w)
        ctx.chunk = chunk
        return F.linear(x, w)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        V = w.shape[0]
        gx = torch.zeros(gy.shape[:-1] + (w.shape[1],), dtype=torch.float32, device=gy.device)
        for s in range(0, V, ctx.chunk):
            gx += (gy[..., s : s + ctx.chunk] @ w[s : s + ctx.chunk]).float()
        gw = gy.reshape(-1, V).t() @ x.reshape(-1, x.shape[-1])
        return gx.to(x.dtype), gw.to(w.dtype), None


class MTPLayer(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        c = copy.deepcopy(cfg)
        c.indexer_types = list(cfg.indexer_types) + ["full"] * (layer_idx + 1 - len(cfg.indexer_types))
        c.mlp_layer_types = list(cfg.mlp_layer_types) + ["sparse"] * (layer_idx + 1 - len(cfg.mlp_layer_types))
        H, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.enorm = M.HYV4RMSNorm(H, eps=eps)
        self.hnorm = M.HYV4RMSNorm(H, eps=eps)
        self.eh_proj = nn.Linear(2 * H, H, bias=False)
        self.input_layernorm = M.HYV4RMSNorm(H, eps=eps)
        self.self_attn = M.HYV4Attention(c, layer_idx)
        self.post_attention_layernorm = M.HYV4RMSNorm(H, eps=eps)
        self.mlp = M.HYV4MoE(c)
        self.final_layernorm = M.HYV4RMSNorm(H, eps=eps)

    def forward(self, emb, h, mask, pe, pos):
        x = self.eh_proj(torch.cat([self.enorm(emb), self.hnorm(h)], dim=-1))
        a, _, _ = self.self_attn(
            hidden_states=self.input_layernorm(x),
            attention_mask=mask,
            position_embeddings=pe,
            position_ids=pos,
            prev_topk_indices=None,
        )
        x = x + a
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.final_layernorm(x)


class Root(nn.Module):
    """主干 + （可选）MTP。forward 返回 (总 loss, 主 loss, MTP loss, 逐 token 主 loss)。"""

    def __init__(self, cfg, mtp: bool, head_chunk: int):
        super().__init__()
        self.hf = M.HYV4ForCausalLM(cfg)
        self.mtp = MTPLayer(cfg, cfg.num_hidden_layers) if mtp else None
        self.head_chunk = head_chunk

    def head(self, h):
        w = self.hf.lm_head.weight
        x = h.to(w.dtype)
        if self.head_chunk:
            return _ChunkedHeadFn.apply(x, w, self.head_chunk)
        return F.linear(x, w)

    def forward(self, ids, labels, mtp_factor):
        m = self.hf.model
        cfg = m.config
        emb = m.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        mask = create_causal_mask(
            config=cfg,
            inputs_embeds=emb,
            attention_mask=None,
            past_key_values=None,
            position_ids=pos,
            allow_is_causal_skip=False,
        )
        pe = m.rotary_emb(emb, position_ids=pos)
        h = emb.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
        topk = None
        for layer in m.layers:
            h, topk = layer(h, attention_mask=mask, position_embeddings=pe, position_ids=pos, prev_topk_indices=topk)
        h = m.norm(m.hc_head(h))
        logits = self.hf.lm_head(h)  # 输出头单元的 forward 换成 head()，见 install_head
        tok_loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), reduction="none")
        loss = tok_loss.mean()
        mtp_loss = torch.zeros((), device=ids.device)
        if self.mtp is not None:
            emb_next = torch.roll(emb, -1, dims=1)  # 位置 i 用 emb(t_{i+1})；最后一位无效
            hm = self.mtp(emb_next, h, mask, pe, pos)
            lm = self.hf.lm_head(hm)
            mtp_labels = torch.roll(labels, -1, dims=1)  # labels 已是 t_{i+1}，再挪一位是 t_{i+2}
            mtp_loss = F.cross_entropy(lm[:, :-1].flatten(0, 1).float(), mtp_labels[:, :-1].flatten())
        return loss + mtp_factor * mtp_loss, loss, mtp_loss, tok_loss


def install_head(root: Root):
    """把输出头模块的 forward 换成（分段 dgrad 的）head()，FSDP 的前后钩子仍挂在输出头模块上。"""
    lm = root.hf.lm_head
    lm.forward = lambda x: root.head(x)


def ckpt_to_model_name(n: str) -> str:
    for a, b in RENAMES:
        n = n.replace(a, b)
    if n.startswith("model.mtp_layers.0."):
        return "mtp." + n[len("model.mtp_layers.0.") :]
    return "hf." + n


def keep_fp32(name: str) -> bool:
    """HF `_keep_in_fp32_modules_strict` 里除输出头以外的模块（输出头单独作 FP32 的 FSDP 单元）。"""
    return (
        ".attn_hc." in name
        or ".ffn_hc." in name
        or ".hc_head." in name
        or name.endswith(".sinks")
        or ".indexer.weights_proj." in name
        or ".indexer.k_norm." in name
    )


def build(cfg_path: str, n_layers: int, mtp: bool, head_chunk: int, experts_impl: str = "eager"):
    apply_patches()
    cfg = M.HYV4Config.from_pretrained(cfg_path)
    cfg.num_hidden_layers = n_layers
    for k in ("indexer_types", "layer_types", "mlp_layer_types"):
        setattr(cfg, k, list(getattr(cfg, k))[:n_layers])
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = experts_impl
    with torch.device("meta"):
        root = Root(cfg, mtp, head_chunk)
    attns = [l.self_attn for l in root.hf.model.layers] + ([root.mtp.self_attn] if mtp else [])
    for at in attns:
        at.q_a_layernorm.variance_epsilon = 1e-5
        at.kv_a_layernorm.variance_epsilon = 1e-5
        if at.indexer is not None:
            at.indexer.k_norm.eps = 1e-6
    return cfg, root, attns
