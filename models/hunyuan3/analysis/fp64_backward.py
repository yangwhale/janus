"""混元 3（截 4 层）单层反向的 FP64 参照：给定层输入 x 与输出端余切 dy，全程 FP64 求 dx = (∂layer/∂x)ᵀ · dy。

用途：逐层隔离回放的反向（L2.layer{k}.bwd）。GPU 自己重跑的差只有反向原子加的不确定性，越靠后的层越小（第 3 层 1.3e-5），
TPU 只改执行方式的重跑差得更少（1e-5 – 1e-4），两边都量不出「两种硬件各自的 BF16 舍入顺序不同」这一份。和前向一样，
换一把对真值的尺子：R = rel_l2(被测 dx, FP64 dx) / rel_l2(GPU dx, FP64 dx)，被测不比 GPU 离真值远就算对齐
（fwd_rratio.py --mode bwd）。

语义与 fp64_forward.layer_forward 逐式相同（同一份 BF16 权重、HF 配置）；用 torch 的 float64 自动求导写成可微的版本：
路由按 FP64 分数（scores + expert_bias）选专家，选谁不可微（与真实训练相同），路由权重（选中的 sigmoid 分数归一化 × 缩放）可微。
"""
from __future__ import annotations

import fp64_forward as F
import numpy as np
import torch


def _t(a):
    return torch.as_tensor(np.asarray(a, np.float64), dtype=torch.float64)


def rope_tables(c, pos):
    d, theta = c["head_dim"], c["rope_parameters"]["rope_theta"]
    inv = theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    ang = _t(pos)[:, None] * inv[None, :]
    return torch.cat([torch.cos(ang)] * 2, -1)[:, None, :], torch.cat([torch.sin(ang)] * 2, -1)[:, None, :]


def layer_torch(c: dict, W, i: int, x, cos, sin, bias, counts=None):
    """第 i 层的可微 FP64 前向。W(HF 全名) → FP64 张量（可带梯度）；x [S, D]；bias：路由偏置（不可微）。
    counts 给了就把本层每个专家选中的 token 数累加进去（numpy [E]，算偏置更新用）。"""
    H, KV, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    eps = c["rms_norm_eps"]
    p = f"model.layers.{i}."
    S = x.shape[0]

    def rms(v, w):
        return v * torch.rsqrt((v * v).mean(-1, keepdim=True) + eps) * w

    def rope(v):
        h = d // 2
        return v * cos + torch.cat([-v[..., h:], v[..., :h]], -1) * sin

    def swiglu(h, pre):
        return (torch.nn.functional.silu(h @ W(pre + "gate_proj.weight").T) * (h @ W(pre + "up_proj.weight").T)) @ W(pre + "down_proj.weight").T

    h1 = rms(x, W(p + "input_layernorm.weight"))
    q = (h1 @ W(p + "self_attn.q_proj.weight").T).reshape(S, H, d)
    k = (h1 @ W(p + "self_attn.k_proj.weight").T).reshape(S, KV, d)
    v = (h1 @ W(p + "self_attn.v_proj.weight").T).reshape(S, KV, d)
    qn, kn = rope(rms(q, W(p + "self_attn.q_norm.weight"))), rope(rms(k, W(p + "self_attn.k_norm.weight")))
    rep = H // KV
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), 1)
    heads = []
    for hh in range(H):
        s = (qn[:, hh] @ kn[:, hh // rep].T) / np.sqrt(d)
        s = s.masked_fill(mask, float("-inf"))
        heads.append(torch.softmax(s, -1) @ v[:, hh // rep])
    core = torch.stack(heads, 1).reshape(S, H * d)
    hres = x + core @ W(p + "self_attn.o_proj.weight").T
    hn = rms(hres, W(p + "post_attention_layernorm.weight"))
    if i < c["first_k_dense_replace"]:
        return hres + swiglu(hn, p + "mlp.")
    E, K = c["num_experts"], c["num_experts_per_tok"]
    scores = torch.sigmoid(hn @ W(p + "mlp.router.gate.weight").T)
    with torch.no_grad():
        sel = torch.as_tensor(F.topk_sets((scores + _t(bias)).numpy(), K))
    if counts is not None:
        counts += np.bincount(sel.numpy().reshape(-1), minlength=E)
    w = torch.gather(scores, 1, sel)
    w = w / w.sum(-1, keepdim=True) * c["router_scaling_factor"]
    routed = torch.zeros_like(hn)
    for e in range(E):
        rows, slot = torch.nonzero(sel == e, as_tuple=True)
        if rows.numel() == 0:
            continue
        routed = routed.index_add(0, rows, w[rows, slot][:, None] * swiglu(hn[rows], p + f"mlp.experts.{e}."))
    return hres + routed + swiglu(hn, p + "mlp.shared_mlp.")


def layer_vjp(hf: F.HF, i: int, x, dy, pos, bias=None) -> np.ndarray:
    """x、dy：[S, D]（任意 dtype）。返回 dx [S, D]（FP64）。"""
    c = hf.cfg
    D = c["hidden_size"]
    x = _t(np.asarray(x, np.float64).reshape(-1, D)).requires_grad_(True)
    dy = _t(np.asarray(dy, np.float64).reshape(-1, D))
    cos, sin = rope_tables(c, pos)
    b = None
    if i >= c["first_k_dense_replace"]:
        b = hf.get(f"model.layers.{i}.mlp.expert_bias") if bias is None else bias
    out = layer_torch(c, lambda n: _t(hf.get(n)), i, x, cos, sin, b)
    (dx,) = torch.autograd.grad(out, x, grad_outputs=dy)
    return dx.numpy()
