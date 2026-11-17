"""诊断（HY4_TPU_LIKE_ATTNK）：带 sink 的稀疏注意力照 TPU 上 tokamax splash 内核（0.0.14，训练镜像里的版本）的
舍入写法算，用来在 GPU 上复现 TPU 一侧注意力内核里的舍入差（01-mini D30「扫描看不到的差」）。只依赖 torch。

基线是 hy4_mcore.sparse_sink_attention 在 hy4_attn_scores_fp32 开时的 autograd：q·k 在 FP32 里算，FP32 softmax
（含 sink 一列），概率舍 BF16 乘 V；反向由 autograd 穿过这几步。四处可以分别换成 splash 的写法
（HY4_TPU_LIKE_ATTNK_PARTS，逗号分隔，默认 all）：

- ``dp``：反向 dP = dO·Vᵀ。基线是 BF16 矩阵乘、输出舍 BF16；splash 的 dq / dkv 内核里是 FP32（preferred_element_type
  float32），不舍。
- ``di``：softmax 反向的修正项。基线（PyTorch softmax 反向）是 Σ_j dP_j·P_j（P 是 FP32 概率，dP 按上一项）；splash 是
  di = Σ_d O·dO（O 是 BF16 的注意力输出，在 _splash_attention_bwd 里升 FP32 后求和）。
- ``ds``：dS 进 dq、dk 的矩阵乘。基线 dS 留 FP32，乘 FP32 的 k / q（scores 在 FP32 里算），dq、dk 最后舍一次 BF16；
  splash 是 ds.astype(bf16) 再乘 BF16 的 k / q（FP32 累加），dq 出内核舍 BF16。MaxText 在进内核前把 q 乘 softmax 缩放
  （BF16），这里同样：dk 乘的是 bf16(q·scale)，dq 是 bf16(bf16(dS)·k)·scale。缩放是 2 的幂时（混元 4 mini：
  (192 + 64)^-0.5 = 1/16）这一步不引入舍入。
- ``pv``：前向 P 进 PV。基线是归一化后的概率舍 BF16 再乘 V；splash 是在线 softmax：kv 按 block_kv_compute（T1 配置
  512，HY4_TPU_LIKE_ATTNK_BLOCK）一块，m 从 sink 起、l 从 1 起，每块 m' = max(m, 块内最大)，e = exp(s − m')，
  α = exp(m − m')，l = α·l + Σe，o = α·o + bf16(e)·V（FP32 累加），最后 o·(1/l) 舍 BF16。内核里那次矩阵乘是 FP32 的 e
  乘 BF16 的 V、不指定精度，实测等于 e 先舍 BF16：这个写法对 TPU 上实跑的 splash 输出逐元素相同的
  比例 99.80%–99.97%（基线写法 89.3%–97.6%；analysis/splash_pv_rounding_check.py，sink 探针 tw-sinkprobe2 的输入）。
- ``pv32``：P 不舍，FP32 的 exp(s − m) 乘 V，最后乘 1/l、舍 BF16。不是 TPU 的写法（上面那次核对里逐元素相同 93.0%–98.5%），
  留作对照。代码包 janus-hy4-attnk-3cc20d891ecf 里它叫 ``pv``，那个包的 all 是 dp、di、ds 加它。

不在开关里、两边本来一致的：dv = Pᵀ·dO（两边 P 都舍 BF16）；sink 的梯度照 GPU 写法 −Σ p_sink·Σ_j bf16(dP_j)·P_j
（TPU 对齐配置开 M47 后也是这个写法）；q·k 本身（两边都是 BF16 输入、FP32 累加）。
"""

from __future__ import annotations

import os

import torch

PARTS = ("dp", "di", "ds", "pv")  # all：照 splash 的四处
EXTRA = ("pv32",)  # 只作对照，不在 all 里


def parts_from_env() -> frozenset[str]:
    """HY4_TPU_LIKE_ATTNK=1 时返回要换的几处（HY4_TPU_LIKE_ATTNK_PARTS，默认 all）；没开返回空集。"""
    if os.environ.get("HY4_TPU_LIKE_ATTNK") != "1":
        return frozenset()
    raw = os.environ.get("HY4_TPU_LIKE_ATTNK_PARTS", "all").strip()
    if raw in ("", "all"):
        return frozenset(PARTS)
    got = frozenset(p.strip() for p in raw.split(",") if p.strip())
    bad = got - set(PARTS) - set(EXTRA)
    if bad:
        raise ValueError(f"HY4_TPU_LIKE_ATTNK_PARTS 里有不认识的项 {sorted(bad)}（可选 {PARTS + EXTRA} 或 all）")
    if {"pv", "pv32"} <= got:
        raise ValueError("HY4_TPU_LIKE_ATTNK_PARTS 里 pv 与 pv32 只能选一个")
    return got


def block_from_env() -> int:
    """splash 前向的 block_kv_compute（HY4_TPU_LIKE_ATTNK_BLOCK，默认 512，同 T1 配置的 sa_block_kv_compute）。"""
    return int(os.environ.get("HY4_TPU_LIKE_ATTNK_BLOCK", "512"))


def _work_dtype(x: torch.Tensor) -> torch.dtype:
    """FP32；输入是 FP64 时用 FP64（单测里验证公式在没有舍入时与基线等价）。"""
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _online_pv(scores, sink, v, f, block: int):
    """splash 前向的在线 softmax 与 PV（见模块说明 ``pv``）。scores [b, h, sq, sk]（被掩的是 -inf），sink [b, h, sq, 1]。"""
    m = sink.clone()
    l = torch.ones_like(m)  # noqa: E741
    o = torch.zeros(*scores.shape[:-1], v.shape[-1], dtype=f, device=scores.device)
    for c0 in range(0, scores.shape[-1], block):
        sb = scores[..., c0 : c0 + block]
        m_new = torch.maximum(m, sb.amax(dim=-1, keepdim=True))
        e = torch.exp(sb - m_new)
        alpha = torch.exp(m - m_new)
        l = alpha * l + e.sum(dim=-1, keepdim=True)  # noqa: E741
        o = alpha * o + torch.matmul(e.to(v.dtype).to(f), v[..., c0 : c0 + block, :].to(f))
        m = m_new
    return (o * (1.0 / l)).to(v.dtype)


class TPULikeSinkAttention(torch.autograd.Function):
    """q / k / v [b, h, s, d]（BF16），allowed [b, 1, sq, sk]（bool），sinks [h]（FP32）。返回 (out, max_logit)：
    out [b, h, sq, dv]（v 的 dtype）；max_logit [h]（FP32，每头缩放后 q·k 在允许位置上的最大值、不含 sink，QK-Clip 用，
    不求导）。"""

    @staticmethod
    def forward(ctx, q, k, v, allowed, sinks, scale: float, parts: frozenset, block: int = 512):
        b, h, sq, _ = q.shape
        f = _work_dtype(q)
        scores = torch.matmul(q.to(f), k.to(f).transpose(-1, -2)) * scale
        scores = scores.masked_fill(~allowed, float("-inf"))
        sink = sinks.to(f).view(1, h, 1, 1).expand(b, h, sq, 1)
        comb = torch.cat([scores, sink], dim=-1)
        m = comb.max(dim=-1, keepdim=True).values
        e = torch.exp(comb - m)
        l = e.sum(dim=-1, keepdim=True)  # noqa: E741
        p_all = e / l
        p32, p_sink = p_all[..., :-1], p_all[..., -1:]
        if "pv" in parts:
            out = _online_pv(scores, sink, v, f, block)
        elif "pv32" in parts:
            out = (torch.matmul(e[..., :-1], v.to(f)) * (1.0 / l)).to(v.dtype)
        else:
            out = torch.matmul(p32.to(v.dtype), v)
        ctx.save_for_backward(q, k, v, p32, p_sink, out)
        ctx.scale, ctx.parts = scale, parts
        max_logit = scores.amax(dim=(0, 2, 3))
        ctx.mark_non_differentiable(max_logit)
        return out, max_logit

    @staticmethod
    def backward(ctx, d_out, _d_max):
        q, k, v, p32, p_sink, out = ctx.saved_tensors
        scale, parts = ctx.scale, ctx.parts
        f = _work_dtype(q)
        d_out = d_out.to(v.dtype)
        dv = torch.matmul(p32.to(v.dtype).transpose(-1, -2), d_out)
        if "dp" in parts:
            dp = torch.matmul(d_out.to(f), v.to(f).transpose(-1, -2))
            dp_bf = dp.to(v.dtype).to(f)
        else:
            dp_bf = torch.matmul(d_out, v.transpose(-1, -2)).to(f)
            dp = dp_bf
        di_gpu = (dp_bf * p32).sum(dim=-1, keepdim=True)
        if "di" in parts:
            di = (out.to(f) * d_out.to(f)).sum(dim=-1, keepdim=True)
        elif "dp" in parts:
            di = (dp * p32).sum(dim=-1, keepdim=True)
        else:
            di = di_gpu
        ds = p32 * (dp - di)
        if "ds" in parts:
            ds_b = ds.to(q.dtype)
            dq = (torch.matmul(ds_b, k).to(f) * scale).to(q.dtype)
            dk = torch.matmul(ds_b.transpose(-1, -2), (q.to(f) * scale).to(q.dtype))
        else:
            g = ds * scale
            dq = torch.matmul(g, k.to(f)).to(q.dtype)
            dk = torch.matmul(g.transpose(-1, -2), q.to(f)).to(k.dtype)
        d_sinks = -(p_sink * di_gpu).sum(dim=(0, 2, 3))
        return dq, dk, dv, None, d_sinks, None, None, None
