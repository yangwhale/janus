"""Megatron-Core QK-Clip 在「某个头的最大 logit ≤ 0」时的行为：最小 CPU 复现（直接调 Megatron-Core 自己的 clip_qk）。

    python qk_clip_nonpositive_repro.py --megatron <Megatron-LM 源码目录> [--json out.json]

- 不建模型、不起分布式：给 MLASelfAttention.clip_qk / SelfAttention.clip_qk 一个只带它们读到的属性的假 self，
  方法本身原样取自给定的 Megatron-LM 源码（同一份代码，在训练里由 megatron.core.optimizer.qk_clip.clip_qk 调用）。
- 场景：同一层两个头，头 0 的最大 logit 超过阈值（触发裁剪），头 1 的最大 logit 是负数（例如注意力全部落在
  sink 上、q·k 全为负）。Megatron-Core 只要有一个头超过阈值就对整层计算 η = clamp(τ / s_max, max=1)，
  负的 s_max 给出负的 η，再取 η^α（α = 0.5）就是 NaN；MLA 的 RoPE 部分乘 η 本身，符号翻转。
- 期望的写法（MaxText 一侧的修法同此）：s_max ≤ τ 的头（含 s_max ≤ 0）η = 1，不裁。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types


def fake_mla(MLA, n_heads, nope, pe, v, q_rank, kv_rank, max_logits, tau, alpha):
    import torch

    cfg = types.SimpleNamespace(
        qk_clip=True,
        qk_clip_threshold=tau,
        qk_clip_alpha=alpha,
        q_lora_rank=q_rank,
        qk_head_dim=nope,
        qk_pos_emb_head_dim=pe,
        v_head_dim=v,
    )
    g = torch.Generator().manual_seed(0)
    s = types.SimpleNamespace(
        config=cfg,
        num_attention_heads_per_partition=n_heads,
        cache_mla_latents=False,
        core_attention=types.SimpleNamespace(current_max_attn_logits=torch.tensor(max_logits)),
        linear_q_up_proj=types.SimpleNamespace(weight=torch.randn(n_heads * (nope + pe), q_rank, generator=g)),
        linear_kv_up_proj=types.SimpleNamespace(weight=torch.randn(n_heads * (nope + v), kv_rank, generator=g)),
    )
    s._clip_q_proj_weight = types.MethodType(MLA._clip_q_proj_weight, s)
    s._clip_kv_proj_weight = types.MethodType(MLA._clip_kv_proj_weight, s)
    return s


def fake_gqa(SA, n_groups, heads_per_group, head_dim, hidden, max_logits, tau, alpha):
    import torch

    cfg = types.SimpleNamespace(qk_clip=True, qk_clip_threshold=tau, qk_clip_alpha=alpha)
    qp, kvp = n_groups * heads_per_group * head_dim, n_groups * head_dim
    g = torch.Generator().manual_seed(0)
    s = types.SimpleNamespace(
        config=cfg,
        num_attention_heads_per_partition=n_groups * heads_per_group,
        num_query_groups_per_partition=n_groups,
        query_projection_size=qp,
        kv_projection_size=kvp,
        core_attention=types.SimpleNamespace(current_max_attn_logits=torch.tensor(max_logits)),
        linear_qkv=types.SimpleNamespace(weight=torch.randn(qp + 2 * kvp, hidden, generator=g)),
    )
    s._clip_linear_qkv = types.MethodType(SA._clip_linear_qkv, s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--megatron", required=True, help="Megatron-LM 源码目录（含 megatron/core）")
    ap.add_argument("--json")
    a = ap.parse_args()
    sys.path.insert(0, a.megatron)
    import torch
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention

    try:
        rev = subprocess.run(
            ["git", "-C", a.megatron, "log", "-1", "--format=%h %s"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        rev = ""
    tau, alpha = 100.0, 0.5
    out = {"megatron": rev, "tau": tau, "alpha": alpha, "cases": {}}

    # MLA：2 个头，nope 4 / pe 2 / v 4，q_lora 8 / kv_lora 8
    for name, ml in (
        ("mla_one_head_above_tau_one_negative", [150.0, -0.5]),
        ("mla_one_head_above_tau_one_zero", [150.0, 0.0]),
        ("mla_control_all_positive", [150.0, 50.0]),
    ):
        s = fake_mla(MLASelfAttention, 2, 4, 2, 4, 8, 8, ml, tau, alpha)
        q0, kv0 = s.linear_q_up_proj.weight.clone(), s.linear_kv_up_proj.weight.clone()
        MLASelfAttention.clip_qk(s)
        eta = s.qk_clip_balancing_eta.view(-1)
        q1 = s.linear_q_up_proj.weight.view(2, 6, -1)
        kv1 = s.linear_kv_up_proj.weight.view(2, 8, -1)
        r = {
            "max_logits": ml,
            "eta": eta.tolist(),
            "q_nope_nan_per_head": [bool(q1[h, :4].isnan().any()) for h in range(2)],
            "q_pe_scale_per_head": [float((q1[h, 4:] / q0.view(2, 6, -1)[h, 4:]).mean()) for h in range(2)],
            "k_nan_per_head": [bool(kv1[h, :4].isnan().any()) for h in range(2)],
            "v_changed": bool((kv1[:, 4:] != kv0.view(2, 8, -1)[:, 4:]).any()),
        }
        out["cases"][name] = r
        print(name, json.dumps(r))

    # GQA（SelfAttention）：2 组 × 2 头；组 1 的两个头都为负
    for name, ml in (
        ("gqa_group_above_tau_group_negative", [150.0, 10.0, -0.5, -1.0]),
        ("gqa_control", [150.0, 10.0, 5.0, 1.0]),
    ):
        s = fake_gqa(SelfAttention, 2, 2, 4, 8, ml, tau, alpha)
        SelfAttention.clip_qk(s)
        w = s.linear_qkv.weight
        r = {
            "max_logits": ml,
            "eta_per_group": s.qk_clip_balancing_eta.view(-1).tolist(),
            "weight_has_nan": bool(w.isnan().any()),
            "nan_rows": int(w.isnan().any(1).sum()),
        }
        out["cases"][name] = r
        print(name, json.dumps(r))

    out["torch"] = torch.__version__
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
