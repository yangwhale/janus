"""GPU（Megatron-Core）一侧的 L1 逐算子标准答案：拿 `hy4_capture.py` 抓到的输入，用 HF 参考实现在 FP64 里重算每个算子。

    python fp64_l1_gpu.py --hf <HF 目录> --l1 <运行目录>/l1 [--step 1] [--layers 0,1,...] [--mtp] \
        [--json out.json] [--save <FP64 输出.safetensors>]

- 只用 CPU。权重从 checkpoint 读成 FP64（BF16 → FP64 精确）；HF 代码里强制的 `.float()` 用 `fp64_ops.keep_fp64` 改成不转。
- 每个算子的输入一律取 GPU 抓到的张量（逐算子隔离，不累积上游误差）；输出与 GPU 抓到的输出比 rel_L2 与最大绝对差。
- 抓数格式见 `frameworks/megatron/README.md`「逐算子与逐层抓数」。名字与 HF 一侧 golden 的抓数相同，所以
  `fp64_ops.py`（HF 一侧）与本脚本（Megatron 一侧）的结果可以逐条对照。
- 路由专家：GPU 抓的是 rank 0 上本地专家（EP 切法下第 0..E/EP−1 个）收到的、按专家排好的 token；FP64 对每个本地专家
  逐段重算 fc1（gate|up）、clamp + SwiGLU × 路由权重（Megatron 在 fc2 之前乘权重，HF 在 fc2 之后乘，数学上相同）、fc2。
- 路由：用 GPU 抓到的 `expert_bias` 重算 sigmoid + bias + top-k + 归一化 + 缩放；选中集合不同的 token 单独计数。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frameworks", "hf"))
import hy4_model as HM  # noqa: E402
from fp64_ops import keep_fp64  # noqa: E402


def stats(gpu: torch.Tensor, ref: torch.Tensor) -> dict:
    g, r = gpu.double().reshape(-1), ref.double().reshape(-1)
    assert g.numel() == r.numel(), (gpu.shape, ref.shape)
    d = (g - r).abs()
    return {
        "rel_l2": float(d.norm() / max(float(r.norm()), 1e-300)),
        "max_abs": float(d.max()) if d.numel() else 0.0,
        "ref_absmax": float(r.abs().max()) if r.numel() else 0.0,
        "gpu_dtype": str(gpu.dtype).replace("torch.", ""),
        "n": int(r.numel()),
    }


def load_model(hf: str, n_layers: int, mtp: bool):
    from safetensors import safe_open

    cfg, root, _ = HM.build(hf, n_layers, mtp, 0)
    root.to_empty(device="cpu")
    root.to(torch.float64)
    idx = json.load(open(os.path.join(hf, "model.safetensors.index.json")))["weight_map"]
    named = dict(root.named_parameters())
    named.update(dict(root.named_buffers()))
    with torch.no_grad():
        for cn, f in idx.items():
            mn = HM.ckpt_to_model_name(cn)
            if mn in named:
                with safe_open(os.path.join(hf, f), framework="pt") as fh:
                    named[mn].copy_(fh.get_tensor(cn).to(torch.float64))
    rot = root.hf.model.rotary_emb
    inv, _ = rot.compute_default_rope_parameters(cfg)
    rot.inv_freq.copy_(inv)
    rot.original_inv_freq.copy_(inv)
    return cfg, root


def bs(t: torch.Tensor, layout: str) -> torch.Tensor:
    """把抓数里少数保留 Megatron [s, b, …] 的张量转成 [b, s, …]。"""
    return t.transpose(0, 1) if layout.startswith("s,b") else t


def router_ref(gate, x, bias):
    """HF HYV4TopkRouter 的 FP64 版，偏置用 GPU 当时的值。返回 logits、稠密的路由权重 [T, E]、选中集合 [T, E] bool。"""
    logits = F.linear(x, gate.weight)
    scores = logits.sigmoid()
    choice = scores + bias
    E = choice.shape[-1]
    gs = choice.view(-1, gate.num_group, E // gate.num_group).topk(2, dim=-1)[0].sum(-1)
    gidx = torch.topk(gs, k=gate.topk_group, dim=-1, sorted=False)[1]
    gmask = torch.zeros_like(gs).scatter_(1, gidx, 1)
    smask = gmask.unsqueeze(-1).expand(-1, gate.num_group, E // gate.num_group).reshape(-1, E)
    choice = choice.masked_fill(~smask.bool(), float("-inf"))
    idx = torch.topk(choice, k=gate.top_k, dim=-1, sorted=False)[1]
    w = scores.gather(1, idx)
    if gate.norm_topk_prob:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    w = w * gate.routed_scaling_factor
    dense = torch.zeros_like(scores).scatter_(1, idx, w)
    sel = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)
    return logits, dense, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--l1", required=True, help="运行目录下的 l1/")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--layers", default="", help="逗号分隔；默认全部层")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--json")
    ap.add_argument(
        "--save", help="FP64 输出存成 safetensors（名字是 json 里 ops 的键，对应的 GPU 张量名在该条的 gpu 字段）"
    )
    a = ap.parse_args()
    from safetensors.torch import load_file, save_file
    from transformers.masking_utils import create_causal_mask

    torch.set_num_threads(min(64, os.cpu_count()))
    G = load_file(os.path.join(a.l1, f"step_{a.step}_fwd.safetensors"))
    meta = json.load(open(os.path.join(a.l1, f"step_{a.step}_index.json")))
    lay = {k: v.get("layout", "") for k, v in meta.get("tensors", meta).items() if isinstance(v, dict)}
    n_layers = 1 + max(int(k.split(".")[3]) for k in G if k.startswith("hf.model.layers."))
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(n_layers))
    cfg, root = load_model(a.hf, n_layers, a.mtp)
    m = root.hf.model
    T = lambda k: bs(G[k], lay.get(k, "")).double()  # noqa: E731
    res = {"l1": os.path.abspath(a.l1), "step": a.step, "ops": {}, "routing": {}, "topk": {}}
    outs: dict[str, torch.Tensor] = {}

    def rec(name, gpu_key, ref):
        if gpu_key not in G:
            res["ops"][name] = {"missing": gpu_key}
            return
        res["ops"][name] = {"gpu": gpu_key, **stats(bs(G[gpu_key], lay.get(gpu_key, "")), ref)}
        outs[name] = ref.detach().contiguous()

    ids = G["input.tokens"].long()
    S = ids.shape[1]
    with torch.no_grad(), keep_fp64():
        emb = m.embed_tokens(ids)
        rec("embed", "hf.model.embed_tokens.out", emb)
        pos = torch.arange(S).unsqueeze(0)
        mask = create_causal_mask(
            config=cfg,
            inputs_embeds=emb,
            attention_mask=None,
            past_key_values=None,
            position_ids=pos,
            allow_is_causal_skip=False,
        )
        pe = m.rotary_emb(emb, position_ids=pos)

        def attention(pre, attn, xin_key):
            xin = T(xin_key)
            rec(f"{pre}.self_attn.gate_proj", f"{pre}.self_attn.gate_proj.out", attn.gate_proj(xin))
            key = f"{pre}.self_attn.topk"
            if key in G:
                gt = G[key].long()
                # Megatron 用 -1 填前 index_topk 个 query 的空位（可选的键不足 k 个）；HF 的 scatter 不收 -1。
                # 换成 query 自己的位置：那一格本来就被选中，重复 scatter 不改结果
                qpos = torch.arange(gt.shape[1]).view(1, -1, 1).expand_as(gt)
                gt = torch.where(gt < 0, qpos, gt)
                ind = attn.indexer
                attn.indexer = None
                o, _, _ = attn(
                    hidden_states=xin,
                    attention_mask=mask,
                    position_embeddings=pe,
                    position_ids=pos,
                    prev_topk_indices=gt,
                )
                attn.indexer = ind
                rec(f"{pre}.self_attn(gpu topk)", f"{pre}.self_attn.out", o)
                if ind is not None:
                    o2, _, tk = attn(
                        hidden_states=xin,
                        attention_mask=mask,
                        position_embeddings=pe,
                        position_ids=pos,
                        prev_topk_indices=None,
                    )
                    k = cfg.index_topk
                    if S > k:
                        a64, agp = tk[0, k:].tolist(), gt[0, k:].tolist()
                        sym = [len(set(x) ^ set(y)) // 2 for x, y in zip(a64, agp)]
                        res["topk"][pre] = {
                            "queries": len(sym),
                            "queries_with_flip": sum(s > 0 for s in sym),
                            "flipped_slots": sum(sym),
                            "slots": len(sym) * k,
                        }
                    rec(f"{pre}.self_attn(own topk)", f"{pre}.self_attn.out", o2)

        def moe(pre, mlp):
            x = T(f"{pre}.mlp.gate.in").reshape(-1, cfg.hidden_size)  # Megatron 的 token 顺序（s 在外、b 在内）
            bias = (
                G[f"{pre}.mlp.gate.expert_bias"].double()
                if f"{pre}.mlp.gate.expert_bias" in G
                else mlp.gate.e_score_correction_bias
            )
            logits, dense, sel = router_ref(mlp.gate, x, bias)
            rec(f"{pre}.mlp.gate.logits", f"{pre}.mlp.gate.logits", logits)
            gsel = G[f"{pre}.mlp.gate.routing_map"].bool()
            same = (gsel == sel).all(-1)
            res["routing"][pre] = {"tokens": int(same.numel()), "tokens_selection_differs": int((~same).sum())}
            gp = G[f"{pre}.mlp.gate.probs"].double()
            res["ops"][f"{pre}.mlp.gate.probs(same selection)"] = stats(gp[same], dense[same])
            outs[f"{pre}.mlp.gate.probs"] = dense
            # 共享专家
            if f"{pre}.mlp.shared_experts.in" in G:
                sx = T(f"{pre}.mlp.shared_experts.in")
                se = mlp.shared_experts
                g_, u_ = se.gate_proj(sx), se.up_proj(sx)
                rec(f"{pre}.mlp.shared_experts.fc1", f"{pre}.mlp.shared_experts.fc1.out", torch.cat([g_, u_], -1))
                if f"{pre}.mlp.shared_experts.fc2.in" in G:
                    rec(
                        f"{pre}.mlp.shared_experts.act",
                        f"{pre}.mlp.shared_experts.fc2.in",
                        se.act_fn(T(f"{pre}.mlp.shared_experts.fc1.out").chunk(2, -1)[0])
                        * T(f"{pre}.mlp.shared_experts.fc1.out").chunk(2, -1)[1],
                    )
                    rec(
                        f"{pre}.mlp.shared_experts.fc2",
                        f"{pre}.mlp.shared_experts.out",
                        se.down_proj(T(f"{pre}.mlp.shared_experts.fc2.in")),
                    )
            # 路由专家（rank 0 上的本地专家）
            k = f"{pre}.mlp.experts.permuted_local_hidden_states"
            if k in G:
                px = G[k].double()
                tpe = G[f"{pre}.mlp.experts.tokens_per_expert"].long().tolist()
                pp = G[f"{pre}.mlp.experts.permuted_probs"].double().reshape(-1, 1)
                ex = mlp.experts
                fc1g = G[f"{pre}.mlp.experts.fc1.out"].double()
                fc2g = G[f"{pre}.mlp.experts.fc2.in"].double()
                fc1r, actr, fc2r = [], [], []
                o = 0
                for e, n in enumerate(tpe):
                    sl = slice(o, o + n)
                    fc1r.append(F.linear(px[sl], ex.gate_up_proj[e]))
                    gate, up = fc1g[sl].chunk(2, -1)
                    act = F.silu(gate.clamp(max=ex.swiglu_limit)) * up.clamp(-ex.swiglu_limit, ex.swiglu_limit)
                    actr.append(act * pp[sl])
                    fc2r.append(F.linear(fc2g[sl], ex.down_proj[e]))
                    o += n
                res["routing"][pre]["local_experts_tokens"] = tpe
                rec(f"{pre}.mlp.experts.fc1", f"{pre}.mlp.experts.fc1.out", torch.cat(fc1r))
                rec(f"{pre}.mlp.experts.clamp_swiglu_x_prob", f"{pre}.mlp.experts.fc2.in", torch.cat(actr))
                rec(f"{pre}.mlp.experts.fc2", f"{pre}.mlp.experts.out", torch.cat(fc2r))
            # 整块（FP64 自己路由）
            rec(f"{pre}.mlp(own routing)", f"{pre}.mlp.out", mlp(T(f"{pre}.post_attention_layernorm.out")))

        def dense_mlp(pre, mlp):
            x = T(f"{pre}.post_attention_layernorm.out")
            if f"{pre}.mlp.fc1.out" in G:
                rec(f"{pre}.mlp.fc1", f"{pre}.mlp.fc1.out", torch.cat([mlp.gate_proj(x), mlp.up_proj(x)], -1))
                g_, u_ = T(f"{pre}.mlp.fc1.out").chunk(2, -1)
                rec(f"{pre}.mlp.act", f"{pre}.mlp.fc2.in", mlp.act_fn(g_) * u_)
                rec(f"{pre}.mlp.fc2", f"{pre}.mlp.out", mlp.down_proj(T(f"{pre}.mlp.fc2.in")))
            rec(f"{pre}.mlp(block)", f"{pre}.mlp.out", mlp(x))

        for L in layers:
            pre, layer = f"hf.model.layers.{L}", m.layers[L]
            streams = T(f"{pre}.attn_hc.in")  # 第 0 层的 in0 是 [b, s, H]（层内才扩成 4 路），所以取 attn_hc 的输入
            post, out = layer.attn_hc(streams)
            rec(f"{pre}.attn_hc.post", f"{pre}.attn_hc.post", post)
            rec(f"{pre}.attn_hc.out", f"{pre}.attn_hc.out", out)
            rec(f"{pre}.input_layernorm", f"{pre}.input_layernorm.out", layer.input_layernorm(T(f"{pre}.attn_hc.out")))
            attention(pre, layer.self_attn, f"{pre}.input_layernorm.out")
            resid = T(f"{pre}.attn_hc.post").unsqueeze(-1) * T(f"{pre}.self_attn.out").unsqueeze(-2) + streams
            rec(f"{pre}.attn_residual", f"{pre}.ffn_hc.in", resid)
            post2, out2 = layer.ffn_hc(T(f"{pre}.ffn_hc.in"))
            rec(f"{pre}.ffn_hc.post", f"{pre}.ffn_hc.post", post2)
            rec(f"{pre}.ffn_hc.out", f"{pre}.ffn_hc.out", out2)
            rec(
                f"{pre}.post_attention_layernorm",
                f"{pre}.post_attention_layernorm.out",
                layer.post_attention_layernorm(T(f"{pre}.ffn_hc.out")),
            )
            (moe if cfg.mlp_layer_types[L] == "sparse" else dense_mlp)(pre, layer.mlp)
            rec(
                f"{pre}.ffn_residual",
                f"{pre}.out0",
                T(f"{pre}.ffn_hc.post").unsqueeze(-1) * T(f"{pre}.mlp.out").unsqueeze(-2) + T(f"{pre}.ffn_hc.in"),
            )
        last = f"hf.model.layers.{n_layers - 1}.out0"
        rec("hc_head", "hf.model.hc_head.out", m.hc_head(T(last)))
        rec("final_norm", "hf.model.norm.out", m.norm(T("hf.model.hc_head.out")))
        lg = F.linear(T("hf.model.norm.out"), root.hf.lm_head.weight)
        rec("lm_head", "hf.lm_head.out", lg)
        if "loss.per_token" in G and "input.labels" in G:
            lab = G["input.labels"].long()
            ce = F.cross_entropy(T("hf.lm_head.out").reshape(-1, lg.shape[-1]), lab.reshape(-1), reduction="none")
            rec("cross_entropy(gpu logits)", "loss.per_token", ce.view_as(G["loss.per_token"]))
        if a.mtp and "hf.model.mtp_layers.0.enorm.in" in G:
            p = "hf.model.mtp_layers.0"
            mt = root.mtp
            rec(f"{p}.enorm", f"{p}.enorm.out", mt.enorm(T(f"{p}.enorm.in")))
            rec(f"{p}.hnorm", f"{p}.hnorm.out", mt.hnorm(T(f"{p}.hnorm.in")))
            rec(f"{p}.eh_proj", f"{p}.eh_proj.out", mt.eh_proj(T(f"{p}.eh_proj.in")))
            if f"{p}.input_layernorm.out" in G:
                rec(f"{p}.input_layernorm", f"{p}.input_layernorm.out", mt.input_layernorm(T(f"{p}.in0")))
                attention(p, mt.self_attn, f"{p}.input_layernorm.out")
                mid = T(f"{p}.in0") + T(f"{p}.self_attn.out")  # MTP 层是普通残差，没有 iHC
                rec(
                    f"{p}.post_attention_layernorm",
                    f"{p}.post_attention_layernorm.out",
                    mt.post_attention_layernorm(mid),
                )
                moe(p, mt.mlp)
                rec(f"{p}.ffn_residual", f"{p}.out0", mid + T(f"{p}.mlp.out"))
            rec(f"{p}.final_layernorm", f"{p}.final_layernorm.out", mt.final_layernorm(T(f"{p}.final_layernorm.in")))
            if "hf.mtp.lm_head.out" in G:
                rec(
                    "mtp.lm_head", "hf.mtp.lm_head.out", F.linear(T(f"{p}.final_layernorm.out"), root.hf.lm_head.weight)
                )
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.json:
        open(a.json, "w").write(txt + "\n")
    if a.save:
        save_file({k: v.double().contiguous() for k, v in outs.items()}, a.save)


if __name__ == "__main__":
    main()
