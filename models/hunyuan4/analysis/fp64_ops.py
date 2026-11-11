"""阶段 A 的 GPU 一侧 L1：每个算子用 GPU 抓到的输入在 FP64 里重算，量 rel_L2(GPU, FP64)；并统计 DSA top-k 选择翻转。

    python fp64_ops.py --hf <截好的 HF 目录> --run <ga1 运行目录> [--layer 0] [--json out.json]

- 只用 CPU。HF 的 `hy_v4` 模块原样拿来，权重从 checkpoint 读成 FP64（BF16 → FP64 是精确的）。
- HF 代码里有几处强制 `.float()` / `.to(torch.float32)`（RMSNorm、iHC、索引器），重算时把「FP64 张量转 FP32 / BF16」
  改成不转，整段在 FP64 里算；这是参照，不改被测。
- 每个算子的输入取 GPU 第 1 步 rank 0 第一个 microbatch 抓到的张量（逐层隔离，不累积上游误差）。
- top-k 翻转：对位置 ≥ index_topk 的 query，比较 GPU 选中集合与 FP64 选中集合的对称差；按 FP64 分数里
  第 k 名与第 k+1 名的分差分桶。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frameworks", "hf"))
import hy4_model as HM  # noqa: E402


@contextlib.contextmanager
def keep_fp64():
    """把 FP64 张量上的 .float() / .to(float32|bfloat16) / .type(float32) 变成不转。"""
    orig_to, orig_float, orig_type = torch.Tensor.to, torch.Tensor.float, torch.Tensor.type

    def to(self, *args, **kw):
        if self.dtype == torch.float64:
            dt = kw.get("dtype")
            for x in args:
                if isinstance(x, torch.dtype):
                    dt = x
            if dt in (torch.float32, torch.bfloat16, torch.float16):
                args = tuple(a for a in args if not isinstance(a, torch.dtype))
                kw.pop("dtype", None)
                return orig_to(self, *args, **kw) if (args or kw) else self
        return orig_to(self, *args, **kw)

    def flt(self, *a, **k):
        return self if self.dtype == torch.float64 else orig_float(self, *a, **k)

    def typ(self, *a, **k):
        if self.dtype == torch.float64 and a and a[0] in (torch.float32, torch.bfloat16):
            return self
        return orig_type(self, *a, **k)

    torch.Tensor.to, torch.Tensor.float, torch.Tensor.type = to, flt, typ
    try:
        yield
    finally:
        torch.Tensor.to, torch.Tensor.float, torch.Tensor.type = orig_to, orig_float, orig_type


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--json")
    ap.add_argument("--save-npz", help="把每个算子的 FP64 输出存下（给 TPU 一侧算 R 比的分子用）；输出头存 FP32")
    a = ap.parse_args()
    outs = {}
    from safetensors import safe_open
    from transformers.masking_utils import create_causal_mask

    torch.set_num_threads(min(64, os.cpu_count()))
    L = a.layer
    cfg, root, _ = HM.build(a.hf, L + 1, False, 0)
    root.to_empty(device="cpu")
    root.to(torch.float64)
    idx = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
    named = dict(root.named_parameters())
    named.update(dict(root.named_buffers()))
    with torch.no_grad():
        for cn, f in idx.items():
            mn = HM.ckpt_to_model_name(cn)
            if mn in named:
                with safe_open(os.path.join(a.hf, f), framework="pt") as fh:
                    named[mn].copy_(fh.get_tensor(cn).to(torch.float64))
    rot = root.hf.model.rotary_emb
    inv, _ = rot.compute_default_rope_parameters(cfg)
    rot.inv_freq.copy_(inv)
    rot.original_inv_freq.copy_(inv)

    f = safe_open(os.path.join(a.run, "step_1", "fwd.safetensors"), framework="np")
    g = {k: f.get_tensor(k) for k in f.keys()}
    p = f"hf.model.layers.{L}"
    T = lambda k: torch.from_numpy(np.asarray(g[k], np.float64))  # noqa: E731
    layer = root.hf.model.layers[L]
    m = root.hf.model
    res = {"run": os.path.basename(os.path.normpath(a.run)), "layer": L, "ops": {}}
    ids = torch.from_numpy(np.asarray(g["input.tokens"], np.int64))
    S = ids.shape[1]
    with torch.no_grad(), keep_fp64():
        emb = m.embed_tokens(ids)
        res["ops"]["embed"] = rel_l2(g["hf.model.embed_tokens.out"], emb.numpy())
        outs["embed"] = emb.numpy()
        pos = torch.arange(S).unsqueeze(0)
        mask = create_causal_mask(
            config=cfg, inputs_embeds=emb, attention_mask=None, past_key_values=None, position_ids=pos, allow_is_causal_skip=False
        )
        pe = m.rotary_emb(emb, position_ids=pos)
        streams = emb.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous() if L == 0 else T(f"hf.model.layers.{L - 1}.out0")
        post, out = layer.attn_hc(streams)
        res["ops"]["attn_hc.post"] = rel_l2(g[f"{p}.attn_hc.post"], post.numpy())
        outs["attn_hc.post"] = post.numpy()
        res["ops"]["attn_hc.out"] = rel_l2(g[f"{p}.attn_hc.out"], out.numpy())
        outs["attn_hc.out"] = out.numpy()
        x = layer.input_layernorm(T(f"{p}.attn_hc.out"))
        res["ops"]["input_layernorm"] = rel_l2(g[f"{p}.input_layernorm.out"], x.numpy())
        outs["input_layernorm"] = x.numpy()
        xin = T(f"{p}.input_layernorm.out")
        gate = layer.self_attn.gate_proj(xin)
        res["ops"]["self_attn.gate_proj"] = rel_l2(g[f"{p}.self_attn.gate_proj.out"], gate.numpy())
        outs["self_attn.gate_proj"] = gate.numpy()
        # 注意力整块：FP64 自己选 top-k；另跑一遍「用 GPU 的 top-k」把索引器与注意力分开
        att, _, topk = layer.self_attn(hidden_states=xin, attention_mask=mask, position_embeddings=pe, position_ids=pos, prev_topk_indices=None)
        res["ops"]["self_attn(own topk)"] = rel_l2(g[f"{p}.self_attn.out"], att.numpy())
        outs["self_attn(own topk)"] = att.numpy()
        gpu_topk = torch.from_numpy(np.asarray(g[f"{p}.self_attn.topk"], np.int64))
        ind = layer.self_attn.indexer
        layer.self_attn.indexer = None
        att2, _, _ = layer.self_attn(
            hidden_states=xin, attention_mask=mask, position_embeddings=pe, position_ids=pos, prev_topk_indices=gpu_topk
        )
        layer.self_attn.indexer = ind
        res["ops"]["self_attn(gpu topk)"] = rel_l2(g[f"{p}.self_attn.out"], att2.numpy())
        outs["self_attn(gpu topk)"] = att2.numpy()
        # top-k 翻转：FP64 重算索引器分数
        k = cfg.index_topk
        q0 = slice(k, S)
        a64 = topk[0, q0].numpy()
        agp = gpu_topk[0, q0].numpy()
        sym = [len(set(x) ^ set(y)) // 2 for x, y in zip(a64, agp)]
        res["topk"] = {
            "queries": int(len(sym)),
            "queries_with_flip": int(sum(s > 0 for s in sym)),
            "flipped_slots": int(sum(sym)),
            "slots": int(len(sym) * k),
            "flip_fraction_of_slots": float(sum(sym) / (len(sym) * k)),
            "max_flips_in_one_query": int(max(sym)),
        }
        # 残差、FFN 一侧
        resid = (T(f"{p}.attn_hc.post").unsqueeze(-1) * T(f"{p}.self_attn.out").unsqueeze(-2) + streams).to(torch.float64)
        post2, out2 = layer.ffn_hc(resid)
        res["ops"]["ffn_hc.post"] = rel_l2(g[f"{p}.ffn_hc.post"], post2.numpy())
        outs["ffn_hc.post"] = post2.numpy()
        res["ops"]["ffn_hc.out"] = rel_l2(g[f"{p}.ffn_hc.out"], out2.numpy())
        outs["ffn_hc.out"] = out2.numpy()
        y = layer.post_attention_layernorm(T(f"{p}.ffn_hc.out"))
        res["ops"]["post_attention_layernorm"] = rel_l2(g[f"{p}.post_attention_layernorm.out"], y.numpy())
        outs["post_attention_layernorm"] = y.numpy()
        mo = layer.mlp(T(f"{p}.post_attention_layernorm.out"))
        res["ops"]["mlp"] = rel_l2(g[f"{p}.mlp.out"], mo.numpy())
        outs["mlp"] = mo.numpy()
        hc = m.hc_head(T(f"{p}.out0"))
        res["ops"]["hc_head"] = rel_l2(g["hf.model.hc_head.out"], hc.numpy())
        outs["hc_head"] = hc.numpy()
        nrm = m.norm(T("hf.model.hc_head.out"))
        res["ops"]["final_norm"] = rel_l2(g["hf.model.norm.out"], nrm.numpy())
        outs["final_norm"] = nrm.numpy()
        lg = torch.nn.functional.linear(T("hf.model.norm.out"), root.hf.lm_head.weight)
        res["ops"]["lm_head"] = rel_l2(g["hf.lm_head.out"], lg.numpy())
        outs["lm_head"] = lg.numpy()
    if a.save_npz:
        outs["lm_head"] = outs["lm_head"].astype(np.float32)
        outs["self_attn.topk_fp64"] = topk.numpy()
        np.savez(a.save_npz, **outs)
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.json:
        open(a.json, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
