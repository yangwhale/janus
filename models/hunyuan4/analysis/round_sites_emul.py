"""诊断：CPU FP64 前向里按类别加 BF16 舍入点（模块输出舍一次 BF16 再转回），看主 loss、MTP loss 移多少。

为什么：06-rw3 第 1 步 GPU 主 loss 对 FP64 +5.3e-5，同一套代码的 CPU BF16 前向复现 +5.21e-5（`kd_bf16_vs_fp64.py`）。
要把它拆到算子：在 `step1_ref.py` 的 FP64 前向里只加一类 BF16 舍入点，看能复现多少。类别（逗号分隔，可组合）：
- `head_in`：输出头（`output_layer`，GPU 上是 FP32 头）的输入舍 BF16；
- `linear`：除输出头以外所有线性层（名字以 Linear 结尾的模块：注意力、MLP、路由专家、MTP 投影）的输出舍 BF16；
- `norm`：所有归一化模块（名字含 Norm）的输出舍 BF16；
- `core_attn`：注意力核（core_attention，含 sink 的稀疏注意力）的输出舍 BF16；
- `attn_probs`：`hy4_mcore.sparse_sink_attention` 里 softmax 之后的注意力概率舍 BF16（GPU 上 `.to(v.dtype)` 是 BF16），
  再乘 v；其余照 FP64；
- `layer_out`：每个解码层（含 MTP 里的那层）的输出（iHC 的多路残差流）舍 BF16，即层与层之间的残差流按 BF16 存。
这只是 GPU 舍入点的近似（GPU 上有些线性层输出是 FP32，例如索引器 weights_proj、路由），读数按「这一类大概占多少」用。

用法：参数原样转给 step1_ref.py（只前向、不加 --audit）：
    python round_sites_emul.py --sites head_in -- --hf ... --tokens ... --dtype float64 --out <目录> --forward-only ...
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "frameworks", "megatron"))

SITES = ("head_in", "linear", "norm", "core_attn", "attn_probs", "layer_out")


def main():
    argv = sys.argv[1:]
    sites = []
    if argv and argv[0] == "--sites":
        sites, argv = argv[1].split(","), argv[2:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    assert sites and set(sites) <= set(SITES), sites
    assert "--audit" not in argv, "舍入模拟不能和 --audit 一起用"
    import step1_ref as S
    import torch

    rnd = S._unforced(lambda t: t.to(torch.bfloat16).to(t.dtype))

    def rnd_out(out):
        if isinstance(out, torch.Tensor):
            return rnd(out) if out.is_floating_point() else out
        if isinstance(out, tuple):
            return tuple(rnd_out(o) if isinstance(o, torch.Tensor) else o for o in out)
        return out

    hooked = {s: [] for s in sites}
    if "attn_probs" in sites:
        import hy4_mcore as H

        def ssa(query, key, value, topk, sinks, scale, scores_fp32=False, return_max_logit=False):
            # 同 hy4_mcore.sparse_sink_attention，只多一处：概率舍 BF16（FP64 前向里 .to(v.dtype) 不舍入）
            sq, b, h, _ = query.shape
            sk = key.shape[0]
            q, k, v = (t.permute(1, 2, 0, 3) for t in (query, key, value))
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
            allowed = H.causal_sparse_mask(topk, sq, sk).unsqueeze(1)
            scores = scores.masked_fill(~allowed, float("-inf"))
            max_logit = scores.detach().float().amax(dim=(0, 2, 3)) if return_max_logit else None
            sink = sinks.float().view(1, h, 1, 1).expand(b, h, sq, 1)
            comb = torch.cat([scores.float(), sink], dim=-1)
            comb = comb - comb.max(dim=-1, keepdim=True).values
            probs = rnd(torch.softmax(comb, dim=-1)[..., :-1].to(v.dtype))
            out = torch.matmul(probs, v).permute(2, 0, 1, 3).reshape(sq, b, h * v.shape[-1])
            hooked["attn_probs"].append(1)
            return (out, max_logit) if return_max_logit else out

        H.sparse_sink_attention = ssa
    orig_build = S.build

    def build(a, dt):
        out = orig_build(a, dt)
        model = out[2]
        for name, mod in model.named_modules():
            cls = type(mod).__name__
            if name.endswith("output_layer"):
                if "head_in" in sites:
                    mod.register_forward_pre_hook(lambda m, args: (rnd(args[0]),) + tuple(args[1:]))
                    hooked["head_in"].append(name)
                continue
            if "linear" in sites and cls.endswith("Linear"):
                mod.register_forward_hook(lambda m, i, o: rnd_out(o))
                hooked["linear"].append(name)
            elif "norm" in sites and "Norm" in cls:
                mod.register_forward_hook(lambda m, i, o: rnd_out(o))
                hooked["norm"].append(name)
            elif "layer_out" in sites and cls.endswith("TransformerLayer"):
                mod.register_forward_hook(lambda m, i, o: rnd_out(o))
                hooked["layer_out"].append(name)
            elif "core_attn" in sites and name.endswith("core_attention"):
                mod.register_forward_hook(lambda m, i, o: rnd_out(o))
                hooked["core_attn"].append(name)
        print("ROUND_SITES", json.dumps({s: len(v) for s, v in hooked.items()}), flush=True)
        return out

    S.build = build
    sys.argv = ["step1_ref.py", *argv]
    try:
        S.main()
    finally:
        out = argv[argv.index("--out") + 1]
        with open(os.path.join(out, "round_sites.json"), "w") as fo:
            hooked["attn_probs"] = len(hooked.get("attn_probs", []))
            json.dump({"sites": sites, "modules": hooked}, fo, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
