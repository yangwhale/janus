"""诊断：CPU FP64 前向里只模拟 MLA 主注意力 RoPE（q_pe、k_pe）的 BF16 写法，看主 loss、MTP loss 移多少。

为什么：06-rw3 第 1 步索引器 KD 的 GPU 偏差约六成来自索引器 RoPE 的 BF16 写法（`kd_bf16_vs_fp64.py`）：Megatron-Core
`_apply_rotary_pos_emb_bshd` 把 cos / sin 转成输入的 BF16，乘、加也在 BF16 里做（`apply_rope_fusion=False`）。MLA 主注意力的
q_pe、k_pe 走同一个函数。GPU 主 loss 对 FP64 +5.3e-5，CPU BF16 前向复现 +5.21e-5；这里看这一处能占多少。

在 `step1_ref.py` 的 FP64 前向里替换 `multi_latent_attention.apply_rotary_pos_emb`（只影响 MLA，索引器用的是 hy4_mcore
导入的那份）。mode：
- `full16`：照原版 BF16 舍入点：输入舍 BF16，cos / sin 舍 BF16，t·cos、rot(t)·sin、两者之和各舍一次；
- `io16`：只把输入与结果各舍一次 BF16，中间 FP64（对照：存储舍入 vs 计算舍入）。

用法：参数原样转给 step1_ref.py（只前向、不加 --audit）：
    python mla_rope16_emul.py --mode full16 -- --hf ... --tokens ... --dtype float64 --out <目录> --forward-only ...
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "frameworks", "megatron"))


def main():
    argv = sys.argv[1:]
    mode = "full16"
    if argv and argv[0] == "--mode":
        mode, argv = argv[1], argv[2:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    assert mode in ("full16", "io16")
    assert "--audit" not in argv, "舍入模拟不能和 --audit 一起用"
    import megatron.core.transformer.multi_latent_attention as MLA
    import step1_ref as S
    import torch

    rnd = S._unforced(lambda t: t.to(torch.bfloat16).to(t.dtype))
    count = {"calls": 0}

    def emul(t, freqs, config=None, cu_seqlens=None, mscale=1.0, cp_group=None, mla_rotary_interleaved=False, **kw):
        assert cu_seqlens is None and not config.rotary_interleaved
        count["calls"] += 1
        fr = freqs
        if fr.dim() == t.dim() + 1 and fr.size(-2) == 1:
            fr = fr.squeeze(-2)
        assert fr.shape[-1] == t.shape[-1]
        x = rnd(t)
        if mla_rotary_interleaved:
            x = torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
        cos, sin = torch.cos(fr) * mscale, torch.sin(fr) * mscale
        x1, x2 = torch.chunk(x, 2, dim=-1)
        rot = torch.cat((-x2, x1), dim=-1)
        if mode == "full16":
            return rnd(rnd(x * rnd(cos)) + rnd(rot * rnd(sin)))
        return rnd(x * cos + rot * sin)

    MLA.apply_rotary_pos_emb = emul
    sys.argv = ["step1_ref.py", *argv]
    try:
        S.main()
    finally:
        out = argv[argv.index("--out") + 1]
        with open(os.path.join(out, "mla_rope16.json"), "w") as fo:
            json.dump({"mode": mode, "calls": count["calls"]}, fo)


if __name__ == "__main__":
    main()
