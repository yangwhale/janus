"""诊断：CPU FP64 前向里只加一处舍入——各注意力层（含 MTP）进注意力的 key 的 RoPE 段 k_pe 舍一次 BF16，看主 loss、MTP loss 移多少。

为什么：06-rw3 第 1 步索引器 KD 的 GPU 偏差拆分里，第 0 层教师 key 的效应全在 k_pe（64 维，64 个头共用一份）上，
FP64 里只把 k_pe 舍一次 BF16 就复现（`kd_gpu_dump_vs_fp64.py`）。同一处舍入是否也解释 GPU 第 1 步主 loss 对 FP64
的 +5.3e-5：在 `step1_ref.py` 的 FP64 前向里，把 `hy4_mcore.sparse_sink_attention` 收到的 key 的后 qk_rope_head_dim
维舍成 BF16 再转回（只改注意力本身；索引器 KD 的教师 key 不经这里，KD 记录值不受影响），其余照 FP64。

用法：参数原样转给 `step1_ref.py`（只前向、不加 --audit：_unforced 只退出最上面一层 dispatch mode）：
    python kpe_round_emul.py [--kpe-layers 0,1,2,mtp0] -- --hf ... --tokens ... --dtype float64 --out <目录> --forward-only ...
输出：step1_ref 的 summary.json，外加 `<out>/kpe_round.json`（作用的层、舍入次数）。
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "frameworks", "megatron"))


def main():
    argv = sys.argv[1:]
    layers = "all"
    if argv and argv[0] == "--kpe-layers":
        layers, argv = argv[1], argv[2:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    assert "--audit" not in argv, "舍入模拟不能和 --audit 一起用"
    import hy4_mcore as H
    import step1_ref as S
    import torch

    rnd = S._unforced(lambda t: t.to(torch.bfloat16).to(t.dtype))
    orig = H.sparse_sink_attention
    cur = {"name": None}
    count = {"calls": 0}
    rope = {"dim": None}

    def ssa(query, key, value, topk, sinks, scale, scores_fp32=False, return_max_logit=False):
        if layers == "all" or cur["name"] in layers.split(","):
            d = rope["dim"]
            key = torch.cat([key[..., :-d], rnd(key[..., -d:])], dim=-1)
            count["calls"] += 1
        return orig(query, key, value, topk, sinks, scale, scores_fp32, return_max_logit)

    H.sparse_sink_attention = ssa
    orig_build = S.build

    def build(a, dt):
        out = orig_build(a, dt)
        hf, cfg, model, L, mtp, coeff = out
        rope["dim"] = cfg.qk_pos_emb_head_dim
        cores = [(str(i), layer.self_attention.core_attention) for i, layer in enumerate(model.decoder.layers)]
        if mtp:
            cores.append(("mtp0", model.mtp.layers[0].mtp_model_layer.self_attention.core_attention))
        for name, c in cores:
            f = c.forward

            def fwd(*x, _n=name, _f=f, **k):
                cur["name"] = _n
                try:
                    return _f(*x, **k)
                finally:
                    cur["name"] = None

            c.forward = fwd
        print("KPE_ROUND", layers, "rope_dim", rope["dim"], flush=True)
        return out

    S.build = build
    sys.argv = ["step1_ref.py", *argv]
    try:
        S.main()
    finally:
        out = argv[argv.index("--out") + 1]
        with open(os.path.join(out, "kpe_round.json"), "w") as fo:
            json.dump({"layers": layers, "rope_dim": rope["dim"], "rounded_calls": count["calls"]}, fo)


if __name__ == "__main__":
    main()
