"""tokamax splash 的前向 PV 怎么舍：用 TPU 上实跑的 splash 输出判（CPU 上算，只要 torch 与 safetensors）。

输入是 sink 探针 `tw-sinkprobe2`（frameworks/maxtext/sink_probe.py，v5p，训练镜像，T1 的块大小 512，因果掩码，
sa_use_base2_exp 关的那一组）存下的 q（已乘 1/16）、k、v、sink 与 TPU 的输出 o。同一份输入在 CPU 上按四种写法重算 o，
逐元素看与 TPU 的 BF16 输出是否相同：

- gpu：Megatron-Core（hy4_mcore，attn_scores_fp32）的写法：FP32 softmax（含 sink）归一化后的概率舍 BF16，再乘 V（FP32 累加），
  输出舍 BF16。
- fp32：概率不舍，FP32 的 exp(s − m) 乘 V，最后乘 1/l、舍 BF16（HY4_TPU_LIKE_ATTNK 的 ``pv32``）。
- rowmax：同 fp32，但 exp(s − m)（m 是整行最大值、含 sink）先舍 BF16 再乘 V：不分块的近似。
- block：splash 内核的在线 softmax：kv 按 512 一块，m 从 sink 起、l 从 1 起；每块 m' = max(m, 块内最大)，
  e = exp(s − m')（FP32），α = exp(m − m')，l = α·l + Σe，o = α·o + bf16(e)·V（e 先舍 BF16 再进矩阵乘，FP32 累加）；
  最后 o·(1/l) 舍 BF16（``pvbf``）。

    python splash_pv_rounding_check.py --probe sink_probe.safetensors --out result.json
"""

from __future__ import annotations

import argparse
import json

import torch


def _scores(q, k, rows):
    s = q[:, rows].double() @ k.double().transpose(-1, -2)  # BF16 输入的乘积在 FP64 里精确，求和舍一次 FP32
    s = s.float()
    sq = q.shape[1]
    causal = torch.ones(sq, k.shape[1], dtype=torch.bool).tril()[rows]
    return s.masked_fill(~causal, float("-inf"))


def emulate(q, k, v, sinks, block: int, chunk: int = 256) -> dict:
    h, sq, _ = q.shape
    outs = {m: torch.empty(h, sq, v.shape[-1]) for m in ("gpu", "fp32", "rowmax", "block")}
    sink = sinks.float().view(h, 1, 1)
    vf = v.float()
    for r0 in range(0, sq, chunk):
        rows = slice(r0, r0 + chunk)
        s = _scores(q, k, rows)  # [h, c, sk]
        comb = torch.cat([s, sink.expand(h, s.shape[1], 1)], dim=-1)
        m = comb.max(dim=-1, keepdim=True).values
        e = torch.exp(comb - m)
        l = e.sum(dim=-1, keepdim=True)  # noqa: E741
        p = e / l
        outs["gpu"][:, rows] = (p[..., :-1].bfloat16().float() @ vf).bfloat16().float()
        outs["fp32"][:, rows] = ((e[..., :-1] @ vf) * (1.0 / l)).bfloat16().float()
        outs["rowmax"][:, rows] = ((e[..., :-1].bfloat16().float() @ vf) * (1.0 / l)).bfloat16().float()
        mb = sink.expand(h, s.shape[1], 1).clone()
        lb = torch.ones_like(mb)
        ob = torch.zeros(h, s.shape[1], v.shape[-1])
        for c0 in range(0, k.shape[1], block):
            sb = s[..., c0 : c0 + block]
            m_new = torch.maximum(mb, sb.max(dim=-1, keepdim=True).values)
            eb = torch.exp(sb - m_new)
            alpha = torch.exp(mb - m_new)
            lb = alpha * lb + eb.sum(dim=-1, keepdim=True)
            ob = alpha * ob + eb.bfloat16().float() @ vf[:, c0 : c0 + block]
            mb = m_new
        outs["block"][:, rows] = (ob * (1.0 / lb)).bfloat16().float()
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, help="sink_probe.py 存的 safetensors")
    ap.add_argument("--layers", default="1,2,4,6,mtp0")
    ap.add_argument("--block", type=int, default=512, help="splash 的 block_kv_compute（T1 配置 512）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors.torch import load_file

    d = load_file(a.probe)
    res = {"probe": a.probe, "block_kv_compute": a.block, "mode": "natural（sa_use_base2_exp 关）", "layers": {}}
    for L in a.layers.split(","):
        q, k, v = (d[f"{L}.{t}"].bfloat16() for t in ("q", "k", "v"))
        tpu = d[f"{L}.natural.o"].float()
        outs = emulate(q, k, v, d[f"{L}.sinks"], a.block)
        rec = {}
        for name, o in outs.items():
            rec[name] = {
                "equal_frac": float((o == tpu).float().mean()),
                "rel_l2_vs_tpu": float((o.double() - tpu.double()).norm() / tpu.double().norm()),
            }
        res["layers"][L] = rec
        print(L, {n: (round(r["equal_frac"], 4), f"{r['rel_l2_vs_tpu']:.2e}") for n, r in rec.items()}, flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
