"""索引器 KD 对输入随机噪声的响应：KD 平均移多少、是否随噪声的平方增长（二阶、方向固定的偏差）。

为什么：06-rw3 第 1 步 GPU 的 KD 对 FP64 偏高（第 1 层 +2.2e-3），拆分（`kd_gpu_dump_vs_fp64.py`）指向 KD 输入里的
BF16 激活噪声，没有单个写错的算子。若偏差来自噪声的二阶效应，则：换不同的随机噪声（种子）得到几乎同一个移动量；
噪声加倍，移动量约乘 4；移动量与 logits 的大小有关（mini 的逐头最大 logit 约 0.13，真实宽度 7–14.5）。

做法：CPU FP64 参照第 0 行的 KD 输入（`step1_ref.py --detail-rows 0` 的 choices.safetensors），给选定的输入乘
(1 + σ·ε)（ε 标准正态，逐元素独立），用 `kd_bf16_vs_fp64.teacher / student / kl_sum` 算 KD（FP64，FP64 的 top-k）。

用法：
    python kd_noise_scaling.py --choices <choices.safetensors> --out kd_noise_scaling.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kd_bf16_vs_fp64 import kl_sum, student, teacher  # noqa: E402

TARGETS = {"student_qk": ("q", "k"), "key": ("key",), "q": ("q",), "k": ("k",)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--choices", required=True)
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--plan", default="1:student_qk,1:q,1:k,0:key,0:student_qk")
    p.add_argument("--sigmas", default="1e-3,2e-3,4e-3,8e-3")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    from safetensors import safe_open

    f64 = torch.float64
    f = safe_open(a.choices, "pt")
    sig = [float(s) for s in a.sigmas.split(",")]
    out = {"choices": a.choices, "row": a.row, "sigmas": sig, "seeds": a.seeds, "runs": []}
    cache = {}
    for item in a.plan.split(","):
        L, tgt = item.split(":")
        if L not in cache:
            x = {n: f.get_tensor(f"r{a.row}.dsa.{L}.{n}") for n in ("query", "key", "q", "k", "w", "topk")}
            x["topk"] = x["topk"].long()
            T = teacher(x["query"], x["key"], x["topk"], f64)
            S = student(x["q"], x["k"], x["w"], x["topk"], f64)
            cache[L] = (x, T, S, kl_sum(T, S, x["topk"]))
        x, T0, S0, ref = cache[L]
        for s in sig:
            shifts = []
            for seed in range(a.seeds):
                g = torch.Generator().manual_seed(1000 * seed + 7)
                y = dict(x)
                for n in TARGETS[tgt]:
                    y[n] = x[n] * (1 + s * torch.randn(x[n].shape, generator=g, dtype=f64))
                T = teacher(y["query"], y["key"], x["topk"], f64) if tgt == "key" else T0
                S = S0 if tgt == "key" else student(y["q"], y["k"], y["w"], x["topk"], f64)
                shifts.append(kl_sum(T, S, x["topk"]) / ref - 1)
            m = sum(shifts) / len(shifts)
            rec = {"layer": L, "target": tgt, "sigma": s, "rel_shift": shifts, "mean": m, "mean_over_sigma2": m / s**2}
            out["runs"].append(rec)
            print(json.dumps(rec), flush=True)
    with open(a.out, "w") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
