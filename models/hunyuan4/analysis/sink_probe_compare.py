"""sink_probe.py 的 TPU 输出对 FP64（CPU）：每层的前向 logsumexp / o（sa_use_base2_exp 开 / 关）与 sink 梯度（tokamax 原样 /
splash_dsinks_fp32），按四要素写；「GPU」一列是在 CPU 上按 GPU 的写法算的（Megatron-Core：q·k 与 softmax 在 FP32，dsink 由
autograd 穿过 FP32 softmax，唯一的 BF16 舍入是 dp = do·vᵀ 每个位置一次），不是 GPU 跑出来的数——GPU 没抓这几个量。
输入（q、k、v、do 的 BF16 值、sink）都取 TPU 输出里存的同一份。

  python models/hunyuan4/analysis/sink_probe_compare.py <sink_probe.json> <输出 json>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import ml_dtypes
import numpy as np


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def fp64_attention(q, k, v, sinks, do):
    """causal softmax with a per-head sink logit, all float64; plus the GPU-form dsink (dp rounded to bf16)."""
    H, S, _ = q.shape
    s = np.einsum("hqd,hkd->hqk", q, k)
    s = np.where(np.tril(np.ones((S, S), bool))[None], s, -np.inf)
    m = np.maximum(s.max(-1), sinks[:, None])
    e = np.exp(s - m[..., None])
    es = np.exp(sinks[:, None] - m)
    den = e.sum(-1) + es
    lse = m + np.log(den)
    p = e / den[..., None]
    o = np.einsum("hqk,hkd->hqd", p, v)
    p_sink = es / den
    dsink = -np.sum(p_sink * np.sum(o * do, -1), -1)
    dp = np.einsum("hqd,hkd->hqk", do, v).astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)
    dsink_gpu = -np.sum(p_sink * np.sum(p * dp, -1), -1)
    return lse, o, dsink, dsink_gpu


def main(src: str, out: str) -> None:
    from safetensors.numpy import load_file

    meta = json.loads(Path(src).read_text())
    A = load_file(str(Path(src).with_suffix(".safetensors")))
    res = {"what": "tokamax splash 精度探针：前向 logsumexp / o 与 sink 梯度对 FP64", "source": meta, "layers": {}}
    for L in meta["layers"]:
        q, k, v, do = (A[f"{L}.{t}"].astype(np.float64) for t in ("q", "k", "v", "do"))
        sinks = A[f"{L}.sinks"].astype(np.float64)
        lse, o, dsink, dsink_gpu = fp64_attention(q, k, v, sinks, do)
        r = {}
        for tag in ("base2", "natural"):
            lse_t = A[f"{L}.{tag}.lse"].astype(np.float64)
            r[tag] = {
                "lse": {
                    "abs_max": float(np.abs(lse_t - lse).max()),
                    "abs_mean_signed": float((lse_t - lse).mean()),
                    "rel_l2": rel(lse_t, lse),
                },
                "o": {"rel_l2": rel(A[f"{L}.{tag}.o"], o)},
                "dsink": {
                    "metric": "sink 梯度 · rel_L2 对 FP64（每层 8 个头为一个向量）",
                    "gpu_form": rel(dsink_gpu, dsink),
                    "tpu_tokamax": rel(A[f"{L}.{tag}.dsinks_tokamax"], dsink),
                    "tpu_fp32": rel(A[f"{L}.{tag}.dsinks_fp32"], dsink),
                    "R_tokamax": rel(A[f"{L}.{tag}.dsinks_tokamax"], dsink) / max(rel(dsink_gpu, dsink), 1e-300),
                    "R_fp32": rel(A[f"{L}.{tag}.dsinks_fp32"], dsink) / max(rel(dsink_gpu, dsink), 1e-300),
                    "tpu_vs_gpu_form": {
                        "tokamax": rel(A[f"{L}.{tag}.dsinks_tokamax"], dsink_gpu),
                        "fp32": rel(A[f"{L}.{tag}.dsinks_fp32"], dsink_gpu),
                    },
                    "signed_mean_rel": {
                        "gpu_form": float(np.mean((dsink_gpu - dsink) / np.abs(dsink))),
                        "tpu_tokamax": float(np.mean((A[f"{L}.{tag}.dsinks_tokamax"] - dsink) / np.abs(dsink))),
                        "tpu_fp32": float(np.mean((A[f"{L}.{tag}.dsinks_fp32"] - dsink) / np.abs(dsink))),
                    },
                },
            }
        if f"{L}.dsinks_gpuorder" in A:  # M42b：TPU 上按 GPU 写法重算的 sink 梯度
            g = A[f"{L}.dsinks_gpuorder"].astype(np.float64)
            r["gpu_order_m42b"] = {
                "metric": "sink 梯度（MaxText splash_dsinks_gpu_order，TPU 算）",
                "vs_gpu_form_rel_l2": rel(g, dsink_gpu),
                "vs_gpu_form_elementwise_rel_max": float(np.max(np.abs(g - dsink_gpu) / np.abs(dsink_gpu))),
                "rel_l2_vs_fp64": rel(g, dsink),
                "gpu_form_vs_fp64": rel(dsink_gpu, dsink),
                "R": rel(g, dsink) / max(rel(dsink_gpu, dsink), 1e-300),
                "signed_mean_rel": float(np.mean((g - dsink) / np.abs(dsink))),
            }
        res["layers"][L] = r
        print(
            L,
            json.dumps(
                {
                    t: {
                        "lse_abs_max": r[t]["lse"]["abs_max"],
                        "o": r[t]["o"]["rel_l2"],
                        **{x: r[t]["dsink"][x] for x in ("gpu_form", "tpu_tokamax", "tpu_fp32")},
                    }
                    for t in ("base2", "natural")
                }
                | (
                    {"m42b": {x: r["gpu_order_m42b"][x] for x in ("vs_gpu_form_rel_l2", "R")}}
                    if "gpu_order_m42b" in r
                    else {}
                ),
                ensure_ascii=False,
            ),
        )
    text = re.sub(r"gs://[^/\"]+", "gs://<POC 桶>", json.dumps(res, ensure_ascii=False, indent=1))
    Path(out).write_text(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
