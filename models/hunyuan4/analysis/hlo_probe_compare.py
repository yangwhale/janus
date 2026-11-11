"""hlo_probe.py 的 TPU 输出：对 FP64，并在 numpy 里按不同的舍入 / 加法顺序重算，看 TPU 的结果逐位等于哪一种写法。
- M28：Σ_k pre_k·x_k，k = 0..3，乘积在 FP32 里精确舍入一次；顺序 (((0+1)+2)+3) 与两两 ((0+1)+(2+3))；另报 FP64 的 rel_L2；
- M9：RMSNorm，单次舍入 bf16(x·rsqrt(ms+eps)·scale) 与两次舍入 bf16(bf16(x·rsqrt(ms+eps))·bf16(scale))，在 FP32 里算。

  python models/hunyuan4/analysis/hlo_probe_compare.py <hlo_probe.json>[,<另一次运行的 hlo_probe.json>…] <输出 json>
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


def bf16(x):
    return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def frac_equal(a, b):
    return float(np.mean(np.asarray(a, np.float32).view(np.uint32) == np.asarray(b, np.float32).view(np.uint32)))


def one(src: str) -> dict:
    from safetensors.numpy import load_file

    meta = json.loads(Path(src).read_text())
    A = load_file(str(Path(src).with_suffix(".safetensors")))
    x, pre = A["m28.x"].astype(np.float32), A["m28.pre"].astype(np.float32)
    terms = [pre[..., k : k + 1] * x[:, :, k] for k in range(x.shape[2])]  # FP32 乘积（各舍一次）
    seq = ((terms[0] + terms[1]) + terms[2]) + terms[3]
    pair = (terms[0] + terms[1]) + (terms[2] + terms[3])
    # 融合乘加（乘积不舍入，加完舍一次）：乘积在 float64 里精确（FP32 × BF16 值），加完舍成 FP32
    p64, x64 = pre.astype(np.float64), x.astype(np.float64)
    fma = terms[0]
    for k in range(1, x.shape[2]):
        fma = (fma.astype(np.float64) + p64[..., k : k + 1] * x64[:, :, k]).astype(np.float32)
    ref = np.einsum("bsk,bskd->bsd", pre.astype(np.float64), x.astype(np.float64))
    m28 = {}
    for name in ("einsum_highest", "unrolled", "reduce_sum"):
        t = A[f"m28.{name}"]
        m28[name] = {
            "rel_l2_vs_fp64": rel(t, ref),
            "frac_equal_sequential": frac_equal(t, seq),
            "frac_equal_pairwise": frac_equal(t, pair),
            "frac_equal_sequential_fma": frac_equal(t, fma),
        }
    m28["numpy_sequential_rel_l2_vs_fp64"] = rel(seq, ref)
    m28["numpy_pairwise_rel_l2_vs_fp64"] = rel(pair, ref)
    x0, scale = A["m9.x"].astype(np.float32), A["m9.scale"].astype(np.float32)
    normed = x0 * (1.0 / np.sqrt(np.mean(x0.astype(np.float32) ** 2, -1, keepdims=True) + np.float32(1e-5))).astype(
        np.float32
    )
    single = bf16(normed * bf16(scale))
    double = bf16(bf16(normed) * bf16(scale))
    ref9 = x0.astype(np.float64) / np.sqrt(np.mean(x0.astype(np.float64) ** 2, -1, keepdims=True) + 1e-5) * bf16(scale)
    m9 = {}
    for name in ("default", "single"):
        t = A[f"m9.{name}"]
        m9[name] = {
            "rel_l2_vs_fp64": rel(t, ref9),
            "frac_equal_numpy_single_rounding": frac_equal(t, single),
            "frac_equal_numpy_double_rounding": frac_equal(t, double),
        }
    m9["numpy_single_vs_double_frac_equal"] = frac_equal(single, double)
    return {
        "run": Path(src).parent.name,
        "xla_flags": meta.get("xla_flags"),
        "device_kind": meta.get("device_kind"),
        "m28": m28,
        "m9": m9,
        "tpu_bit_identical": {
            "m9_default_vs_single": meta.get("m9_bit_identical"),
            "m28": meta.get("m28_bit_identical"),
        },
    }


def main(srcs: str, out: str) -> None:
    res = {
        "what": "M28 / M9 机制核实：TPU 结果逐位等于哪种舍入 / 加法顺序（numpy FP32 重算），与对 FP64 的 rel_L2",
        "runs": [one(s) for s in srcs.split(",")],
    }
    text = re.sub(r"gs://[^/\"]+", "gs://<POC 桶>", json.dumps(res, ensure_ascii=False, indent=1))
    Path(out).write_text(text + "\n")
    print(text[:3000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
