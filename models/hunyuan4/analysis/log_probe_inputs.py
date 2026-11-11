"""FP32 log 精度探针的确定输入（TPU：frameworks/maxtext/log_ulp_probe.py；GPU：analysis/gpu_log_probe.py，两边共用这个函数）。"""
from __future__ import annotations

import numpy as np


def make_inputs() -> dict:
    lo, hi = np.float32(0.5).view(np.uint32), np.float32(4.0).view(np.uint32)
    allx = np.arange(lo, hi, dtype=np.uint32).view(np.float32)            # [0.5, 4) 全部 FP32，3 × 2^23 个
    rng = np.random.default_rng(1234)
    logu = np.exp(rng.uniform(np.log(1e-37), np.log(1e37), 1 << 22)).astype(np.float32)
    lse = np.exp(rng.uniform(0.0, np.log(4096.0), 1 << 22)).astype(np.float32)
    return {"all[0.5,4)": allx, "logu[1e-37,1e37]": logu, "logu[1,4096)": lse}
