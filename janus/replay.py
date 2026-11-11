"""离线重放：用参照侧抓下的 FP32 主权重、梯度与配置，在 FP64 里重算一步优化器更新。

只支持第一次更新（动量初值为 0）或给出更新前的动量。语义按 Megatron 默认的 AdamW：
梯度先按全局范数裁剪（coeff = clip / (‖g‖ + 1e-6)，小于 1 才乘），
m = β1·m + (1−β1)·g，v = β2·v + (1−β2)·g²，偏差修正后 p ← p − lr·(m̂ / (√v̂ + eps) + wd·p)；
一维参数（norm 权重、bias）不做权重衰减。
"""
from __future__ import annotations

import numpy as np


def no_weight_decay(name: str, shape) -> bool:
    return len(shape) == 1 or name.endswith(".bias")


def adamw_step(p, g, *, lr, beta1, beta2, eps, wd, step=1, m0=None, v0=None, clip_coeff=1.0):
    """返回 (新参数, m, v)，都是 float64。"""
    p = np.asarray(p, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64) * clip_coeff
    m = (np.zeros_like(p) if m0 is None else np.asarray(m0, dtype=np.float64)) * beta1 + (1 - beta1) * g
    v = (np.zeros_like(p) if v0 is None else np.asarray(v0, dtype=np.float64)) * beta2 + (1 - beta2) * g * g
    mh, vh = m / (1 - beta1 ** step), v / (1 - beta2 ** step)
    new = p - lr * (mh / (np.sqrt(vh) + eps) + wd * p)
    return new, m, v


def clip_coeff(grad_norm: float, clip: float | None) -> float:
    if not clip or clip <= 0 or grad_norm is None:
        return 1.0
    c = clip / (grad_norm + 1e-6)
    return c if c < 1.0 else 1.0
