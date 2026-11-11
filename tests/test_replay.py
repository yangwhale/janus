"""replay.py：FP64 重放 AdamW 的公式、权重衰减规则、梯度裁剪系数。优化器重放是 optimizer_step 的参照，错了会让检查全盲。"""

import numpy as np
import pytest

from janus.replay import adamw_step, clip_coeff, no_weight_decay


def _reference(p, g, lr, b1, b2, eps, wd, steps):
    """逐元素、逐步照公式手算（Python float，不用向量化），当独立参照。"""
    m = v = 0.0
    for t in range(1, steps + 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1**t), v / (1 - b2**t)
        p = p - lr * (mh / (vh**0.5 + eps) + wd * p)
    return p, m, v


def test_first_step_matches_hand_formula():
    p, g = np.array([1.0, -2.0, 0.5]), np.array([0.1, 0.3, -0.2])
    new, m, v = adamw_step(p, g, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.033, step=1)
    for i in range(3):
        rp, rm, rv = _reference(p[i], g[i], 1e-3, 0.9, 0.95, 1e-8, 0.033, 1)
        assert new[i] == pytest.approx(rp, rel=0, abs=1e-15)
        assert m[i] == pytest.approx(rm) and v[i] == pytest.approx(rv)
    # 第一步偏差修正后 m̂ / √v̂ = sign(g)：没有衰减时更新量就是 lr·sign(g)（差 eps 的量级）
    new0, _, _ = adamw_step(p, g, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.0, step=1)
    assert np.allclose(p - new0, 1e-3 * np.sign(g), rtol=1e-6)


def test_second_step_with_previous_moments():
    p, g = np.array([0.7]), np.array([0.05])
    p1, m1, v1 = adamw_step(p, g, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1, step=1)
    p2, _, _ = adamw_step(p1, g, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1, step=2, m0=m1, v0=v1)
    rp, _, _ = _reference(0.7, 0.05, 1e-3, 0.9, 0.95, 1e-8, 0.1, 2)
    assert p2[0] == pytest.approx(rp, rel=1e-14)


def test_wrong_hyperparameter_changes_the_update():
    """工具先自证：β2 差一点、或衰减没关，重放结果就和正确的不一样（optimizer_step 靠这个抓错）。"""
    p, g = np.ones(4), np.full(4, 0.01)
    good, _, _ = adamw_step(p, g, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.0, step=3)
    bad_b2, _, _ = adamw_step(p, g, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0, step=3)
    bad_wd, _, _ = adamw_step(p, g, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.033, step=3)
    assert not np.array_equal(good, bad_b2) and not np.array_equal(good, bad_wd)


def test_no_weight_decay_rule():
    assert no_weight_decay("decoder.layers.0.input_layernorm.weight", (4096,))  # 一维
    assert no_weight_decay("decoder.layers.0.mlp.linear_fc1.bias", (8, 8))  # 名字是 bias
    assert not no_weight_decay("decoder.layers.0.mlp.linear_fc1.weight", (8, 8))


def test_clip_coeff():
    assert clip_coeff(10.0, None) == 1.0 and clip_coeff(10.0, 0.0) == 1.0 and clip_coeff(None, 1.0) == 1.0
    assert clip_coeff(0.5, 1.0) == 1.0  # 范数小于阈值不裁
    assert clip_coeff(4.0, 1.0) == pytest.approx(1.0 / (4.0 + 1e-6))
    p, g = np.zeros(2), np.array([3.0, 4.0])  # 裁剪系数乘在梯度上
    a, _, _ = adamw_step(p, g, lr=1.0, beta1=0.0, beta2=0.0, eps=0.0, wd=0.0, clip_coeff=0.5)
    b, _, _ = adamw_step(p, g * 0.5, lr=1.0, beta1=0.0, beta2=0.0, eps=0.0, wd=0.0)
    assert np.array_equal(a, b)
