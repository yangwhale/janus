"""analysis/ops.py：L1 单算子的 CPU 参照（FP64 正确舍入、GEMM 舍入模拟、R-ratio、打平、解析 VJP）。含「故意改坏会被发现」的用例。"""

import ml_dtypes
import numpy as np
import pytest

from janus import metrics as M
from janus.analysis import ops as O

BF16 = ml_dtypes.bfloat16


rng = np.random.default_rng(0)


def _case(out, truth, **kw):
    return dict(name=kw.pop("name", "c"), out=out, fp64=truth, **kw)


def test_round_fp32_matches_numpy():
    x = rng.standard_normal(10000) * np.exp(rng.uniform(-80, 80, 10000))
    x = np.concatenate([x, [0.0, -0.0, np.inf, -np.inf, 1e-45, 3e-39, 3.5e38, -3.5e38]])
    with np.errstate(over="ignore"):
        ref = x.astype(np.float32)
    assert np.array_equal(O.round_from_f64(x, np.float32).view(np.uint32), ref.view(np.uint32))



def test_round_bf16_single_rounding():
    # 1 + 2^-8 + 2^-30：经 FP32 中转会先舍成正好的中点 1 + 2^-8，再就近取偶成 1.0（两次舍入错）；一次舍入应得 1 + 2^-7
    x = np.array([1 + 2.0 ** -8 + 2.0 ** -30])
    assert float(O.round_from_f64(x, BF16)[0]) == 1 + 2.0 ** -7
    # 正好的中点就近取偶
    assert float(O.round_from_f64(np.array([1 + 2.0 ** -8]), BF16)[0]) == 1.0
    assert float(O.round_from_f64(np.array([1 + 3 * 2.0 ** -8]), BF16)[0]) == 1 + 2.0 ** -6
    # 次正规数（格距 2^-133）与溢出
    assert float(O.round_from_f64(np.array([1.5 * 2.0 ** -133]), BF16)[0]) == 2.0 ** -132
    assert np.isinf(O.round_from_f64(np.array([3.4e38]), BF16)[0])
    assert float(O.round_from_f64(np.array([3.389e38]), BF16)[0]) == float(ml_dtypes.finfo(BF16).max)
    # 随机值：结果必须是相邻两个 BF16 格点里更近的那个
    v = rng.standard_normal(5000) * 10.0
    r = O.round_from_f64(v, BF16).astype(np.float64)
    up = np.nextafter(O.round_from_f64(v, BF16), np.array(np.inf, dtype=BF16)).astype(np.float64)
    dn = np.nextafter(O.round_from_f64(v, BF16), np.array(-np.inf, dtype=BF16)).astype(np.float64)
    assert np.all(np.abs(r - v) <= np.abs(up - v)) and np.all(np.abs(r - v) <= np.abs(dn - v))



def test_same_bits_nan_and_signed_zero():
    a = np.array([np.nan, 0.0, 1.0], np.float32)
    b = np.array([np.nan, -0.0, 1.0], np.float32)
    assert O.same_bits(a, b).tolist() == [True, False, True]
    with pytest.raises(TypeError):
        O.same_bits(a, a.astype(np.float64))



def test_r_ratio_sees_extra_rounding_cosine_does_not():
    """多舍一次：中间结果先舍到 BF16。R-ratio 明显 > 1.2，而两边 cosine 都在 0.9999 以上、看不出差别。"""
    x = rng.standard_normal((256, 512)).astype(BF16)
    w = rng.standard_normal((64, 512)).astype(BF16)
    s = rng.uniform(0.5, 2.0, 64)
    truth = (x.astype(np.float64) @ w.astype(np.float64).T) * s
    ref = O.round_from_f64(truth, BF16)                                     # 一次舍入
    mid = O.round_from_f64(x.astype(np.float64) @ w.astype(np.float64).T, BF16)
    test = O.round_from_f64(mid.astype(np.float64) * s, BF16)              # 多舍一次
    r = O.r_ratio(test.astype(np.float64), ref.astype(np.float64), truth)
    assert r["r_ratio"] > 1.2
    assert r["cosine_test"] > 0.9999 and r["cosine_ref"] > 0.9999
    same = O.r_ratio(ref.astype(np.float64), ref.astype(np.float64), truth)
    assert same["r_ratio"] == 1.0



def test_gemm_emulate_modes():
    x = rng.standard_normal((8, 64)).astype(BF16)
    w = rng.standard_normal((5, 64)).astype(BF16)
    xf, wf = x.astype(np.float32), w.astype(np.float32)
    seq = np.zeros((8, 5), np.float32)
    for k in range(64):
        seq = seq + xf[:, k:k + 1] * wf[:, k][None, :]
    assert np.array_equal(O.gemm_emulate(x, w, "seq"), seq)
    assert np.array_equal(O.gemm_emulate(x, w, "block", block=64), seq)
    exact = x.astype(np.float64) @ w.astype(np.float64).T
    assert np.array_equal(O.gemm_emulate(x, w, "fp64"), exact.astype(np.float32))
    # 张量核模型：位数给足、一组装下全部 K、RNE 写回 ⇒ 等于正确舍入的点积
    assert np.array_equal(O.gemm_emulate(x, w, "tc", group=64, frac_bits=40, rounding="rne"), exact.astype(np.float32))
    # 截断位数很少时与正确舍入明显不同（模型真的在截）
    assert not np.array_equal(O.gemm_emulate(x, w, "tc", group=16, frac_bits=12), exact.astype(np.float32))
    # 手算：1 + 1.5·2^-24。正确舍入（0.75 格）进位成 1 + 2^-23；张量核模型保留下来但写回时向零截断，得 1.0
    a = np.array([[1.0, 1.5 * 2.0 ** -24]]).astype(BF16)
    b = np.array([[1.0, 1.0]]).astype(BF16)
    assert float(O.gemm_emulate(a, b, "fp64")[0, 0]) == 1 + 2.0 ** -23
    assert float(O.gemm_emulate(a, b, "tc", group=16, frac_bits=25)[0, 0]) == 1.0
    assert float(O.gemm_emulate(a, b, "tc", group=16, frac_bits=25, rounding="rne")[0, 0]) == 1 + 2.0 ** -23
    assert O.gemm_emulate(x, w, "fp64", rows=[1, 3]).shape == (2, 5)



def test_topk_stable_and_ties():
    s = np.array([[1.0, 3.0, 3.0, 2.0, 3.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    assert O.topk_stable(s, 2).tolist() == [[1, 2], [0, 1]]
    assert O.boundary_ties(s, 2).tolist() == [0]



def test_rmsnorm_and_router_vjp_match_finite_difference():
    x = rng.standard_normal((4, 32))
    w = rng.uniform(0.5, 1.5, 32)
    u = rng.standard_normal((4, 32))
    v = rng.standard_normal((4, 32))
    gx, gw = O.rmsnorm_vjp_f64(x, w, 1e-5, u)
    fd = O.directional_fd(lambda t: O.rmsnorm_f64(t, w, 1e-5), x, v, u)
    assert abs(np.sum(gx * v) - fd) <= 1e-7 * abs(fd)
    vw = rng.standard_normal(32)
    fdw = O.directional_fd(lambda t: O.rmsnorm_f64(x, t, 1e-5), w, vw, u)
    assert abs(np.sum(gw * vw) - fdw) <= 1e-7 * abs(fdw)

    z = rng.standard_normal((6, 16))
    bias = rng.standard_normal(16) * 0.01
    _, idx = O.router_probs_f64(z, bias, 4, 2.5)
    up = rng.standard_normal((6, 4))
    gz = O.router_probs_vjp_f64(z, idx, 2.5, up)
    vz = rng.standard_normal((6, 16))
    fdz = O.directional_fd(lambda t: O.router_probs_f64(t, bias, 4, 2.5, idx=idx)[0], z, vz, up)
    assert abs(np.sum(gz * vz) - fdz) <= 1e-7 * abs(fdz)
    # 改坏解析梯度（漏掉归一化那一项）会被数值梯度发现
    s = O.sigmoid_f64(z)
    sel = np.take_along_axis(s, idx, -1)
    bad = np.zeros_like(z)
    np.put_along_axis(bad, idx, 2.5 * up / sel.sum(-1, keepdims=True) * sel * (1 - sel), -1)
    assert abs(np.sum(bad * vz) - fdz) > 1e-3 * abs(fdz)



def test_extreme_formulas():
    assert np.all(O.rmsnorm_f64(np.zeros((1, 8)), np.ones(8), 1e-5) == 0.0)
    big = np.full((1, 8), 1e30)
    assert np.allclose(O.rmsnorm_f64(big, np.ones(8), 1e-5), 1.0)
    assert O.sigmoid_f64(np.array([-1000.0]))[0] == 0.0 and O.sigmoid_f64(np.array([1000.0]))[0] == 1.0
    assert np.allclose(O.softmax_f64(np.array([[1e30, 1e30, 0.0]])), [[0.5, 0.5, 0.0]])



def test_ulp_against_cr_is_integer_distance():
    """对正确舍入的 ULP 距离：正确舍入本身是 0，差一格是 1（不是 CUDA 手册里 |误差| ≤ 0.5 那种连续口径）。"""
    t = np.array([1.0 + 2.0 ** -25])
    cr = O.round_from_f64(t, np.float32)
    one = np.nextafter(cr, np.float32(2))
    assert int(M.ulp_distance(cr, cr)[0]) == 0 and int(M.ulp_distance(one, cr)[0]) == 1
