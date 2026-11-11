import math

import ml_dtypes
import numpy as np
import pytest

from janus import metrics as M


def test_rel_l2_hand_computed():
    a, b = np.array([1.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])
    assert M.rel_l2(a, b) == pytest.approx(1.0 / math.sqrt(14.0), rel=1e-12)


def test_rel_l2_independent_of_element_count():
    rng = np.random.default_rng(0)
    b = rng.standard_normal(1000)
    a = b * (1 + 1e-3)
    big_a, big_b = np.tile(a, 100), np.tile(b, 100)
    assert M.rel_l2(a, b) == pytest.approx(M.rel_l2(big_a, big_b), rel=1e-12)
    assert M.rel_l2(a, b) == pytest.approx(1e-3, rel=1e-9)


def test_rel_l2_zero_reference():
    assert M.rel_l2(np.zeros(3), np.zeros(3)) == 0.0
    assert M.rel_l2(np.ones(3), np.zeros(3)) == math.inf


def test_cosine_and_norm_ratio():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 2.0])
    assert M.cosine(a, b) == pytest.approx(0.0, abs=1e-15)
    assert M.cosine(b, 2 * b) == pytest.approx(1.0, rel=1e-15)
    assert M.norm_ratio(2 * b, b) == pytest.approx(2.0, rel=1e-15)


def test_mean_shift():
    b = np.array([1.0, 3.0])      # mean 2, std 1
    a = np.array([2.0, 4.0])      # mean 3
    assert M.mean_shift(a, b) == pytest.approx(1.0, rel=1e-12)


def test_max_abs_location():
    b = np.zeros((2, 3))
    a = b.copy()
    a[1, 2] = -0.5
    v, at = M.max_abs(a, b)
    assert v == 0.5 and at == (1, 2)


def test_out_of_tol_ratio():
    b = np.array([1.0, 1.0, 1.0, 1.0])
    a = np.array([1.0, 1.0005, 1.01, 2.0])
    assert M.out_of_tol_ratio(a, b, atol=0.0, rtol=1e-3) == pytest.approx(0.5)


def test_nonfinite_positions_must_match():
    a = np.array([1.0, np.nan])
    assert M.finite_check(a, np.array([1.0, np.nan])).ok
    assert not M.finite_check(a, np.array([1.0, 2.0])).ok
    with pytest.raises(ValueError):
        M.rel_l2(a, np.array([1.0, 2.0]))
    assert M.rel_l2(a, np.array([1.0, np.nan])) == 0.0


def test_ulp_adjacent_float32_and_bf16():
    x = np.array([1.0, -3.5, 1e-30], dtype=np.float32)
    y = np.nextafter(x, np.float32(np.inf)).astype(np.float32)
    assert list(M.ulp_distance(y, x)) == [1, 1, 1]
    bx = np.array([1.0, 0.5], dtype=ml_dtypes.bfloat16)
    by = (bx.view(np.uint16) + np.uint16(1)).view(ml_dtypes.bfloat16)
    assert list(M.ulp_distance(by, bx)) == [1, 1]


def test_ulp_signed_zero_and_cross_zero():
    z = np.array([0.0], dtype=np.float32)
    nz = np.array([-0.0], dtype=np.float32)
    assert M.ulp_distance(z, nz)[0] == 0
    s = M.ulp_summary(np.array([1.0, -1e-3], dtype=np.float32), np.array([1.0, 1e-3], dtype=np.float32))
    assert s.cross_zero == 1 and s.n == 1 and s.zero_frac == 1.0


def test_ulp_requires_same_dtype():
    with pytest.raises(TypeError):
        M.ulp_distance(np.zeros(2, np.float32), np.zeros(2, np.float16))


def test_topk_agreement_and_margin():
    a = np.array([[0, 1, 2], [3, 4, 5]])
    b = np.array([[1, 0, 2], [3, 4, 6]])
    t = M.topk_agreement(a, b)
    assert t.set_frac == 0.5 and t.order_frac == 0.0 and list(t.rows_differ) == [1]
    scores = np.array([[0.9, 0.5, 0.49, 0.1]])
    assert M.topk_margin(scores, 2)[0] == pytest.approx(0.01, rel=1e-9)


def test_series_stats():
    ref = {s: 2.0 for s in range(5)}
    test = {0: 2.0, 1: 2.002, 2: 1.998, 3: 2.004, 4: 2.0}
    st = M.series_stats(test, ref, (0, 5))
    assert st.d0 == 0.0
    assert st.signed_mean == pytest.approx(0.0005, rel=1e-9)     # (1e-3 − 1e-3 + 2e-3 + 0) / 4
    assert st.mare == pytest.approx(0.001, rel=1e-9)
    assert st.max_step == pytest.approx(0.002, rel=1e-9) and st.max_step_at == 3


def test_series_stats_nonfinite_and_missing():
    ref = {0: 1.0, 1: 1.0, 2: 1.0}
    st = M.series_stats({0: 1.0, 1: float("nan"), 2: 1.0}, ref, (0, 3))
    assert st.nonfinite_steps == [1]
    with pytest.raises(ValueError):
        M.series_stats({0: 1.0}, ref, (0, 3))


# ---- 改坏数据必须被发现

@pytest.mark.parametrize("poison", ["flip_bit", "nan", "shift"])
def test_poison_is_detected(poison):
    rng = np.random.default_rng(1)
    b = rng.standard_normal(4096).astype(np.float32)
    a = b.copy()
    if poison == "flip_bit":
        a.view(np.uint32)[7] ^= np.uint32(1 << 20)
        assert M.ulp_distance(a, b).max() > 0 and M.rel_l2(a, b) > 0
    elif poison == "nan":
        a[11] = np.nan
        assert not M.finite_check(a, b).ok
    else:
        a = np.roll(a, 1)
        assert M.rel_l2(a, b) > 0.5


def test_pair_stats_matches_individual():
    rng = np.random.default_rng(3)
    a = rng.normal(2.0, 1.0, (64, 300)).astype(np.float32)
    b = (a + rng.normal(0, 1e-3, a.shape)).astype(np.float32)
    old = M.CHUNK
    M.CHUNK = 1000                      # 强制分很多块
    try:
        st = M.pair_stats(a, b)
    finally:
        M.CHUNK = old
    assert abs(st["rel_l2"] - M.rel_l2(a, b)) <= 1e-12 * max(1.0, M.rel_l2(a, b))
    assert abs(st["cosine"] - M.cosine(a, b)) < 1e-12
    assert abs(st["mean_shift"] - M.mean_shift(a, b)) < 1e-9
    assert st["max_abs"] == M.max_abs(a, b)[0] and st["max_abs_at"] == M.max_abs(a, b)[1]
    assert abs(st["norm_ratio"] - M.norm_ratio(a, b)) < 1e-12
    bad = a.copy()
    bad[0, 0] = np.nan
    assert M.pair_stats(bad, b) is None


# ---------------------------------------------------------------- 1 ULP 比例与分桶

def test_ulp_buckets_hand_computed():
    b = np.ones(10, np.float32)
    u = b.view(np.uint32).copy()
    u += np.array([0, 0, 0, 1, 1, 3, 6, 12, 20, 100], np.uint32)     # 0,0,0 | 1,1 | 3 | 6 | 12 | 20,100
    a = u.view(np.float32)
    s = M.ulp_summary(a, b)
    assert s.zero_frac == pytest.approx(0.3) and s.one_frac == pytest.approx(0.2)
    assert s.buckets == pytest.approx({"0": 0.3, "1": 0.2, "2-4": 0.1, "5-8": 0.1, "9-16": 0.1, "17+": 0.2})
    assert sum(s.buckets.values()) == pytest.approx(1.0)


def test_ulp_buckets_bf16_counts_bf16_grid():
    b = np.array([1.0, 2.0], np.float32).astype(ml_dtypes.bfloat16)
    a = (b.view(np.uint16) + np.uint16(1)).view(ml_dtypes.bfloat16)
    s = M.ulp_summary(a, b)
    assert s.one_frac == 1.0 and s.buckets["1"] == 1.0


# ---------------------------------------------------------------- 逐元素相对误差、std 比、零比例差

def test_rel_err_p99_symmetric_denominator_and_eps():
    b = np.ones(1000)
    a = b.copy()
    a[:20] = 1.5                                      # 2% 元素差 0.5 → 相对误差 0.5 / 1.5
    assert M.rel_err_p99(a, b) == pytest.approx(1 / 3)
    assert M.rel_err_p99(b, a) == pytest.approx(M.rel_err_p99(a, b))   # 对称
    # 两边都接近 0 的元素被 eps（1e-3 × RMS(b)）兜住，不会给出巨大的相对误差
    b2 = np.ones(1000)
    b2[:20] = 1e-12
    a2 = b2.copy()
    a2[:20] = -1e-12
    assert M.rel_err_p99(a2, b2) < 1e-8
    assert M.rel_err_p99(np.zeros(5), np.zeros(5)) == 0.0


def test_std_ratio_and_zero_frac_diff():
    rng = np.random.default_rng(0)
    b = rng.standard_normal(10000)
    assert M.std_ratio(b * 1.1 + 5.0, b) == pytest.approx(1.1)        # 平移不改 std 比
    assert M.std_ratio(np.zeros(3), np.zeros(3)) == 1.0
    assert M.std_ratio(np.arange(3.0), np.zeros(3)) == math.inf
    a = b.copy()
    a[:500] = 0.0
    assert M.zero_frac_diff(a, b) == pytest.approx(0.05)
    assert M.zero_frac_diff(np.array([0.0, -0.0, 1.0]), np.array([0.0, 2.0, 1.0])) == pytest.approx(1 / 3)


# ---------------------------------------------------------------- 点积误差归一

def test_normalized_dot_error():
    x = np.array([1e8, 1.0, -1e8])
    y = np.ones(3)
    ref = float(np.dot(x.astype(np.float64), y))           # 1.0（正负抵消）
    abs_sum = float(np.dot(np.abs(x), np.abs(y)))          # 2e8 + 1
    s_hat = float(np.float32(np.float32(1e8) + np.float32(1.0)) - np.float32(1e8))   # FP32 顺序累加丢掉 1 → 0
    e = M.normalized_dot_error(s_hat, ref, abs_sum)
    assert e == pytest.approx(1.0 / abs_sum)               # 按 |s| 归一会是 100%，按 Σ|xy| 归一只有 5e-9
    arr = M.normalized_dot_error(np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([4.0, 0.0]))
    assert arr[0] == 0.0 and arr[1] == math.inf
    assert M.normalized_dot_error(0.0, 0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        M.normalized_dot_error(1.0, 1.0, -1.0)


# ---------------------------------------------------------------- logits KL

def test_kl_direction_and_hand_value():
    ref = np.log(np.array([[0.5, 0.5]]))
    tst = np.log(np.array([[0.9, 0.1]]))
    want = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)      # KL(参照 ‖ 被测)
    assert M.kl_div(tst, ref) == pytest.approx(want, rel=1e-12)
    other = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)    # 反方向不同
    assert abs(M.kl_div(tst, ref) - other) > 0.1
    assert M.kl_div(ref, ref) == 0.0


def test_kl_chunked_matches_unchunked_and_masks():
    rng = np.random.default_rng(1)
    b = rng.standard_normal((7, 3, 11)).astype(ml_dtypes.bfloat16)
    a = (b.astype(np.float32) + 0.01 * rng.standard_normal(b.shape)).astype(ml_dtypes.bfloat16)
    assert M.kl_div(a, b, chunk_rows=2) == pytest.approx(M.kl_div(a, b, chunk_rows=1000), rel=1e-12)
    m = np.zeros((1, 3))
    m[0, 2] = -np.inf                                      # 两边都屏蔽同一个 token：按概率 0 处理
    assert M.kl_div(m, m) == 0.0
    t = m.copy()
    t[0, 1] = -np.inf
    r = np.zeros((1, 3))
    r[0, 2] = -np.inf
    with pytest.raises(ValueError):                        # 位置不一致
        M.kl_div(t, r)


def test_metrics_are_computed_in_fp64():
    """指标在 FP64 里算（docs/dev/architecture.md 不变式第 3 条）：FP32 输入、差在末几位时，结果对 math.fsum 的精确参照
    到 1e-12。同一个公式全程用 FP32 算，误差在 1e-9 以上——这一句是自证：说明这个用例分得开两种实现。"""
    rng = np.random.default_rng(0)
    b = (1e3 + rng.standard_normal(200_000)).astype(np.float32)
    a = (b.astype(np.float64) + 1e-4 * rng.standard_normal(b.size)).astype(np.float32)
    x, y = a.astype(np.float64), b.astype(np.float64)
    exact = math.sqrt(math.fsum((x - y) ** 2)) / math.sqrt(math.fsum(y**2))
    assert abs(M.rel_l2(a, b) - exact) / exact < 1e-12
    fp32 = float(np.linalg.norm(a - b) / np.linalg.norm(b))   # 全程 FP32（numpy 对 float32 输入按 float32 累加）
    assert abs(fp32 - exact) / exact > 1e-9
