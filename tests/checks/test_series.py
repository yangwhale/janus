import math

import numpy as np
import pytest

from janus import metrics as M
from janus.checks import REGISTRY, Threshold
from janus.checks.base import ERROR, FAIL, INCONCLUSIVE, PASS
from janus.checks.series import first_differ_step, loss_series, noise_from_pairs


def curve(seed, n=101, scale=1e-4, bias=0.0):
    rng = np.random.default_rng(seed)
    base = 3.0 - np.linspace(0, 1, n)
    return {s: float(base[s] * (1 + bias + scale * rng.standard_normal())) for s in range(n)}


def test_loss_series_numeric_and_nonfinite():
    g = curve(0, scale=0)
    t = curve(1)
    th = Threshold("numeric", {"signed_mean": 1e-3, "max_step": 1e-2}, source="推断：示例")
    r = REGISTRY["loss_series"]("l", "L4", t, g, th, params={"window": "0:101"})
    assert r.status == PASS and r.metrics["max_step_at"] is not None
    t[50] = float("nan")
    r = REGISTRY["loss_series"]("l", "L4", t, g, th, params={"window": "0:101"})
    assert r.status == FAIL and 50 in r.metrics["nonfinite_steps"]


def test_loss_series_hard_limit():
    g = curve(0, scale=0)
    t = dict(g)
    t[30] = g[30] * 1.05
    th = Threshold("numeric", {"signed_mean": 1.0}, source="推断：示例")
    r = REGISTRY["loss_series"]("l", "L4", t, g, th, params={"window": "0:101", "hard_limit": {"max_step": 0.02}})
    assert r.status == FAIL and "硬上限" in r.message


def test_loss_series_missing_window_is_error():
    r = REGISTRY["loss_series"]("l", "L4", curve(0), curve(1), Threshold("exact"), params={})
    assert r.status == ERROR


def test_multi_golden_calibrated_from_golden_pairs():
    goldens = {f"g{i}": curve(10 + i) for i in range(1, 4)}
    mid = {s: float(np.mean([g[s] for g in goldens.values()])) for s in range(101)}   # 落在 golden 中间
    ok = REGISTRY["multi_golden"]("m", "L4", mid, goldens, Threshold("calibrated"), params={"window": "0:101"})
    assert ok.status in (PASS, INCONCLUSIVE) and ok.metrics["goldens"] == 3
    biased = REGISTRY["multi_golden"]("m", "L4", curve(21, bias=5e-3), goldens, Threshold("calibrated"),
                                      params={"window": "0:101"})
    assert biased.status == FAIL


def test_multi_golden_baseline_compares_to_one_golden():
    """单 golden 模式：判定量只对 baseline 算；噪声仍取全部 golden 两两组合；baseline 不在列表里报 error。"""
    goldens = {f"g{i}": curve(10 + i) for i in range(1, 6)}
    t = {s: v * (1 + 2e-4) for s, v in goldens["g1"].items()}
    worst = REGISTRY["multi_golden"]("m", "L4", t, goldens, Threshold("calibrated"), params={"window": "0:101"})
    one = REGISTRY["multi_golden"]("m", "L4", t, goldens, Threshold("calibrated"),
                                   params={"window": "0:101", "baseline": "g1"})
    assert one.metrics["baseline"] == "g1" and [d["name"] for d in one.details] == ["g1"]
    assert one.metrics["signed_mean_max"] == pytest.approx(2e-4, rel=1e-6)
    assert one.metrics["signed_mean_max"] <= worst.metrics["signed_mean_max"]
    assert one.threshold["noise"]["refs"] == 10                  # 5 份 golden 两两 10 对
    bad = REGISTRY["multi_golden"]("m", "L4", t, goldens, Threshold("calibrated"),
                                   params={"window": "0:101", "baseline": "gx"})
    assert bad.status == ERROR


def test_first_step_tol_needs_source_and_judges():
    g = curve(0, scale=0)
    t = dict(g)
    t[0] = g[0] * (1 + 1e-3)
    th = Threshold("calibrated", noise={"mare": [1.0] * 3, "signed_mean": [1.0] * 3, "max_step": [1.0] * 3})
    r = REGISTRY["loss_series"]("l", "L4", t, g, th, params={"window": "0:101", "first_step_tol": 1e-4})
    assert r.status == ERROR
    r = REGISTRY["loss_series"]("l", "L4", t, g, th, params={"window": "0:101", "first_step_tol": 1e-4,
                                                              "first_step_source": "推断：示例"})
    assert r.status == FAIL


def test_multi_golden_in_band():
    goldens = {"a": {s: 1.0 for s in range(11)}, "b": {s: 2.0 for s in range(11)}}
    r = REGISTRY["multi_golden"]("m", "L4", {s: 1.5 for s in range(11)}, goldens, Threshold("calibrated"),
                                 params={"window": "0:11"})
    assert r.metrics["in_band_frac"] == 1.0


def test_scalar_series_ulp_floor():
    ref = {s: float(np.float32(1e-4 * s)) for s in range(10)}
    t = dict(ref)
    t[5] = float(np.nextafter(np.float32(ref[5]), np.float32(1)))      # 差 1 格：在 2 格容差内
    assert REGISTRY["scalar_series"]("s", "L4", t, ref, params={"window": "0:10"}).status == PASS
    t[6] = ref[6] * 1.001
    r = REGISTRY["scalar_series"]("s", "L4", t, ref, params={"window": "0:10"})
    assert r.status == FAIL and r.metrics["first_bad"] == 6


def test_scalar_series_equals_registered_value():
    """equals：两边逐步相同还不够，两边都要等于登记值（设置核对，如常数学习率）。"""
    ref = {s: 1e-5 for s in range(1, 11)}
    p = {"window": "1:11", "equals": 1e-5, "abs_ulp": 0}
    r = REGISTRY["scalar_series"]("L0.lr", "L0", dict(ref), ref, params=p)
    assert r.status == PASS and r.metrics["equals"] == 1e-5
    both_wrong = {s: 2e-5 for s in range(1, 11)}      # 两边相同，但不是登记值
    assert REGISTRY["scalar_series"]("s", "L0", both_wrong, both_wrong, params={"window": "1:11"}).status == PASS
    r = REGISTRY["scalar_series"]("s", "L0", both_wrong, both_wrong, params=p)
    assert r.status == FAIL and r.metrics["first_bad"] == 1
    ref_only = dict(ref)
    ref_only[4] = 1.1e-5                              # 只有参照一侧偏离
    r = REGISTRY["scalar_series"]("s", "L0", dict(ref), ref_only, params=p)
    assert r.status == FAIL and r.metrics["first_bad"] == 4


def test_noise_verdict():
    noise = {"mare": [1e-3, 1.1e-3, 0.9e-3], "signed_mean": [1e-4, 2e-4, 1.5e-4]}
    assert REGISTRY["noise_verdict"]("n", "L4", {"mare": 1e-3, "signed_mean": 1e-4}, noise).status == PASS
    r = REGISTRY["noise_verdict"]("n", "L4", {"mare": 1e-3, "signed_mean": 5e-4}, noise)
    assert r.status == FAIL and r.r2 == pytest.approx(2.5)


def test_noise_from_pairs_shapes():
    out = noise_from_pairs([(curve(1), curve(2)), (curve(1), curve(3))], (0, 101))
    assert len(out["mare"]) == 2 and all(v >= 0 for v in out["signed_mean"])


def test_mean_se_autocorrelation():
    rng = np.random.default_rng(0)
    white = rng.normal(0, 1, 2000)
    assert 0.9 < M.mean_se(white) * math.sqrt(2000) < 1.4
    ar = np.zeros(2000)  # AR(1)，ρ = 0.8：有效样本约 n/9，SE 约放大 3 倍
    e = rng.normal(0, 1, 2000)
    for i in range(1, 2000):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    assert 2.5 < M.mean_se(ar) / (ar.std(ddof=1) / math.sqrt(2000)) < 3.5
    assert M.mean_se([1.0]) == 0.0 and M.mean_se([2.0, 2.0, 2.0]) == 0.0


def _noisy(seed, bias=0.0, n=200):
    rng = np.random.default_rng(seed)
    return {s: 2.0 * (1 + bias + 1e-3 * rng.normal()) for s in range(n)}


def test_signed_mean_se_floor():
    ref = _noisy(1)
    pairs = [(_noisy(10 + i), ref) for i in range(3)]
    noise = noise_from_pairs(pairs, (1, 200))
    assert len(noise["signed_mean_se"]) == 3
    # 噪声参照的带符号均值人为压小：不开下限时上限过紧
    noise["signed_mean"] = [1e-7, 1e-7, 1e-7]
    test = _noisy(99)
    th = Threshold("calibrated", noise=noise)
    off = loss_series("c", "L4", test, ref, th, params={"window": "1:200"})
    on = loss_series("c", "L4", test, ref, th, params={"window": "1:200", "signed_mean_se_k": 2})
    assert off.r2 > on.r2
    assert on.threshold["noise"]["floor"]["signed_mean"] == 2 * max(noise["signed_mean_se"])
    assert on.metrics["signed_mean_se"] > 0
    # 下限不掩盖真偏移：+1% 系统偏差仍 fail
    biased = loss_series("c", "L4", _noisy(99, bias=0.01), ref, th, params={"window": "1:200", "signed_mean_se_k": 2})
    assert biased.status == "fail"


# ---------------------------------------------------------------- 噪声参照剔重与留一法

def test_noise_from_pairs_drops_bitwise_identical_pairs():
    a, b, c = curve(1), curve(2), curve(3)
    dup = dict(a)                                         # 与 a 逐位相同（如两次确定性运行）
    out = noise_from_pairs([(a, b), (a, dup), (b, c)], (1, 101), names=["a~b", "a~dup", "b~c"])
    assert out["dropped"] == ["a~dup"]
    assert len(out["mare"]) == 2                          # 剔掉后只剩 2 条
    # 剔除后参照不足 3 条：calibrated 最多 inconclusive
    th = Threshold("calibrated", noise=out)
    r = loss_series("l", "L4", curve(4), a, th, params={"window": "1:101"})
    assert r.status in (INCONCLUSIVE, FAIL) and r.metrics["noise_dropped"] == ["a~dup"]
    assert r.threshold["noise"]["refs"] == 2


def test_noise_leave_one_out_flags_outlier_pair():
    pairs = [(curve(1), curve(2)), (curve(3), curve(4)), (curve(5), curve(6)), (curve(7), curve(8, scale=1e-3))]
    out = noise_from_pairs(pairs, (1, 101), names=["p1", "p2", "p3", "wide"])
    loo = {r["pair"]: r for r in out["loo"]}
    assert set(loo) == {"p1", "p2", "p3", "wide"}
    assert loo["wide"]["mare"] > 3                        # 噪声大一个量级的那条，用其余三条的上限判明显超
    assert all(loo[p]["mare"] < 1 for p in ("p1", "p2", "p3"))   # 其余三条的上限被 wide 撑着
    r = loss_series("l", "L4", curve(9), curve(1), Threshold("calibrated", noise=out), params={"window": "1:101"})
    assert {x["pair"] for x in r.metrics["noise_loo"]} == set(loo)


# ---------------------------------------------------------------- 步号偏移探测

def test_probe_offset_finds_shift():
    from janus.checks.series import probe_offset
    g = curve(0, scale=0)
    t = {s - 1: v for s, v in g.items()}                  # 被测第 s 步其实是 golden 第 s+1 步
    d, cand = probe_offset(t, g, 1, (-2, 2))
    assert d == 1 and cand[1] == 0.0
    assert probe_offset(g, g, 1, (-2, 2))[0] == 0


def test_auto_offset_fails_when_convention_wrong():
    g = curve(0, scale=0)
    t = {s + 1: v for s, v in g.items()}                  # 约定写错一格：换算后整体后移一步
    th = Threshold("calibrated", noise={"signed_mean": [1.0] * 3, "mare": [1.0] * 3, "max_step": [1.0] * 3})
    p = {"window": "2:100", "first_step": 1, "auto_offset": [-2, 2], "step_offset": 1}
    r = loss_series("l", "L4", t, g, th, params=p)
    assert r.status == FAIL and "步号偏移" in r.message and "step +0" in r.message
    assert r.metrics["offset_probe"]["ref"]["delta"] == -1
    # 没开探测时，首步检查也必须红（窗口错位缺步时也先判首步，不报 error）
    p2 = {"window": "1:101", "first_step": 1, "first_step_tol": 0.0, "first_step_source": "测试"}
    r = loss_series("l", "L4", t, g, th, params=p2)
    assert r.status == FAIL and "首步差" in r.message
    # 对齐时探测给 0，不影响判定
    r = loss_series("l", "L4", g, g, th, params={**p, "window": "1:101", "step_offset": 0})
    assert r.status == PASS and r.metrics["offset_probe"]["ref"]["delta"] == 0


def test_first_differ_step_and_loss_series_exact():
    a = {1: 2.0, 2: 1.9, 3: 1.8, 4: 1.7}
    b = dict(a)
    assert first_differ_step(a, b, (1, 5)) is None
    b[3] = np.nextafter(1.8, 2.0)
    assert first_differ_step(a, b, (1, 5)) == 3
    r = REGISTRY["loss_series"]("l", "regress", b, a, Threshold("exact"), params={"window": "1:5", "first_step": 1})
    assert r.status == FAIL and r.metrics["first_differ_step"] == 3
    assert first_differ_step(a, {1: 2.0}, (1, 3)) == 2                 # 缺步也算不同
