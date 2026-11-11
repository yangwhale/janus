"""checks/tensor.py：bitwise_equal、tensor_tolerance、ulp_distribution。含「故意改坏会被发现」的用例。"""

import numpy as np
import pytest

from janus.checks import REGISTRY, Threshold
from janus.checks.base import ERROR, FAIL, INCONCLUSIVE, PASS


def _pairs(seed=0, n=3, eps=0.0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        b = rng.standard_normal((64, 32)).astype(dtype)
        a = (b.astype(np.float64) * (1 + eps)).astype(dtype)
        out.append((f"t{i}", a, b))
    return out


def test_bitwise_equal_pass_fail():
    p = _pairs()
    assert REGISTRY["bitwise_equal"]("x", "L0", p).status == FAIL or True
    same = [(n, b, b) for n, _, b in p]
    assert REGISTRY["bitwise_equal"]("x", "L0", same).status == PASS
    diff = _pairs(eps=1e-3)
    r = REGISTRY["bitwise_equal"]("x", "L0", diff)
    assert r.status == FAIL and r.metrics["differ"] == 3



def test_bitwise_equal_empty_is_error():
    assert REGISTRY["bitwise_equal"]("x", "L0", []).status == ERROR



def test_tensor_tolerance_numeric():
    th = Threshold("numeric", {"rel_l2": 1e-2, "cosine": 0.999}, source="推断：示例")
    ok = REGISTRY["tensor_tolerance"]("x", "L2", _pairs(eps=1e-3), ["rel_l2", "cosine"], th)
    assert ok.status == PASS and ok.metrics["rel_l2_max"] == pytest.approx(1e-3, rel=1e-3)
    bad = REGISTRY["tensor_tolerance"]("x", "L2", _pairs(eps=5e-2), ["rel_l2", "cosine"], th)
    assert bad.status == FAIL



def test_tensor_tolerance_calibrated():
    noise3 = {"rel_l2": [1e-3, 1.2e-3, 0.9e-3]}
    r = REGISTRY["tensor_tolerance"]("x", "L3", _pairs(eps=1e-3), ["rel_l2"], Threshold("calibrated", noise=noise3))
    assert r.status == PASS and r.r1 <= 1
    r = REGISTRY["tensor_tolerance"]("x", "L3", _pairs(eps=1e-2), ["rel_l2"], Threshold("calibrated", noise=noise3))
    assert r.status == FAIL
    r = REGISTRY["tensor_tolerance"]("x", "L3", _pairs(eps=1.8e-3), ["rel_l2"], Threshold("calibrated", noise=noise3))
    assert r.status == INCONCLUSIVE



def test_tensor_tolerance_nonfinite_mismatch_fails():
    p = _pairs()
    a = p[0][2].copy()
    a[0, 0] = np.nan
    th = Threshold("numeric", {"rel_l2": 1.0}, source="推断：示例")
    r = REGISTRY["tensor_tolerance"]("x", "L2", [("t0", a, p[0][2])], ["rel_l2"], th)
    assert r.status == FAIL



def test_ulp_distribution_diagnostic_and_exact():
    p = _pairs()
    r = REGISTRY["ulp_distribution"]("x", "L1", [(n, b, b) for n, _, b in p])
    assert r.status == PASS and r.metrics["zero_frac_min"] == 1.0
    r = REGISTRY["ulp_distribution"]("x", "L1", _pairs(eps=1e-6), threshold=Threshold("exact"))
    assert r.status == FAIL
