"""checks/base.py：注册表、门槛判定（数值门槛要 source、calibrated 要 3 条参照才能 pass）。"""

import numpy as np
import pytest

from janus.checks import REGISTRY, Threshold
from janus.checks.base import FAIL, INCONCLUSIVE, PASS, judge_calibrated


def _pairs(seed=0, n=3, eps=0.0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        b = rng.standard_normal((64, 32)).astype(dtype)
        a = (b.astype(np.float64) * (1 + eps)).astype(dtype)
        out.append((f"t{i}", a, b))
    return out


def test_registry_has_first_batch():
    for t in ["bitwise_equal", "tensor_tolerance", "ulp_distribution", "discrete_agreement", "coverage", "poison_selftest"]:
        assert t in REGISTRY



def test_numeric_threshold_requires_source():
    with pytest.raises(ValueError):
        Threshold("numeric", {"rel_l2": 1e-2})



def test_calibrated_needs_three_refs_for_pass():
    assert judge_calibrated(1e-3, [2e-3, 2e-3])[0] == INCONCLUSIVE
    assert judge_calibrated(1e-3, [2e-3, 2e-3, 2e-3])[0] == PASS
    assert judge_calibrated(1e-2, [2e-3])[0] == FAIL
