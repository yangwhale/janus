"""checks/meta.py：coverage、poison_selftest（工具自证：盲的检查必须被 poison 抓出来）。"""

import numpy as np
import pytest

from janus.checks import REGISTRY, Threshold
from janus.checks.base import ERROR, FAIL, PASS


def _pairs(seed=0, n=3, eps=0.0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        b = rng.standard_normal((64, 32)).astype(dtype)
        a = (b.astype(np.float64) * (1 + eps)).astype(dtype)
        out.append((f"t{i}", a, b))
    return out


def test_coverage():
    r = REGISTRY["coverage"]("x", "L3", ["grad.a", "grad.b"], ["grad.a"])
    assert r.status == ERROR and r.coverage["missing"] == ["grad.b"]
    r = REGISTRY["coverage"]("x", "L3", ["grad.*"], ["grad.a", "grad.b"])
    assert r.status == PASS



@pytest.mark.parametrize("target,kwargs", [
    ("bitwise_equal", {}),
    ("tensor_tolerance", {"metrics": ["rel_l2", "max_abs"],
                          "threshold": Threshold("numeric", {"rel_l2": 1e-6, "max_abs": 1e-6}, source="推断：自证")}),
    ("ulp_distribution", {"threshold": Threshold("exact")}),
])
def test_poison_selftest_float_types(target, kwargs):
    r = REGISTRY["poison_selftest"]("x", "L0", target, _pairs(), kwargs)
    assert r.status == PASS, r.details



def test_poison_selftest_discrete():
    rng = np.random.default_rng(3)
    idx = rng.integers(0, 192, size=(16, 8))
    r = REGISTRY["poison_selftest"]("x", "L2", "discrete_agreement", [("topk", idx, idx)],
                                    {"threshold": Threshold("exact")}, injections=["flip_bit", "nan", "scale"])
    assert r.status == PASS, r.details



def test_poison_selftest_catches_a_blind_check():
    # 一个「什么都放行」的门槛：自证必须判 fail
    lax = {"metrics": ["cosine"], "threshold": Threshold("numeric", {"cosine": -1.0}, source="推断：故意放水")}
    r = REGISTRY["poison_selftest"]("x", "L0", "tensor_tolerance", _pairs(), lax, injections=["scale"])
    assert r.status == FAIL
