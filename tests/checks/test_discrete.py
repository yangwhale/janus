"""checks/discrete.py：discrete_agreement 的 exact 与 topk。"""

import numpy as np

from janus.checks import REGISTRY, Threshold
from janus.checks.base import FAIL, PASS


def _pairs(seed=0, n=3, eps=0.0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        b = rng.standard_normal((64, 32)).astype(dtype)
        a = (b.astype(np.float64) * (1 + eps)).astype(dtype)
        out.append((f"t{i}", a, b))
    return out


def test_discrete_exact_and_topk():
    a = np.array([[0, 1, 2], [3, 4, 5]])
    b = np.array([[1, 0, 2], [3, 4, 6]])
    r = REGISTRY["discrete_agreement"]("x", "L2", [("r", a, b)], Threshold("exact"), params={"mode": "topk"})
    assert r.status == FAIL and r.metrics["set_frac_min"] == 0.5
    r = REGISTRY["discrete_agreement"]("x", "L2", [("r", a, a)], Threshold("exact"), params={"mode": "topk"})
    assert r.status == PASS
    ints = np.arange(10)
    assert REGISTRY["discrete_agreement"]("x", "L0", [("h", ints, ints)], Threshold("exact")).status == PASS
