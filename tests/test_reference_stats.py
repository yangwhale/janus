"""参考档统计：janus/metrics.py 的 reference_stats，以及 tensor_tolerance / op_replay 把它记进 metrics（只报不判）。"""

import numpy as np

from janus import metrics as M
from janus.checks import REGISTRY, Threshold


def test_reference_stats_definitions():
    b = np.array([1.0, -2.0, 0.0, 4.0])
    a = b + np.array([0.0, 0.5, 0.0, 0.01])
    st = M.reference_stats(a, b)
    rms = np.sqrt(np.mean(b**2))
    tol = M.REF_ATOL_REL * rms + M.REF_RTOL * np.abs(b)
    assert st["oot_frac"] == np.mean(np.abs(a - b) > tol) == 0.25
    assert np.isclose(st["mean_diff"], np.mean(np.abs(a - b)) / np.mean(np.abs(b)))
    assert np.isclose(st["norm_max"], 0.5 / 4.0)
    assert np.isclose(st["rel_l2"], M.rel_l2(a, b)) and np.isclose(st["cosine"], M.cosine(a, b))
    z = M.reference_stats(np.zeros(3), np.zeros(3))
    assert z["oot_frac"] == z["mean_diff"] == z["norm_max"] == 0.0


def test_tensor_tolerance_reports_reference_stats_without_judging():
    rng = np.random.default_rng(0)
    b = rng.standard_normal(1000).astype(np.float32)
    a = b * 1.05  # 5% 的系统差：超差元素很多
    th = Threshold("calibrated", noise={"rel_l2": [1.0] * 3, "cosine": [0.5] * 3})
    r = REGISTRY["tensor_tolerance"]("x", "L2", [("t", a, b)], ["rel_l2", "cosine"], th)
    assert r.status == "pass"  # 判定只看目录里的指标
    assert r.metrics["oot_frac_max"] > 0.5 and np.isclose(r.metrics["mean_diff_max"], 0.05, rtol=1e-3)
    assert "norm_max_max" in r.metrics


def test_op_replay_vs_ref():
    b = np.linspace(-1, 1, 64).astype(np.float32)
    out = b + 1e-3
    case = {"name": "o", "out": out, "fp64": b.astype(np.float64), "vsref": b}
    r = REGISTRY["op_replay"](
        "L2.layer0.fwd", "L2", [case, {"name": "p", "out": out, "fp64": b.astype(np.float64)}], None
    )
    assert r.metrics["vs_ref_cases"] == 1 and np.isclose(r.metrics["vs_ref_norm_max"], 1e-3, rtol=1e-3)
    other = out + 1.0  # vsref_out：同一输入上的另一份被测输出
    r = REGISTRY["op_replay"]("x", "L1", [{**case, "vsref_out": other}], None)
    assert np.isclose(r.metrics["vs_ref_norm_max"], 1.001, rtol=1e-3)
