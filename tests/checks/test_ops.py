"""checks/ops.py：op_replay 检查（逐位、poison、非有限值、R-ratio 门槛、topk 期望、演练必须报警）。"""

import ml_dtypes
import numpy as np

from janus.analysis import ops as O
from janus.checks import REGISTRY, Threshold
from janus.checks.base import ERROR, FAIL, PASS

BF16 = ml_dtypes.bfloat16


rng = np.random.default_rng(0)


def _case(out, truth, **kw):
    return dict(name=kw.pop("name", "c"), out=out, fp64=truth, **kw)


def test_op_replay_exact_and_poison():
    truth = rng.standard_normal(1000)
    good = O.round_from_f64(truth, np.float32)
    fn = REGISTRY["op_replay"]
    r = fn("o", "L1", [_case(good, truth, bitref=good.copy())], Threshold("exact"))
    assert r.status == PASS and r.metrics["cr_match_frac"] == 1.0 and r.metrics["ulp_max"] == 0
    bad = good.copy()
    bad.view(np.uint32)[7] += 1                       # 改坏一个元素的末位
    r = fn("o", "L1", [_case(bad, truth, bitref=good)], Threshold("exact"))
    assert r.status == FAIL and r.metrics["ulp_max"] == 1 and r.metrics["bit_match_frac"] < 1.0
    th = Threshold("numeric", {"ulp_max": 0}, source="推断：测试")
    assert fn("o", "L1", [_case(bad, truth)], th).status == FAIL
    th = Threshold("numeric", {"ulp_max": 2, "cr_match_frac": 0.99}, source="推断：测试")
    assert fn("o", "L1", [_case(bad, truth)], th).status == PASS



def test_op_replay_nonfinite_rratio_and_errors():
    fn = REGISTRY["op_replay"]
    truth = np.array([1.0, np.inf, 2.0])
    out = np.array([1.0, 0.0, 2.0], np.float32)       # 应为 inf 却给了 0（如 RMSNorm 平方溢出）
    r = fn("o", "L1", [_case(out, truth)], Threshold("exact"))
    assert r.status == FAIL and r.metrics["nonfinite_mismatch"] == 1
    t = rng.standard_normal(4096)
    ref = O.round_from_f64(t, BF16)
    test = O.round_from_f64(O.round_from_f64(t * 3, BF16).astype(np.float64) / 3, BF16)
    th = Threshold("numeric", {"r_ratio": 1.2}, source="推断：起点")
    r = fn("o", "L1", [_case(test, t, ref=ref)], th)
    assert r.status == FAIL and r.metrics["r_ratio"] > 1.2
    assert fn("o", "L1", [_case(ref, t, ref=ref)], th).status == PASS
    assert fn("o", "L1", [], th).status == ERROR
    assert fn("o", "L1", [_case(ref, t, bitref=ref.astype(np.float32))], th).status == ERROR
    assert fn("o", "L1", [_case(ref, t)], None).status == PASS      # 无门槛 = 只诊断



def test_op_replay_expect_topk():
    fn = REGISTRY["op_replay"]
    a = np.array([[0, 1], [2, 3], [4, 5]])
    b = np.array([[1, 0], [2, 4], [4, 5]])
    r = fn("o", "L1", [{"name": "tie", "out": a, "expect": b}], Threshold("exact"))
    assert r.status == FAIL and r.details[0]["rows_differ"] == 1 and r.details[0]["rows_differ_first"] == [1]
    assert fn("o", "L1", [{"name": "tie", "out": a, "expect": a}], Threshold("exact")).status == PASS



def test_op_replay_drill_expect_fail():
    """演练：故意多舍一次的实现，R-ratio 门槛必须报警；报警了检查才算 pass。干净实现当演练用会判 fail。"""
    fn = REGISTRY["op_replay"]
    t = rng.standard_normal(4096)
    ref = O.round_from_f64(t, BF16)
    bad = O.round_from_f64(O.round_from_f64(t * 3, BF16).astype(np.float64) / 3, BF16)
    th = Threshold("numeric", {"r_ratio": 1.2}, source="推断：起点")
    r = fn("o", "L1", [_case(bad, t, ref=ref)], th, params={"expect": "fail"})
    assert r.status == PASS and r.metrics["drill_judged"] == FAIL
    r = fn("o", "L1", [_case(ref, t, ref=ref)], th, params={"expect": "fail"})
    assert r.status == FAIL and r.metrics["drill_judged"] == PASS
