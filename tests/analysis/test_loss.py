"""analysis/loss.py：逐步相对差与分段统计（HTML 报告 loss 一节的数）。"""

import numpy as np
import pytest

from janus.analysis import loss


def test_rel_diff_uses_common_steps_and_abs_denominator():
    st, d = loss.rel_diff({1: 2.0, 2: 3.0, 5: 1.0}, {1: 1.0, 2: -3.0, 3: 9.0})
    assert st == [1, 2] and d.tolist() == [1.0, 2.0]


def test_segments_split_by_edges():
    ref = {s: 1.0 for s in range(1, 31)}
    test = {s: 1.0 + (1e-3 if s >= 11 else 0.0) + (5e-3 if s == 25 else 0.0) for s in range(1, 31)}
    seg = loss.segments(test, ref, [1, 11, 31])
    assert [s["window"] for s in seg] == ["1:11", "11:31"]
    assert seg[0]["signed_mean"] == 0.0 and seg[0]["max_step"] == 0.0
    assert seg[1]["steps"] == 20 and seg[1]["max_step_at"] == 25
    assert seg[1]["max_step"] == pytest.approx(6e-3) and seg[1]["signed_mean"] == pytest.approx(1e-3 + 5e-3 / 20)
    assert np.isfinite(seg[1]["se"])
    assert loss.segments(test, ref, [100, 200]) == []  # 没有步落在段里：不出这一段
