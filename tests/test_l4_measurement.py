"""L4 的两处测量口径：Run.series 的派生列 param_norm_from_groups，与序列相对差的 denominator=mean。"""

import json
import math

from janus.store import Run


def _run(tmp_path, rows):
    """只用 metrics.jsonl 的 Run（绕过 open 的 meta 校验，series 只读这一份文件）。"""
    d = tmp_path / "r"
    d.mkdir()
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    r = Run.__new__(Run)
    r.path = d
    return r


def test_param_norm_from_groups(tmp_path):
    rows = [
        {"step": 1, "param_norm": 5.0000001, "group_sq.a": 9.0, "group_sq.b.1": 16.0},
        {"step": 2, "param_norm": 5.0, "group_sq.a": 1.0, "group_sq.b.1": 3.0},
    ]
    r = _run(tmp_path, rows)
    s = r.series("param_norm_from_groups")
    assert s == {1: 5.0, 2: math.sqrt(4.0)}
    assert r.series("param_norm") == {1: 5.0000001, 2: 5.0}


def test_recorded_column_wins(tmp_path):
    r = _run(tmp_path, [{"step": 1, "param_norm_from_groups": 7.0, "group_sq.a": 1.0}])
    assert r.series("param_norm_from_groups") == {1: 7.0}


def test_mean_denominator_is_antisymmetric():
    """denominator=mean：交换两边只差一个符号；以 golden 为分母时逐步涨落带来正偏置。"""
    from janus.metrics import series_stats

    a = {s: 1.0 + 0.1 * (-1) ** s for s in range(1, 101)}
    b = {s: 1.0 - 0.1 * (-1) ** s for s in range(1, 101)}
    g_ab = series_stats(a, b, (1, 101), "golden").signed_mean
    g_ba = series_stats(b, a, (1, 101), "golden").signed_mean
    assert g_ab > 0 and g_ba > 0  # 两个方向都偏正：Jensen 偏置
    m_ab = series_stats(a, b, (1, 101), "mean").signed_mean
    m_ba = series_stats(b, a, (1, 101), "mean").signed_mean
    assert abs(m_ab + m_ba) < 1e-15 and abs(m_ab) < 1e-12
