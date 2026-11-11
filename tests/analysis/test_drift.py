"""analysis/drift.py：多步漂移拆解。用合成的逐步原值验证埋进去的漂移能被拆到对的那一组 / 那一桶。"""

import json

import numpy as np
import pytest

from janus.analysis import drift


def _series(n=1000, drift_rate=0.0, offset=0.0, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    out = {}
    for s in range(1, n + 1):
        r = {
            "step": s,
            "lm_loss": (2.0 + np.exp(-s / 300)) * (1 + offset * (s > 100) + 1e-4 * rng.standard_normal() * (s > 1)),
            "param_norm": 3000.0 + 0.1 * s + drift_rate * s,
            "grad_norm": 1.0 + 0.01 * rng.standard_normal(),
        }
        r.update((extra or {}).get(s, {}))
        out[s] = r
    return out


def test_pairs_separates_noise_from_offset():
    S = {f"g{i}": _series(seed=i) for i in range(3)} | {"t": _series(offset=-1.5e-4, drift_rate=2e-5, seed=9)}
    out = drift.pairs(S, {"g": ["g0", "g1", "g2"], "t": ["t"]}, ["噪声:g-g", "被测:t-g"])
    noise, test = out["comparisons"]["噪声"], out["comparisons"]["被测"]
    assert noise["pairs"] == ["g0-g1", "g0-g2", "g1-g2"] and test["pairs"] == ["t-g0", "t-g1", "t-g2"]
    assert max(abs(x) for x in noise["tail_rel"]["lm_loss"]["per_pair"]) < 2e-5
    assert all(-1.7e-4 < x < -1.3e-4 for x in test["tail_rel"]["lm_loss"]["per_pair"])
    assert test["param_norm_abs"]["min"] == pytest.approx(2e-5 * 1000)  # 第 1000 步多涨 0.02
    assert noise["param_norm_abs"]["max"] == 0.0


def test_trend_summaries():
    G = {f"g{i}": _series(seed=i) for i in range(2)}
    res = drift.trend(_series(offset=-2e-4, seed=7), G, steps=1000)
    t = res["lm_loss"]["test"]["g0"]
    assert t["step1"] == 0.0 and -2.1e-4 < t["signed_mean"] < -1.5e-4
    assert set(t["env95"]) == {"1-10", "11-100", "101-300", "301-600", "601-1000"}
    assert t["cum"][1000] < t["cum"][100] < 0  # 累积均值一直往负的一边走
    assert abs(res["lm_loss"]["golden_pairs"]["g0~g1"]["signed_mean"]) < 3e-5
    assert res["router_bias"] == {}  # 没有 router.bias_* 字段


def test_groups_attributes_drift_to_the_right_group():
    def ex(rate):
        return {
            s: {
                "group_sq.attn": 1000.0 + s,
                "group_sq.mlp": 2000.0 + s + rate * s,
                "expert_sq.1": [1.0 + rate * s * (i < 2) for i in range(16)],
            }
            for s in range(1, 301)
        }

    T, R, P = _series(300, extra=ex(0.5)), _series(300, extra=ex(0.0)), _series(300, extra=ex(0.0), seed=3)
    out = drift.groups(T, R, [P], [100, 300, 999])
    assert list(out) == [100, 300]  # 没有的步跳过
    row = out[300]
    assert list(row["groups"]) == ["group_sq.mlp", "group_sq.attn"]  # 按贡献大小排
    assert row["groups"]["group_sq.mlp"]["contrib"] == pytest.approx(0.5 * 300 / (2 * R[300]["param_norm"]))
    assert row["groups"]["group_sq.attn"]["contrib"] == 0.0 and row["groups"]["group_sq.mlp"]["pairs"] == [0.0]
    assert row["experts"]["expert_sq.1"]["top8_share"] == 1.0  # 多涨的全在前两个专家


def test_rows_buckets_by_frequency():
    V = 12
    cnt = np.array([0, 0, 5, 5, 50, 50, 500, 500, 5000, 5000, 50000, 50000])

    def ex(d):
        return {
            100: {
                "output_row_sq": list(1.0 + d * (cnt > 1000)),
                "embedding_row_sq": [1.0] * V,
                "output_row_u": [0.1] * V,
            }
        }

    out = drift.rows(_series(100, extra=ex(0.2)), _series(100, extra=ex(0.0)), None, cnt, [100, 200])
    assert list(out) == [100]
    by = {r["bucket"]: r for r in out[100]["output_row_sq"]}
    assert by["没出现过"]["sum_diff"] == 0.0 and by[">1e4"]["sum_diff"] == pytest.approx(0.4)
    assert by[">1e4"]["pair_sum_diff"] is None and by["1001–1e4"]["rel"] == pytest.approx(0.2)


@pytest.mark.parametrize("sub", sorted(drift.SUB))
def test_subcommand_help(sub, capsys):
    assert drift.main([sub, "--help"]) == 0
    assert capsys.readouterr().out.strip()


def test_dispatch_errors(capsys):
    assert drift.main([]) == 2 and drift.main(["nope"]) == 2
    assert drift.main(["--help"]) == 0


def test_trend_cli_reads_run_dirs(tmp_path, capsys):
    for name, kw in [("g0", {"seed": 0}), ("g1", {"seed": 1}), ("t", {"seed": 2, "offset": 1e-4})]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.jsonl").write_text(
            "\n".join(json.dumps(r) for r in _series(200, **kw).values()) + "\n"
        )
    out = tmp_path / "o.json"
    assert (
        drift.main(
            [
                "trend",
                "--test",
                str(tmp_path / "t"),
                "--golden",
                str(tmp_path / "g0"),
                str(tmp_path / "g1"),
                "--out",
                str(out),
                "--steps",
                "200",
            ]
        )
        == 0
    )
    assert set(json.loads(out.read_text())["lm_loss"]["test"]) == {"g0", "g1"}  # 目录名当运行名
