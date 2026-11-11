"""analysis/curves.py：逐步曲线与按设置分组互比。用合成的逐步原值验证埋的偏移能看出来。"""

import json

import numpy as np
import pytest
import yaml
from helpers import ROOT

from janus.analysis import curves, fault_matrix


def _write_series(path, n=300, offset=0.0, start=100, seed=0, noise=1e-4, param_drift=0.0):
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        for s in range(1, n + 1):
            loss = 3.0 * np.exp(-s / 200) + 1.0
            if s > 1:
                loss *= 1 + noise * rng.standard_normal()
            if s >= start:
                loss *= 1 + offset
            pn = 100.0 + 0.001 * s + param_drift * max(0, s - start)
            f.write(json.dumps({"step": s, "lm_loss": loss, "grad_norm": 1.0 + 0.01 * rng.standard_normal(),
                                "param_norm": pn, "lr": 1e-4}) + "\n")


@pytest.fixture
def series(tmp_path):
    d = tmp_path / "series"
    d.mkdir()
    for i, name in enumerate(["a1", "a2", "a3"]):
        _write_series(d / f"{name}.metrics.jsonl", seed=i + 1)
    for i, name in enumerate(["b1", "b2"]):  # 另一种设置：从第 100 步起整体高 1e-3
        _write_series(d / f"{name}.metrics.jsonl", seed=i + 10, offset=1e-3)
    _write_series(d / "t.metrics.jsonl", seed=99, offset=-5e-4, param_drift=1e-4)  # 被测：偏低、参数范数多涨
    return d


def _spec(tmp_path, **over):
    d = {"groups": {"A": ["a1", "a2", "a3"], "B": ["b1", "b2"]}, "noise_groups": ["A", "B"], "contrast": ["A", "B"],
         "fields": [["lm_loss", "lm_loss"]], "window": 20,
         "settings": {"a1": {"s": "A"}, "a2": {"s": "A"}, "a3": {"s": "A"}, "b1": {"s": "B"}, "b2": {"s": "B"}}, **over}
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True))
    return curves.GroupSpec.from_yaml(p)


def _fault_results(root, statuses):
    for f, st in statuses.items():
        (root / f).mkdir(parents=True)
        (root / f / "results.json").write_text(json.dumps({"checks": [{"id": i, "status": s} for i, s in st.items()]}))


def test_rel_and_rolling():
    st, d = curves.rel({1: 2.0, 2: 4.0, 3: 1.0}, {1: 1.0, 2: -4.0, 4: 1.0})
    assert st.tolist() == [1, 2] and d.tolist() == [1.0, 2.0]       # 分母取绝对值
    assert curves.rolling(np.array([1.0, 2.0, 3.0, 4.0]), 2).tolist() == [1.5, 2.5, 3.5]
    assert curves.rolling(np.array([1.0]), 3).size == 0



def test_compare_writes_three_figures_per_field(series, tmp_path):
    made = curves.compare(("t", series / "t.metrics.jsonl"), [(n, series / f"{n}.metrics.jsonl") for n in ("a1", "a2")],
                          tmp_path / "out", fields=("lm_loss", "missing_field"), prefix="L4.t", steps=(1, 51))
    assert list(made) == ["lm_loss"]                               # 没有的字段跳过
    assert all(p.exists() and p.stat().st_size > 0 for p in made["lm_loss"])
    assert {p.name for p in made["lm_loss"]} == {"L4.t-lm_loss-overlay.png", "L4.t-lm_loss-pairdiff.png", "L4.t-lm_loss-mean.png"}



def test_groups_and_offsets_find_the_setting_offset(series, tmp_path):
    spec = _spec(tmp_path)
    summary = curves.groups(series, tmp_path / "g", spec)
    assert summary["lm_loss"]["pairs"] == 3 + 1                   # 组内两两：A 3 对、B 1 对
    assert len(summary["lm_loss"]["cross_pairs"]) == 3 * 2
    res = curves.offsets(series, tmp_path / "g", spec)
    assert res["window"] == "2:301"
    cross = next(r for r in res["rows"] if r["a"] != r["b"])
    same = [r for r in res["rows"] if r["a"] == r["b"]]
    # A 对 B：B 从第 100 步起高 1e-3，第 2–300 步平均约 −(199/299)·1e-3；组内没有偏移
    assert all(-8e-4 < m < -5e-4 for m in cross["signed_mean"])
    assert all(abs(m) < 1e-4 for r in same for m in r["signed_mean"])
    assert min(abs(z) for z in cross["z"]) > 5



def test_group_spec_rejects_bad_config(tmp_path):
    with pytest.raises(ValueError, match="不认识的键"):
        _spec(tmp_path, typo=1)
    with pytest.raises(ValueError, match="没有在 groups 里定义"):
        _spec(tmp_path, highlight=["C"])



def test_model_configs_load():
    """混元 3 的两份配置能被读入（键名、组名都对）。"""

    root = ROOT / "models/hunyuan3/analysis"
    spec = curves.GroupSpec.from_yaml(root / "phase1-curves.yaml")
    assert set(spec.settings) == {n for g in spec.groups.values() for n in g}
    assert len(fault_matrix.load_spec(root / "phase1-faults.yaml")) == 9
