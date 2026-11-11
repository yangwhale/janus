"""analysis/fault_matrix.py：故障矩阵标出「该报没报」；期望 YAML 的校验。"""

import json

import numpy as np
import pytest
import yaml

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


def test_fault_matrix_marks_missed_expectations(tmp_path, capsys):
    spec = tmp_path / "faults.yaml"
    spec.write_text(yaml.safe_dump({"faults": {"f-a": {"what": "埋 A", "expect": ["L0.x"]},
                                               "f-b": {"what": "埋 B", "expect": ["L2.y"]}}}, allow_unicode=True))
    _fault_results(tmp_path / "r", {"f-a": {"L0.x": "fail", "L2.y": "pass"}, "f-b": {"L0.x": "fail", "L2.y": "pass"}})
    faults = fault_matrix.load_spec(spec)
    results = {f: json.loads((tmp_path / "r" / f / "results.json").read_text()) for f in faults}
    assert fault_matrix.missed(results, faults) == {"f-b": ["L2.y"]}       # f-b 期望 L2.y 报警却没报
    md = fault_matrix.render(results, faults)
    assert "| `f-b` | 埋 B | 1 / 2 | L2.y |" in md and "| `L2.y` | · | ✗ |" in md
    assert fault_matrix.main([str(tmp_path / "r"), "--spec", str(spec)]) == 0
    assert fault_matrix.main([str(tmp_path / "r"), "--spec", str(spec), "--strict"]) == 1
    assert fault_matrix.main([str(tmp_path / "empty"), "--spec", str(spec)]) == 2



def test_fault_spec_validation(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("faults: {f-a: {expect: [x]}}\n")
    with pytest.raises(ValueError, match="缺 what"):
        fault_matrix.load_spec(p)
