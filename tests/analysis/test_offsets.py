"""analysis/offsets.py：loss 偏移归因。合成的被测从第 100 步起偏低、参数范数多涨，两样都要看得出。"""

import json

import numpy as np
import pytest
import yaml

from janus.analysis import curves, offsets


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


def test_offset_attribution_sees_offset_and_drift(series):
    load = offsets.load
    res = offsets.attribute(("t", load(series / "t.metrics.jsonl")),
                            {n: load(series / f"{n}.metrics.jsonl") for n in ("a1", "a2", "a3")},
                            segments=[(2, 100), (100, 301)], tail=(150, 301), norm_steps=[1, 100, 300], lag_window=(2, 301))
    seg = res["segments"]["signed_mean"]
    assert abs(seg["t-a1"][0]) < 1e-4 and -6e-4 < seg["t-a1"][1] < -4e-4   # 第 100 步前没偏，之后偏 −5e-4
    assert abs(seg["a1-a2"][1]) < 1e-4                                      # 噪声参照没有偏移
    assert res["tail"]["stats"]["t-a1"]["neg_frac"] > 0.9
    assert res["lag_check"]["0"] < res["lag_check"]["1"]                    # 同步号对比的差最小
    assert res["param_norm_abs"]["300"]["test_minus_golden"] == pytest.approx(1e-4 * 200)
    assert "lr" not in res["other_fields"]["rel_mean"] and "param_norm" in res["other_fields"]["rel_mean"]



def test_offsets_cli_writes_json(series, tmp_path, capsys):
    out = tmp_path / "o.json"
    rc = offsets.main(["--test", f"t={series / 't.metrics.jsonl'}", "--golden", f"a1={series / 'a1.metrics.jsonl'}",
                       "--golden", f"a2={series / 'a2.metrics.jsonl'}", "--json", str(out),
                       "--segments", "2:100", "100:301", "--tail", "150:301", "--lag-window", "2:301"])
    assert rc == 0 and json.loads(out.read_text())["golden"] == ["a1", "a2"]


def test_rel_skips_zero_reference():
    """参照为 0 的步（如某层专家负载最小值为 0）跳过，不除零。"""
    x = {1: {"f": 1.0}, 2: {"f": 2.0}, 3: {"f": 3.0}}
    y = {1: {"f": 2.0}, 2: {"f": 0.0}, 3: {"f": 3.0}}
    assert offsets.rel(x, y, "f", [1, 2, 3]).tolist() == [-0.5, 0.0]
