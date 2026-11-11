"""analysis/serial.py：整网串跑找台阶——按执行顺序排观测点，找第一个不再逐位相同的。"""

import numpy as np

from janus.analysis import serial


class FakeRun:
    def __init__(self, tensors):
        self.t = tensors

    def names(self, step, pattern=None):
        return list(self.t)

    def tensor(self, step, name):
        return self.t[name]


def test_execution_order():
    names = ["layers.1.out", "logits", "layers.0.attn.qkv.out", "embed.out", "layers.0.out", "unknown.x"]
    assert serial.execution_order(names) == [
        "embed.out",
        "layers.0.attn.qkv.out",
        "layers.0.out",
        "layers.1.out",
        "logits",
    ]


def test_table_marks_first_step_change():
    rng = np.random.default_rng(0)
    ref = {
        n: rng.standard_normal((4, 8)).astype(np.float32)
        for n in ["embed.out", "layers.0.attn.qkv.out", "layers.0.out", "logits"]
    }
    test = {n: v.copy() for n, v in ref.items()}
    test["layers.0.out"][0, 0] += 1e-3  # 从这里开始不逐位
    test["logits"] = test["logits"] * np.float32(1.001)
    rows = serial.table(FakeRun(test), FakeRun(ref))
    by = {r["name"]: r for r in rows}
    assert [r["name"] for r in rows] == ["embed.out", "layers.0.attn.qkv.out", "layers.0.out", "logits"]
    assert by["embed.out"]["same_frac"] == 1.0 and by["layers.0.attn.qkv.out"]["rel_l2"] == 0.0
    assert by["layers.0.out"]["same_frac"] == 31 / 32 and by["layers.0.out"].get("note") == "第一个不再逐位相同的点"
    assert "note" not in by["logits"]  # 只标第一个
    assert "第一个不再逐位相同的点" in serial.table_html(rows)


def test_table_reports_dtype_mismatch():
    ref = {"embed.out": np.zeros((2, 2), np.float32)}
    test = {"embed.out": np.zeros((2, 2), np.float64)}
    rows = serial.table(FakeRun(test), FakeRun(ref))
    assert rows[0]["same_frac"] == 0.0 and "dtype" in rows[0]["note"]
