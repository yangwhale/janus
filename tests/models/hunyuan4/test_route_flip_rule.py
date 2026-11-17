"""路由翻转定责（analysis/route_flip_rule.py，PROTOCOL 2026-11-12 事前登记）与两个判档工具里的接法。"""

import importlib.util
import sys
from pathlib import Path

import pytest

_AN = Path(__file__).resolve().parents[3] / "models" / "hunyuan4" / "analysis"
sys.path.insert(0, str(_AN))

from route_flip_rule import route_flip_check  # noqa: E402


def test_attributed_when_injected_reference_explains_gap():
    # QK-Clip 第 4 层的读数：TPU 5.55e-3、灌 TPU 选择后 2.88e-3、GPU 中位数 2.97e-3
    r = route_flip_check(5.55e-3, 2.88e-3, 2.97e-3)
    assert r["attributed"] and r["R"] > 1.2 and r["R_flip"] <= 1.2 and r["closer"]


def test_not_attributed_when_r_flip_still_high():
    r = route_flip_check(5.0e-3, 4.0e-3, 2.0e-3)  # R 2.5，R_flip 2.0
    assert not r["attributed"] and "R_flip > 1.2" in r["reason"]


def test_not_attributed_when_injection_moves_reference_away():
    r = route_flip_check(3.0e-3, 3.2e-3, 2.9e-3)  # R_flip 1.10 ≤ 1.2，但灌后更远
    assert not r["attributed"] and not r["closer"]


def test_not_applicable_when_already_within_limit():
    r = route_flip_check(2.0e-3, 1.0e-3, 2.0e-3)
    assert not r["attributed"] and r["R"] <= 1.2


def test_zero_gpu_distance_is_not_applicable():
    r = route_flip_check(1e-3, 1e-4, 0.0)
    assert not r["attributed"]


def test_boundary_inclusive():
    r = route_flip_check(3.0, 1.2, 1.0)  # R_flip 恰好 1.2：算过
    assert r["attributed"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _AN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    spec.loader.exec_module(mod)
    return mod


def _row(name, over, r, attributed=None):
    row = {
        "tensor": name,
        "gpu_noise_max": 1.0,
        "tpu_vs_gpu_min": over,
        "R_vs_ref": r,
        "gpu_vs_ref": 1e-2,
        "tpu_vs_ref": r * 1e-2,
    }
    if attributed is not None:
        row["route_flip_attributed"] = attributed
    return row


def test_step1_judge_excludes_route_flip_attributed():
    m = _load("step1_vs_fp64")
    rows = [_row("a", 2.5, 1.5, attributed=True), _row("b", 1.5, 1.3), _row("c", 3.0, 0.9)]
    j = m.judge(rows)
    assert j["route_flip_attributed"] == ["a"]
    assert j["unattributed_over_2N"] == [] and j["unattributed_1_2N"] == ["b"] and j["status"] == "inconclusive"
    assert j["n_over_2N"] == 2  # 计数不变，只改定责


def test_step1_judge_unchanged_without_field():
    m = _load("step1_vs_fp64")
    rows = [_row("a", 2.5, 1.5), _row("c", 3.0, 0.9)]
    j = m.judge(rows)
    assert j["status"] == "fail" and j["unattributed_over_2N"] == ["a"] and "route_flip_attributed" not in j


def test_dashboard_step1_judge_matches():
    spec = importlib.util.spec_from_file_location(
        "hy4_build", Path(__file__).resolve().parents[3] / "models" / "hunyuan4" / "dashboard" / "build.py"
    )
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    rows = [_row("a", 2.5, 1.5, attributed=True), _row("c", 3.0, 0.9)]
    st, un2, un1, zbad, over2, over1 = b._step1_judge(rows)
    assert st == "pass" and not un2 and len(over2) == 2
    rows[0]["route_flip_attributed"] = False
    assert b._step1_judge(rows)[0] == "fail"
