"""旧导入路径的转发：给 DeprecationWarning，拿到的就是新模块本身（0.2 删除这些转发时一并删本文件）。"""

import importlib
import sys

import pytest

MOVED = {
    "janus.report_html": "janus.report.html",
    "janus.alignment": "janus.report.tables",
    "janus.figures": "janus.report.figures",
}


@pytest.mark.parametrize("old,new", sorted(MOVED.items()))
def test_old_module_names_forward(old, new):
    sys.modules.pop(old, None)
    with pytest.warns(DeprecationWarning, match=new):
        mod = importlib.import_module(old)
    assert mod is importlib.import_module(new)


def test_package_code_does_not_use_old_names():
    """包内与模型层代码都已改用新路径：旧名只留给并行分支与外部脚本过渡。"""
    from helpers import ROOT as root

    bad = []
    for p in [*root.joinpath("janus").rglob("*.py"), *root.joinpath("models").rglob("*.py")]:
        if p.stem in ("report_html", "alignment", "figures") and p.parent.name == "janus":
            continue  # 转发模块本身
        text = p.read_text(encoding="utf-8")
        for old in MOVED:
            short = old.split(".")[1]
            if f"import {old}" in text or f"from {old} " in text or f"from janus import {short}" in text:
                bad.append(f"{p.relative_to(root)}: {old}")
    assert not bad, "\n".join(bad)


def test_private_aliases_of_public_helpers():
    """跨模块用的辅助函数改成了公开名；旧的下划线名留作别名（模型层脚本还在用），指向同一个函数。"""
    from janus.checks import base, series, tensor

    assert series._stats is series.window_stats and series._judge is series.judge_series
    assert tensor._tensor_row is tensor.tensor_row and base._is_higher_better is base.is_higher_better
