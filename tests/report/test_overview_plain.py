"""janus/report/overview.py 的简版总览（detail_stem / plain_legend）：每组一张「检查项｜判定符号｜一行说明」的表。"""

import pytest

from janus.report.overview import Overview


def _ov(tmp_path):
    return Overview(
        tmp_path, "T", [], [""], "", out_dir=tmp_path, checks_md=False, detail_stem="detail", plain_legend=["图例一行"]
    )


def test_plain_table_uses_say_and_column_say(tmp_path):
    spec = {"column_say": {"mini": "TPU 对 GPU"}}
    it_ok = {"name": "jargon-a", "say": "人话 A"}
    it_flag = {"name": "jargon-c", "say": "人话 C", "mini_say": "判法事后变更"}
    it_bad = {"name": "jargon-b", "say": "人话 B", "mini_say": "差在哪一句"}
    plain = [
        (
            "第一组",
            [
                (it_ok, {"mini": {"status": "pass"}}),
                (it_bad, {"mini": {"status": "fail"}}),
                (it_flag, {"mini": {"status": "pass"}}),
            ],
            ["![图](x.png)"],
        )
    ]
    _ov(tmp_path).write_plain(spec, ["mini"], plain, {"mini": {"pass": 1, "fail": 1}})
    md = (tmp_path / "dashboard.md").read_text()
    assert md.startswith("# T · 对齐总览\n\n图例一行\n")
    assert "| 检查项 | TPU 对 GPU | 说明 |" in md
    assert "| 人话 A | ✅ |  |" in md
    assert "| 人话 B | ❌ | 差在哪一句 |" in md
    assert "| 人话 C | ✅ | 判法事后变更 |" in md
    assert "\n![图](x.png)\n" in md
    assert "jargon" not in md
    assert "| TPU 对 GPU | 1 | 0 | 1 | 0 |" in md


def test_plain_requires_say_for_non_pass(tmp_path):
    plain = [("组", [({"name": "x"}, {"mini": {"status": "inconclusive"}})], [])]
    with pytest.raises(SystemExit, match="mini_say"):
        _ov(tmp_path).write_plain({}, ["mini"], plain, {"mini": {}})
