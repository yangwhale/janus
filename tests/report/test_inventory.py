"""janus/report/inventory.py：检查清单只从 results.json 取数；有非 ✅ 格子必须写原因；引用不存在的结果直接失败。"""

import json

import pytest

from janus.report import inventory as INV

EXACT = {
    "id": "L0.weights.loaded",
    "type": "bitwise_equal",
    "level": "L0",
    "status": "pass",
    "threshold": {"kind": "exact"},
    "metrics": {"tensors": 50, "differ": 0},
}
RRATIO = {
    "id": "L1.ops.op.qkv",
    "type": "op_replay",
    "level": "L1",
    "status": "fail",
    "threshold": {"kind": "numeric", "values": {"r_ratio_own_inputs": 1.2}},
    "metrics": {"r_ratio_own_inputs": 1.3, "cases": 4},
}
SERIES = {
    "id": "L4.loss",
    "type": "multi_golden",
    "level": "L4",
    "status": "pass",
    "threshold": {"kind": "calibrated", "noise": {"refs": 45, "upper": {"signed_mean": 1e-4, "max_step": 1e-3}}},
    "metrics": {"signed_mean_max": 5e-5, "max_step_max": 5e-4, "window": "1:1001"},
}


def _model(tmp_path, files):
    for adir, name, checks in files:
        d = tmp_path / "alignments" / adir / "results"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.results.json").write_text(
            json.dumps({"schema_version": 1, "summary": {"pass": 1}, "checks": checks})
        )
        (d / f"{name}.report.md").write_text("r\n")
    return INV.Sources(tmp_path)


def _spec(why=None, src=None):
    row = {
        "name": "qkv",
        "what": "投影",
        "objects": "`layers.{0..3}.attn.qkv.out`",
        "src": src or {"a": "p:L1.ops.op.qkv"},
    }
    if why:
        row["why"] = why
    return {
        "columns": {"a": "平台 A"},
        "evidence_sources": [{"match": "p", "source": "生产程序"}],
        "header": ["# 清单"],
        "levels": [
            {
                "id": "L1",
                "title": "L1 · 单算子",
                "purpose": "单算子",
                "lead": ["L1 共 {rows} 项、{checks} 条、张量级 {tensors} 次，覆盖 {modules} 个算子"],
                "modules": {
                    "columns": {"a": "平台 A", "up": "更高一级"},
                    "count": ["a"],
                    "rows": [
                        {"name": "qkv", "does": "投影", "checks": {"a": ["p:L1.ops.op.qkv"], "up": "L2 观测点"}},
                        {"name": "词嵌入", "does": "查表", "checks": {"up": "L3"}},
                    ],
                },
                "tables": [{"rows": [row]}],
            },
            {
                "id": "L4",
                "title": "L4 · 多步",
                "purpose": "多步",
                "lead": [],
                "tables": [{"rows": [{"name": "loss", "what": "逐步 loss", "src": {"a": "p:L4.loss"}}]}],
            },
        ],
    }


def test_unit_count():
    assert INV.unit_count(EXACT) == (50, "个张量")
    assert INV.unit_count(RRATIO) == (4, "个用例")
    assert INV.unit_count(SERIES) == (1000, "步")
    assert INV.unit_count({"coverage": {"compared": 367}}) == (367, "个名字")
    assert INV.unit_count({"metrics": {"rel_l2": 0.1}}) is None
    assert INV.tensor_total({"个张量": 50, "个用例": 4, "步": 1000}) == 54


def test_render_counts_links_and_reasons(tmp_path):
    src = _model(tmp_path, [("01-a", "p", [EXACT, RRATIO, SERIES])])
    md = INV.render(_spec(why="一个 token 路由翻转（台账 #5）。"), src, tmp_path)
    assert "L1 共 1 项、1 条、张量级 4 次，覆盖 1 个算子" in md  # 词嵌入一行没有平台检查，不算覆盖
    assert "| 1 | qkv | 投影 | [`L1.ops.op.qkv`](alignments/01-a/results/p.report.md) ❌ | L2 观测点 |" in md
    assert "(alignments/01-a/results/p.report.md)" in md
    assert "比较单元：4 个用例" in md
    assert "**qkv**（平台 A ❌）：一个 token 路由翻转（台账 #5）。" in md
    assert "| 合计 |  | 2 | 2 | 4 |" in md  # 2 项、2 条检查、张量级 4 次（L4 的步数不算）


def test_non_pass_without_reason_fails(tmp_path):
    src = _model(tmp_path, [("01-a", "p", [RRATIO])])
    spec = _spec()
    spec["levels"] = spec["levels"][:1]
    with pytest.raises(SystemExit, match="要在 checks.yaml 写 why"):
        INV.render(spec, src, tmp_path)


def test_missing_result_fails(tmp_path):
    src = _model(tmp_path, [("01-a", "p", [RRATIO])])
    spec = _spec(why="x", src={"a": "p:L1.nope"})
    with pytest.raises(SystemExit, match="引用的结果不存在"):
        INV.render(spec, src, tmp_path)


def test_claimed_and_unlisted(tmp_path):
    src = _model(tmp_path, [("01-a", "p", [EXACT, RRATIO, SERIES]), ("01-a", "q", [EXACT])])
    spec = _spec(why="x")
    got = INV.claimed(spec, src)
    assert ("p", "L1.ops.op.qkv") in got["a"] and ("p", "L4.loss") in got["a"]
    assert ("p", "L0.weights.loaded") not in got["a"]
    with pytest.raises(SystemExit, match="unlisted_reasons"):  # 清单外的结果必须写原因
        INV.unlisted(spec, src, tmp_path, tmp_path, ["01-a"])
    spec["unlisted_reasons"] = {"q": "旧配置"}
    rows = INV.unlisted(spec, src, tmp_path, tmp_path, ["01-a"])
    assert any("`q`" in r and "旧配置" in r for r in rows) and not any("`p`" in r for r in rows)


def test_info_rows_counted_apart(tmp_path):
    src = _model(tmp_path, [("01-a", "p", [EXACT, RRATIO, SERIES])])
    spec = _spec(why="x")
    spec["levels"][1]["tables"][0]["rows"][0]["info"] = True  # L4 那一行标成只报
    md = INV.render(spec, src, tmp_path)
    assert "loss（只报）" in md
    l4 = next(line for line in md.splitlines() if line.startswith("| [L4]"))
    assert "只报 1" in l4 and "✅" not in l4  # 只报行不进 ✅ / 🟡 / ❌ 计数


def test_evidence_source_column(tmp_path):
    src = _model(tmp_path, [("01-a", "p", [EXACT, RRATIO, SERIES])])
    spec = _spec(why="x")
    md = INV.render(spec, src, tmp_path)
    assert "| 证据来源 |" in md and "| 生产程序 |" in md
    spec["evidence_sources"] = []
    with pytest.raises(SystemExit, match="evidence_sources"):  # 每一行都要有证据来源
        INV.render(spec, src, tmp_path)


def test_virtual_pending_wildcard_and_module_status(tmp_path):
    # 现算的检查（virtual）、带 * 的结果名、还没有结果的格子（todo / wip）、算子表的判定列与计数
    raw = tmp_path / "alignments" / "01-a" / "results" / "raw.json"
    _model(tmp_path, [("01-a", "s-1", [SERIES]), ("01-a", "s-2", [SERIES])])
    raw.write_text("{}")
    src = INV.Sources(tmp_path, virtual={"v": (raw, [RRATIO])})
    assert src.resolve("s-*") == "s-2" and src.resolve("nope-*") is None
    spec = _spec(why="x", src={"a": "v:L1.ops.op.qkv"})
    spec["evidence_sources"] = [{"match": "*", "source": "现算"}]
    mod = spec["levels"][0]["modules"]
    mod["status"] = "判定"
    mod["rows"][0]["checks"]["a"] = ["v:L1.ops.op.qkv"]
    mod["rows"][1]["checks"]["a"] = "尚未覆盖"
    spec["levels"][0]["lead"] = [
        "共 {modules_total} 个：过 {modules_pass}、待定或不过 {modules_open}、尚未覆盖 {modules_none}"
    ]
    spec["levels"][1]["tables"][0]["rows"].append(
        {"name": "重跑", "what": "待结果", "src": {"a": "todo: 等重跑"}},
    )
    spec["levels"][1]["tables"][0]["rows"][0]["src"] = {"a": "s-*:L4.loss"}
    md = INV.render(spec, src, tmp_path)
    assert "共 2 个：过 0、待定或不过 1、尚未覆盖 1" in md
    assert "[`L1.ops.op.qkv`](alignments/01-a/results/raw.json) ❌ | L2 观测点 | ❌ |" in md
    assert "| 尚未覆盖 | L3 | 尚未覆盖 |" in md
    assert "⬜ 未开始（等重跑）" in md
    assert "(alignments/01-a/results/s-2.report.md)" in md  # 带 * 的取最后一个
    assert ("s-2", "L4.loss") in INV.claimed(spec, src)["a"]


def test_unlisted_patterns_and_grouping(tmp_path):
    src = _model(
        tmp_path,
        [
            ("01-a", "p", [EXACT, RRATIO, SERIES]),
            ("01-a", "q1", [EXACT]),
            ("01-a", "q2", [EXACT]),
            ("01-a", "r", [EXACT]),
        ],
    )
    spec = _spec(why="x")
    spec["unlisted_reasons"] = {"q*": "旧配置", "r": "旧配置"}
    rows = INV.unlisted(spec, src, tmp_path, tmp_path, ["01-a"])
    assert sum("旧配置" in r for r in rows) == 3  # 通配匹配；不并行时一文件一行
    spec["unlisted_group"] = True
    rows = INV.unlisted(spec, src, tmp_path, tmp_path, ["01-a"])
    hit = [r for r in rows if "旧配置" in r]
    assert len(hit) == 1 and all(f"`{n}`" in hit[0] for n in ("q1", "q2", "r"))


def test_unit_override_and_same_as_above(tmp_path):
    src = _model(tmp_path, [("01-a", "p", [EXACT, RRATIO, SERIES])])
    spec = _spec(why="x")
    spec["levels"][0]["tables"][0]["rows"][0]["unit"] = "个参数张量"
    mod = spec["levels"][0]["modules"]
    mod["same_as_above"] = True
    mod["rows"].insert(1, {"name": "o", "does": "投影", "checks": {"a": ["p:L1.ops.op.qkv"]}})
    md = INV.render(spec, src, tmp_path)
    assert "比较单元：4 个参数张量" in md
    assert "| 2 | o | 投影 | 同 #1 | — |" in md
