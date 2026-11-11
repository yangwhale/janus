"""report/html.py：四级门槛总表的判法（低一级的硬检查 fail / error 拦下本级）。整份 HTML 报告由 tests/test_run_full.py 端到端跑。"""

from janus.report import html


def _c(cid, level, status, hard=True):
    return {"id": cid, "level": level, "status": status, "hard": hard}


def test_gate_table_verdict_per_level():
    res = {
        "checks": [
            _c("L0.a", "L0", "pass"),
            _c("L2.a", "L2", "pass"),
            _c("L2.b", "L2", "inconclusive"),
            _c("L3.a", "L3", "fail"),
            _c("L3.soft", "L3", "fail", hard=False),
            _c("L4.a", "L4", "fail", hard=False),
        ]
    }
    t = html._gate_table(res)
    rows = {r.split("</td>")[0].split("<td>")[-1]: r for r in t.split("<tr>")[2:]}
    verdict = {
        k: ("fail" if ">fail</td>" in r else "inconclusive" if ">inconclusive</td>" in r else "pass")
        for k, r in rows.items()
    }
    assert list(verdict.values()) == ["pass", "inconclusive", "fail", "pass"]  # L0 / L2 / L3 / L4（软检查不拦）
    l3 = list(rows.values())[2]
    assert "L3.a" in l3 and "L3.soft" not in l3  # 只列拦下本级的硬检查
    assert len(rows) == 4  # L1 没有检查：不出行


def test_setup_fonts_never_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("JANUS_CJK_FONT", str(tmp_path / "missing.ttf"))
    html.setup_fonts()  # 找不到字体时退回默认字体，不抛
