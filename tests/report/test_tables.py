"""janus/report/tables.py：逐级大表只从 results.json 取数，标记之间可重复生成。"""
import json

from janus.report import tables as AL


def _write(tmp, name, checks):
    d = tmp / "results"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.results.json").write_text(json.dumps({"schema_version": 1, "checks": checks}))
    (d / f"{name}.report.md").write_text("r\n")


SERIES = {"id": "L4.loss", "type": "multi_golden", "level": "L4", "status": "fail",
          "threshold": {"kind": "calibrated", "noise": {"refs": 3, "upper": {"signed_mean": 1e-5, "mare": 1e-5, "max_step": 1e-4}}},
          "metrics": {"d0_max": 2e-6, "signed_mean_max": 1.5e-5, "mare_max": 3e-5, "max_step_max": 5e-5}}
TENSOR = {"id": "L2.layer0.bwd", "type": "tensor_tolerance", "level": "L2", "status": "inconclusive",
          "threshold": {"kind": "calibrated", "noise": {"refs": 28, "upper": {"rel_l2": 1e-3, "cosine": 1e-6}}},
          "metrics": {"rel_l2_max": 1.5e-3, "cosine_min": 1 - 5e-7}}
EXACT = {"id": "L0.weights.loaded", "type": "bitwise_equal", "level": "L0", "status": "pass",
         "threshold": {"kind": "exact"}, "metrics": {"tensors": 50, "differ": 0}}


def test_rows_and_verdicts(tmp_path):
    _write(tmp_path, "p", [SERIES, TENSOR, EXACT])
    res = AL.Results(tmp_path)
    items = [{"name": "loss", "src": "p:L4.loss", "level": "L4"}, {"name": "bwd", "src": "p:L2.layer*.bwd", "level": "L2"},
             {"name": "w", "src": "p:L0.weights.loaded", "level": "L0"}, {"name": "未跑", "src": "todo", "level": "L1"},
             {"name": "缺文件", "src": "q:x", "level": "L3"}]
    rows = AL.build_rows(items, res, {"L4.loss": ["figures/a.png"]})
    assert [r.level for r in rows] == sorted(r.level for r in rows)          # 按级排序
    loss = [r for r in rows if r.check_id == "L4.loss"]
    by = {r.metric: r for r in loss}
    assert by["首步差"].status == "na" and by["首步差"].value == "+2.00e-06"
    assert by["带符号均值"].ratio == "1.50×N" and by["带符号均值"].status == "inconclusive"
    assert by["MARE"].status == "fail" and by["最大单步"].status == "pass"
    assert "[图·a](figures/a.png)" in loss[0].evidence and loss[0].evidence[0] == "[p](results/p.report.md)"
    bwd = next(r for r in rows if r.check_id == "L2.layer0.bwd")
    assert bwd.metric == "rel_L2" and bwd.ratio == "1.50×N" and bwd.status == "inconclusive"   # 最差的是 rel_l2（1.5 > 0.5）
    w = next(r for r in rows if r.check_id == "L0.weights.loaded")
    assert w.status == "pass" and "differ=0" in w.value
    assert {r.item: r.status for r in rows if not r.check_id} == {"未跑": "todo", "缺文件": "todo"}


def test_wip_with_evidence(tmp_path):
    (tmp_path / "results").mkdir()
    res = AL.Results(tmp_path)
    rows = AL.build_rows([{"name": "回放", "src": "wip: 证据在 results/x.md", "level": "L1"}], res)
    assert rows[0].status == "wip" and rows[0].evidence == ["证据在 results/x.md"]
    assert "🟡" in rows[0].md() or AL.SYM["wip"] in rows[0].md()


def test_write_section_idempotent(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# 标题\n\n导言\n\n## 结论\n\n手写结论\n")
    AL.write_section(readme, "表 1\n")
    AL.write_section(readme, "表 2\n")
    s = readme.read_text()
    assert s.count(AL.BEGIN) == 1 and "表 2" in s and "表 1" not in s
    assert s.index(AL.BEGIN) < s.index("## 结论") and "手写结论" in s


def test_expect_fail_flips_verdict(tmp_path):
    """故障演练（expect: fail）：检查报警了才算通过；没报警就是演练失败。"""
    fired = {"id": "L4.health.grad_norm", "type": "health_signals", "level": "L4", "status": "fail",
             "threshold": {"kind": "numeric", "values": {"spike_ratio": 3}}, "metrics": {"fields": 1, "problems": 1},
             "message": "grad_norm@5=非有限"}
    quiet = dict(fired, id="L4.health.loss", status="pass", message="健康")
    _write(tmp_path, "d", [fired, quiet])
    res = AL.Results(tmp_path)
    rows = AL.build_rows([{"name": "梯度 NaN", "src": "d:L4.health.grad_norm", "level": "L4", "expect": "fail"},
                          {"name": "loss 没报", "src": "d:L4.health.loss", "level": "L4", "expect": "fail"}], res)
    by = {r.item: r.status for r in rows}
    assert by == {"梯度 NaN": "pass", "loss 没报": "fail"}
