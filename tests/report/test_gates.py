"""janus/report/gates.py：两档门槛的说明文字与参考档现算（参考档只从 metrics 算，缺指标写「未记录」）。"""

from janus.report import gates as G
from janus.report import tables as AL


def _tensor(cid, rel_l2, status="fail", cos=0.99998, n=1e-3, refs=3):
    return {
        "id": cid,
        "type": "tensor_tolerance",
        "level": "L2",
        "status": status,
        "threshold": {"kind": "calibrated", "noise": {"refs": refs, "upper": {"rel_l2": n, "cosine": 1e-7}}},
        "metrics": {"rel_l2_max": rel_l2, "cosine_min": cos},
    }


def _loss(cid="L4.loss", d0=1e-5, sm=5e-5, status="inconclusive"):
    return {
        "id": cid,
        "type": "multi_golden",
        "level": "L4",
        "status": status,
        "threshold": {
            "kind": "calibrated",
            "noise": {"refs": 10, "upper": {"signed_mean": 4e-5, "mare": 1e-4, "max_step": 1e-3}},
        },
        "metrics": {"d0_max": d0, "signed_mean_max": sm, "mare_max": 5e-5, "max_step_max": 5e-4},
    }


RRATIO = {
    "id": "L2.layer4.fwd",
    "type": "op_replay",
    "level": "L2",
    "status": "fail",
    "threshold": {"kind": "numeric", "values": {"r_ratio": 1.2}, "source": "推断"},
    "metrics": {"rel_l2": 0.0146, "r_ratio": 1.2163},
    "message": "超线：['r_ratio']",
}


def test_module_gate_partial_and_fail():
    v = G.reference_verdict(_tensor("L2.layer5.bwd", 0.00688))
    assert v.status == "partial" and v.sym == "☑️"
    assert "rel_L2 < 0.01：0.00688（0.688×）" in v.text and "超差元素比例、均值差未记录" in v.text
    assert "只报" in v.text  # cos ≥ 0.99995 只有个别例子用过：只报不判
    v = G.reference_verdict(_tensor("L2.layer4.bwd", 0.0146))
    assert v.status == "fail" and "1.46×" in v.text and "不过" in v.text


def test_basis_mismatch_is_missing_not_judged():
    # op_replay 的 rel_L2 是对 FP64 真值，口径不同：不判，写「所需指标未记录」
    v = G.reference_verdict(RRATIO)
    assert v.status == "missing" and "所需指标未记录" in v.text and "FP64" in v.text


def test_loss_gate():
    v = G.reference_verdict(_loss())
    assert v.status == "pass" and "首步差 ≤ 2e-4" in v.text and "带符号均值 ≤ 2e-4：5.00e-05（0.25×）" in v.text
    assert "无 NaN / Inf" in v.text  # 有窗口统计 ⇒ 窗口内没有非有限步
    assert G.reference_verdict(_loss(sm=-3e-4)).status == "fail"  # 带符号均值按绝对值判
    assert G.reference_verdict(_loss(d0=5e-4)).status == "fail"
    nan = {**_loss(status="fail"), "metrics": {}, "message": "对 a 出现非有限值：第 [5] 步"}
    assert G.reference_verdict(nan).status == "missing"
    assert G.reference_verdict(_loss(), only="mare").status == "none"
    assert G.reference_verdict(_loss(), only="d0").status == "pass"


def test_param_norm_gate():
    # 项目负责人设定的参考档：带符号均值 ≤ 5e-6、最大单步 ≤ 1.25e-5（loss_series 的 metrics 不带 _max 后缀）
    pn = {
        "id": "L4.param_norm",
        "type": "loss_series",
        "level": "L4",
        "status": "fail",
        "threshold": {"kind": "calibrated", "noise": {"refs": 10, "upper": {"signed_mean": 1.24e-7}}},
        "metrics": {"signed_mean": 4.27e-7, "mare": 4.37e-7, "max_step": 9.23e-7},
    }
    v = G.reference_verdict(pn)
    assert v.status == "pass" and "带符号均值 ≤ 5e-6" in v.text and "最大单步 ≤ 1.25e-5" in v.text
    assert G.reference_verdict({**pn, "metrics": {"signed_mean": -2.6e-6, "max_step": 6.3e-6}}).status == "pass"
    assert G.reference_verdict({**pn, "metrics": {"signed_mean": -3.7e-5, "max_step": 4.9e-5}}).status == "fail"
    assert G.ref_threshold_text("L4.param_norm") == "带符号均值 ≤ 5e-6、最大单步 ≤ 1.25e-5"


def test_no_gate():
    assert G.reference_verdict({**_loss(), "id": "L4.grad_norm"}).status == "none"
    assert G.reference_verdict(_tensor("L3.step1.optim_state", 0.1)).text == "无对应门槛"


def test_strict_parts_human_text():
    m, s = G.strict_parts(RRATIO)
    assert m == "1.216（R 比）" and s == "≤ 1.2 → 1.014×，不过"
    m, s = G.strict_parts(_tensor("L2.layer5.bwd", 0.00688, n=1.32e-5, cos=0.9999))
    assert m == "0.00688（rel_L2），cos 0.999900" and s.startswith("N=1.32e-05（3 对）→ 521×N") and s.endswith("不过")
    assert "最差的是 1−cos" in s
    _, s = G.strict_parts(_tensor("L2.layer0.bwd", 1e-3, status="inconclusive", refs=2, n=1e-3))
    assert "参照不足" in s and "待定" in s
    m, s = G.strict_parts(
        {"id": "x", "status": "fail", "threshold": {"kind": "exact"}, "metrics": {"tensors": 41, "differ": 41}}
    )
    assert m == "41 / 41 个张量不同" and s == "逐位 / 逐项相同，不过"


def test_summarize_checks_cell():
    cs = [_tensor("L2.layer4.bwd", 0.0048, n=1.8e-3), _tensor("L2.layer5.bwd", 0.0069, n=1.3e-5)]
    st, ref, note = AL.summarize_checks(cs)
    assert st == "fail" and ref.status == "partial"
    assert note.startswith("2 条取最差 `L2.layer5.bwd`：实测 0.0069（rel_L2）")
    assert "｜严格档 N=1.30e-05" in note
    assert "｜参考档 rel_L2 < 0.01：0.0069" in note
    st, ref, note = AL.summarize_checks([RRATIO], expect="fail")
    assert st == "pass" and ref is None and "按期望报警" in note


def test_table_columns_and_notes(tmp_path):
    import json

    d = tmp_path / "results"
    d.mkdir()
    (d / "p.results.json").write_text(json.dumps({"checks": [_loss(), RRATIO]}))
    rows = AL.build_rows(
        [{"name": "loss", "src": "p:L4.loss", "level": "L4"}, {"name": "fwd", "src": "p:L2.layer4.fwd", "level": "L2"}],
        AL.Results(tmp_path),
    )
    fwd = next(r for r in rows if r.check_id == "L2.layer4.fwd")
    assert fwd.md().count("|") == 7 and "| ❌ | ❔ |" in fwd.md()
    assert fwd.note.startswith("实测 1.216（R 比）｜严格档 ≤ 1.2 → 1.014×，不过｜参考档 ")
    d0 = next(r for r in rows if r.metric == "首步差")
    assert d0.status == "na" and d0.ref.status == "pass"  # 严格档只报不判，参考档要判
    out = AL.render(rows)
    assert "| 级 | 检查项 | 严格档 | 参考档 | 说明 | 证据 |" in out and "参考档（有对应门槛的" in out


def test_module_gate_full_with_reference_stats():
    c = _tensor("L2.layer5.bwd", 0.00688)
    c["metrics"].update(oot_frac_max=0.00997, mean_diff_max=0.0031)
    v = G.reference_verdict(c)
    assert v.status == "pass" and "超差元素比例 < 1%：0.00997" in v.text and "未记录" not in v.text
    c["metrics"]["oot_frac_max"] = 0.012
    assert G.reference_verdict(c).status == "fail"


def test_op_replay_uses_vs_ref_only():
    c = {
        **RRATIO,
        "metrics": {
            **RRATIO["metrics"],
            "vs_ref_rel_l2": 0.004,
            "vs_ref_oot_frac": 0.002,
            "vs_ref_mean_diff": 0.003,
            "vs_ref_cosine": 0.99999,
            "vs_ref_cases": 11,
            "cases": 11,
        },
    }
    v = G.reference_verdict(c)
    assert v.status == "pass" and "rel_L2 < 0.01：0.004" in v.text  # 不拿对 FP64 的 rel_l2=0.0146 去判
    k = {**c, "id": "L1.ops.op.qkv", "metrics": {**c["metrics"], "vs_ref_cases": 4, "cases": 8}}
    v = G.reference_verdict(k)
    assert (
        v.status == "fail" and v.text.startswith("4 / 8 个用例有同输入的对照") and "归一化最大误差未记录" in v.text
    )  # 0.004 > 2^-10


def test_combine_status():
    assert G.combine_status(["pass", "missing"]) == "partial"
    assert G.combine_status(["missing", "missing"]) == "missing"
    assert G.combine_status(["pass", "fail", "missing"]) == "fail"
    assert G.combine_status(["none", "pass"]) == "pass"
    assert G.combine_status(["none"]) == "none"


def test_cell_measure_and_threshold_text():
    c = _tensor("L2.layer5.bwd", 0.00688, n=1.32e-5)
    cm = G.cell_measure(c)
    assert cm["metric"] == "rel_L2" and cm["value"] == "0.00688" and cm["how"] == "521×N"
    assert cm["thr"] == "≤ N=1.32e-05 过，> 2N 不过" and cm["short_thr"] == "N=1.32e-05"
    # rel_L2 自己在 N 以内、整条检查却不过（别的指标超了）→ 写决定判定的那个指标
    c2 = _tensor("x", 5e-4, n=1e-3, cos=0.999)
    assert G.cell_measure(c2)["metric"] == "1−cos"
    assert G.cell_measure(RRATIO) == {
        "metric": "R 比",
        "value": "1.216",
        "how": "1.014×",
        "thr": "R 比 ≤ 1.2",
        "short_thr": "≤ 1.2",
    }
    # 带两边绝对距离的 R 比；门槛里写了参照名字时，门槛栏带上它
    rr = {
        "id": "x",
        "status": "pass",
        "threshold": {"kind": "numeric", "values": {"r_ratio": 1.2}, "ref": "CPU FP64"},
        "metrics": {"r_ratio": 0.98, "rel_l2_gpu_vs_ref": 5.27e-3, "rel_l2_tpu_vs_ref": 5.17e-3},
    }
    assert G.cell_measure(rr)["thr"] == "对 CPU FP64 的 R 比 ≤ 1.2"
    rr["threshold"].pop("ref")
    assert G.cell_measure(rr)["thr"] == "R 比 ≤ 1.2"
    ex = {"id": "x", "status": "fail", "threshold": {"kind": "exact"}, "metrics": {"tensors": 41, "differ": 41}}
    assert G.cell_measure(ex)["value"] == "41 / 41 不同"
    det = {
        "id": "x",
        "status": "fail",
        "threshold": {"kind": "exact"},
        "metrics": {"fields": 6, "tensors_differ": 0},
        "message": "不可复现：['lm_loss@1']",
    }
    assert G.cell_measure(det)["value"].startswith("不可复现")  # 没过却数出 0 个不同：用 message
    assert G.ref_threshold_text("L4.loss") == "首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf"
    assert G.ref_threshold_text("L2.layer0.fwd") == "rel_L2 < 0.01、超差 < 1%、均值差 < 0.01"
    assert G.ref_threshold_text("L4.grad_norm") == "无"


def test_overview_cell_skips_drill_cases_as_worst():
    drill = {**RRATIO, "id": "L1.poison", "status": "pass", "metrics": {"r_ratio": 9e3, "drill_judged": "fail"}}
    ok = {**RRATIO, "status": "pass", "metrics": {"r_ratio": 1.05}}
    cell = AL.overview_cell([drill, ok])
    assert cell["check_id"] == RRATIO["id"] and cell["value"] == "1.05"
    cell = AL.overview_cell([RRATIO], expect="fail")
    assert cell["value"] == "按期望报警" and cell["ref"] is None


def test_optimizer_update_gate():
    ok = G.reference_verdict({**_tensor("L3.step1.updated", 0.01, cos=0.99995), "type": "tensor_tolerance"})
    assert ok.status == "pass" and "cos ≥ 0.9999" in ok.text
    bad = G.reference_verdict({**_tensor("L1.optim.adamw.replay", 0.03, cos=0.9998), "type": "optimizer_step"})
    assert bad.status == "fail"


def test_module_gate_excludes_routed_layers():
    """逐层参考档只用在没有路由的层：混元 3 的 MoE 层（第 1–3 层）写「不适用」并给原因，不判。"""
    v = G.reference_verdict(_tensor("L2.layer2.fwd", 0.02))
    assert v.status == "none" and "不适用" in v.text and "路由" in v.text
    assert G.ref_threshold_text("L2.layer2.fwd").startswith("不适用")
    assert G.reference_verdict(_tensor("L2.layer0.fwd", 0.004)).status in ("pass", "partial")  # 第 0 层照常判


def test_signed_value_shows_sign(monkeypatch):
    c = {
        "id": "L4.grad_norm",
        "type": "multi_golden",
        "status": "inconclusive",
        "threshold": {"kind": "calibrated", "noise": {"refs": 28, "upper": {"signed_mean": 1e-4}}},
        "metrics": {"signed_mean": -1.5e-4, "window": "1:1001"},
    }
    assert G.cell_measure(c)["value"].lstrip("−-") == G.cell_measure(c)["value"]  # 默认写绝对值
    monkeypatch.setattr(G, "SIGNED_VALUE", True)
    assert G.cell_measure(c)["value"].startswith("-")


def test_r_ratio_cell_names_below_floor_tensors():
    c = {
        "id": "L3.optim.adam_v",
        "type": "optimizer_step",
        "level": "L3",
        "status": "pass",
        "threshold": {"kind": "numeric", "values": {"r_ratio": 1.2}},
        "metrics": {
            "r_ratio": 4.55,
            "rel_l2_gpu_vs_ref": 2.23e-8,
            "rel_l2_tpu_vs_ref": 1.02e-7,
            "tensors": 99,
            "below_floor": 99,
        },
    }
    assert "99 / 99 个张量低于分辨率下限，按下限过" in G.cell_measure(c)["value"]
    c["metrics"]["below_floor"] = 0
    assert "分辨率下限" not in G.cell_measure(c)["value"]


def test_router_stats_signed_mean_says_abs(monkeypatch):
    monkeypatch.setattr(G, "SIGNED_VALUE", True)
    c = {
        "id": "L4.router.bias_std",
        "type": "router_stats",
        "status": "inconclusive",
        "threshold": {
            "kind": "calibrated",
            "noise": {"refs": 28, "upper": {"signed_mean": 1.6e-3, "mare": 5e-3, "max_step": 3e-2}},
        },
        "metrics": {"signed_mean_max": 1.82e-3, "mare_max": 4.9e-3, "max_step_max": 2.4e-2},
    }
    assert "各层取绝对值最大" in G.cell_measure(c)["metric"]
