"""L1 单算子检查：op_replay。把算子回放与合成用例的产出统一成 CheckResult。规格见 docs/reference/checks/op_replay.md。"""
from __future__ import annotations

import math

import numpy as np

from janus import metrics as M
from janus.analysis import ops as O
from janus.checks.base import ERROR, FAIL, PASS, CheckResult, Threshold, judge_numeric, register

# 汇总指标 → (逐用例的字段, 汇总方式)。汇总名就是门槛里写的名字。
_AGG = {
    "rel_l2": ("rel_l2", max),
    "cr_match_frac": ("cr_match_frac", min),
    "ulp_max": ("ulp_max", max),
    "ulp_p999": ("ulp_p999", max),
    "nonfinite_mismatch": ("nonfinite_mismatch", sum),
    "bit_match_frac": ("bit_match_frac", min),
    "emulate_match_frac": ("emulate_match_frac", min),
    "r_ratio": ("r_ratio", max),
    "expect_match_frac": ("expect_match_frac", min),
    # 参考档：被测对参照（GPU 或参照 kernel）的统计，只报不判（门槛写在汇总名上也能判）
    "vs_ref_rel_l2": ("vs_ref_rel_l2", max),
    "vs_ref_cosine": ("vs_ref_cosine", min),
    "vs_ref_oot_frac": ("vs_ref_oot_frac", max),
    "vs_ref_mean_diff": ("vs_ref_mean_diff", max),
    "vs_ref_norm_max": ("vs_ref_norm_max", max),
}


def _float_row(case: dict) -> dict:
    out = np.asarray(case["out"])
    row = {"name": case["name"], "dtype": str(out.dtype), "n": int(out.size)}
    truth = case.get("fp64")
    if truth is not None:
        truth = np.asarray(truth, dtype=np.float64)
        cr = O.round_from_f64(truth, out.dtype)
        fc = M.finite_check(out, cr)
        row["nonfinite_mismatch"] = fc.mismatch
        row["cr_match_frac"] = O.match_frac(out, cr)
        d = M.ulp_distance(out, cr)
        x, y = out.astype(np.float64).ravel(), cr.astype(np.float64).ravel()
        cross = np.sign(x) * np.sign(y) < 0
        v = d[(d >= 0) & ~cross]
        if v.size:
            row["ulp_max"] = int(v.max())
            row["ulp_p999"] = float(np.percentile(v, 99.9))
            row["ulp_hist"] = {b: int(np.count_nonzero(sel)) for b, sel in
                               (("0", v == 0), ("1", v == 1), ("2-4", (v >= 2) & (v <= 4)),
                                ("5-16", (v >= 5) & (v <= 16)), ("17+", v >= 17))}
        row["cross_zero"] = int(cross.sum())
        if fc.ok:
            # rel_l2 只在两边都有限的位置上算；非有限值位置不一致已经单独计数
            keep = np.isfinite(out.astype(np.float64)) & np.isfinite(truth)
            row["rel_l2"] = M.rel_l2(out.astype(np.float64)[keep], truth[keep]) if keep.any() else 0.0
            if case.get("ref") is not None:
                ref = np.asarray(case["ref"])
                row.update(O.r_ratio(out.astype(np.float64)[keep], ref.astype(np.float64)[keep], truth[keep]))
    vs = case.get("vsref", case.get("ref"))
    if vs is not None:
        # 参考档口径：被测对参照的同名输出。vsref_out 给了就用它（同一输入上的另一份被测输出），否则用 out
        vo = np.asarray(case.get("vsref_out", out))
        vs = np.asarray(vs)
        if vo.shape == vs.shape and M.finite_check(vo, vs).ok:
            row.update({f"vs_ref_{k}": v for k, v in M.reference_stats(vo, vs).items()})
            if case.get("vsref_note"):
                row["vs_ref_note"] = case["vsref_note"]
    if case.get("bitref") is not None:
        row["bit_match_frac"] = O.match_frac(out, np.asarray(case["bitref"]))
    if case.get("emulate") is not None:
        row["emulate_match_frac"] = O.match_frac(out, np.asarray(case["emulate"]))
    return row


def _int_row(case: dict) -> dict:
    """整数输出（top-k 下标）：与期望逐行比集合。"""
    out, exp = np.asarray(case["out"]), np.asarray(case["expect"])
    M._check_shape(out, exp)
    t = M.topk_agreement(out, exp)
    return {"name": case["name"], "dtype": str(out.dtype), "n": int(t.n), "expect_match_frac": t.set_frac,
            "rows_differ": int(t.rows_differ.size), "rows_differ_first": [int(i) for i in t.rows_differ[:10]]}


@register("op_replay")
def op_replay(cid, level, cases, threshold: Threshold | None = None, hard=True, params=None, **_):
    """cases：[{name, out, fp64?, bitref?, emulate?, ref?, expect?, extra?}]。

    out：被测输出（原始 dtype）。fp64：FP64 真值 → rel_l2、对正确舍入的逐位相同率 cr_match_frac、ULP 分布。
    bitref：应当逐位相同的另一份结果（训练时抓的输出、CPU 上的同一运算）→ bit_match_frac。
    emulate：CPU 模拟 GPU 舍入方式的结果 → emulate_match_frac（100% 才算猜对）。
    ref：参照实现的输出 → R-ratio 三方比。vsref（默认 = ref）：参考档的对照输出 → vs_ref_*（rel_l2、cosine、超差元素比例、
    平均差、归一化最大差，只报）；vsref_out：同一输入上的另一份被测输出（默认 out），vsref_note 写它从哪来。expect：整数输出的期望（top-k 打平规则）→ expect_match_frac。
    extra：原样写进明细（开关实测值、输入说明等）。
    门槛：None → 只诊断，判 pass；exact → 给了 bitref / emulate / expect 的必须 100%，否则对正确舍入 100%；
    数值 → 写在汇总指标上（_AGG 的键）。
    params.expect = "fail"：演练——用例是故意改坏的实现，门槛判 fail 时本检查才 pass（原判定记在 drill_judged）。
    params.worst_by：worst 按哪个指标排（默认 _AGG 里第一个有的）。
    """
    res = CheckResult(cid, "op_replay", level, hard)
    if not cases:
        res.status, res.message = ERROR, "没有用例"
        return res
    for c in cases:
        try:
            row = _int_row(c) if c.get("expect") is not None else _float_row(c)
        except (TypeError, ValueError) as e:
            res.status, res.message = ERROR, f"{c.get('name')}：{e}"
            return res
        row.update(c.get("extra") or {})
        res.details.append(row)
    agg = {}
    for key, (field, how) in _AGG.items():
        vals = [r[field] for r in res.details if field in r and r[field] is not None]
        if vals:
            agg[key] = how(vals)
    res.metrics = dict(agg, cases=len(cases))
    if "vs_ref_rel_l2" in agg:
        has = [r for r in res.details if "vs_ref_rel_l2" in r]
        res.metrics["vs_ref_cases"] = len(has)
        res.metrics["vs_ref_worst"] = max(has, key=lambda r: r["vs_ref_rel_l2"])["name"]   # rel_L2 最大的用例
    if threshold is None:
        res.threshold, res.status = {}, PASS
        res.message = "诊断：只报数，不判"
    elif threshold.kind == "exact":
        res.threshold = {"kind": "exact"}
        keys = [k for k in ("bit_match_frac", "emulate_match_frac", "expect_match_frac") if k in agg] or ["cr_match_frac"]
        bad = [k for k in keys if agg.get(k) != 1.0] + (["nonfinite_mismatch"] if agg.get("nonfinite_mismatch") else [])
        res.status = PASS if not bad else FAIL
        res.message = f"逐位全中（{', '.join(keys)}）" if not bad else f"没有逐位全中：{', '.join(f'{k}={agg.get(k)}' for k in bad)}"
    else:
        res.threshold = {"kind": "numeric", "values": threshold.values, "source": threshold.source}
        res.status, bad = judge_numeric(agg, threshold.values)
        res.message = "全部在门槛内" if res.status == PASS else f"超线：{bad}"
    if (params or {}).get("expect") == "fail" and res.status in (PASS, FAIL):
        # 演练：被测是故意改坏的实现，门槛必须报警。报警了 → 演练通过；没报警 → 演练失败
        judged = res.status
        res.metrics["drill_judged"] = judged
        res.status = PASS if judged == FAIL else FAIL
        res.message = (f"演练：门槛按期望报警（{res.message}）" if judged == FAIL
                       else "演练失败：故意改坏的实现没有被门槛发现")
    first = next((k for k in (params or {}).get("worst_by", []) + list(_AGG) if k in agg), None)
    if first:
        field, how = _AGG[first]
        rows = [r for r in res.details if field in r and r[field] is not None and not
                (isinstance(r[field], float) and math.isnan(r[field]))]
        rows.sort(key=lambda r: r[field], reverse=how is not min)
        res.worst = [{"name": r["name"], "metric": first, "value": r[field], "location": None} for r in rows[:5]]
    return res
