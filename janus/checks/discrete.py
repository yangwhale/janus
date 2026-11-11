"""离散类检查：discrete_agreement（整数结果、路由选择）。"""
from __future__ import annotations

import numpy as np

from janus import metrics as M
from janus.checks.base import ERROR, FAIL, PASS, CheckResult, Threshold, judge_calibrated, judge_numeric, register


@register("discrete_agreement")
def discrete_agreement(cid, level, pairs, threshold: Threshold, hard=True, params=None, **_):
    """pairs: [(名字, 被测, 参照)]。

    params.mode = "exact"（默认，整数张量逐元素比）或 "topk"（[行, k] 下标，比集合与顺序）。
    topk 模式可给 params.scores = {名字: (被测分数, 参照分数)}，报不一致行第 k 与 k+1 名的分差。
    exact 门槛 = 完全相同；数值门槛写在 exact_match_frac / set_frac / order_frac 上（越大越好）。
    """
    params = params or {}
    mode = params.get("mode", "exact")
    res = CheckResult(cid, "discrete_agreement", level, hard)
    if not pairs:
        res.status, res.message = ERROR, "select 选空"
        return res
    rows = []
    for name, a, b in pairs:
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            res.status, res.message = ERROR, f"{name} 形状不同 {a.shape} vs {b.shape}"
            return res
        if mode == "exact":
            rows.append({"name": name, "exact_match_frac": M.exact_match_frac(a, b)})
        else:
            t = M.topk_agreement(a, b)
            row = {"name": name, "set_frac": t.set_frac, "order_frac": t.order_frac, "rows": t.n,
                   "rows_differ": int(t.rows_differ.size)}
            sc = params.get("scores", {}).get(name)
            if sc is not None and t.rows_differ.size:
                k = a.shape[-1]
                margin = M.topk_margin(sc[1], k, t.rows_differ)
                row["margin_at_differ_max"] = float(np.max(margin))
                row["margin_at_differ_median"] = float(np.median(margin))
            rows.append(row)
    res.details = rows
    key = "exact_match_frac" if mode == "exact" else "set_frac"
    agg = {key: min(r[key] for r in rows)}
    if mode == "topk":
        agg["order_frac"] = min(r["order_frac"] for r in rows)
    res.metrics = {f"{k}_min": v for k, v in agg.items()}
    if threshold.kind == "exact":
        res.threshold = {"kind": "exact"}
        res.status = PASS if all(v == 1.0 for v in agg.values()) else FAIL
    elif threshold.kind == "numeric":
        res.threshold = {"kind": "numeric", "values": threshold.values, "source": threshold.source}
        res.status, bad = judge_numeric(agg, threshold.values)
    else:
        st, n, r = judge_calibrated(agg[key], threshold.noise.get(key, []), higher_better=True)
        res.threshold = {"kind": "calibrated", "noise": {"refs": len(threshold.noise.get(key, [])), "upper": {key: n}}}
        res.status, res.r1 = st, r
    res.worst = sorted(({"name": r["name"], "metric": key, "value": r[key], "location": None} for r in rows),
                       key=lambda d: d["value"])[:5]
    res.message = f"{key} 最低 {agg[key]:.6f}"
    return res
