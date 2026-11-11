"""张量类检查：bitwise_equal、tensor_tolerance、ulp_distribution。"""
from __future__ import annotations

import numpy as np

from janus import metrics as M
from janus.checks.base import (
    ERROR,
    FAIL,
    INCONCLUSIVE,
    PASS,
    CheckResult,
    Threshold,
    judge_calibrated,
    judge_numeric,
    register,
    worst_of,
)

TENSOR_METRICS = ("rel_l2", "cosine", "mean_shift", "max_abs", "out_of_tol_ratio", "norm_ratio",
                  "std_ratio", "zero_frac_diff", "rel_err_p99", "kl")


def _pairs_or_error(res: CheckResult, pairs):
    if not pairs:
        res.status, res.message = ERROR, "select 选空：没有可比较的张量"
        return False
    return True


@register("bitwise_equal")
def bitwise_equal(cid, level, pairs, threshold=None, hard=True, **_):
    """pairs: [(名字, 被测数组, 参照数组)]。逐元素位模式相同才 pass。"""
    res = CheckResult(cid, "bitwise_equal", level, hard, threshold={"kind": "exact"})
    if not _pairs_or_error(res, pairs):
        return res
    differ = 0
    for name, a, b in pairs:
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape:
            d = {"name": name, "equal": False, "reason": f"dtype/形状不同 {a.dtype}{a.shape} vs {b.dtype}{b.shape}"}
        else:
            ne = int(np.count_nonzero(a.view(np.uint8).reshape(a.shape + (-1,)) != b.view(np.uint8).reshape(b.shape + (-1,)))
                     ) if a.size else 0
            d = {"name": name, "equal": ne == 0, "differ_bytes": ne}
        differ += 0 if d["equal"] else 1
        res.details.append(d)
    res.metrics = {"tensors": len(pairs), "differ": differ}
    res.status = PASS if differ == 0 else FAIL
    res.worst = [{"name": d["name"], "metric": "equal", "value": False, "location": None}
                 for d in res.details if not d["equal"]][:5]
    res.message = "全部逐位相同" if differ == 0 else f"{differ} / {len(pairs)} 个张量不同"
    return res


def tensor_row(name, a, b, metrics, atol=0.0, rtol=0.0):
    """一对张量上的一组指标（一行明细）。张量类检查与引擎算噪声参照共用，保证两边是同一把尺子。"""
    row = {"name": name}
    fc = M.finite_check(a, b)
    if not fc.ok:
        row["nonfinite_mismatch"] = fc.mismatch
        return row
    if "kl" in metrics:                        # kl 要按原始形状分行，先于拉直算
        row["kl"] = M.kl_div(a, b)
        metrics = [m for m in metrics if m != "kl"]
        if not metrics:
            return row
    fast = {"rel_l2", "cosine", "mean_shift", "max_abs", "norm_ratio"}
    if set(metrics) <= fast and not M.np.issubdtype(M.np.asarray(a).dtype, M.np.integer):
        st = M.pair_stats(a, b)
        if st is not None:
            for m in metrics:
                row[m] = st[m]
                if m == "max_abs":
                    row["max_abs_at"] = st["max_abs_at"]
            return row
    shape = M.np.shape(a)
    if not M.np.issubdtype(M.np.asarray(a).dtype, M.np.integer):
        a, b = M._finite_pair(a, b)            # 只转一次 FP64、只剔一次非有限值，后面各指标共用
    for m in metrics:
        if m == "max_abs" and isinstance(a, M._Clean) and a.size == int(M.np.prod(shape)):
            d = M.np.abs(a - b)
            i = int(M.np.argmax(d)) if d.size else 0
            row[m] = float(d[i]) if d.size else 0.0
            row["max_abs_at"] = tuple(int(k) for k in M.np.unravel_index(i, shape)) if d.size else None
            continue
        if m == "rel_l2":
            row[m] = M.rel_l2(a, b)
        elif m == "cosine":
            row[m] = M.cosine(a, b)
        elif m == "mean_shift":
            row[m] = M.mean_shift(a, b)
        elif m == "max_abs":
            row[m], row["max_abs_at"] = M.max_abs(a, b)
        elif m == "out_of_tol_ratio":
            row[m] = M.out_of_tol_ratio(a, b, atol, rtol)
        elif m == "norm_ratio":
            row[m] = M.norm_ratio(a, b)
        elif m == "std_ratio":
            row[m] = M.std_ratio(a, b)
        elif m == "zero_frac_diff":
            row[m] = M.zero_frac_diff(a, b)
        elif m == "rel_err_p99":
            row[m] = M.rel_err_p99(a, b)
        else:
            raise ValueError(f"未知指标 {m}")
    return row


AGG_NAME = {"cosine": "min", "norm_ratio": "dev_max", "std_ratio": "dev_max"}


def aggregate(m: str, vals: list[float]) -> float:
    """全体张量的最差值：cosine 取最小；norm_ratio、std_ratio 取 |比值 − 1| 的最大（偏大偏小都算）；其余取最大。"""
    if m == "cosine":
        return min(vals)
    if m in ("norm_ratio", "std_ratio"):
        return max(abs(v - 1.0) for v in vals)
    return max(vals)


@register("tensor_tolerance")
def tensor_tolerance(cid, level, pairs, metrics, threshold: Threshold, hard=True, params=None, **_):
    """每个张量按 metrics 算指标，取全体最差值判门槛。

    calibrated 时 threshold.noise[指标] 是噪声参照上「全体最差值」的列表。
    非有限值位置不一致的张量直接 fail。
    """
    params = params or {}
    res = CheckResult(cid, "tensor_tolerance", level, hard)
    if not _pairs_or_error(res, pairs):
        return res
    for m in metrics:
        if m not in TENSOR_METRICS:
            res.message = f"未知指标 {m}"
            return res
    rows = [tensor_row(n, a, b, metrics, params.get("atol", 0.0), params.get("rtol", 0.0)) for n, a, b in pairs]
    res.details = rows
    bad_finite = [r["name"] for r in rows if "nonfinite_mismatch" in r]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in rows if m in r]
        if not vals:
            continue
        agg[m] = aggregate(m, vals)
    res.metrics = {f"{m}_{AGG_NAME.get(m, 'max')}": v for m, v in agg.items()}
    res.metrics["tensors"] = len(rows)
    if not bad_finite and not params.get("no_reference_stats"):
        # 参考档用的统计（超差元素比例、平均差、归一化最大差），只报不判；定义见 janus/metrics.py 的 reference_stats
        ref = [M.reference_stats(a, b, with_l2=False) for _, a, b in pairs]
        for r, st in zip(rows, ref):
            r.update({f"ref_{k}": v for k, v in st.items()})
        res.metrics.update({f"{k}_max": max(st[k] for st in ref) for k in ("oot_frac", "mean_diff", "norm_max")})
    if bad_finite:
        res.status, res.message = FAIL, f"{len(bad_finite)} 个张量非有限值位置不一致：{bad_finite[:3]}"
        return res
    if threshold.kind == "exact":
        res.status, res.message = ERROR, "tensor_tolerance 不接受 exact，用 bitwise_equal"
        return res
    if threshold.kind == "numeric":
        res.threshold = {"kind": "numeric", "values": threshold.values, "source": threshold.source}
        status, bad = judge_numeric(agg, threshold.values)
        res.status = status
        res.message = "全部在门槛内" if status == PASS else f"超线：{bad}"
    else:
        statuses, ups, rs = [], {}, []
        for m in metrics:
            st, n, r = judge_calibrated(agg[m], threshold.noise.get(m, []), higher_better=(m == "cosine"))
            statuses.append(st)
            ups[m] = n
            if r is not None:
                rs.append(r)
        res.threshold = {"kind": "calibrated", "noise": {"refs": min((len(threshold.noise.get(m, [])) for m in metrics), default=0),
                                                         "upper": ups}}
        res.r1 = max(rs) if rs else None
        res.status = FAIL if FAIL in statuses else (INCONCLUSIVE if INCONCLUSIVE in statuses else PASS)
        res.message = {PASS: "在噪声参照内", FAIL: "超过噪声参照两倍", INCONCLUSIVE: "介于噪声参照一到两倍之间，或参照不足"}[res.status]
    main = metrics[0]
    res.worst = worst_of(rows, main, higher_better=(main == "cosine"))
    return res


@register("ulp_distribution")
def ulp_distribution(cid, level, pairs, threshold: Threshold | None = None, hard=False, **_):
    """ULP 分布，默认只诊断（hard=False）。给了数值门槛时按 p999 / max / zero_frac 判。"""
    res = CheckResult(cid, "ulp_distribution", level, hard)
    if not _pairs_or_error(res, pairs):
        return res
    rows = []
    for name, a, b in pairs:
        try:
            s = M.ulp_summary(a, b)
        except TypeError as e:
            res.status, res.message = ERROR, f"{name}: {e}"
            return res
        fc = M.finite_check(a, b)
        rows.append({"name": name, "zero_frac": s.zero_frac, "one_frac": s.one_frac, "p50": s.p50, "p99": s.p99,
                     "p999": s.p999, "max": s.max, "cross_zero": s.cross_zero, "n": s.n,
                     "nonfinite_mismatch": fc.mismatch, "buckets": s.buckets})
    res.details = rows
    # 全体元素合在一起的分桶占比（按各张量元素数加权）
    n_all = sum(r["n"] for r in rows)
    pooled = {k: (sum(r["buckets"][k] * r["n"] for r in rows) / n_all if n_all else 0.0) for k, _, _ in M.ULP_BUCKETS}
    res.metrics = {"zero_frac_min": min(r["zero_frac"] for r in rows), "one_frac": pooled["1"], "buckets": pooled,
                   "p999_max": max(r["p999"] for r in rows), "max_max": max(r["max"] for r in rows),
                   "cross_zero": sum(r["cross_zero"] for r in rows),
                   "nonfinite_mismatch": sum(r["nonfinite_mismatch"] for r in rows)}
    # 「0 ULP 比例在哪个张量掉下来」：按 zero_frac 升序列最差的
    res.worst = worst_of(rows, "zero_frac", higher_better=True)
    if res.metrics["nonfinite_mismatch"]:
        res.status = FAIL if (threshold is not None and threshold.kind != "calibrated") else PASS
        res.message = f"非有限值位置不一致：{res.metrics['nonfinite_mismatch']} 个元素"
        if res.status == FAIL:
            return res
    if threshold is None or threshold.kind == "calibrated":
        res.status = PASS
        res.message = "诊断：只报分布，不判" + (f"；{res.message}" if res.metrics["nonfinite_mismatch"] else "")
        res.threshold = {"kind": "none"}
        return res
    if threshold.kind == "exact":
        res.threshold = {"kind": "exact"}
        res.status = PASS if res.metrics["max_max"] == 0 and res.metrics["cross_zero"] == 0 else FAIL
    else:
        res.threshold = {"kind": "numeric", "values": threshold.values, "source": threshold.source}
        agg = {"p999": res.metrics["p999_max"], "max": res.metrics["max_max"], "zero_frac": res.metrics["zero_frac_min"]}
        res.status, bad = judge_numeric(agg, threshold.values)
        res.message = "在门槛内" if res.status == PASS else f"超线：{bad}"
    return res


_tensor_row = tensor_row  # 旧名
