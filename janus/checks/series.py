"""序列类检查：loss_series、multi_golden、scalar_series、noise_verdict。"""
from __future__ import annotations

import itertools
import math

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
)

SERIES_KEYS = ("d0", "signed_mean", "mare", "max_step")
# calibrated 只判这三个：首步差 d0 是单个样本，GPU 自比时常为 0（前向确定），不能当噪声尺子；
# d0 用 params.first_step_tol（数值，必须配 first_step_source）单独判。
CALIBRATED_KEYS = ("signed_mean", "mare", "max_step")


def parse_window(w) -> tuple[int, int]:
    if isinstance(w, (list, tuple)):
        return int(w[0]), int(w[1])
    a, _, b = str(w).partition(":")
    return int(a), int(b)


def window_stats(test, ref, window, denominator, first_step):
    """把步号平移成「第一步 = 0」后调 series_stats；返回绝对值化的判定量与原始统计。"""
    off = first_step
    t = {s - off: v for s, v in test.items()}
    r = {s - off: v for s, v in ref.items()}
    a, b = window
    st = M.series_stats(t, r, (a - off, b - off), denominator)
    vals = {"d0": abs(st.d0) if st.d0 is not None else None, "signed_mean": abs(st.signed_mean),
            "mare": st.mare, "max_step": st.max_step, "signed_mean_se": st.signed_mean_se}
    raw = {"d0": st.d0, "signed_mean": st.signed_mean, "mare": st.mare, "max_step": st.max_step,
           "max_step_at": st.max_step_at + off if st.max_step_at >= 0 else None,
           "window": f"{a}:{b}", "denominator": denominator, "nonfinite_steps": [s + off for s in st.nonfinite_steps],
           "signed_mean_se": st.signed_mean_se}
    return vals, raw


def _se_floor(threshold: Threshold, params) -> dict:
    """params.signed_mean_se_k = k 时，带符号均值的上限不低于 k × 噪声对标准误的最大值。

    几条噪声参照的 |带符号均值| 本身是围绕 0 的随机数，取最大值做上限时，参照恰好都小会让上限偏紧；
    标准误按每条噪声对的逐步离散度估，比几个样本的最大值稳。默认关闭。"""
    k = (params or {}).get("signed_mean_se_k")
    se = [v for v in (threshold.noise or {}).get("signed_mean_se", []) if v is not None]
    if k is None or not se:
        return {}
    return {"signed_mean": float(k) * max(se)}


def judge_series(res: CheckResult, vals: dict, threshold: Threshold, keys, floors=None):
    """序列类检查共用的判定：exact / 数值 / calibrated，写进 res 的 status、threshold、message。"""
    if threshold.kind == "exact":
        res.threshold = {"kind": "exact"}
        res.status = PASS if all((vals.get(k) or 0.0) == 0.0 for k in keys) else FAIL
        return
    if threshold.kind == "numeric":
        res.threshold = {"kind": "numeric", "values": threshold.values, "source": threshold.source}
        res.status, bad = judge_numeric({k: vals[k] for k in threshold.values if vals.get(k) is not None}, threshold.values)
        res.message = "在门槛内" if res.status == PASS else f"超线：{bad}"
        return
    statuses, ups, rs = [], {}, {}
    for k in keys:
        if k not in threshold.noise or vals.get(k) is None:
            continue
        st, n, r = judge_calibrated(vals[k], threshold.noise[k], floor=(floors or {}).get(k, 0.0))
        statuses.append(st)
        ups[k], rs[k] = n, r
    if not statuses:
        res.status, res.message = INCONCLUSIVE, "没有可用的噪声参照"
        return
    res.threshold = {"kind": "calibrated", "noise": {"refs": min(len(threshold.noise[k]) for k in ups), "upper": ups}}
    if floors:
        res.threshold["noise"]["floor"] = floors
    res.r1, res.r2 = rs.get("mare"), rs.get("signed_mean")
    res.status = FAIL if FAIL in statuses else (INCONCLUSIVE if INCONCLUSIVE in statuses else PASS)
    res.message = {PASS: "在噪声参照内", FAIL: "超过噪声参照两倍",
                   INCONCLUSIVE: "介于噪声参照一到两倍之间，或参照不足"}[res.status]


def _first_step(res: CheckResult, vals, params) -> bool:
    """首步单独判：给了 first_step_tol 时 |d0| 超过即 fail。返回是否已判 fail。"""
    tol = params.get("first_step_tol")
    if tol is None or vals.get("d0") is None:
        return False
    if not params.get("first_step_source"):
        res.status, res.message = ERROR, "first_step_tol 必须配 first_step_source"
        return True
    if vals["d0"] > float(tol):
        res.status, res.message = FAIL, f"首步差 {vals['d0']:.3e} 超过 {tol}（{params['first_step_source']}）"
        return True
    return False


def _hard_limit(res: CheckResult, vals, params):
    lim = (params.get("hard_limit") or {}).get("max_step")
    if lim is not None and vals["max_step"] > lim:
        res.status = FAIL
        res.message = f"单步硬上限超线：{vals['max_step']:.3e} > {lim}"
        return True
    return False


@register("loss_series")
def loss_series(cid, level, test, ref, threshold: Threshold, hard=True, params=None, **_):
    """test / ref：步号 → 值（被测已按 meta.steps.convention 换算到 golden 步号）。params：window（必填）、
    denominator（golden）、first_step（窗口外单独报的首步，默认 0）、hard_limit.max_step、
    auto_offset（[lo, hi]，步号偏移探测，见 probe_offset）、step_offset（约定里的 k，只用于报告）。
    判定量取绝对值：d0、signed_mean、mare、max_step。顺序：偏移探测 → 首步 → 窗口统计。"""
    params = params or {}
    res = CheckResult(cid, "loss_series", level, hard)
    if _offset_check(res, test, {params.get("ref_name", "ref"): ref}, params):
        return res
    if _first_step(res, {"d0": _d0_direct(test, ref, int(params.get("first_step", 0)),
                                          params.get("denominator", "golden"))}, params):
        return res
    try:
        vals, raw = window_stats(test, ref, parse_window(params["window"]), params.get("denominator", "golden"),
                           int(params.get("first_step", 0)))
    except (KeyError, ValueError) as e:
        res.status, res.message = ERROR, f"取数或参数错误：{e}"
        return res
    res.metrics = {**raw, **res.metrics, "first_differ_step": first_differ_step(test, ref, parse_window(params["window"]))}
    _noise_notes(res, threshold)
    if raw["nonfinite_steps"]:
        res.status, res.message = FAIL, f"出现非有限值：第 {raw['nonfinite_steps'][:5]} 步"
        return res
    if _first_step(res, vals, params) or _hard_limit(res, vals, params):
        return res
    judge_series(res, vals, threshold, CALIBRATED_KEYS if threshold.kind == "calibrated" else SERIES_KEYS,
           _se_floor(threshold, params))
    return res


def first_differ_step(test: dict, ref: dict, window) -> int | None:
    """窗口内第一个「两边值不逐位相同」的步（一边缺这一步也算）；全部相同给 None。日常回归的二分判据从这里取。"""
    lo, hi = window
    for s in range(lo, hi):
        a, b = test.get(s), ref.get(s)
        if a is None or b is None or np.float64(a).tobytes() != np.float64(b).tobytes():
            return s
    return None


def _identical_in_window(a: dict, b: dict, window) -> bool:
    """两条序列在窗口内逐步完全相等（同一步都存在且值相等）。缺步时返回 False，留给 window_stats 报错。"""
    lo, hi = window
    for s in range(lo, hi):
        if s not in a or s not in b or not (a[s] == b[s]):
            return False
    return hi > lo


def noise_from_pairs(pairs, window, denominator="golden", first_step=0, names=None):
    """噪声参照：每一对运行（参照 A、参照 B）算一遍判定量。返回 指标 → [值]，另带两项说明（判定时不读）：

    - dropped：窗口内逐位相同的参照对（名字）。它们是同一条轨迹的重复（如两次确定性运行），噪声为 0，
      留着会把上限拉低、把「参照条数」虚增，所以剔除；剔除后不足 3 条时 calibrated 最多给 inconclusive。
    - loo：留一法互判。对每条保留下来的参照对，用其余参照对的最大值当上限算 r = 值 / 上限（只报不判；
      其余参照只有 2 条以下，本来也给不出 pass）。r 明显大于 1 的参照对说明参照本身不齐，上限由它一条撑着。
    """
    names = list(names) if names is not None else [f"pair{i}" for i in range(len(pairs))]
    out = {k: [] for k in SERIES_KEYS + ("signed_mean_se",)}
    kept, dropped = [], []
    for nm, (a, b) in zip(names, pairs):
        if _identical_in_window(a, b, window):
            dropped.append(nm)
            continue
        vals, _ = window_stats(a, b, window, denominator, first_step)
        kept.append((nm, vals))
        for k in out:
            if vals[k] is not None:
                out[k].append(vals[k])
    loo = []
    for i, (nm, vals) in enumerate(kept):
        others = [v for j, (_, v) in enumerate(kept) if j != i]
        row = {"pair": nm}
        for k in CALIBRATED_KEYS:
            ref_vals = [o[k] for o in others if o[k] is not None]
            if not ref_vals or vals[k] is None:
                row[k] = None
                continue
            n = max(ref_vals)
            row[k] = (0.0 if vals[k] == 0.0 else math.inf) if n == 0.0 else vals[k] / n
        loo.append(row)
    out["dropped"] = dropped
    out["loo"] = loo
    return out


def _noise_notes(res: CheckResult, threshold: Threshold):
    """把 noise_from_pairs 的剔除与留一法结果写进 metrics（只报不判）。"""
    noise = threshold.noise or {}
    if "dropped" in noise:
        res.metrics["noise_dropped"] = list(noise["dropped"])
    if noise.get("loo"):
        res.metrics["noise_loo"] = noise["loo"]


def probe_offset(test: dict, ref: dict, first_step: int, rng) -> tuple[int | None, dict]:
    """步号偏移探测：被测首步（first_step，已按约定换算到 golden 步号）的值依次与参照第 first_step + δ 步比，
    δ ∈ [rng[0], rng[1]]，取 |Δ| 最小的 δ。并列时取 |δ| 最小的（0 优先）。返回 (δ, {δ: |Δ|})；
    被测没有首步或参照一步都对不上时 δ 为 None。"""
    lo, hi = int(rng[0]), int(rng[1])
    if first_step not in test:
        return None, {}
    t = float(test[first_step])
    cand = {}
    for d in range(lo, hi + 1):
        r = ref.get(first_step + d)
        if r is not None and math.isfinite(float(r)) and math.isfinite(t):
            cand[d] = abs(t - float(r))
    if not cand:
        return None, cand
    best = min(cand.values())
    return min((d for d, v in cand.items() if v == best), key=lambda d: (abs(d), d)), cand


def _offset_check(res: CheckResult, test: dict, refs: dict, params) -> bool:
    """params.auto_offset = [lo, hi] 时探测步号偏移；任一份参照上探测到的偏移不为 0（即与 meta.steps.convention 不一致）
    → fail，报探测到的偏移（换回被测原始步号的写法：golden_step = step + 约定 k + δ）。返回是否已判 fail。"""
    rng = params.get("auto_offset")
    if not rng:
        return False
    fs, k = int(params.get("first_step", 0)), int(params.get("step_offset", 0))
    found = {}
    for name, ref in refs.items():
        d, cand = probe_offset(test, ref, fs, rng)
        found[name] = d
        res.metrics.setdefault("offset_probe", {})[name] = {"delta": d, "abs_diff": {str(x): v for x, v in cand.items()}}
    bad = {n: d for n, d in found.items() if d not in (0, None)}
    if bad:
        n, d = next(iter(bad.items()))
        res.status = FAIL
        res.message = (f"步号偏移：探测到 golden_step = step {k + d:+d}（对 {n}），与约定 golden_step = step {k:+d} 不一致"
                       f"；首步 |Δ| 在偏移 {d:+d} 处最小")
        return True
    return False


def _d0_direct(test: dict, ref: dict, fs: int, denominator: str):
    """首步相对差，不经过窗口统计（窗口因错位缺步时也能先判首步）。缺步给 None。"""
    if fs not in test or fs not in ref:
        return None
    t, r = float(test[fs]), float(ref[fs])
    den = abs(r) if denominator == "golden" else abs(t) if denominator == "test" else abs(t + r) / 2
    if den == 0.0:
        return 0.0 if t == r else math.inf
    return abs(t - r) / den


@register("multi_golden")
def multi_golden(cid, level, test, goldens: dict, threshold: Threshold, hard=True, params=None, noise_pairs=None,
                 noise_names=None, **_):
    """被测对每一份 golden 各算一遍 loss_series 的判定量，取最差判门槛。

    params.baseline = <golden 名>（单 golden 模式，正式验收用）：被测只对这一份指定的 golden 算判定量，
    其余 golden 只用来组噪声参照（默认取全部 golden 两两组合），不取平均、不取各份里的最差。

    calibrated 且没给 threshold.noise 时，噪声参照用 noise_pairs（名字 noise_names；默认取 golden 两两组合），
    逐位相同的参照对剔除、留一法互判见 noise_from_pairs。params.auto_offset 对每份比较用的 golden 做步号偏移探测（见 loss_series）。
    另报「落在 golden 带内的步数比例」（每步被测是否在各 golden 的最小、最大值之间；单 golden 模式下仍用全部 golden）。
    """
    params = params or {}
    res = CheckResult(cid, "multi_golden", level, hard)
    if not goldens:
        res.status, res.message = ERROR, "没有 golden"
        return res
    base = params.get("baseline")
    if base is not None and base not in goldens:
        res.status, res.message = ERROR, f"baseline {base} 不在 golden 列表 {list(goldens)} 里"
        return res
    cmp_goldens = {base: goldens[base]} if base is not None else goldens
    if _offset_check(res, test, cmp_goldens, params):
        return res
    fs0, den0 = int(params.get("first_step", 0)), params.get("denominator", "golden")
    d0s = [d for d in (_d0_direct(test, g, fs0, den0) for g in cmp_goldens.values()) if d is not None]
    if _first_step(res, {"d0": max(d0s) if d0s else None}, params):
        return res
    try:
        window = parse_window(params["window"])
        den, fs = params.get("denominator", "golden"), int(params.get("first_step", 0))
        per = {name: window_stats(test, g, window, den, fs) for name, g in cmp_goldens.items()}
    except (KeyError, ValueError) as e:
        res.status, res.message = ERROR, f"取数或参数错误：{e}"
        return res
    for name, (_, raw) in per.items():
        res.details.append({"name": name, **raw})
        if raw["nonfinite_steps"]:
            res.status, res.message = FAIL, f"对 {name} 出现非有限值：第 {raw['nonfinite_steps'][:5]} 步"
            return res
    worst = {k: max((v[0][k] for v in per.values() if v[0][k] is not None), default=None) for k in SERIES_KEYS}
    a, b = window
    steps = [s for s in range(a, b) if s != fs]
    inside = 0
    for s in steps:
        gv = [g[s] for g in goldens.values()]
        inside += min(gv) <= test[s] <= max(gv)
    if base is not None:
        res.metrics["baseline"] = base
    res.metrics = {**{f"{k}_max": v for k, v in worst.items()}, "goldens": len(goldens),
                   "in_band_frac": inside / len(steps) if steps else None, "window": f"{a}:{b}", "denominator": den,
                   **res.metrics}
    if _first_step(res, worst, params) or _hard_limit(res, worst, params):
        return res
    th = threshold
    if th.kind == "calibrated" and not th.noise:
        if noise_pairs is not None:
            pairs, pnames = noise_pairs, noise_names
        else:
            combos = list(itertools.combinations(goldens.items(), 2))
            pairs, pnames = [(x[1], y[1]) for x, y in combos], [f"{x[0]}~{y[0]}" for x, y in combos]
        th = Threshold("calibrated", noise=noise_from_pairs(pairs, window, den, fs, names=pnames))
    _noise_notes(res, th)
    judge_series(res, worst, th, CALIBRATED_KEYS if th.kind == "calibrated" else SERIES_KEYS, _se_floor(th, params))
    return res


@register("scalar_series")
def scalar_series(cid, level, test, ref, params=None, hard=True, threshold=None, **_):
    """逐步标量（学习率等）：|t − r| ≤ rtol·|r| + abs_ulp·ulp(r)（按 dtype 数格，默认 float32、2 格）。

    params.equals 给定时改为设置核对：被测与参照每一步都要等于这个登记值（容差同上，以登记值为 r），
    用于学习率这类我们自己设的常数——两边相同还不够，必须是登记的那个值。"""
    params = params or {}
    res = CheckResult(cid, "scalar_series", level, hard)
    rtol, n_ulp = float(params.get("rtol", 0.0)), int(params.get("abs_ulp", 2))
    dt = np.dtype(params.get("dtype", "float32"))
    want = params.get("equals")
    try:
        a, b = parse_window(params["window"])
        bad = []
        for s in range(a, b):
            t, r = float(test[s]), float(ref[s])
            pairs = [(t, r)] if want is None else [(t, float(want)), (r, float(want))]
            for x, y in pairs:
                tol = rtol * abs(y) + n_ulp * float(np.spacing(dt.type(abs(y))))
                if not (math.isfinite(x) and math.isfinite(y)) or abs(x - y) > tol:
                    bad.append(s)
                    break
    except (KeyError, ValueError) as e:
        res.status, res.message = ERROR, f"取数或参数错误：{e}"
        return res
    res.threshold = {"kind": "numeric", "values": {"rtol": rtol, "abs_ulp": n_ulp},
                     "source": params.get("source", "规格默认值")}
    res.metrics = {"steps": b - a, "bad_steps": len(bad), "first_bad": bad[0] if bad else None}
    if want is not None:
        res.metrics["equals"] = float(want)
    res.status = PASS if not bad else FAIL
    if want is None:
        res.message = "逐步在容差内" if not bad else f"{len(bad)} 步超容差，最早第 {bad[0]} 步"
    else:
        res.message = (f"两边每一步都等于登记值 {want}" if not bad
                       else f"{len(bad)} 步有一边不等于登记值 {want}，最早第 {bad[0]} 步")
    return res


@register("noise_verdict")
def noise_verdict(cid, level, values: dict, noise: dict, hard=True, **_):
    """values：指标 → 被测值（越小越好）；noise：指标 → 噪声参照值列表。逐指标按 calibrated 规则判，取最严。"""
    res = CheckResult(cid, "noise_verdict", level, hard)
    if not values:
        res.status, res.message = ERROR, "没有被测值"
        return res
    judge_series(res, values, Threshold("calibrated", noise=noise), list(values))
    res.metrics = dict(values)
    return res


# 旧名：模型层脚本（如 models/hunyuan3/regression/calibrate.py）按它们导入
_stats = window_stats
_judge = judge_series
