"""两档门槛的说明文字：严格档（janus 自己的判定）与参考档（同类 GPU→TPU 迁移项目的实践门槛）。

严格档的判定只来自 results.json 里检查的 status，这里只把门槛、实测、离门槛多远写成人话；
参考档按 reference_gates.yaml 从同一份 metrics 现算，结果里没有所需指标就写「未记录」，不拿口径不同的数去判。
两档的含义与参考值见 docs/reference/judging.md「两档门槛」。
"""

from __future__ import annotations

import fnmatch
import functools
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from janus.checks.base import MIN_NOISE_REFS, is_higher_better

GATES_FILE = Path(__file__).with_name("reference_gates.yaml")
SERIES_TYPES = {"loss_series", "multi_golden", "router_stats"}
# 参考档状态：pass 全部判过｜partial 记录了的都过、有门槛项没记录｜fail 有一项不过｜missing 所需指标都没记录｜none 无对应门槛
REF_SYM = {"pass": "✅", "partial": "☑️", "fail": "❌", "missing": "❔", "none": "—"}
REF_ORDER = ("fail", "missing", "partial", "pass", "none")  # 说明栏挑哪条检查来写参考档：取最靠前的
STRICT_WORD = {"pass": "过", "fail": "不过", "error": "出错", "inconclusive": "待定"}
METRIC_ZH = {
    "signed_mean": "带符号均值",
    "mare": "MARE",
    "max_step": "最大单步",
    "d0": "首步差",
    "rel_l2": "rel_L2",
    "cosine": "1−cos",
    "set_frac": "集合不一致",
    "kl": "KL",
    "norm_ratio": "范数比偏离",
    "std_ratio": "标准差比偏离",
    "zero_frac_diff": "零元素比例差",
    "rel_err_p99": "逐元素相对误差 p99",
    "max_abs": "max_abs",
    "r_ratio": "R 比",
    "r_ratio_own_inputs": "R 比·各用自己的输入",
    "spike_ratio": "突跳倍数",
    "n_unattributed": "超 2N 且未定责的张量数",
    "n_unattributed_over_n": "超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数",
}
UNIT_ZH = {
    "tensors": "个张量",
    "params": "个参数",
    "units": "个单元",
    "fields": "个字段",
    "steps": "步",
    "arrays": "个数组",
    "entries": "项",
    "objects": "个对象",
    "cases": "个用例",
}

# 只在没有上面任何一个计数键时才用的分母（配置逐项对照的 items、逐参数对拍的 compared）
UNIT_ALT = {"compared": "个参数", "items": "项"}


# 数值格式（模型的 build.py 可以在生成前改，各模型分进程生成、互不影响）：
# SCI_BELOW：绝对值小于它的写科学计数；SHORT_EXP：指数不补零（3.5e-4 而不是 3.50e-04 里的 e-04）。
SCI_BELOW = 1e-3
SHORT_EXP = False
UNICODE_MINUS = False
SIGNED_VALUE = False  # 格子里带符号均值写带符号的实测（判定仍按绝对值）；混元 4 打开  # 负号写成「−」
R_DIGITS = 4  # 1 附近的比值类实测保留几位有效数字


def _sci(x) -> str:
    t = f"{x:.2e}"
    if SHORT_EXP:
        m, e = t.split("e")
        t = f"{m}e{int(e)}"
    return _minus(t)


def _minus(t: str) -> str:
    """开头的负号换成「−」（指数里的减号不动）。"""
    return "−" + t[1:] if UNICODE_MINUS and t.startswith("-") else t


def fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if math.isinf(x):
            return "∞"
        if x == 0.0:
            return "0"
        return _minus(f"{x:.3g}") if SCI_BELOW <= abs(x) < 1e4 else _sci(x)
    return str(x)


def _fe(x) -> str:
    """绝对距离一律写科学计数，两边并排时量级一眼可比。"""
    return "0" if not x else _sci(x)


def fmt_ratio(r) -> str:
    """离门槛多远：贴线（0.9–1.1）时多给一位，1.005× 和 1.01× 要分得开。"""
    if r is None:
        return "—"
    if not math.isfinite(r):
        return "∞"
    if r >= 1e4:
        m, e = f"{r:.1e}".split("e")
        return f"{m}e{int(e)}"
    return f"{r:.4g}" if 0.9 < r < 1.1 else f"{r:.3g}"


def fmt_near1(x) -> str:
    """1 附近的比值类实测（R 比、集合相同比例）多给一位：1.0016 不该写成 1，1.2163 对 1.2 的线要看得出差多少。"""
    return f"{x:.{R_DIGITS}g}" if isinstance(x, float) and 0.5 < x < 2 and x != 1.0 else fmt(x)


# 数值门槛里越大越好的指标按原值写（calibrated 里才换成离 1 的距离）
NUMERIC_ZH = {"set_frac": "集合相同比例", "cosine": "cos"}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def measured(metrics: dict, key: str):
    """噪声上限的键（rel_l2、cosine、norm_ratio…）→ 实测里对应的值（取最差的那份汇总）。"""
    for k in (f"{key}_max", f"{key}_dev_max", f"{key}_min", key):
        if k in metrics and _num(metrics[k]):
            x = float(metrics[k])
            if is_higher_better(key):
                x = 1.0 - x
            return abs(x) if key == "signed_mean" else x
    return None


def ratio(x, n):
    if x is None or n is None:
        return None
    if n == 0.0:
        return 0.0 if x == 0.0 else math.inf
    return x / n


# ---------------------------------------------------------------- 严格档


def calibrated_metrics(c: dict) -> list[tuple]:
    """calibrated 检查的逐指标 (键, N, 实测, 实测/N)。"""
    ups = ((c.get("threshold") or {}).get("noise") or {}).get("upper") or {}
    m = c.get("metrics") or {}
    out = []
    for k, n in ups.items():
        x = measured(m, k)
        out.append((k, n, x, ratio(x, n)))
    return out


def worst_calibrated(c: dict):
    cand = calibrated_metrics(c)
    return max(cand, key=lambda t: -1 if t[3] is None else t[3]) if cand else None


def numeric_criteria(c: dict) -> list[tuple]:
    """数值门槛的逐项 (键, 门槛文字, 实测, 实测/门槛)；实测是 None 的项也列出。"""
    vals = (c.get("threshold") or {}).get("values")
    m = c.get("metrics") or {}
    out = []
    if not isinstance(vals, dict):
        return out
    for k, lim in vals.items():
        x = m.get(k, m.get(f"{k}_max", m.get(f"{k}_min")))
        x = x if _num(x) else None
        if isinstance(lim, dict):
            t, r = "、".join(f"{'≥' if b == 'min' else '≤'} {fmt(v)}" for b, v in lim.items()), None
        elif is_higher_better(k):
            t, r = f"≥ {fmt(lim)}", None
        else:
            t, r = f"≤ {fmt(lim)}", (ratio(float(x), float(lim)) if x is not None and _num(lim) else None)
        out.append((k, t, x, r))
    return out


def exact_counts(c: dict):
    """逐位类检查可数时 → (不同的个数, 总数, 单位)；不可数 → None。"""
    m = c.get("metrics") or {}
    den = next((k for k in UNIT_ZH if _num(m.get(k)) and m.get(k)), None)
    if den is None:
        den = next((k for k in UNIT_ALT if _num(m.get(k)) and m.get(k)), None)
    if den is None:
        return None
    bad = next(
        (k for k in ("differ", "bad", "must_equal_differ", "tensors_differ", "unmatched") if _num(m.get(k))), None
    )
    if bad is not None:
        if int(m[bad]) == 0 and c.get("status") not in ("pass", None):
            return None  # 没过却数出 0 个不同：不过的原因不在这个计数里（如逐步标量不可复现），用 message
        return int(m[bad]), int(m[den]), UNIT_ZH.get(den) or UNIT_ALT[den]
    if _num(m.get("same")):
        return int(m[den]) - int(m["same"]), int(m[den]), UNIT_ZH.get(den) or UNIT_ALT[den]
    return None


def exact_measured(c: dict) -> str:
    """逐位类检查的实测写成一句话：「41 / 41 个张量不同」；没有可数的就用结果里的 message。"""
    m = c.get("metrics") or {}
    bad = next(
        (k for k in ("differ", "bad", "must_equal_differ", "tensors_differ", "unmatched") if _num(m.get(k))), None
    )
    den = next((k for k in UNIT_ZH if _num(m.get(k)) and m.get(k)), None)
    if bad and den:
        n_bad, n = int(m[bad]), int(m[den])
        if n_bad == 0 and c.get("status") == "pass":
            return f"全部 {n:,} {UNIT_ZH[den]}一致"
        if n_bad:
            return f"{n_bad:,} / {n:,} {UNIT_ZH[den]}不同"
    msg = str(c.get("message") or "").strip()
    if len(msg) > 70:
        msg = msg[:70] + "…"
    return msg or "见报告"


def strict_parts(c: dict, metric: str | None = None, status: str | None = None) -> tuple[str, str]:
    """→ (实测, 严格档)。metric 指定 calibrated 的某个指标（L4 逐步量逐个列），不给就取最差的那个。
    status 覆盖判定词（逐指标判时用），不给就用检查自己的 status。"""
    th = c.get("threshold") or {}
    kind = th.get("kind")
    st = status or c.get("status", "error")
    word = STRICT_WORD.get(st, st)
    m = c.get("metrics") or {}
    if kind == "calibrated":
        noise = th.get("noise") or {}
        refs = noise.get("refs") or 0
        cand = calibrated_metrics(c)
        if metric == "d0":
            d0 = m.get("d0_max", m.get("d0"))
            return (f"{d0:+.2e}（首步差）" if d0 else "0（首步差）"), "只报，不判"
        worst = worst_calibrated(c)
        # 不指定指标时先写主判指标 rel_L2（误差占信号的比例，最好读）；最差的不是它时再补一句最差的那个
        pick = next((t for t in cand if t[0] == (metric or "rel_l2")), None) or worst
        if pick is None:
            return "—", f"参照不足，{word}"
        k, n, x, r = pick
        short = f"，参照不足（至少 {MIN_NOISE_REFS} 对）" if refs < MIN_NOISE_REFS else ""
        extra = ""
        if metric is None and worst is not None and worst[0] != k and worst[3] is not None:
            extra = f"；最差的是 {METRIC_ZH.get(worst[0], worst[0])} {fmt(_signed(c, worst[0], worst[2]))} = {fmt_ratio(worst[3])}×N"
        return (
            f"{fmt(_signed(c, k, x))}（{METRIC_ZH.get(k, k)}）{cos_text(m)}",
            f"N={fmt(n)}（{refs} 对{short}）→ {fmt_ratio(r)}×N{extra}，{word}",
        )
    if kind == "numeric":
        crit = [q for q in numeric_criteria(c) if q[2] is not None]
        if crit:
            k, t, x, r = max(crit, key=lambda q: -1 if q[3] is None else q[3])
            how = f"{fmt_ratio(r)}×，" if r is not None else ""
            name = NUMERIC_ZH.get(k) if is_higher_better(k) else None
            return f"{fmt_near1(x)}（{name or METRIC_ZH.get(k, k)}）", f"{t} → {how}{word}"
        t = "、".join(f"{k} {q}" for k, q, _, _ in numeric_criteria(c)) or "按规则"
        return exact_measured(c), f"{t}，{word}"
    if kind == "exact":
        return exact_measured(c), f"逐位 / 逐项相同，{word}"
    return exact_measured(c), "只诊断，不判"


def cos_text(m: dict) -> str:
    """结果里记了 cosine 时，在实测后面并列写 cos 本身（最差的那份；1 − cos 很小时 cos 写到 6 位小数才看得出差）。"""
    for k in ("cosine_min", "cosine"):
        v = m.get(k)
        if _num(v):
            return f"，cos {float(v):.6f}"
    return ""


def _metric_status(r, refs) -> str:
    return "fail" if r > 2 else ("pass" if r <= 1 and refs >= MIN_NOISE_REFS else "inconclusive")


def _signed(c: dict, key: str, x):
    """SIGNED_VALUE 打开时，带符号均值写结果里记的带符号值（判定仍按绝对值）；结果只记了绝对值时原样返回。"""
    if not (SIGNED_VALUE and key == "signed_mean" and _num(x)):
        return x
    m = c.get("metrics") or {}
    raw = next((m[q] for q in ("signed_mean_max", "signed_mean") if _num(m.get(q))), None)
    return float(raw) if raw is not None and abs(abs(float(raw)) - x) <= 1e-12 * max(x, 1e-300) else x


def cell_measure(c: dict) -> dict:
    """总览格子用的短写法：{metric 指标名, value 实测, how 离门槛多远, thr 本格的严格门槛}。
    calibrated 取「决定判定的那个指标」：rel_L2 的逐指标判定与整条检查一致就用 rel_L2（最好读），否则用最差的那个。"""
    th = c.get("threshold") or {}
    kind = th.get("kind")
    if kind == "calibrated":
        refs = (th.get("noise") or {}).get("refs") or 0
        cand = [t for t in calibrated_metrics(c) if t[3] is not None]
        if not cand:
            return {"metric": "—", "value": "参照不足", "how": "", "thr": "≤ N 过，> 2N 不过", "short_thr": ""}
        if c.get("type") in SERIES_TYPES:
            pick = max(cand, key=lambda t: t[3])
        else:
            rel = next((t for t in cand if t[0] == "rel_l2"), None)
            pick = rel if rel and _metric_status(rel[3], refs) == c.get("status") else max(cand, key=lambda t: t[3])
        k, n, x, r = pick
        x = _signed(c, k, x)
        name = METRIC_ZH.get(k, k)
        if SIGNED_VALUE and k == "signed_mean" and c.get("type") == "router_stats":
            name += "，各层取绝对值最大"  # router_stats 只记了各层里绝对值最大的那个，没有符号
        return {
            "metric": name,
            "value": fmt(x),
            "how": f"{fmt_ratio(r)}×N",
            "thr": f"≤ N={fmt(n)} 过，> 2N 不过",
            "short_thr": f"N={fmt(n)}",
        }
    if kind == "numeric":
        crit = [q for q in numeric_criteria(c) if q[2] is not None]
        m = c.get("metrics") or {}
        if crit and "r_ratio" in m and "rel_l2_gpu_vs_ref" in m and "rel_l2_tpu_vs_ref" in m:
            # R 比带两边的绝对距离：「GPU 对参照 a、TPU 对参照 b → R」，只看比值看不出两边的误差是不是都在舍入以下
            t = next(q[1] for q in crit if q[0] == "r_ratio")
            ref = (c.get("threshold") or {}).get("ref")  # 参照的名字（可选），写进门槛栏
            return {
                "metric": "R 比",
                "value": (
                    "两边对参照都是 0（逐位）"
                    if not m["rel_l2_gpu_vs_ref"] and not m["rel_l2_tpu_vs_ref"]
                    else f"GPU {_fe(m['rel_l2_gpu_vs_ref'])}、TPU {_fe(m['rel_l2_tpu_vs_ref'])} → R {fmt_near1(m['r_ratio'])}"
                )
                # 逐张量判时低于分辨率下限的按下限过：整体 R 可以超线而格子仍过，写明是哪几个
                + (
                    f"（{m['below_floor']} / {m['tensors']} 个张量低于分辨率下限，按下限过）"
                    if m.get("below_floor") and m.get("tensors") and c.get("status") == "pass"
                    else ""
                ),
                "how": "",
                "thr": f"对 {ref} 的 R 比 {t}" if ref else f"R 比 {t}",
                "short_thr": t,
            }
        if crit:
            k, t, x, r = max(crit, key=lambda q: -1 if q[3] is None else q[3])
            name = (NUMERIC_ZH.get(k) if is_higher_better(k) else None) or METRIC_ZH.get(k, k)
            return {
                "metric": name,
                "value": fmt_near1(x),
                "how": f"{fmt_ratio(r)}×" if r is not None else "",
                "thr": f"{name.split('·')[0]} {t}",
                "short_thr": t,
            }
        t = "、".join(f"{k} {q}" for k, q, _, _ in numeric_criteria(c)) or "按规则"
        return {"metric": "逐项", "value": exact_measured(c), "how": "", "thr": t, "short_thr": t}
    if kind == "exact":
        cnt = exact_counts(c)
        msg = exact_measured(c)
        val = f"{cnt[0]:,} / {cnt[1]:,} 不同" if cnt else (msg[:28] + "…" if len(msg) > 28 else msg)
        return {"metric": "逐位", "value": val, "how": "", "thr": "逐位", "short_thr": ""}
    return {"metric": "只诊断", "value": exact_measured(c), "how": "", "thr": "只报，不判", "short_thr": ""}


def ref_threshold_text(check_id: str) -> str:
    """参考档门槛的短写法（门槛栏用）：「rel_L2 < 0.01、超差 < 1%、均值差 < 0.01」；没有 → 「无」。"""
    _, g = gate_for(check_id)
    if g is None:
        return "不适用（受路由影响）" if exclusion_reason(check_id) else "无"
    out = []
    for q in g["criteria"]:
        if q.get("optional"):
            continue
        lab = q.get("short", q["label"])
        out.append(
            f"无 {lab}"
            if q["op"] == "==" and q["value"] == 0
            else f"{lab} {OP[q['op']]} {q.get('show', fmt(q['value']))}"
        )
    return "、".join(out)


# ---------------------------------------------------------------- 参考档


@functools.lru_cache(maxsize=1)
def reference_gates() -> dict:
    return yaml.safe_load(GATES_FILE.read_text())["gates"]


def gate_for(check_id: str):
    for name, g in reference_gates().items():
        if any(fnmatch.fnmatchcase(check_id, p) for p in g["applies"]):
            if excluded(check_id, g):
                return None, None
            return name, g
    return None, None


def excluded(check_id: str, g: dict) -> bool:
    return any(fnmatch.fnmatchcase(check_id, p) for p in g.get("exclude") or [])


def exclusion_reason(check_id: str) -> str | None:
    """这条检查落在某一档的 applies 里、又被它的 exclude 排除时，返回不适用的原因。"""
    for g in reference_gates().values():
        if any(fnmatch.fnmatchcase(check_id, p) for p in g["applies"]) and excluded(check_id, g):
            return g.get("exclude_reason") or "参考档不适用"
    return None


@dataclass
class RefVerdict:
    status: str  # pass / partial / fail / missing / none
    text: str  # 「rel_L2 < 0.01：6.88e-3（0.688×）→ 过（超差元素比例、均值差未记录）」

    @property
    def sym(self) -> str:
        return REF_SYM[self.status]


NONE = RefVerdict("none", "无对应门槛")


def _ref_value(c: dict, key: str, basis: bool = True):
    """basis：这类检查的普通指标就是参考档口径（被测对参照）；否则只认 vs_ref_<键>（op_replay 另记的被测对参照）。"""
    m = c.get("metrics") or {}
    if not basis:
        v = m.get(f"vs_ref_{key}")
        return float(v) if _num(v) else None
    if key == "nonfinite_steps":
        v = m.get("nonfinite_steps")
        if isinstance(v, list):
            return len(v)
        if c.get("type") in SERIES_TYPES and measured(m, "signed_mean") is not None:
            return (
                0  # series 检查遇到非有限步即判 fail、不出窗口统计（janus/checks/series.py）；有窗口统计 ⇒ 窗口内没有
            )
        return None
    x = None
    for k in (f"{key}_max", f"{key}_min", key):
        if _num(m.get(k)):
            x = float(m[k])
            break
    return abs(x) if x is not None and key in ("d0", "signed_mean") else x


OP = {"<": "<", "<=": "≤", ">=": "≥", "==": "="}


def _thr(q: dict) -> str:
    return f"{q['label']} {OP[q['op']]} {q.get('show', fmt(q['value']))}"


def _passes(x, op, v) -> bool:
    return {"<": x < v, "<=": x <= v, ">=": x >= v, "==": x == v}[op]


def reference_verdict(c: dict, only: str | None = None) -> RefVerdict:
    """按参考档判一条检查。only：只看这一个指标（逐级大表里 L4 逐步量一行一个指标）。"""
    _, g = gate_for(c.get("id", ""))
    if g is None:
        why = exclusion_reason(c.get("id", ""))
        return RefVerdict("none", why) if why else NONE
    crit = g["criteria"]
    if only is not None:
        crit = [q for q in crit if q["metric"] == only]
        if not crit:
            return RefVerdict("none", "这个量无对应门槛")
    basis = c.get("type") in g["basis"]
    m = c.get("metrics") or {}
    if not basis and not any(k.startswith("vs_ref_") for k in m):
        shown = [_thr(q) for q in crit if not q.get("optional")]
        return RefVerdict("missing", f"{'、'.join(shown)}：所需指标未记录（{g['basis_note']}）")
    parts, missing, fails, judged = [], [], 0, 0
    for q in crit:
        x = _ref_value(c, q["metric"], basis)
        thr = _thr(q)
        if x is None:
            if not q.get("optional"):
                missing.append(q["label"])
            continue
        if q["op"] == "==":
            parts.append(
                f"{thr}：{fmt(int(x))}"
                if q["metric"] != "nonfinite_steps"
                else ("无 NaN / Inf" if x == 0 else f"{int(x)} 步非有限")
            )
        elif q["op"] == ">=":
            parts.append(f"{thr}：{x:.6f}")
        else:
            parts.append(f"{thr}：{fmt(x)}（{fmt_ratio(ratio(x, q['value']))}×）")
        if q.get("optional"):
            parts[-1] += "，只报"
            continue
        judged += 1
        fails += not _passes(x, q["op"], q["value"])
    if not judged:
        return RefVerdict("missing", f"所需指标未记录（{'、'.join(missing)}）")
    status = "fail" if fails else ("partial" if missing else "pass")
    tail = f"（{'、'.join(missing)}未记录）" if missing else ""
    head = ""
    if not basis and _num(m.get("vs_ref_cases")) and _num(m.get("cases")) and m["vs_ref_cases"] < m["cases"]:
        head = f"{int(m['vs_ref_cases'])} / {int(m['cases'])} 个用例有同输入的对照："  # 其余用例输入不同，不比
    where = f"（rel_L2 最大的是 `{m['vs_ref_worst']}`）" if not basis and m.get("vs_ref_worst") else ""
    return RefVerdict(status, f"{head}{'；'.join(parts)}，{'不过' if fails else '过'}{tail}{where}")


def combine_status(statuses: list[str]) -> str:
    """一格选中多条检查时的参考档：有一条不过 → 不过；全部未记录 → 未记录；判了的都过但有未记录 / 部分 → 部分；全过 → 过。"""
    judged = [s for s in statuses if s != "none"]
    if not judged:
        return "none"
    if "fail" in judged:
        return "fail"
    if all(s == "missing" for s in judged):
        return "missing"
    if any(s in ("missing", "partial") for s in judged):
        return "partial"
    return "pass"


def note(measured_text: str, strict_text: str, ref: RefVerdict | None) -> str:
    """说明栏的统一写法：实测｜严格档｜参考档。ref 是 None 表示参考档不适用（演练格）。"""
    out = f"实测 {measured_text}｜严格档 {strict_text}"
    return out + (f"｜参考档 {ref.text}" if ref is not None else "")
