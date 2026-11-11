"""检查框架：结果、门槛判定、注册表。字段与 docs/reference/result-schema.md 一致。"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

PASS, FAIL, INCONCLUSIVE, ERROR = "pass", "fail", "inconclusive", "error"
MIN_NOISE_REFS = 3

REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        if name in REGISTRY:
            raise ValueError(f"检查类型重复注册: {name}")
        REGISTRY[name] = fn
        return fn
    return deco


@dataclass
class Threshold:
    """门槛。kind: exact / numeric / calibrated。

    numeric：values 是 指标 → 上限（cosine 这类越大越好的指标写 {"cosine": {"min": 0.999}}），必须有 source。
    calibrated：noise 是 指标 → 各条噪声参照上的值列表。
    """
    kind: str
    values: dict = field(default_factory=dict)
    source: str | None = None
    noise: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ("exact", "numeric", "calibrated"):
            raise ValueError(f"门槛写法不合法: {self.kind}")
        if self.kind == "numeric" and not self.source:
            raise ValueError("数值门槛必须写 source")


@dataclass
class CheckResult:
    id: str
    type: str
    level: str
    hard: bool = True
    status: str = ERROR
    threshold: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    r1: float | None = None
    r2: float | None = None
    worst: list = field(default_factory=list)
    coverage: dict | None = None
    evidence: list = field(default_factory=list)
    details: list = field(default_factory=list)   # 逐单元明细；写结果时单独落盘
    message: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("details")
        return {k: v for k, v in d.items() if v is not None}


def is_higher_better(metric: str) -> bool:
    """越大越好的指标（cosine、*_frac）：门槛是下限，calibrated 先换成「离 1 的距离」。"""
    return metric in ("cosine", "zero_frac", "exact_match_frac", "set_frac", "order_frac") or metric.endswith("_match_frac")


_is_higher_better = is_higher_better  # 旧名，别的模块里还有人按它导入


def judge_numeric(metrics: dict, values: dict) -> tuple[str, list[str]]:
    """数值门槛：每个指标都在门槛内才 pass。返回 (状态, 超线的指标)。"""
    bad = []
    for name, lim in values.items():
        if name not in metrics:
            return ERROR, [f"缺指标 {name}"]
        x = metrics[name]
        if isinstance(lim, dict):
            lo, hi = lim.get("min", -math.inf), lim.get("max", math.inf)
        elif is_higher_better(name):
            lo, hi = lim, math.inf
        else:
            lo, hi = -math.inf, lim
        if not (lo <= x <= hi) or (isinstance(x, float) and math.isnan(x)):
            bad.append(name)
    return (FAIL if bad else PASS), bad


def judge_calibrated(x: float, noise_values: list[float], higher_better: bool = False, floor: float = 0.0):
    """按噪声参照判。返回 (状态, 上限 N, 比值 r)。

    越小越好的指标：N = max(噪声)，r = x / N；越大越好的（cosine）换成「离 1 的距离」再判。
    r ≤ 1 pass；r > 2 fail；其间 inconclusive。参照少于 3 条时最多 inconclusive。
    floor：上限的下限，N = max(N, floor)（见 loss_series 的 signed_mean_se_k）。
    """
    vals = [v for v in noise_values if v is not None and math.isfinite(v)]
    if not vals:
        return INCONCLUSIVE, None, None
    if higher_better:
        x, vals = 1.0 - x, [1.0 - v for v in vals]
    n = max(max(vals), floor)
    if n == 0.0:
        r = 0.0 if x == 0.0 else math.inf
    else:
        r = x / n
    if r > 2.0:
        status = FAIL
    elif r <= 1.0:
        status = PASS if len(vals) >= MIN_NOISE_REFS else INCONCLUSIVE
    else:
        status = INCONCLUSIVE
    return status, n, r


def worst_of(details: list[dict], metric: str, k: int = 5, higher_better: bool = False) -> list[dict]:
    rows = [d for d in details if metric in d and d[metric] is not None]
    rows.sort(key=lambda d: d[metric], reverse=not higher_better)
    return [{"name": d.get("name"), "metric": metric, "value": d[metric], "location": d.get(f"{metric}_at")}
            for d in rows[:k]]
