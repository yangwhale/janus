"""元检查：coverage（比全了没有）、poison_selftest（工具自证）。"""
from __future__ import annotations

import fnmatch

import numpy as np

from janus.checks.base import ERROR, FAIL, PASS, REGISTRY, CheckResult, register


@register("coverage")
def coverage(cid, level, expected, compared, hard=True, **_):
    """expected：期望比较的名字清单（可含通配）；compared：实际比较过的名字。缺一个就 error。"""
    res = CheckResult(cid, "coverage", level, hard, threshold={"kind": "exact"})
    exp = set()
    for pat in expected:
        hits = [c for c in compared if fnmatch.fnmatchcase(c, pat)]
        exp.update(hits if any(ch in pat for ch in "*?[") else [pat])
    missing = sorted(e for e in exp if e not in set(compared))
    if not expected:
        res.status, res.message = ERROR, "期望清单为空"
        return res
    res.coverage = {"expected": len(exp), "compared": len(set(compared) & exp), "missing": missing[:50]}
    res.status = PASS if not missing else ERROR
    res.message = "全部比到" if not missing else f"缺 {len(missing)} 个：{missing[:3]}"
    return res


# 注入方式：返回改坏后的数组（不改原数组）
def _flip_bit(x: np.ndarray) -> np.ndarray:
    y = np.array(x, copy=True)
    flat = y.reshape(-1).view(np.uint8)
    flat[flat.size // 2] ^= np.uint8(0x40)
    return y


def _set_nan(x: np.ndarray) -> np.ndarray:
    y = np.array(x, copy=True)
    if np.issubdtype(y.dtype, np.integer):
        y.reshape(-1)[0] += 1
    else:
        y.reshape(-1)[y.size // 3] = np.nan
    return y


def _shift_one(x: np.ndarray) -> np.ndarray:
    return np.roll(np.array(x, copy=True), 1, axis=0)


def _scale(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(np.asarray(x).dtype, np.integer):
        return np.asarray(x) + 1
    return (np.asarray(x).astype(np.float64) * 1.5).astype(np.asarray(x).dtype)


INJECTIONS = {"flip_bit": _flip_bit, "nan": _set_nan, "shift": _shift_one, "scale": _scale}


@register("poison_selftest")
def poison_selftest(cid, level, target_type: str, pairs, target_kwargs: dict, injections=None, hard=True, **_):
    """对目标检查类型做注入：先确认干净数据（被测 = 参照）判 pass，再逐种注入，必须判 fail。"""
    res = CheckResult(cid, "poison_selftest", level, hard, threshold={"kind": "exact"})
    if target_type not in REGISTRY:
        res.status, res.message = ERROR, f"未知类型 {target_type}"
        return res
    fn = REGISTRY[target_type]
    clean = fn(f"{cid}.clean", level, [(n, b, b) for n, _, b in pairs], **target_kwargs)
    rows = [{"name": "clean", "status": clean.status}]
    ok = clean.status == PASS
    for inj in injections or list(INJECTIONS):
        poisoned = [(n, INJECTIONS[inj](b), b) for n, _, b in pairs]
        r = fn(f"{cid}.{inj}", level, poisoned, **target_kwargs)
        rows.append({"name": inj, "status": r.status})
        ok = ok and r.status in (FAIL, ERROR)
    res.details = rows
    res.metrics = {"cases": len(rows)}
    res.status = PASS if ok else FAIL
    res.message = "干净数据通过、每种注入都被发现" if ok else f"自证失败：{rows}"
    return res
