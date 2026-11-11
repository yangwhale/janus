"""其余检查类型：optimizer_step、router_stats、config_diff、data_fingerprint、checkpoint_equal、
replica_consistency、overlap_replay、artifact_check、run_complete、health_signals。"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np

from janus.checks.base import ERROR, FAIL, PASS, CheckResult, Threshold, register
from janus.checks.series import judge_series, parse_window, window_stats
from janus.checks.tensor import tensor_tolerance


@register("optimizer_step")
def optimizer_step(cid, level, pairs, threshold: Threshold, lr: float, hard=True, metrics=None, **_):
    """更新后参数逐参数比（从 golden 的 FP32 主权重与梯度起跑一步之后）。所选步学习率必须 > 0。"""
    if not (lr is not None and lr > 0):
        r = CheckResult(cid, "optimizer_step", level, hard, status=ERROR,
                        message=f"所选步学习率 {lr} 不大于 0：参数不会动，这一步验证不了优化器")
        return r
    r = tensor_tolerance(cid, level, pairs, metrics or ["rel_l2", "max_abs"], threshold, hard=hard)
    r.type = "optimizer_step"
    r.metrics["lr"] = lr
    return r


@register("router_stats")
def router_stats(cid, level, test: dict, ref: dict, threshold: Threshold, hard=True, params=None, **_):
    """test / ref：{统计名（如 router.bias_std.3）: {步号: 值}}。每个统计按 loss_series 的判定量比，取最差。"""
    params = params or {}
    res = CheckResult(cid, "router_stats", level, hard)
    missing = sorted(set(ref) - set(test))
    if not ref or missing:
        res.status, res.message = ERROR, f"缺统计：{missing[:5]}" if missing else "没有统计"
        return res
    window, fs = parse_window(params["window"]), int(params.get("first_step", 0))
    worst = {"signed_mean": 0.0, "mare": 0.0, "max_step": 0.0}
    for name in sorted(ref):
        vals, raw = window_stats(test[name], ref[name], window, params.get("denominator", "golden"), fs)
        res.details.append({"name": name, **raw})
        for k in worst:
            worst[k] = max(worst[k], vals[k])
    res.metrics = {f"{k}_max": v for k, v in worst.items()} | {"stats": len(ref)}
    judge_series(res, worst, threshold, list(worst))
    return res


@register("config_diff")
def config_diff(cid, level, test_cfg: dict, ref_cfg: dict, mapping: list, hard=True, **_):
    """mapping 每项两种写法：
    - 两边对照：{test: 键路径, ref: 键路径, must_equal: bool, transform: 可选 'float'|'str'}。
      must_equal 的字段不同即 fail；其余不同只列出。键缺失：must_equal 的判 error。
      两边都是字典时逐子键比，ignore（通配列表）匹配到的子键不比，details 只列不同的子键。
    - 等于给定值：{test: 键路径, equals: 值, ref: 可选键路径}。被测与参照两边都必须等于给定值（ref 省略时与 test 同路径），
      任一边不等即 fail，任一边缺键即 error。用于 dropout = 0、随机路由关这类「两边相同也不够，必须是这个值」的断言。"""
    res = CheckResult(cid, "config_diff", level, hard, threshold={"kind": "exact"})

    def get(d, path):
        cur = d
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                raise KeyError(path)
            cur = cur[k]
        return cur

    bad, diff, missing = [], [], []
    n_equals = 0
    for m in mapping:
        if "equals" in m:
            n_equals += 1
            want = m["equals"]
            for side, cfg, path in (("test", test_cfg, m["test"]), ("ref", ref_cfg, m.get("ref", m["test"]))):
                try:
                    v = get(cfg, path)
                except KeyError as e:
                    missing.append(f"{side}:{e}")
                    continue
                if m.get("transform") == "float":
                    v, want_v = float(v), float(want)
                else:
                    want_v = want
                same = v == want_v
                row = {"name": f"{side}:{path} == {want!r}", side: v, "expected": want, "equal": same, "must_equal": True}
                res.details.append(row)
                if not same:
                    bad.append(row["name"])
            continue
        try:
            a, b = get(test_cfg, m["test"]), get(ref_cfg, m["ref"])
        except KeyError as e:
            (missing if m.get("must_equal") else diff).append(str(e))
            continue
        if m.get("transform") == "float":
            a, b = float(a), float(b)
        if isinstance(a, dict) and isinstance(b, dict):
            # 整个字典比（如展开后的完整配置、运行进程的环境变量）：ignore 里的通配匹配到的子键不比（运行名、主机名、路径），
            # 其余子键逐个比，details 只列不同的子键，避免把几百个相同的键都写进结果
            import fnmatch
            ign = list(m.get("ignore") or [])
            keep = lambda k: not any(fnmatch.fnmatchcase(str(k), pat) for pat in ign)  # noqa: E731
            sub = sorted(k for k in set(a) | set(b) if keep(k) and (k not in a or k not in b or a[k] != b[k]))
            same = not sub
            row = {"name": f"{m['test']} ↔ {m['ref']}", "equal": same, "must_equal": bool(m.get("must_equal")),
                   "keys": len([k for k in set(a) | set(b) if keep(k)]), "ignored": ign,
                   "differ": [{"key": k, "test": a.get(k, "<缺>"), "ref": b.get(k, "<缺>")} for k in sub[:20]],
                   "differ_count": len(sub)}
            res.details.append(row)
            if not same:
                (bad if m.get("must_equal") else diff).append(f"{row['name']}（{len(sub)} 个子键：{sub[:3]}）")
            continue
        same = a == b
        row = {"name": f"{m['test']} ↔ {m['ref']}", "test": a, "ref": b, "equal": same, "must_equal": bool(m.get("must_equal"))}
        res.details.append(row)
        if not same:
            (bad if m.get("must_equal") else diff).append(row["name"])
    res.metrics = {"fields": len(mapping), "must_equal_differ": len(bad), "other_differ": len(diff), "missing": len(missing),
                   "equals_rules": n_equals}
    res.status = ERROR if missing else (FAIL if bad else PASS)
    res.message = f"缺字段 {missing}" if missing else (f"必须相同（或等于给定值）的字段不符：{bad}" if bad
                                                    else f"必须相同的字段都相同；其他不同 {len(diff)} 项")
    return res


FINGERPRINT_FIELDS = ("tokens", "labels", "loss_mask", "position_ids")


def _steps_of(units: dict) -> set:
    """单元名「步.r<rank>.m<mb>.<字段>」或「步.nonpad_total」里的步号集合；步号不是整数的单元不计。"""
    out = set()
    for k in units:
        head = str(k).split(".", 1)[0]
        if head.lstrip("-").isdigit():
            out.add(int(head))
    return out


def _ranges(steps: list) -> str:
    """[4, 5, 6, 9] → 「4–6、9」。"""
    parts, i = [], 0
    while i < len(steps):
        j = i
        while j + 1 < len(steps) and steps[j + 1] == steps[j] + 1:
            j += 1
        parts.append(str(steps[i]) if i == j else f"{steps[i]}–{steps[j]}")
        i = j + 1
    return "、".join(parts)


@register("data_fingerprint")
def data_fingerprint(cid, level, test_tokens: dict, ref_tokens: dict, hard=True, first_loss=None, missing=None,
                     fields=None, **_):
    """test_tokens / ref_tokens：{单元名: token 数组或 sha256}，单元名形如「步.r<rank>.m<microbatch>.<字段>」。
    逐位相同；可选 first_loss=(被测, 参照) 首步 loss 逐位比。
    missing：取数时发现缺失的期望字段（「运行:单元.字段」），非空即 error——缺字段等于少比了，不能报一致。
    fields：期望字段，用来在 metrics.units_per_field 里数每个字段比了多少单元。"""
    res = CheckResult(cid, "data_fingerprint", level, hard, threshold={"kind": "exact"})
    keys = sorted(set(test_tokens) | set(ref_tokens), key=str)
    if missing:
        res.status = ERROR
        res.metrics = {"missing_fields": len(missing)}
        res.message = f"期望字段缺失 {len(missing)} 处：{missing[:3]}"
        return res
    if not keys:
        res.status, res.message = ERROR, "没有 token"
        return res
    # 缺整步：一侧有某步的记录、另一侧这一步一条都没有 → error（步数不够是少比了，不是 token 不同）
    steps_t, steps_r = _steps_of(test_tokens), _steps_of(ref_tokens)
    miss = {"test": sorted(steps_r - steps_t), "ref": sorted(steps_t - steps_r)}
    if miss["test"] or miss["ref"]:
        res.status = ERROR
        res.metrics = {"missing_steps": {k: v for k, v in miss.items() if v}}
        res.message = "；".join(f"{'被测' if k == 'test' else '参照'}缺整步 {len(v)} 步：{_ranges(v)}"
                               for k, v in miss.items() if v) + "（另一侧有记录；步数不够，不是 token 不同）"
        return res
    if fields:
        res.metrics["units_per_field"] = {f: sum(1 for k in keys if str(k).endswith("." + f)) for f in fields}
    differ = []
    for k in keys:
        a, b = test_tokens.get(k), ref_tokens.get(k)
        same = a is not None and b is not None and np.array_equal(np.asarray(a), np.asarray(b))
        h = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16] if a is not None else None
        res.details.append({"name": str(k), "equal": same, "sha256_16": h})
        if not same:
            differ.append(str(k))
    loss_ok = True
    if first_loss is not None:
        loss_ok = float(first_loss[0]) == float(first_loss[1])
        res.metrics["first_loss"] = {"test": first_loss[0], "ref": first_loss[1]}
    res.metrics.update({"units": len(keys), "differ": len(differ)})
    res.status = PASS if not differ and loss_ok else FAIL
    res.message = "token 与首步 loss 逐位相同" if res.status == PASS else (f"token 不同：{differ[:3]}" if differ else "首步 loss 不同")
    return res


@register("checkpoint_equal")
def checkpoint_equal(cid, level, test_dir=None, ref_dir=None, hard=True, pattern="**/*", manifests=None, **_):
    """两份检查点目录逐文件、逐字节比（不抽样）。文件集合不同也 fail。
    manifests=(被测清单, 参照清单)：改比参数清单 {分组: {参数名: {sha256, …}}}（抓数时每个参数全量算的 sha256），逐参数报。"""
    res = CheckResult(cid, "checkpoint_equal", level, hard, threshold={"kind": "exact"})
    if manifests is not None:
        mt, mr = manifests
        keys_t = {f"{g}/{n}" for g, d in mt.items() for n in d}
        keys_r = {f"{g}/{n}" for g, d in mr.items() for n in d}
        only = sorted(keys_t ^ keys_r)
        differ = []
        for k in sorted(keys_t & keys_r):
            g, n = k.split("/", 1)
            ht, hr = mt[g][n]["sha256"], mr[g][n]["sha256"]
            if ht != hr:
                differ.append(k)
                res.details.append({"name": k, "equal": False, "test": ht[:16], "ref": hr[:16]})
        res.metrics = {"params": len(keys_t & keys_r), "differ": len(differ), "only_one_side": len(only)}
        if not keys_r:
            res.status, res.message = ERROR, "参照清单为空"
            return res
        res.status = PASS if not differ and not only else FAIL
        res.message = "全部参数逐位相同" if res.status == PASS else f"不同 {len(differ)} 个：{differ[:3]}；只在一边 {only[:3]}"
        return res
    t, r = Path(test_dir), Path(ref_dir)
    if not t.is_dir() or not r.is_dir():
        res.status, res.message = ERROR, "目录不存在"
        return res
    tf = {p.relative_to(t).as_posix() for p in t.glob(pattern) if p.is_file()}
    rf = {p.relative_to(r).as_posix() for p in r.glob(pattern) if p.is_file()}
    only = sorted(tf ^ rf)
    differ = []
    for f in sorted(tf & rf):
        ht = hashlib.sha256((t / f).read_bytes()).hexdigest()
        hr = hashlib.sha256((r / f).read_bytes()).hexdigest()
        if ht != hr:
            differ.append(f)
        res.details.append({"name": f, "equal": ht == hr, "sha256": ht})
    res.metrics = {"files": len(tf & rf), "differ": len(differ), "only_one_side": len(only)}
    res.status = PASS if not differ and not only else FAIL
    res.message = "全部逐字节相同" if res.status == PASS else f"不同 {differ[:3]}，只在一边 {only[:3]}"
    return res


@register("replica_consistency")
def replica_consistency(cid, level, replicas: dict, hard=True, **_):
    """replicas：{副本名: 数组或校验和字符串}。全部相同才 pass。
    分组模式：{对象名: {副本名: 数组或校验和}}，逐对象判，任一对象副本不一致即 fail。"""
    res = CheckResult(cid, "replica_consistency", level, hard, threshold={"kind": "exact"})
    if replicas and all(isinstance(v, dict) for v in replicas.values()):
        bad = []
        for obj, reps in sorted(replicas.items()):
            if len(reps) < 2:
                continue
            r = replica_consistency(cid, level, reps, hard)
            if r.status != PASS:
                bad.append(obj)
                res.details.append({"name": obj, "groups": r.message})
        n = sum(1 for v in replicas.values() if len(v) >= 2)
        res.metrics = {"objects": n, "inconsistent": len(bad)}
        if n == 0:
            res.status, res.message = ERROR, "没有至少两个副本的对象"
            return res
        res.status = PASS if not bad else FAIL
        res.message = f"{n} 个对象副本全部一致" if not bad else f"{len(bad)} 个对象副本不一致：{bad[:3]}"
        return res
    if len(replicas) < 2:
        res.status, res.message = ERROR, "副本少于 2 个"
        return res

    def digest(v):
        return v if isinstance(v, str) else hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
    ds = {k: digest(v) for k, v in replicas.items()}
    groups = {}
    for k, h in ds.items():
        groups.setdefault(h, []).append(k)
    res.details = [{"name": k, "sha256": h} for k, h in ds.items()]
    res.metrics = {"replicas": len(ds), "distinct": len(groups)}
    res.status = PASS if len(groups) == 1 else FAIL
    res.message = "副本一致" if len(groups) == 1 else f"{len(groups)} 组不同：{list(groups.values())[:3]}"
    return res


@register("overlap_replay")
def overlap_replay(cid, level, before: dict, after: dict, hard=True, fields=None, **_):
    """重启前后重叠步的原始标量逐位比。before / after：{步号: {字段: 值}}。报第一处不同的步。"""
    res = CheckResult(cid, "overlap_replay", level, hard, threshold={"kind": "exact"})
    steps = sorted(set(before) & set(after))
    if not steps:
        res.status, res.message = ERROR, "没有重叠步"
        return res
    first = None
    for s in steps:
        keys = fields or sorted(set(before[s]) & set(after[s]))
        diff = [k for k in keys if before[s].get(k) != after[s].get(k)
                and not (isinstance(before[s].get(k), float) and math.isnan(before[s][k]) and math.isnan(after[s].get(k, 0.0)))]
        if diff and first is None:
            first = (s, diff)
    res.metrics = {"overlap_steps": len(steps), "first_differ_step": first[0] if first else None,
                   "first_differ_fields": first[1] if first else []}
    res.status = PASS if first is None else FAIL
    res.message = "重叠步全部逐位相同" if first is None else f"第 {first[0]} 步起不同：{first[1][:5]}"
    return res


HLO_METADATA = re.compile(r",?\s*metadata=\{[^{}]*\}")
HLO_STACK = re.compile(r",?\s*stack_frame_id=\d+")


def hlo_fingerprint(text: str) -> str:
    """编译产物指纹（R21）：去掉 metadata={…}（op_name、源码位置）与 stack_frame_id 之后的 sha256。
    改源码行号、改函数名不改指纹；改融合、调度、算子、精度都改指纹。"""
    t = HLO_STACK.sub("", HLO_METADATA.sub("", text))
    return hashlib.sha256(t.encode()).hexdigest()


@register("artifact_check")
def artifact_check(cid, level, text: str, rules: list, hard=True, ref_text=None, fingerprint=False, ignore_lines=None, **_):
    """编译产物文本上的计数检查。rules：[{name, regex, min?, max?, match_ref?}]，计数不在 [min, max] 内即 fail；
    match_ref: true 的规则另要求计数与参照产物（ref_text，日常回归里是 LKG 的同一份文件）上的计数相等。
    fingerprint: true 时另比去掉元数据后的指纹（hlo_fingerprint）：被测与参照不同即 fail（这一条通常 hard: false，只报）。
    ignore_lines：正则列表，匹配到的整行在算指纹之前去掉（两边都去，计数规则不受影响），只用于和被测平台无关的行。"""
    res = CheckResult(cid, "artifact_check", level, hard, threshold={"kind": "numeric", "values": rules,
                                                                   "source": "目录文件里的规则"})
    if any(r.get("match_ref") for r in rules) and ref_text is None:
        res.status, res.message = ERROR, "有 match_ref 规则但没有参照产物（目录里要写 compare: lkg）"
        return res
    if fingerprint and ref_text is None:
        res.status, res.message = ERROR, "fingerprint 要参照产物（目录里要写 compare: lkg）"
        return res
    if ignore_lines:
        pats = [re.compile(x) for x in ignore_lines]

        def drop(t):
            return "\n".join(line for line in t.splitlines() if not any(p.search(line) for p in pats))
        ftext, fref = drop(text), (drop(ref_text) if ref_text is not None else None)
    else:
        ftext, fref = text, ref_text
    bad = []
    for rule in rules:
        n = len(re.findall(rule["regex"], text))
        lo, hi = rule.get("min", 0), rule.get("max", math.inf)
        ok = lo <= n <= hi
        row = {"name": rule["name"], "count": n, "min": lo, "max": hi}
        if rule.get("match_ref"):
            row["ref_count"] = len(re.findall(rule["regex"], ref_text))
            ok = ok and n == row["ref_count"]
        row["ok"] = ok
        res.details.append(row)
        if not ok:
            bad.append(f"{rule['name']}={n}" + (f"（参照 {row['ref_count']}）" if "ref_count" in row else ""))
    res.metrics = {"rules": len(rules), "bad": len(bad)}
    if fingerprint:
        ft, fr = hlo_fingerprint(ftext), hlo_fingerprint(fref)
        res.metrics.update({"fingerprint": ft[:16], "ref_fingerprint": fr[:16], "fingerprint_equal": ft == fr})
        if ignore_lines:
            res.metrics["ignored_lines"] = {"test": len(text.splitlines()) - len(ftext.splitlines()),
                                            "ref": len(ref_text.splitlines()) - len(fref.splitlines())}
        if ft != fr:
            bad.append("指纹不同")
            lt = set(HLO_STACK.sub("", HLO_METADATA.sub("", ftext)).splitlines())
            lr = set(HLO_STACK.sub("", HLO_METADATA.sub("", fref)).splitlines())
            res.metrics.update({"lines_only_test": len(lt - lr), "lines_only_ref": len(lr - lt)})
            res.details += [{"name": "只在被测", "line": x[:300]} for x in sorted(lt - lr)[:10]]
            res.details += [{"name": "只在参照", "line": x[:300]} for x in sorted(lr - lt)[:10]]
    res.status = PASS if not bad else FAIL
    res.message = ("全部规则满足" + ("，指纹与参照相同" if fingerprint else "")) if not bad else f"不满足：{bad}"
    return res


@register("run_complete")
def run_complete(cid, level, steps: list, meta: dict, hard=True, params=None, **_):
    """运行完整（日常回归 R01）。steps：metrics.jsonl 里按文件顺序的步号；meta：运行的 meta.json。
    params.steps（计划步数，必填）、params.first（首步，默认 1）。fail 的条件：重复步（重启后追加写会出现）、缺步、
    步号不是 first…first+steps−1、meta.steps 与计划不符、meta.restarts（有就读）不为 0。DONE 由读运行目录时保证。"""
    params = params or {}
    res = CheckResult(cid, "run_complete", level, hard, threshold={"kind": "exact"})
    if "steps" not in params:
        res.status, res.message = ERROR, "params.steps（计划步数）必填"
        return res
    n, first = int(params["steps"]), int(params.get("first", 1))
    want = list(range(first, first + n))
    seen, dup = set(), []
    for s in steps:
        (dup.append(s) if s in seen else seen.add(s))
    missing = sorted(set(want) - seen)
    extra = sorted(seen - set(want))
    ms = meta.get("steps") or {}
    meta_bad = [] if (ms.get("first") == first and ms.get("last") == first + n - 1) else [f"meta.steps={ms}"]
    restarts = meta.get("restarts", 0)
    res.metrics = {"planned": n, "logged": len(steps), "duplicate": len(dup), "missing": len(missing),
                   "extra": len(extra), "restarts": restarts}
    problems = []
    if dup:
        problems.append(f"重复步 {sorted(set(dup))[:5]}（中途重启后追加写？）")
    if missing:
        problems.append(f"缺 {len(missing)} 步：{_ranges(missing)}")
    if extra:
        problems.append(f"计划外的步 {extra[:5]}")
    if meta_bad:
        problems.append(f"{meta_bad[0]} 与计划 {first}…{first + n - 1} 不符")
    if restarts:
        problems.append(f"中途重启 {restarts} 次")
    res.status = PASS if not problems else FAIL
    res.message = f"{n} 步全部完成、无重复、无重启" if not problems else "；".join(problems)
    return res


@register("health_signals")
def health_signals(cid, level, series: dict, hard=True, params=None, **_):
    """series：{字段: {步号: 值}}（如 lm_loss、raw_grad_norm、param_norm）。
    非有限值 → fail；某步相对前 window 步中位数的倍数超过 spike_ratio（默认 10）→ fail。"""
    params = params or {}
    win, ratio = int(params.get("window", 20)), float(params.get("spike_ratio", 10.0))
    res = CheckResult(cid, "health_signals", level, hard, threshold={"kind": "numeric", "values": {"spike_ratio": ratio},
                                                                    "source": params.get("source", "规格默认值")})
    empty = [name for name, ser in series.items() if not ser]
    if not series or empty:                    # 字段拼错或这次运行没记：没有数据不能判「健康」
        res.status, res.message = ERROR, f"没有数据的字段：{empty or '（一个字段都没选）'}"
        return res
    problems = []
    for name, ser in series.items():
        steps = sorted(ser)
        vals = [float(ser[s]) for s in steps]
        for i, (s, v) in enumerate(zip(steps, vals)):
            if not math.isfinite(v):
                problems.append(f"{name}@{s}=非有限")
                break
            if i >= win:
                med = float(np.median(vals[i - win:i]))
                if med > 0 and v / med > ratio:
                    problems.append(f"{name}@{s}={v:.3g}（中位数 {med:.3g} 的 {v / med:.1f} 倍）")
                    break
    res.metrics = {"fields": len(series), "problems": len(problems)}
    res.status = PASS if not problems else FAIL
    res.message = "健康" if not problems else "；".join(problems[:5])
    return res


@register("determinism")
def determinism(cid, level, series_a: dict, series_b: dict, pairs=None, hard=True, **_):
    """同配置两次运行：指定标量序列逐步逐位相同，且（给了 pairs 时）张量逐位相同。用于基线可复现与「观测不改数值」。"""
    res = CheckResult(cid, "determinism", level, hard, threshold={"kind": "exact"})
    empty = [f for f, sa in series_a.items() if not sa]
    if not series_a or empty:                  # 被测这个字段一步都没有：循环什么都不比，不能判「逐位可复现」
        res.status, res.message = ERROR, f"被测没有数据的字段：{empty}" if empty else "没有序列"
        return res
    bad = []
    for field, sa in series_a.items():
        sb = series_b.get(field, {})
        for s in sorted(sa):
            if s not in sb or sa[s] != sb[s]:
                bad.append(f"{field}@{s}")
                break
    tensors_differ = 0
    if pairs:
        tensors_differ = sum(1 for _, a, b in pairs if not (np.asarray(a).dtype == np.asarray(b).dtype
                                                           and np.array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8))))
    res.metrics = {"fields": len(series_a), "first_differ": bad[:5], "tensors": len(pairs or []), "tensors_differ": tensors_differ}
    res.status = PASS if not bad and tensors_differ == 0 else FAIL
    res.message = "逐位可复现" if res.status == PASS else f"不可复现：{bad[:3]}，张量不同 {tensors_differ}"
    return res


def _norm_dtype(d: str) -> str:
    """dtype 名归一：torch.bfloat16 / BF16 / bfloat16 → bfloat16；F32 / torch.float32 → float32。"""
    from janus.dtypes import ST_TO_NP
    d = str(d)
    if d.startswith("torch."):
        d = d[len("torch."):]
    if d in ST_TO_NP:
        return str(ST_TO_NP[d])
    return {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16", "half": "float16", "float": "float32"}.get(d, d)


@register("dtype_manifest")
def dtype_manifest(cid, level, manifests: dict, rules: list, hard=True, **_):
    """参数 / buffer 的 dtype 清单检查。manifests：{标签（如 init/rank0）: {名字: {dtype, …}}}（params_sha 清单）。
    rules：[{pattern, dtype, exclude?}] 按顺序取第一条匹配的规则（fnmatch，exclude 匹配的不算命中），dtype 不符 → fail；
    没有任何规则覆盖的名字 → error（规则没写全就不能说清单合格）。清单为空 → error。"""
    import fnmatch
    res = CheckResult(cid, "dtype_manifest", level, hard, threshold={"kind": "exact"})
    names = [(tag, n, ent) for tag, d in manifests.items() for n, ent in d.items()]
    if not names:
        res.status, res.message = ERROR, "dtype 清单为空"
        return res
    bad, unmatched, hits = [], [], [0] * len(rules)
    for tag, n, ent in names:
        rule_i = next((i for i, r in enumerate(rules) if fnmatch.fnmatchcase(n, r["pattern"])
                       and not (r.get("exclude") and fnmatch.fnmatchcase(n, r["exclude"]))), None)
        if rule_i is None:
            unmatched.append(f"{tag}/{n}")
            continue
        hits[rule_i] += 1
        got, want = _norm_dtype(ent.get("dtype")), _norm_dtype(rules[rule_i]["dtype"])
        if got != want:
            bad.append(f"{tag}/{n}")
            res.details.append({"name": f"{tag}/{n}", "dtype": got, "expected": want, "rule": rules[rule_i]["pattern"]})
    res.metrics = {"entries": len(names), "wrong_dtype": len(bad), "unmatched": len(unmatched),
                   "rule_hits": {r["pattern"] + (f" !{r['exclude']}" if r.get("exclude") else ""): h for r, h in zip(rules, hits)}}
    if unmatched:
        res.status, res.message = ERROR, f"{len(unmatched)} 个名字没有规则覆盖：{unmatched[:3]}"
    elif bad:
        res.status = FAIL
        res.message = f"dtype 不符 {len(bad)} 个：" + "；".join(f"{d['name']} 是 {d['dtype']}，应为 {d['expected']}" for d in res.details[:3])
    else:
        res.status, res.message = PASS, f"{len(names)} 项 dtype 全部符合规则"
    return res


FAKE_FP32_MAX_FRAC = 0.99
FAKE_FP32_SOURCE = ("推断：真 FP32 张量的低 16 位近似均匀，全为 0 的概率约 2⁻¹⁶ ≈ 1.5e-5；"
                    "BF16 舍入后存成 FP32 的张量这一比例是 1。0.99 两头都留足余量（容得下少量元素恰好落在 BF16 网格上）")


@register("fake_fp32")
def fake_fp32(cid, level, tensors, hard=True, params=None, **_):
    """「假 FP32」：标着 float32、实际只有 BF16 精度（低 16 位全 0）的张量。
    tensors：[(名字, 数组)]，只看 float32 的。每个张量在非零有限元素里数低 16 位全为 0 的比例（±0 本来就全 0，不算），
    超过 params.max_frac（默认 0.99，source 见 FAKE_FP32_SOURCE）→ fail。没有 float32 张量 → error。"""
    params = params or {}
    lim = float(params.get("max_frac", FAKE_FP32_MAX_FRAC))
    source = params.get("source") or (FAKE_FP32_SOURCE if "max_frac" not in params else None)
    res = CheckResult(cid, "fake_fp32", level, hard, threshold={"kind": "numeric", "values": {"low16_zero_frac": lim},
                                                               "source": source})
    if source is None:
        res.status, res.message = ERROR, "改了 max_frac 必须写 params.source"
        return res
    rows, skipped = [], []
    for name, a in tensors:
        a = np.asarray(a)
        if a.dtype != np.float32:
            skipped.append(name)
            continue
        u = a.reshape(-1).view(np.uint32)
        keep = (u & np.uint32(0x7FFFFFFF)) != 0                      # 去掉 ±0
        keep &= (u & np.uint32(0x7F800000)) != np.uint32(0x7F800000)  # 去掉 Inf / NaN
        n = int(np.count_nonzero(keep))
        z = int(np.count_nonzero((u[keep] & np.uint32(0xFFFF)) == 0)) if n else 0
        rows.append({"name": name, "n": n, "low16_zero_frac": (z / n) if n else 0.0})
    res.details = rows
    if not rows:
        res.status, res.message = ERROR, "没有 float32 张量可查" + (f"（跳过非 float32 {len(skipped)} 个）" if skipped else "")
        return res
    bad = [r for r in rows if r["n"] and r["low16_zero_frac"] > lim]
    res.metrics = {"low16_zero_frac_max": max(r["low16_zero_frac"] for r in rows), "tensors": len(rows),
                   "fake": len(bad), "skipped_non_fp32": len(skipped)}
    res.worst = [{"name": r["name"], "metric": "low16_zero_frac", "value": r["low16_zero_frac"], "location": None}
                 for r in sorted(rows, key=lambda r: -r["low16_zero_frac"])[:5]]
    res.status = FAIL if bad else PASS
    res.message = (f"{len(bad)} 个张量疑似 BF16 精度存成 FP32：{[r['name'] for r in bad[:3]]}" if bad
                   else f"{len(rows)} 个 float32 张量低 16 位全 0 的比例最高 {res.metrics['low16_zero_frac_max']:.2e}")
    return res
