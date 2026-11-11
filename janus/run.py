"""janus run：读目录 → 取数 → 跑检查 → 写 results.json、details/、report.md。退出码见 result-schema.md。"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

from janus import __version__
from janus.catalog import Catalog, CatalogError, load
from janus.catalog import fingerprint as catalog_fingerprint
from janus.checks import REGISTRY, Threshold
from janus.checks.base import ERROR, FAIL, INCONCLUSIVE, PASS
from janus.checks.series import parse_window
from janus.checks.tensor import aggregate, tensor_row
from janus.report.markdown import render as _render_report  # 旧名：models/<模型>/frameworks/megatron/l1_ops.py 按它导入
from janus.store import Run, StoreError

# results.json 的格式版本（docs/reference/result-schema.md）。2：catalog 记内容指纹 content_sha256（content-v1），
# 改注释与说明文字不再让结果对不上；1：catalog.sha256 是目录原文件的哈希（独立脚本写的结果仍是这种）。
RESULTS_SCHEMA_VERSION = 2


def _pair_worst(job):
    """进程池里算一对噪声运行在选中张量上的各指标最差值（模块级函数，方便进程池序列化）。"""
    root, a, b, sel, metrics, tol = job
    ra, rb = Run.open(Path(root) / a), Run.open(Path(root) / b)
    step = int(sel["step"])
    rows = [tensor_row(n, ra.tensor(step, n), rb.tensor(step, n), metrics, *tol) for n in Runner._names(ra, sel)]
    out = {}
    for m in metrics:
        vals = [r[m] for r in rows if m in r]
        if vals:
            out[m] = aggregate(m, vals)
    return out


class Runner:
    def __init__(self, cat: Catalog):
        self.cat = cat
        self.root = Path(cat.runs["root"])
        self._runs: dict[str, Run] = {}
        self._rows: dict = {}

    def run(self, name: str) -> Run:
        if name not in self._runs:
            r = Run.open(self.root / name)
            r.enable_cache()
            self._runs[name] = r
        return self._runs[name]

    # ---- 取数
    def _ref_name(self, c) -> str:
        cmp = c.get("compare", "golden")
        if cmp == "lkg":
            return self.cat.runs["lkg"]
        if cmp == "self":                          # 同一次运行里比两组张量（配合 select.ref_rename）
            return self.cat.runs["test"]
        return self.cat.runs["golden"][0]

    def _tensor_pairs(self, test: Run, ref: Run, sel) -> list:
        """被测张量与参照的同名张量配对。select.ref_rename = {from, to}：参照名 = 被测名把 from 换成 to，
        用于同一次运行里比两组张量（如 main_param_after.* 对 main_param.*：跳过更新的步参数应不变；
        <名>.recompute 对 <名>：重算的前向应与首次前向逐位相同）。"""
        step = int(sel["step"])
        names = self._names(test, sel)
        rn = sel.get("ref_rename")
        conv = (lambda n: n.replace(rn["from"], rn["to"])) if rn else (lambda n: n)
        return [(n, test.tensor(step, n), ref.tensor(step, conv(n))) for n in names]

    @staticmethod
    def _test_series(test: Run, field: str) -> dict:
        """被测序列按 meta.steps.convention 换算成 golden 步号（golden_step = step + k）。"""
        return {test.to_golden_step(s): v for s, v in test.series(field).items()}

    @staticmethod
    def _step_params(test: Run) -> dict:
        """首步（换算到 golden 步号）与约定偏移 k，给序列类检查用。"""
        return {"first_step": test.to_golden_step(int(test.meta["steps"].get("first", 0))),
                "step_offset": test.to_golden_step(0)}

    def _threshold(self, c, noise=None) -> Threshold:
        th = c["threshold"]
        if th == "exact":
            return Threshold("exact")
        if th == "calibrated":
            return Threshold("calibrated", noise=noise or {})
        if th == "pending":                    # 待标定：按「没有噪声参照」判，结果最多 inconclusive（硬底线照样判 fail）
            return Threshold("calibrated", noise={})
        return Threshold("numeric", values=th, source=c.get("source"))

    @staticmethod
    def _names(run: Run, sel) -> list[str]:
        """select.name 通配选张量；select.exclude（可选，一个通配或通配列表）再剔掉一批，如专家分发缓冲区 *.experts.out。"""
        import fnmatch
        names = run.names(int(sel["step"]), sel.get("name", "*"))
        exc = sel.get("exclude") or []
        exc = [exc] if isinstance(exc, str) else list(exc)
        return [n for n in names if not any(fnmatch.fnmatchcase(n, e) for e in exc)]

    def _row(self, a: str, b: str, step: int, name: str, metrics, tol=(0.0, 0.0)) -> dict:
        """一对运行上一个张量的指标，按 (运行对, 步, 名字, 指标, 容差) 缓存：foreach 展开后同一张量会被多条检查用到。"""
        key = (a, b, step, name, tuple(metrics), tuple(tol))
        if key not in self._rows:
            self._rows[key] = tensor_row(name, self.run(a).tensor(step, name), self.run(b).tensor(step, name), metrics, *tol)
        return self._rows[key]

    @staticmethod
    def _tol(c) -> tuple[float, float]:
        """out_of_tol_ratio 的 (atol, rtol)：被测与噪声参照必须用同一把尺子。"""
        p = c.get("params") or {}
        return float(p.get("atol", 0.0)), float(p.get("rtol", 0.0))

    def _tensor_noise(self, c) -> dict:
        """张量类 calibrated：每对噪声运行按同样的指标算全体最差值。
        参照对多时（如 8 份运行两两 28 对）按参照对并行（进程池，JANUS_WORKERS 控制，默认 min(对数, 16)）。"""
        pairs = self.cat.runs.get("noise") or []
        workers = int(os.environ.get("JANUS_WORKERS", min(len(pairs), 16)))
        if workers > 1 and len(pairs) > 1:
            return self._tensor_noise_parallel(c, pairs, workers)
        sel, metrics = c["select"], c.get("metrics", ["rel_l2"])
        out = {m: [] for m in metrics}
        step = int(sel["step"])
        for a, b in self.cat.runs.get("noise") or []:
            rows = [self._row(a, b, step, n, metrics, self._tol(c)) for n in self._names(self.run(a), sel)]
            for m in metrics:
                vals = [r[m] for r in rows if m in r]
                if vals:
                    out[m].append(aggregate(m, vals))
        return out

    def _tensor_noise_parallel(self, c, pairs, workers) -> dict:
        from concurrent.futures import ProcessPoolExecutor
        sel, metrics = c["select"], c.get("metrics", ["rel_l2"])
        jobs = [(str(self.root), a, b, dict(sel), list(metrics), self._tol(c)) for a, b in pairs]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            worst = list(ex.map(_pair_worst, jobs))
        out = {m: [] for m in metrics}
        for w in worst:
            for m in metrics:
                if w.get(m) is not None:
                    out[m].append(w[m])
        return out

    def _discrete_noise(self, c) -> dict:
        """离散类 calibrated：每对噪声运行算同样的一致比例（取最低）。"""
        from janus import metrics as M
        mode = (c.get("params") or {}).get("mode", "exact")
        key = "exact_match_frac" if mode == "exact" else "set_frac"
        vals = []
        for a, b in self.cat.runs.get("noise") or []:
            pr = self._tensor_pairs(self.run(a), self.run(b), c["select"])
            if not pr:
                continue
            if mode == "exact":
                vals.append(min(M.exact_match_frac(x, y) for _, x, y in pr))
            else:
                vals.append(min(M.topk_agreement(x, y).set_frac for _, x, y in pr))
        return {key: vals}

    def execute(self, c: dict):
        fn = REGISTRY[c["type"]]
        sel, typ, hard = c["select"], c["type"], c.get("hard", True)
        common = dict(hard=hard)
        test = self.run(self.cat.runs["test"])
        evidence = [{"path": str(test.path), "sha256": test.meta_sha256()}]
        if typ in ("bitwise_equal", "ulp_distribution", "tensor_tolerance", "discrete_agreement"):
            ref = self.run(self._ref_name(c))
            evidence.append({"path": str(ref.path), "sha256": ref.meta_sha256()})
            pairs = self._tensor_pairs(test, ref, sel)
            if typ == "bitwise_equal":
                res = fn(c["id"], c["level"], pairs, **common)
            elif typ == "ulp_distribution":
                th = None if c.get("threshold") in (None, "calibrated") else self._threshold(c)
                res = fn(c["id"], c["level"], pairs, threshold=th, **common)
            elif typ == "tensor_tolerance":
                noise = self._tensor_noise(c) if c["threshold"] == "calibrated" else None
                res = fn(c["id"], c["level"], pairs, c.get("metrics", ["rel_l2", "cosine", "max_abs"]),
                         self._threshold(c, noise), params=c.get("params"), **common)
            else:
                if (c.get("params") or {}).get("reference") == "fp64_replay":
                    pairs = self._router_fp64_pairs(test, sel)
                    evidence.pop()
                noise = self._discrete_noise(c) if c["threshold"] == "calibrated" else None
                res = fn(c["id"], c["level"], pairs, self._threshold(c, noise), params=c.get("params"), **common)
        elif typ == "coverage":
            # select.exclude 两边同样剔掉（如 GPU 专家分发缓冲区 *.experts.out，TPU 没有对应的量）
            res = fn(c["id"], c["level"], self._names(self.run(self._ref_name(c)), sel), self._names(test, sel), **common)
        elif typ in ("loss_series", "scalar_series"):
            ref = self.run(self._ref_name(c))
            evidence.append({"path": str(ref.path), "sha256": ref.meta_sha256()})
            field = sel["field"]
            params = {"window": sel.get("window"), **self._step_params(test), "ref_name": ref.name, **(c.get("params") or {})}
            # select.test_field：被测换一列读（golden 与噪声对仍读 field），用于被测侧有更准的同一物理量（例：派生的 param_norm_from_groups）
            tfield = sel.get("test_field", field)
            if tfield != field:
                params["test_field"] = tfield
            ts = self._test_series(test, tfield)
            if typ == "loss_series":
                noise = None
                if c["threshold"] == "calibrated":
                    from janus.checks.series import noise_from_pairs
                    npairs = self.cat.runs.get("noise") or []
                    # 噪声参照对是同一侧的两次运行，步号约定相同，不换算
                    noise = noise_from_pairs([(self.run(a).series(field), self.run(b).series(field)) for a, b in npairs],
                                             parse_window(params["window"]), params.get("denominator", "golden"),
                                             int(params["first_step"]), names=[f"{a}~{b}" for a, b in npairs])
                res = fn(c["id"], c["level"], ts, ref.series(field), self._threshold(c, noise),
                         params=params, **common)
            else:
                res = fn(c["id"], c["level"], ts, ref.series(field), params=params, **common)
        elif typ == "multi_golden":
            field = sel["field"]
            goldens = {g: self.run(g).series(field) for g in self.cat.runs["golden"]}
            for g in goldens:
                evidence.append({"path": str(self.run(g).path), "sha256": self.run(g).meta_sha256()})
            npairs = self.cat.runs.get("noise") or []
            # noise_from: golden —— 目录里 runs.noise 是给别的检查用的另一种噪声（如回归的 TPU 执行噪声），
            # 这一条仍用 golden 两两组合当参照，不跟着换
            src = (c.get("params") or {}).get("noise_from", "runs")
            if src not in ("runs", "golden"):
                raise CatalogError(f"noise_from 只能是 runs 或 golden：{src}")
            if src == "golden":
                npairs = []
            pairs = [(self.run(a).series(field), self.run(b).series(field)) for a, b in npairs]
            params = {"window": sel.get("window"), **self._step_params(test), **(c.get("params") or {})}
            res = fn(c["id"], c["level"], self._test_series(test, field), goldens, self._threshold(c),
                     params=params, noise_pairs=pairs or None, noise_names=[f"{a}~{b}" for a, b in npairs] or None,
                     **common)
        else:
            res = self._execute_more(c, fn, test, evidence, common)
        res.evidence = evidence
        if c.get("threshold") == "pending":
            res.threshold = {"kind": "pending", "note": c.get("pending_note", "待标定")}
            if res.status == INCONCLUSIVE:
                res.message = f"门槛待标定（{c.get('pending_note', '噪声参照还没有')}），只报被测值"
        return res

    def _execute_more(self, c, fn, test: Run, evidence: list, common: dict):
        """其余类型的取数：逐个按 select.kind 从运行目录里拿数据。"""
        sel, typ, params = c["select"], c["type"], c.get("params") or {}

        def ref_run():
            r = self.run(self._ref_name(c))
            evidence.append({"path": str(r.path), "sha256": r.meta_sha256()})
            return r
        if typ == "config_diff":
            return fn(c["id"], c["level"], test.meta, ref_run().meta, params["mapping"], **common)
        if typ == "data_fingerprint":
            ref = ref_run()
            a, b = parse_window(sel.get("steps", sel.get("window", "1:2")))

            from janus.checks.extra import FINGERPRINT_FIELDS
            fields = list(params.get("fields", FINGERPRINT_FIELDS))

            def flat(run):
                out, tot, missing = {}, {}, []
                for k, rec in run.data_records().items():
                    st = int(k.split(".")[0])
                    if not (a <= st < b):
                        continue
                    for f in fields:
                        if f in rec:
                            out[f"{k}.{f}"] = rec[f]
                        else:                  # 期望字段缺失：报 error，不跳过（跳过等于少比了还报一致）
                            missing.append(f"{run.name}:{k}.{f}")
                    tot[st] = tot.get(st, 0) + int(rec.get("nonpad", 0))
                out.update({f"{st}.nonpad_total": np.int64(v) for st, v in tot.items()})
                return out, missing
            first = test.meta["steps"].get("first", 1)
            fl = None
            if params.get("first_loss", True):
                lt, lr_ = test.series("lm_loss").get(first), ref.series("lm_loss").get(test.to_golden_step(first))
                if lt is not None and lr_ is not None:
                    fl = (lt, lr_)
            (ft, mt), (fr, mr) = flat(test), flat(ref)
            return fn(c["id"], c["level"], ft, fr, first_loss=fl, missing=mt + mr, fields=fields, **common)
        if typ == "checkpoint_equal":
            if sel.get("manifest", True):
                # select.test_run / select.ref_run：这一条改比两个指定的运行（如转换后的检查点清单对 HF 原件清单），
                # 不用目录的 runs.test / golden
                if sel.get("test_run"):
                    test = self.run(sel["test_run"])
                    evidence[:] = [{"path": str(test.path), "sha256": test.meta_sha256()}]
                if sel.get("ref_run"):
                    ref = self.run(sel["ref_run"])
                    evidence.append({"path": str(ref.path), "sha256": ref.meta_sha256()})
                else:
                    ref = ref_run()
                tag, src = sel.get("which", "init"), sel.get("source", "params")
                return fn(c["id"], c["level"], manifests=(test.manifest(src, tag), ref.manifest(src, tag)), **common)
            return fn(c["id"], c["level"], sel["test_dir"], sel["ref_dir"], pattern=sel.get("pattern", "**/*"), **common)
        if typ == "dtype_manifest":
            if sel.get("run"):                     # select.run：查指定运行的清单（默认查被测）
                test = self.run(sel["run"])
                evidence[:] = [{"path": str(test.path), "sha256": test.meta_sha256()}]
            which = sel.get("which", "final")
            tags = which if isinstance(which, list) else [which]
            src = sel.get("source", "params")
            man = {(f"{t}/rank{rk}" if src == "params" else f"{t}/{rk}"): d
                   for t in tags for rk, d in test.manifest(src, t).items()}
            return fn(c["id"], c["level"], man, params["rules"], **common)
        if typ == "fake_fp32":
            step = int(sel["step"])
            return fn(c["id"], c["level"], [(n, test.tensor(step, n)) for n in self._names(test, sel)],
                      params=params, **common)
        if typ == "replica_consistency":
            groups: dict[str, dict] = {}
            for rank, d in test.param_manifest(sel.get("which", "final")).items():
                for name, ent in d.items():
                    groups.setdefault(name, {})[f"rank{rank}"] = ent["sha256"]
            pat, exc = sel.get("name", "*"), sel.get("exclude")
            import fnmatch
            groups = {k: v for k, v in groups.items()
                      if fnmatch.fnmatchcase(k, pat) and not (exc and fnmatch.fnmatchcase(k, exc))}
            return fn(c["id"], c["level"], groups, **common)
        if typ == "overlap_replay":
            ref = ref_run()
            fields = sel.get("fields")
            return fn(c["id"], c["level"], ref.metrics(), test.metrics(), fields=fields, **common)
        if typ == "artifact_check":
            f = test.path / sel["file"]
            if not f.is_file():
                raise StoreError(f"{test.path} 没有编译产物 {sel['file']}")
            ref_text = None
            if c.get("compare") == "lkg":          # 计数对 LKG：规则写 match_ref: true 的要和 LKG 的同一条计数相等
                rf = ref_run().path / sel["file"]
                if not rf.is_file():
                    raise StoreError(f"LKG 没有编译产物 {sel['file']}")
                ref_text = rf.read_text()
            return fn(c["id"], c["level"], f.read_text(), params.get("rules", []), ref_text=ref_text,
                      fingerprint=bool(params.get("fingerprint")), ignore_lines=params.get("ignore_lines"), **common)
        if typ == "run_complete":
            return fn(c["id"], c["level"], test.raw_steps(), test.meta, params=params, **common)
        if typ == "health_signals":
            return fn(c["id"], c["level"], {f: test.series(f) for f in sel["fields"]}, params=params, **common)
        if typ == "determinism":
            ref = ref_run()
            sa = {f: test.series(f) for f in sel["fields"]}
            sb = {f: ref.series(f) for f in sel["fields"]}
            if sel.get("window"):                  # 只比窗口内的步（如第 1 步：window "1:2"）；窗口内被测缺步照样报不同
                a, b = parse_window(sel["window"])
                sa = {f: {s: v for s, v in d.items() if a <= s < b} for f, d in sa.items()}
                sb = {f: {s: v for s, v in d.items() if a <= s < b} for f, d in sb.items()}
                if not any(sa.values()):
                    raise StoreError(f"被测在窗口 {sel['window']} 内没有 {sel['fields']}")
            pairs = self._tensor_pairs(test, ref, sel) if "step" in sel else None
            return fn(c["id"], c["level"], sa, sb, pairs=pairs, **common)
        if typ == "router_stats":
            ref = ref_run()
            prefix = sel["prefix"]
            names = sorted(k for k in ref.metrics()[min(ref.metrics())] if k.startswith(prefix))
            st = {n: self._test_series(test, n) for n in names}
            sr = {n: ref.series(n) for n in names}
            p = {"window": sel["window"], **self._step_params(test), **params}
            noise = None
            if c["threshold"] == "calibrated":
                from janus.checks.series import window_stats
                noise = {"signed_mean": [], "mare": [], "max_step": []}
                for a, b in self.cat.runs.get("noise") or []:
                    ra, rb = self.run(a), self.run(b)
                    worst = {k: 0.0 for k in noise}
                    for n in names:
                        v, _ = window_stats(ra.series(n), rb.series(n), parse_window(sel["window"]), "golden", int(p["first_step"]))
                        worst = {k: max(worst[k], v[k]) for k in worst}
                    for k in noise:
                        noise[k].append(worst[k])
            return fn(c["id"], c["level"], st, sr, self._threshold(c, noise), params=p, **common)
        if typ == "optimizer_step":
            src = test if (c.get("params") or {}).get("replay_from", "test") == "test" else ref_run()
            return self._optimizer_step(c, fn, test, src, common)
        if typ == "poison_selftest":
            ref = ref_run()
            pairs = self._tensor_pairs(test, ref, sel)
            tk = dict(params.get("target_kwargs") or {})
            th = tk.pop("threshold", None)
            if isinstance(th, dict):
                tk["threshold"] = Threshold("numeric", values=th, source=params.get("source", "自证用"))
            elif th == "exact":
                tk["threshold"] = Threshold("exact")
            return fn(c["id"], c["level"], params["target"], pairs, tk, injections=params.get("injections"), **common)
        raise CatalogError(f"{typ} 还不能从目录里调用")

    def _router_fp64_pairs(self, run: Run, sel) -> list:
        """路由选中集合对 FP64 重算：用同一次运行抓下的路由输入（MLP 前归一化输出）、路由权重与 expert_bias，
        在 FP64 里算 logits → sigmoid → 加偏置 → top-k，与 GPU 实际选中的专家比。"""
        step = int(sel["step"])
        out = []
        for n in run.names(step, sel.get("name", "*")):
            layer = n.split(".")[1]
            gpu = run.tensor(step, n)
            x = run.tensor(step, f"layers.{layer}.mlp_norm.out").astype(np.float64).reshape(-1, gpu.shape[0] and run.tensor(step, f"layers.{layer}.mlp_norm.out").shape[-1])
            w = run.tensor(step, f"param.decoder.layers.{layer}.mlp.router.weight").astype(np.float64)
            b = run.tensor(step, f"buffer.decoder.layers.{layer}.mlp.router.expert_bias").astype(np.float64)
            scores = 1.0 / (1.0 + np.exp(-(x @ w.T))) + b
            k = gpu.shape[-1]
            ref = np.sort(np.argpartition(-scores, k - 1, axis=-1)[:, :k], axis=-1).astype(gpu.dtype)
            out.append((n, gpu, ref))
        return out

    def _optimizer_step(self, c, fn, test: Run, ref: Run, common: dict):
        """被测更新后的 FP32 主权重（与动量）对「参照的主权重 + 梯度，FP64 重放一步 AdamW」。"""
        from janus.replay import adamw_step, clip_coeff, no_weight_decay
        sel = c["select"]
        if "steps" in sel:                         # 连续多步：每步用上一步更新后的动量接着重放
            a, b = parse_window(sel["steps"])
            base = {k: v for k, v in sel.items() if k != "steps"}
            results = [self._optimizer_step({**c, "select": {**base, "step": s, "_prev": s - 1 if s > a else None}},
                                            fn, test, ref, common) for s in range(a, b)]
            worst = max(results, key=lambda r: (r.status == FAIL, r.metrics.get("rel_l2_max", 0.0)))
            worst.metrics["steps"] = f"{a}:{b}"
            worst.details = [{"name": f"step {a + i}", "status": r.status, **{k: v for k, v in r.metrics.items() if isinstance(v, (int, float))}}
                             for i, r in enumerate(results)]
            worst.status = next((st for st in (FAIL, ERROR, INCONCLUSIVE) if any(r.status == st for r in results)), PASS)
            worst.message = f"连续 {b - a} 步重放：" + ("全部在门槛内" if worst.status == PASS else worst.message)
            return worst
        step = int(sel["step"])
        prev = sel.get("_prev")
        oc = ref.meta.get("optimizer_config") or {}
        lr = ref.series("lr").get(step)
        # 权重衰减取实际生效的值：每步记录的 wd > 目录 params.weight_decay > 调度器配置 > optimizer 配置
        wd_cfg = ref.series("wd").get(step) if "wd" in ref.metrics().get(step, {}) else None
        if wd_cfg is None:
            wd_cfg = (c.get("params") or {}).get("weight_decay")
        if wd_cfg is None:
            wd_cfg = (ref.meta.get("scheduler_config") or {}).get("start_weight_decay")
        if wd_cfg is None:
            wd_cfg = oc.get("weight_decay", 0.0)
        coeff = clip_coeff(ref.series("grad_norm").get(step), oc.get("clip_grad"))
        pairs = []
        for n in ref.names(step, "main_param." + sel.get("name", "*")):
            pname = n[len("main_param."):]
            p, g = ref.tensor(step, n), ref.tensor(step, "grad." + pname)
            wd = 0.0 if no_weight_decay(pname, p.shape) else float(wd_cfg)
            m0 = ref.tensor(prev, f"exp_avg.{pname}") if prev else None
            v0 = ref.tensor(prev, f"exp_avg_sq.{pname}") if prev else None
            new, m, v = adamw_step(p, g, lr=lr, beta1=float(oc.get("adam_beta1", 0.9)), beta2=float(oc.get("adam_beta2", 0.999)),
                                   eps=float(oc.get("adam_eps", 1e-8)), wd=wd, step=step, clip_coeff=coeff, m0=m0, v0=v0)
            # 比「更新量」而不是更新后的参数：参数本身比更新量大三个数量级，比参数会把更新错藏进舍入里
            pairs.append((f"update.{pname}", test.tensor(step, f"main_param_after.{pname}").astype(np.float64) - p, new - p))
            if sel.get("states", True):
                pairs.append((f"exp_avg.{pname}", test.tensor(step, f"exp_avg.{pname}"), m))
                pairs.append((f"exp_avg_sq.{pname}", test.tensor(step, f"exp_avg_sq.{pname}"), v))
        res = fn(c["id"], c["level"], pairs, self._threshold(c), lr=lr, hard=common["hard"],
                 metrics=c.get("metrics", ["rel_l2", "max_abs"]))
        res.metrics["clip_coeff"] = coeff
        res.metrics["weight_decay"] = float(wd_cfg)
        return res


def janus_commit() -> str | None:
    """janus 自己的提交号，写进 results.json 的 janus.commit，复现时用。

    只有从 git 工作区装（pip install -e）的才拿得到；janus/ 目录下有未提交的改动时加 "+dirty"——
    那次结果不是任何一个提交能复现的。拿不到（不是 git 工作区、没有 git）返回 None。"""
    import subprocess
    pkg = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "-C", str(pkg), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", str(pkg), "status", "--porcelain", "--", "."],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return head.stdout.strip() + ("+dirty" if dirty else "")


def build_parser(prog: str = "janus run"):
    """`janus run` 的参数。`janus.cli` 和 `python -m janus run` 共用这一份。"""
    import argparse
    ap = argparse.ArgumentParser(prog=prog, description="读一份检查项目录，跑完全部检查，写 results.json、details/、report.md。"
                                 "退出码：0 没有硬检查失败；1 有硬检查 fail；2 有硬检查 error 或目录不合规。")
    ap.add_argument("catalog", type=Path, help="检查项目录（YAML），格式见 docs/reference/catalog-schema.md")
    ap.add_argument("--out", type=Path, default=Path("janus-results"), help="结果目录（默认 ./janus-results）")
    ap.add_argument("--test", help="换被测运行名，目录文件不用改（例如逐个跑故障注入运行）")
    ap.add_argument("--lkg", help="日常回归：换 LKG 运行名（目录里 compare: lkg 的检查都对它比）")
    ap.add_argument("--root", help="换运行根目录，目录文件不用改（results.json 里记的是实际路径）")
    ap.add_argument("--html", action="store_true", help="另出 report.html（要 matplotlib：pip install -e '.[report]'）")
    ap.add_argument("--render-only", action="store_true", help="不跑检查，按 --out 里已有的 results.json 重画报告")
    ap.add_argument("--verbose", action="store_true", help="逐条打印检查结果与耗时")
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:                      # `python -m janus run …` 与旧写法 `python -m janus <目录> …` 都认
        argv = argv[1:]
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:                      # argparse 出错或 --help：按 janus 的退出码约定返回，不抛
        return 0 if e.code == 0 else 2
    cat_path, out = args.catalog, args.out
    try:
        cat = load(cat_path)
    except CatalogError as e:
        print(f"目录不合规：{e}", file=sys.stderr)
        return 2
    except OSError as e:                         # 目录文件不存在、读不了：同样按「目录不合规」退出码 2
        print(f"读不了目录文件：{e}", file=sys.stderr)
        return 2
    if args.test:
        cat.data["runs"]["test"] = args.test
    if args.lkg:
        cat.data["runs"]["lkg"] = args.lkg
    if args.root:
        cat.data["runs"]["root"] = args.root
    runner = Runner(cat)
    if args.render_only:                       # 只按已有 results.json 重画报告（改了报告样式不必重跑检查）
        if not (out / "results.json").is_file():
            print(f"--render-only 要 {out / 'results.json'}，没有这个文件（先不带 --render-only 跑一次）", file=sys.stderr)
            return 2
        results = json.loads((out / "results.json").read_text())
        (out / "report.md").write_text(_render_report(cat, results))
        if args.html:                          # 与正常运行一样：给了 --html 才写 report.html
            from janus.report.html import render_html
            (out / "report.html").write_text(render_html(cat, results, runner, out))
        print(f"已重画 {out}")
        return 0
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    checks = []
    out.mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(exist_ok=True)
    import time
    # 先把要读的运行目录打开一遍：不合规（如 side 写错）时给一条可读的提示；检查照跑，各自报 error，退出码 2
    names = [cat.runs["test"], *([cat.runs["lkg"]] if cat.runs.get("lkg") else []), *cat.runs.get("golden", []),
             *(n for pr in cat.runs.get("noise") or [] for n in pr)]
    for n in dict.fromkeys(names):
        try:
            runner.run(n)
        except StoreError as e:
            print(f"运行目录不合规（{n}）：{e}", file=sys.stderr)
    for c in cat.checks:
        t0 = time.time()
        try:
            res = runner.execute(c)
        except Exception as e:  # noqa: BLE001  一条检查出任何错都记成这条的 error（退出码 2），不让整次运行崩掉、没有 results.json
            from janus.checks.base import CheckResult
            known = (StoreError, CatalogError, KeyError, ValueError, TypeError)
            msg = str(e) if isinstance(e, known) else f"{type(e).__name__}: {e}"
            res = CheckResult(c["id"], c["type"], c["level"], c.get("hard", True), status=ERROR, message=msg)
        d = res.to_json()
        if res.details:
            rel = f"details/{c['id']}.jsonl"
            with open(out / rel, "w") as f:
                for row in res.details:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            d["details"] = rel
        checks.append(d)
        if args.verbose:
            print(f"{d['status']:<13} {c['id']:<36} {time.time() - t0:6.1f}s  {str(d.get('message', ''))[:110]}", flush=True)
    summary = {k: sum(1 for c in checks if c["status"] == k) for k in (PASS, FAIL, INCONCLUSIVE, ERROR)}
    summary["hard_fail"] = any(c["status"] == FAIL and c.get("hard", True) for c in checks)
    results = {
        "schema_version": RESULTS_SCHEMA_VERSION, "janus": {"version": __version__, "commit": janus_commit()},
        "catalog": {**catalog_fingerprint(cat_path),
                    **({"includes": cat.data["_includes"]} if cat.data.get("_includes") else {})},
        "runs": {n: {"path": str(r.path), "meta_sha256": r.meta_sha256()} for n, r in runner._runs.items()},
        "started_at": started, "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": summary, "checks": checks,
    }
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    (out / "report.md").write_text(_render_report(cat, results))
    if args.html:
        from janus.report.html import render_html
        (out / "report.html").write_text(render_html(cat, results, runner, out))
    print(f"pass {summary['pass']} / fail {summary['fail']} / inconclusive {summary['inconclusive']} / error {summary['error']} → {out}")
    if any(c["status"] == ERROR and c.get("hard", True) for c in checks):
        return 2
    return 1 if summary["hard_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
