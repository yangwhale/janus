"""优化器回放（TPU 一侧）：把 GPU golden 每一步的梯度、更新前的 FP32 主权重原样喂给 MaxText 的优化器
（T1 配置：optax Muon + muon_mcore_* 开关 + 标量参数普通 Adam，hy4-mini-t1.yml），逐步逐张量比更新量。
单步（g2s-a 第 1 步）或连续多步（g2o-a 第 1–8 步：第 1 步从零状态开始，此后状态由同一串 GPU 梯度累出来）。

混元 3 的做法（analysis/adamw_step_check.py、catalogs/phase2-st8-opt8.yaml）是用抓下的状态按公式在 FP64 里重算更新、对抓下的
更新后参数比；这里同一个思路换成 Muon，并且被测是 MaxText 自己的优化器代码（optimizers.get_optimizer，同训练里一样建）：
1. GPU `export_hf_grads.py --kind grad / main_param / main_param_after` 把抓数换成 HF 名字与布局（路由专家只有 rank 0 本地的 0–3 号）；
2. HF → MaxText 布局（与 convert_hf_to_maxtext.py 同一段代码），缺的专家补 0（Muon 对每个专家各自正交化、Adam 逐元素，互不影响）；
3. 裁剪：GPU 记下的裁剪前 grad_norm，按 MaxText 的公式（optax.clip_by_global_norm：g / ‖g‖ × 1.0）缩放；
4. 在 TPU（和同一进程的 CPU，FP32）上跑 tx.init / tx.update / optax.apply_updates，得到 FP32 主权重的增量 Δ = p_after − p；
   变体 ns_bf16 再开 muon_mcore_ns_bf16（NS 在 BF16 张量上迭代，对上 GPU 的 emerging_optimizers medium 精度）；
   变体 adam_acc 再开 muon_adam_accurate（Adam 一支的除法、sqrt、1 − β^t 用精确的 FP32 算法）；
5. FP64 参照：numpy FP64 里按同一个分区（MaxText 的 Muon 维度表，含 GLU 融合）、Megatron-Core 的公式连续算：
   Muon：buf ← buf + (1−β)(g − buf)，方向 g + β(buf − g)（Nesterov 不做偏差修正），X / max(‖X‖_F, 1e-7)，
   NS 5 步（3.4445, −4.7750, 2.0315），乘 √max(1, fan_out / fan_in)，× −lr；Adam：−lr · m̂ / (√v̂ + eps)（偏差修正）；p 加 Δ 不舍入；
6. 换回 HF 名字逐张量比：R = rel_L2(Δ_TPU, Δ_FP64) ÷ rel_L2(Δ_GPU, Δ_FP64)。两边的 Δ 都含「FP32 主权重加更新时的舍入」，口径相同。

  python optim_replay.py --gpu-prefix gs://…/g2s-a.s1 --gpu-run gs://…/golden-t1/g2s-a --hf-config <config.json> --out optim_replay.json
  python optim_replay.py --gpu-prefix gs://…/g2o-a --steps 1-8 --gpu-run gs://…/golden-t1/g2o-a --out optim_replay.json
  # SCRIPT=optim_replay.py 用 run.sh 提交（--stage mini、YML=hy4-mini-t1.yml）；--cpu 只在 CPU 上算（验证脚本）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VARIANTS = {  # 变体名 → 在配置原样之上多开的开关（MaxText 覆盖项）
    "ns_bf16": ["muon_mcore_ns_bf16=true"],
    "adam_acc": ["muon_adam_accurate=true"],
    "ns_bf16_adam_acc": ["muon_mcore_ns_bf16=true", "muon_adam_accurate=true"],
}

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "model"))
import mt_common  # noqa: E402

R_LIMIT = 1.2
NS = (3.4445, -4.7750, 2.0315)


def fetch(path: str) -> str:
    if not path.startswith("gs://"):
        return path
    loc = Path(tempfile.mkdtemp()) / Path(path).name
    subprocess.run(["gcloud", "storage", "cp", "-q", path, str(loc)], check=True)
    return str(loc)


def load_st(path: str) -> dict:
    from safetensors.numpy import load_file
    return load_file(fetch(path))


def gpu_grad_norm(run: str, step: int) -> float:
    txt = Path(fetch(run.rstrip("/") + "/metrics.jsonl")).read_text()
    for line in txt.splitlines():
        r = json.loads(line)
        if int(r.get("step", -1)) == step:
            return float(r["grad_norm"])
    raise SystemExit(f"{run} 的 metrics.jsonl 里没有第 {step} 步")


# ---------------------------------------------------------------- FP64 参照（numpy，Megatron-Core 公式，MaxText 的分区）

def _ns64(x: np.ndarray) -> np.ndarray:
    """Newton-Schulz 5 步，FP64；行多于列时先转置（同 optax / emerging_optimizers）。"""
    t = x.shape[0] > x.shape[1]
    if t:
        x = x.T
    x = x / max(np.linalg.norm(x), 1e-7)
    a, b, c = NS
    for _ in range(5):
        A = x @ x.T
        x = a * x + (b * A + c * A @ A) @ x
    return x.T if t else x


def _ns_bf16(x: np.ndarray, iterate_bf16: bool) -> np.ndarray:
    """GPU 一侧 NS 精度的两种模拟：iterate_bf16=False 只把矩阵乘的操作数舍成 BF16（FP32 累加、迭代量 FP32，
    即 TPU 默认精度的做法）；True 把每一步的中间量都舍成 BF16（迭代量存 BF16）。只作诊断。"""
    import ml_dtypes
    r = (lambda v: v.astype(ml_dtypes.bfloat16).astype(np.float64))
    mm = (lambda u, w: (r(u).astype(np.float32) @ r(w).astype(np.float32)).astype(np.float64))
    t = x.shape[0] > x.shape[1]
    if t:
        x = x.T
    x = x / max(np.linalg.norm(x), 1e-7)
    x = r(x) if iterate_bf16 else x.astype(np.float32).astype(np.float64)
    a, b, c = NS
    for _ in range(5):
        A = mm(x, x.T)
        if iterate_bf16:
            A = r(A)
            x = r(a * x + r(r(b * A + r(c * r(mm(A, A)))) @ x))
        else:
            x = (a * x + mm(b * A + c * mm(A, A), x)).astype(np.float32).astype(np.float64)
    return x.T if t else x


def gpu_ns_probe(G, P0, P1, names, lr: float) -> list[dict]:
    """GPU 的 Muon 增量离 FP64 远时，看是不是 NS 的精度：同一梯度在三种精度下算第 1 步方向，比 GPU 的增量（torch 布局 [out, in]）。"""
    from janus import metrics as M
    out = []
    for n in names:
        if n not in G:
            continue
        g = np.asarray(G[n], np.float64)
        if g.ndim != 2:
            continue
        dg = np.asarray(P1[n], np.float64) - np.asarray(P0[n], np.float64)
        sc = -lr * np.sqrt(max(1.0, g.shape[0] / g.shape[1]))
        u64, uop, uit = sc * _ns64(g), sc * _ns_bf16(g, False), sc * _ns_bf16(g, True)
        out.append({"name": n, "gpu_vs_fp64": M.rel_l2(dg, u64), "bf16_operands_fp32_iterate_vs_fp64": M.rel_l2(uop, u64),
                    "bf16_iterate_vs_fp64": M.rel_l2(uit, u64), "gpu_vs_bf16_iterate": M.rel_l2(dg, uit)})
    return out


# ---------------------------------------------------------------- MaxText 一侧

def build(hf_config: str, stage: str, extra: list[str]):
    """配置、抽象模型、HF → MaxText 映射（convert_hf_to_maxtext.build 同一段代码）。"""
    import convert_hf_to_maxtext as C
    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text(Path(hf_config).read_text())
    return C.build(d, stage, extra)


def hf_to_mt(config, abstract, mapping, hooks, slash_of, hf: dict, shapes: dict) -> dict:
    """HF 名字的张量 → MaxText 叶子（只取 params 集合）。HF 里缺的张量或缺的专家补 0。"""
    from maxtext.checkpoint_conversion.to_maxtext import _get_hf_loading_function

    def getter(name):
        full = tuple(shapes[name])
        if name not in hf:
            return np.zeros(full, np.float32)
        x = np.asarray(hf[name], np.float32)
        if x.shape != full:      # 路由专家只有前几个
            y = np.zeros(full, np.float32)
            y[: x.shape[0]] = x
            x = y
        return x

    out = {}
    for k, sds in abstract.items():
        if not slash_of[k].startswith("params/"):
            continue
        fn = _get_hf_loading_function(mapping[k], getter, hooks.get(k), tuple(sds.shape), config, k)
        out[slash_of[k]] = np.asarray(fn(), np.float32)
    return out


class Ref64:
    """FP64 参照的优化器状态，连续多步：Megatron-Core 的公式、MaxText 的分区（Muon 维度表与 GLU 融合）。"""

    def __init__(self, mdn_flat: dict, fuse_glu: bool, lr, beta, b1, b2, eps):
        self.mdn, self.fuse = mdn_flat, fuse_glu
        self.lr, self.beta, self.b1, self.b2, self.eps = lr, beta, b1, b2, eps
        self.buf, self.m, self.v, self.t = {}, {}, {}, 0

    def _pair(self, path):
        other = tuple("wi_1" if s == "wi_0" else s for s in path)
        return other if self.fuse and "wi_0" in path and self.mdn.get(other) is not None else None

    def step(self, g_mt: dict, coef: float) -> dict:
        self.t += 1
        out = {}
        for path, d in self.mdn.items():
            key = "params/" + "/".join(path)
            if d is not None and self.fuse and "wi_1" in path and \
                    self.mdn.get(tuple("wi_0" if s == "wi_1" else s for s in path)) is not None:
                continue  # 由 wi_0 一起算
            g = g_mt[key].astype(np.float64) * coef
            if d is None:
                m = self.m.get(key, 0.0) * self.b1 + (1 - self.b1) * g
                v = self.v.get(key, 0.0) * self.b2 + (1 - self.b2) * g * g
                self.m[key], self.v[key] = m, v
                out[key] = -self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)
                continue
            pair = self._pair(path)
            if pair is not None:
                k1 = "params/" + "/".join(pair)
                g = np.concatenate([g, g_mt[k1].astype(np.float64) * coef], axis=-1)
            buf = self.buf.get(key, 0.0)
            buf = buf + (1 - self.beta) * (g - buf)                    # buf.lerp_(g, 1 − β)
            self.buf[key] = buf
            u = g + self.beta * (buf - g)                               # g.lerp(buf, β)：Nesterov，不做偏差修正
            upd = muon64_dir(u, d.reduction_axis, d.output_axis, self.lr)
            if pair is not None:
                n0 = g_mt[key].shape[-1]
                out[key], out["params/" + "/".join(pair)] = upd[..., :n0], upd[..., n0:]
            else:
                out[key] = upd
        return out


def muon64_dir(u, red, out, lr):
    """方向 u（已含动量）→ −lr · √max(1, fan_out / fan_in) · NS(u)，FP64；MaxText 的轴约定（批维、归约轴、输出轴）。"""
    nd = u.ndim
    red = tuple(r % nd for r in red)
    out = tuple(o % nd for o in out)
    batch = tuple(sorted(set(range(nd)) - set(red) - set(out)))
    perm = batch + red + out
    sh = [u.shape[i] for i in perm]
    fi, fo = int(np.prod([u.shape[i] for i in red])), int(np.prod([u.shape[i] for i in out]))
    m = u.transpose(perm).reshape(-1, fi, fo)
    o = np.stack([_ns64(m[i]) for i in range(m.shape[0])]).reshape(sh).transpose(np.argsort(perm))
    return -lr * np.sqrt(max(1.0, fo / fi)) * o


def parse_steps(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="mini")
    p.add_argument("--gpu-prefix", required=True,
                   help="export_hf_grads.py 的产物前缀：单步 <前缀>.{grad,…}.hf.safetensors；多步 <前缀>.s<k>.{grad,…}.hf.safetensors")
    p.add_argument("--gpu-run", required=True, help="GPU 运行目录（读 metrics.jsonl 的 grad_norm）")
    p.add_argument("--steps", default="1", help="连续重放的步，如 1-8；第 1 步从零状态开始")
    p.add_argument("--variants", default="off,ns_bf16",
                   help="TPU 上的优化器变体：off = 配置原样；其余见 VARIANTS（在配置上多开的开关）")
    p.add_argument("--hf-config", default="/data/config.json")
    p.add_argument("--out", required=True)
    p.add_argument("--cpu", action="store_true", help="只在 CPU 上跑（验证脚本）")
    a, _rest = p.parse_known_args(argv)
    steps = parse_steps(a.steps)
    if steps[0] != 1 or steps != list(range(1, steps[-1] + 1)):
        raise SystemExit("连续重放要从第 1 步开始、不跳步（状态由同一串梯度累出来）")
    import jax
    if not a.cpu and not jax.distributed.is_initialized():
        try:
            jax.distributed.initialize()
        except Exception as e:  # noqa: BLE001
            print("jax.distributed.initialize 跳过：", e)
    if jax.process_index() == 0:   # 先建运行目录：中途失败时 run.sh 也能认出交结果的 Pod、取回日志
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from hf_view import HFView
    from make_tiny_hf import hf_shapes
    from maxtext.optimizers import optimizers
    from maxtext.utils import maxtext_utils, model_creation_utils

    from janus import metrics as M

    t0 = time.time()
    extra = ["hardware=cpu", "skip_jax_distributed_system=True", "run_name=optim_replay", "attention=dot_product"]
    config, abstract, mapping, hooks, hf_cfg, slash_of = build(a.hf_config, a.stage, extra)
    bad = mt_common.effective_mismatch(config)
    lr, beta = float(config.learning_rate), float(config.muon_beta)
    b1, b2, eps = float(config.adam_b1), float(config.adam_b2), float(config.adam_eps)
    layers, mtp = mt_common.STAGES[a.stage]
    shapes = hf_shapes(hf_cfg, layers, mtp)
    view = HFView(config, json.loads(Path(a.hf_config).read_text()), a.stage)
    metrics_txt = Path(fetch(a.gpu_run.rstrip("/") + "/metrics.jsonl")).read_text()
    gnorm = {int(json.loads(x)["step"]): float(json.loads(x)["grad_norm"]) for x in metrics_txt.splitlines() if x.strip()}

    # 优化器：同训练（train_utils.create_training_optimizer）；变体 ns_bf16 只多开一个开关
    _, model = model_creation_utils.create_nnx_abstract_model(config, mt_common.make_mesh(config))
    sched = maxtext_utils.create_learning_rate_schedule(config)
    variants = [v for v in a.variants.split(",") if v]
    txs = {}
    for v in variants:
        if v == "off":
            txs[v] = optimizers.get_optimizer(config, sched, model)
        elif v in VARIANTS:
            cfg_v = build(a.hf_config, a.stage, extra + VARIANTS[v])[0]
            txs[v] = optimizers.get_optimizer(cfg_v, sched, model)
        else:
            raise SystemExit(f"不认识的变体 {v}（可选 off、{'、'.join(VARIANTS)}）")
    _, abs_params, _ = nnx.split(model, nnx.Param, ...)
    pure_abs = nnx.to_pure_dict(abs_params)

    def flat(tree, path=()):
        if isinstance(tree, dict):
            for k, v in tree.items():
                yield from flat(v, path + (str(k),))
        else:
            yield path, tree

    mdn_flat = dict(flat(nnx.to_pure_dict(optimizers.get_muon_weight_dimension_numbers(model, config))))
    opt_of = {"params/" + "/".join(k): ("adam" if d is None else "muon") for k, d in mdn_flat.items()}

    def fill(src):
        def rec(node, path):
            if isinstance(node, dict):
                return {k: rec(v, path + (str(k),)) for k, v in node.items()}
            return src["params/" + "/".join(path)]
        return rec(pure_abs, ())

    def make_step(tx, gn):
        @jax.jit
        def f(g, st, prm):
            g = jax.tree.map(lambda t: (t / jnp.asarray(gn, t.dtype)) * 1.0 if gn > 1.0 else t, g)  # optax.clip_by_global_norm
            upd, st = tx.update(g, st, prm)
            return optax.apply_updates(prm, upd), st
        return f

    devs = {"cpu": jax.local_devices(backend="cpu")[0]} if a.cpu else {"tpu": jax.local_devices()[0]}
    arms = [(f"{d}_{v}", dev, txs[v]) for d, dev in devs.items() for v in variants]
    if not a.cpu and "off" in txs:
        arms.append(("cpu_off", jax.local_devices(backend="cpu")[0], txs["off"]))   # 同一进程的 CPU FP32 对照
    states = {}
    ref = Ref64(mdn_flat, bool(config.muon_fuse_glu), lr, beta, b1, b2, eps)

    def to_hf(leaves, dtype=np.float32):
        fl = {"params/model/" + k[len("params/"):]: v for k, v in leaves.items()}
        out = {}
        for h, srcs in view.by_hf.items():
            if h not in keep_names:
                continue
            pieces = {}
            for mt_key, i in srcs:
                full = fl[view.state_key(fl, mt_key, "param")]
                pieces[mt_key] = full if i is None else np.take(full, i, axis=view.scan_axis if full.ndim > view.scan_axis else 0)
            from hy4_map import assemble_hf
            out[h] = np.asarray(assemble_hf(pieces, view.shapes[h], view.rope, view.dein), dtype)
        return out

    rows, probe = [], None
    for k in steps:
        t1 = time.time()
        G = load_st(fetch_name(a.gpu_prefix, k, "grad", multi=len(steps) > 1))
        P0 = load_st(fetch_name(a.gpu_prefix, k, "main_param", multi=len(steps) > 1))
        P1 = load_st(fetch_name(a.gpu_prefix, k, "main_param_after", multi=len(steps) > 1))
        keep_names = set(G)
        experts = {n: x.shape[0] for n, x in G.items() if tuple(x.shape) != tuple(shapes[n])}
        gn = gnorm[k]
        g_mt = hf_to_mt(config, abstract, mapping, hooks, slash_of, G, shapes)
        p_mt = hf_to_mt(config, abstract, mapping, hooks, slash_of, P0, shapes)
        deltas = {}
        for tag, dev, tx in arms:
            g = jax.device_put(nnx.State(fill(g_mt)), dev)
            prm = jax.device_put(nnx.State(fill(p_mt)), dev)
            if tag not in states:
                states[tag] = jax.jit(tx.init)(prm)
            new, states[tag] = make_step(tx, gn)(g, states[tag], prm)
            new = nnx.to_pure_dict(jax.device_get(new))
            deltas[tag] = to_hf({"params/" + "/".join(kk): np.asarray(v, np.float32) - p_mt["params/" + "/".join(kk)]
                                 for kk, v in flat(new)})
        r64 = ref.step(g_mt, min(1.0, 1.0 / (gn + 1e-6)))                    # GPU 的裁剪公式（GPU 是 golden）
        hf64 = to_hf(r64, np.float64)
        for h in sorted(G):
            ne = experts.get(h)
            sl = (slice(0, ne),) if ne else (slice(None),)
            d64 = np.asarray(hf64[h], np.float64)[sl]
            p0 = np.asarray(P0[h], np.float64)
            dg = np.asarray(P1[h], np.float64) - p0
            row = {"step": k, "name": h, "experts": ne, "opt": opt_of.get(next(iter(view.by_hf[h]))[0].replace("-", "/"), "?"),
                   "rel_l2_gpu_vs_fp64": M.rel_l2(dg, d64)}
            for tag in deltas:
                dt = np.asarray(deltas[tag][h], np.float64)[sl]
                row[f"rel_l2_{tag}_vs_fp64"] = M.rel_l2(dt, d64)
                row[f"rel_l2_{tag}_vs_gpu"] = M.rel_l2(dt, dg)
                row[f"R_{tag}"] = row[f"rel_l2_{tag}_vs_fp64"] / max(row["rel_l2_gpu_vs_fp64"], 1e-300)
            floor = np.asarray(p0 + d64, np.float32).astype(np.float64) - p0
            row["rel_l2_fp32_store_floor"] = M.rel_l2(floor, d64)
            rows.append(row)
        if k == 1:
            probe = gpu_ns_probe(G, P0, P1, ["model.layers.0.self_attn.q_b_proj.weight", "model.layers.6.self_attn.o_proj.weight",
                                             "model.layers.6.self_attn.kv_b_proj.weight",
                                             "model.layers.6.self_attn.kv_a_proj_with_mqa.weight"], lr)
        print(f"第 {k} 步（{time.time() - t1:.0f} s）", flush=True)
    summary = {}
    for tag in [t for t, _, _ in arms]:
        rs = [r[f"R_{tag}"] for r in rows]
        worst = max(rows, key=lambda r: r[f"R_{tag}"])
        summary[tag] = {"R_max": max(rs), "worst": {"step": worst["step"], "name": worst["name"]},
                        "n_R_gt_limit": sum(x > R_LIMIT for x in rs),
                        "rel_l2_vs_gpu_muon_max": max(r[f"rel_l2_{tag}_vs_gpu"] for r in rows if r["opt"] == "muon"),
                        "rel_l2_vs_gpu_adam_max": max(r[f"rel_l2_{tag}_vs_gpu"] for r in rows if r["opt"] == "adam")}
    main_tag = arms[0][0]
    fails = [f"{r['step']}:{r['name']}" for r in rows if r[f"R_{main_tag}"] > R_LIMIT]
    rec = {
        "what": "优化器连续多步回放：GPU 每一步的梯度与主权重喂 MaxText 优化器（T1 配置），状态由同一串梯度累出来，逐步逐张量比主权重增量",
        "gpu": {"prefix": a.gpu_prefix, "run": a.gpu_run, "steps": steps, "grad_norm": {k: gnorm[k] for k in steps}},
        "platform": arms[0][1].platform, "device_kind": arms[0][1].device_kind, "arms": [t for t, _, _ in arms],
        "maxtext_config_mismatch_vs_yml": [list(map(str, b)) for b in bad],
        "optimizer": {"lr": lr, "muon_beta": beta, "adam_b1": b1, "adam_b2": b2, "adam_eps": eps,
                      "muon_mcore_nesterov": bool(config.muon_mcore_nesterov), "muon_adam_plain": bool(config.muon_adam_plain),
                      "muon_mcore_ns_normalize": bool(config.muon_mcore_ns_normalize), "muon_fuse_glu": bool(config.muon_fuse_glu),
                      "clip": "TPU：optax.clip_by_global_norm(1.0)，用 GPU 记下的 grad_norm；FP64 参照与 GPU：min(1, 1/(‖g‖+1e-6))"},
        "reference": "FP64（numpy）：Megatron-Core 的公式（buf.lerp_(g, 1−β)、u = g.lerp(buf, β)、NS 5 步 FP64、√max(1, fan_out/fan_in)；"
                     "Adam 带偏差修正），MaxText 的分区；主权重加增量不舍入",
        "rule": "R = rel_L2(Δ_TPU, Δ_FP64) ÷ rel_L2(Δ_GPU, Δ_FP64) ≤ 1.2 过（PROTOCOL.md 第 5 节）；逐步逐张量取最差；判定看第一个变体",
        "summary": summary, "verdict": "pass" if not fails else "fail", "fail_items": fails,
        "gpu_ns_precision_probe": probe, "seconds": time.time() - t0, "rows": rows,
    }
    if jax.process_index() == 0:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({"summary": summary, "verdict": rec["verdict"]}, ensure_ascii=False, default=float))
    return 0


def fetch_name(prefix: str, step: int, kind: str, multi: bool) -> str:
    return f"{prefix}.s{step}.{kind}.hf.safetensors" if multi else f"{prefix}.{kind}.hf.safetensors"


if __name__ == "__main__":
    sys.exit(main())
