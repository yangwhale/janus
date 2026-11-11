"""逐元素函数的 ULP（TPU 一侧）：混元 3 GPU 一侧 l1_ops.py「elementwise / basic / extreme」那组合成用例搬到 TPU（JAX / XLA）上，
外加混元 4 用到、混元 3 没比的 log、relu、clamp 与 log_softmax / log-sum-exp（LSE）。

每个函数在 FP32 与 BF16 下算，对 FP64 真值正确舍入到输出 dtype 的值数 ULP（janus `op_replay` 检查：ulp_max、cr_match_frac），
另报平均 ULP 与带符号的平均 ULP（系统偏差）。输入分三类：
- 合成：均匀分布、对数均匀分布、[1, 4) 全部 FP32、全部有限 BF16（穷举）；
- 真实（`--inputs`，`--make-inputs` 生成）：GPU golden 第 1 步抓数里的路由 logits、稠密 MLP 的 gate 半边、RMSNorm 的均方，
  以及 TPU 第 0 层索引器 KD 的学生分数（加性掩码）与教师目标（`debug_sow_indexer`）；
- LSE 拆解：jax.nn.log_softmax 在图里那种写法（分数 + 加性掩码）、logsumexp、以及 exp、求和、log 三步各自的误差，
  用来单独确认或否定「TPU FP32 log_softmax 的 LSE 系统偏低约 0.95 ulp」（results/t1-layer0-kd-attribution.json）。

判定（门槛的出处写在每条检查的 source 里）：
- exp、rsqrt、sigmoid、silu 用混元 3 GPU 一侧同一组门槛（CUDA 数学函数附录 expf 2 ULP、rsqrtf 2 ULP，sigmoid / silu 4 ULP 是推断）；
  log 用 CUDA logf 1 ULP。即：TPU 不比 GPU 一侧文档保证的精度差；
- relu、clamp 必须逐位等于 numpy（正确舍入就是原值）；
- BF16、尾部（输出落到 FP32 次正规数）、LSE 只报不判（同混元 3 的 L1.elem.bf16_vs_fp32 / L1.elem.tail）。

  python elementwise_ulp.py --make-inputs --gpu-step <GPU 运行>/step_1 --kd <TPU 运行>/diag_kd_inputs.safetensors --out inputs.npz
  python elementwise_ulp.py --inputs inputs.npz（或 gs://…）--out tpu-elementwise-ulp.json      # TPU 上（SCRIPT=elementwise_ulp.py）
  python elementwise_ulp.py --inputs inputs.npz --out x.json --cpu                              # CPU 上只验证脚本
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ml_dtypes
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # 仓库根：import janus

from janus import metrics as M  # noqa: E402
from janus.analysis import ops as O  # noqa: E402
from janus.checks import REGISTRY, Threshold  # noqa: E402

BF = ml_dtypes.bfloat16
SEED = 20261107
OPS = REGISTRY["op_replay"]
CLAMP = 10.0  # hy4-mini.yml mlp_activations_limit：路由专家 SwiGLU 的截断

GATES = {
    "exp": ({"ulp_max": 2}, "CUDA Programming Guide 数学函数附录：expf 最大 2 ULP（混元 3 L1.elem.exp.fp32 同一门槛）"),
    "rsqrt": ({"ulp_max": 2}, "CUDA Programming Guide 数学函数附录：rsqrtf 最大 2 ULP（混元 3 L1.elem.rsqrt.fp32 同一门槛）"),
    "sigmoid": ({"ulp_max": 4}, "推断（同混元 3）：sigmoid = 1/(1+exp(−x))，expf 2 ULP 经一次加、一次除后的起点"),
    "silu": ({"ulp_max": 4}, "推断（同混元 3）：silu = x·sigmoid(x)，同 sigmoid 再多一次正确舍入的乘法"),
    "log": ({"ulp_max": 1}, "CUDA Programming Guide 数学函数附录：logf 最大 1 ULP"),
    "sqrt": ({"ulp_max": 1}, "CUDA Programming Guide 数学函数附录：sqrtf 默认（-prec-sqrt=true）正确舍入；取 1 ULP"),
    "div": ({"ulp_max": 1}, "CUDA Programming Guide 数学函数附录：x / y 默认（-prec-div=true）正确舍入；取 1 ULP（Adam 的 m̂ / (√v̂ + eps) 用到）"),
}


# ---------------------------------------------------------------- 输入

def make_inputs(gpu_step: str, kd: str | None, out: str) -> None:
    """从 GPU golden 第 1 步抓数（janus 运行目录的 step_1）与 TPU 的 KD 诊断抓数取真实输入，存成一个 npz。"""
    from safetensors.numpy import load_file
    step = Path(gpu_step)
    idx = json.loads((step / "index.json").read_text())["tensors"]
    files = {}

    def get(name):
        e = next(e for e in idx if e["name"] == name)
        if e["file"] not in files:
            files[e["file"]] = load_file(str(step / e["file"]))
        a = files[e["file"]][name]
        return a.view(BF) if e["dtype"] == "BF16" else a

    layers = sorted({int(e["name"].split(".")[1]) for e in idx if e["name"].startswith("layers.") and e["name"].endswith(".in")})
    moe = [L for L in layers if any(e["name"] == f"layers.{L}.mlp.router.logits" for e in idx)]
    d = {}
    lg = np.concatenate([get(f"layers.{L}.mlp.router.logits").reshape(-1, 16) for L in moe]).astype(np.float32)
    d["router_logits"] = lg.ravel()
    fc1 = get("layers.0.mlp.fc1.out")
    fc1 = fc1.reshape(-1, fc1.shape[-1])
    d["mlp_gate_bf16"] = np.ascontiguousarray(fc1[:, : fc1.shape[-1] // 2]).ravel().view(np.uint16)  # npz 不认 BF16：存位模式
    ms = []
    for L in layers:
        x = get(f"layers.{L}.in").astype(np.float64)
        ms.append(((x * x).mean(axis=-1) + 1e-5).ravel())   # hy4-mini normalization_layer_epsilon 1e-5
    d["rmsnorm_ms"] = np.concatenate(ms).astype(np.float32)
    if kd:
        T = load_file(kd)
        k0 = {k.split("/")[-2]: v for k, v in T.items() if "/dense_layers/" in k}
        d["kd_score"] = k0["debug_kd_indexer_score"].reshape(4096, 4096).astype(np.float32)
        d["kd_mask"] = k0["debug_kd_indexer_mask"].reshape(4096, 4096).astype(np.float32)
        tgt = k0["debug_kd_target"].reshape(-1)
        d["kd_target"] = tgt[tgt > 0].astype(np.float32)
    np.savez(out, **d)
    print({k: (v.shape, str(v.dtype)) for k, v in d.items()})


def load_inputs(path: str | None) -> dict:
    if not path:
        return {}
    if path.startswith("gs://"):
        loc = Path(tempfile.mkdtemp()) / "inputs.npz"
        subprocess.run(["gcloud", "storage", "cp", "-q", path, str(loc)], check=True)
        path = str(loc)
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    if "mlp_gate_bf16" in d:
        d["mlp_gate_bf16"] = d["mlp_gate_bf16"].view(BF)
    return d


# ---------------------------------------------------------------- 统计

def _normal(v) -> np.ndarray:
    """正规数、0 或非有限值（不是次正规数）。"""
    v = np.asarray(v, np.float64)
    return (np.abs(v) >= np.finfo(np.float32).tiny) | (v == 0) | ~np.isfinite(v)


def ulp_stats(out, truth) -> dict:
    """平均 ULP、带符号平均 ULP（out − 真值，以输出 dtype 在真值处的一格为单位，不先舍入）。"""
    out = np.asarray(out)
    t = np.asarray(truth, np.float64)
    o = out.astype(np.float64)
    keep = np.isfinite(o) & np.isfinite(t)
    cr = O.round_from_f64(t, out.dtype)
    d = M.ulp_distance(out, cr).astype(np.float64)
    keep &= d >= 0
    sp = np.spacing(np.abs(cr).astype(out.dtype)).astype(np.float64)
    keep_s = keep & (sp > 0)
    signed = (o - t)[keep_s] / sp[keep_s]
    return {"ulp_mean": float(d[keep].mean()) if keep.any() else None,
            "ulp_signed_mean": float(signed.mean()) if signed.size else None}


class Job:
    def __init__(self, dev, cpu_dev, inputs: dict):
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp = jax, jnp
        self.dev, self.cpu_dev, self.inp = dev, cpu_dev, inputs
        self.rng = np.random.default_rng(SEED)
        self.checks, self.summary = [], {}
        self.fns = {
            "exp": (jnp.exp, np.exp),
            "log": (jnp.log, np.log),
            "sigmoid": (jax.nn.sigmoid, O.sigmoid_f64),
            "silu": (jax.nn.silu, O.silu_f64),
            "rsqrt": (jax.lax.rsqrt, lambda v: 1.0 / np.sqrt(v)),
            "relu": (jax.nn.relu, lambda v: np.maximum(v, 0.0)),
            "clamp": (lambda v: jnp.clip(v, -CLAMP, CLAMP), lambda v: np.clip(v, -CLAMP, CLAMP)),
        }

    def run(self, f, x, dev=None):
        """在指定设备上 jit 一个逐元素函数，BF16 输入保持位模式。"""
        jax = self.jax
        dev = dev or self.dev
        xd = jax.device_put(np.ascontiguousarray(x), dev)
        return np.asarray(jax.jit(f)(xd))

    def add(self, cid, cases, gate=None, hard=True):
        t0 = time.time()
        th = Threshold("numeric", values=gate[0], source=gate[1]) if gate else None
        if gate == "exact":
            th = Threshold("exact")
        r = OPS(cid, "L1", cases, th, hard=hard)
        for row, c in zip(r.details, cases):
            row.update(ulp_stats(c["out"], c["fp64"]) if c.get("fp64") is not None else {})
            row.pop("ulp_hist", None)
        means = [d["ulp_mean"] for d in r.details if d.get("ulp_mean") is not None]
        if means:
            r.metrics["ulp_mean_max"] = max(means)
        self.checks.append(r)
        print(f"{r.status:<6} {cid:<34} {time.time() - t0:6.1f}s  {r.message[:90]}", flush=True)
        return r

    def unif(self, lo, hi, n=1 << 22):
        return self.rng.uniform(lo, hi, n).astype(np.float32)

    def logu(self, lo, hi, n=1 << 22):
        return np.exp(self.rng.uniform(np.log(lo), np.log(hi), n)).astype(np.float32)

    # ---- FP32：合成 + 真实输入，对文档门槛判
    def fp32(self):
        I = self.inp
        binade = (np.arange(1 << 24, dtype=np.uint32) + np.float32(1.0).view(np.uint32)).view(np.float32)
        real = {
            "exp": [("real.kd_softmax_in", self._kd_shifted())] if "kd_score" in I else [],
            "log": ([("real.kd_target", I["kd_target"])] if "kd_target" in I else []),
            "sigmoid": [("real.router_logits", I["router_logits"])] if "router_logits" in I else [],
            "silu": [("real.mlp_gate", I["mlp_gate_bf16"].astype(np.float32))] if "mlp_gate_bf16" in I else [],
            "rsqrt": [("real.rmsnorm_ms", I["rmsnorm_ms"])] if "rmsnorm_ms" in I else [],
            "relu": [("real.kd_score_valid", I["kd_score"][I["kd_mask"] == 0])] if "kd_score" in I else [],
            "clamp": [("real.mlp_gate", I["mlp_gate_bf16"].astype(np.float32))] if "mlp_gate_bf16" in I else [],
        }
        synth = {
            "exp": [("unif[-87,88]", self.unif(-87, 88)), ("unif[-10,10]", self.unif(-10, 10)), ("unif[-1,0]", self.unif(-1, 0))],
            "log": [("logu[1e-37,1e37]", self.logu(1e-37, 1e37)), ("unif[0.5,2]", self.unif(0.5, 2)), ("binade[1,4)", binade)],
            "sigmoid": [("unif[-30,30]", self.unif(-30, 30))],
            "silu": [("unif[-30,30]", self.unif(-30, 30))],
            "rsqrt": [("all[1,4)", binade), ("logu[1e-37,1e37]", self.logu(1e-37, 1e37))],
            "relu": [("unif[-30,30]", self.unif(-30, 30))],
            "clamp": [("unif[-30,30]", self.unif(-30, 30))],
        }
        for fn, (jf, f64) in self.fns.items():
            cases = []
            for nm, x in real[fn] + synth[fn]:
                x = np.asarray(x, np.float32)
                out = self.run(jf, x)
                truth = f64(x.astype(np.float64))
                c = dict(name=f"{fn}.fp32.{nm}", out=out, fp64=truth, extra={"input": nm})
                if fn in ("relu", "clamp"):
                    c["bitref"] = np.asarray(f64(x), np.float32)
                cases.append(c)
            self.add(f"L1.elem.{fn}.fp32", cases, "exact" if fn in ("relu", "clamp") else GATES[fn])
        # 同一批输入换成 lax 的 accuracy=HIGHEST（JAX「Numerical accuracy」文档：TPU v5p 的 FP32 exp 默认上限 109.5 ULP、
        # HIGHEST 1.5 ULP；log 62 → log2 HIGHEST 3.0；logistic 124 → HIGHEST 3.5），看默认实现之外有没有现成的精确写法
        hi = jax_hi_fns(self.jax)
        for fn, jf in hi.items():
            f64 = self.fns[fn][1]
            cases = []
            for nm, x in real[fn] + synth[fn]:
                x = np.asarray(x, np.float32)
                cases.append(dict(name=f"{fn}.fp32_highest.{nm}", out=self.run(jf, x), fp64=f64(x.astype(np.float64)),
                                  extra={"input": nm, "impl": "lax accuracy=HIGHEST"}))
            self.add(f"L1.elem.{fn}.fp32_highest", cases, GATES[fn])
        # MaxText 开关 transcendental_accuracy_highest 实际用的写法（maxtext.utils.accurate_math，accurate=True）
        mt = maxtext_accurate_fns()
        for fn, jf in mt.items():
            f64 = self.fns[fn][1]
            cases = []
            for nm, x in real[fn] + synth[fn]:
                x = np.asarray(x, np.float32)
                cases.append(dict(name=f"{fn}.fp32_mt.{nm}", out=self.run(jf, x), fp64=f64(x.astype(np.float64)),
                                  extra={"input": nm, "impl": "maxtext.utils.accurate_math, accurate=True"}))
            self.add(f"L1.elem.{fn}.fp32_mt", cases, GATES[fn])
        # 尾部：输出落到 FP32 次正规数或下溢（TPU 把次正规数冲成 0），单列诊断
        cases = []
        for fn, x in (("exp", self.unif(-104, -87)), ("sigmoid", self.unif(-110, -80)), ("silu", self.unif(-110, -80))):
            jf, f64 = self.fns[fn]
            out = self.run(jf, x)
            truth = f64(x.astype(np.float64))
            cases.append(dict(name=f"{fn}.fp32.tail", out=out, fp64=truth,
                              extra={"input": "下溢区", "out_zero_frac": float(np.mean(out == 0))}))
        self.add("L1.elem.tail", cases, None, hard=False)
        self.backward()

    def backward(self):
        """sigmoid / silu 的反向（上游梯度取 1）：jax 默认实现的自动微分 vs MaxText 精确写法（M39：PyTorch 的解析式）
        对 FP64 解析导数的 ULP；FP32 与 BF16（BF16 进出，对 FP64 导数正确舍入到 BF16）。只报。
        输入取 [-8, 8]：再往外 sigmoid 的导数 y (1 - y) 在 FP32 里 1 - y 先抵消成 0，两种写法都一样，ULP 没有意义。"""
        jax, jnp = self.jax, self.jnp
        try:
            from maxtext.utils import accurate_math as am
        except ImportError:
            return
        x32 = self.unif(-8, 8)
        fns = {"sigmoid": (jax.nn.sigmoid, lambda v: am.sigmoid(v, True)), "silu": (jax.nn.silu, lambda v: am.silu(v, True))}
        for fn, (dflt, acc) in fns.items():
            cases = []
            for dt_name, dt in (("fp32", np.float32), ("bf16", BF)):
                x = x32.astype(dt)
                xt = x.astype(np.float64)
                st = 1.0 / (1.0 + np.exp(-xt))
                t = st * (1.0 - st) if fn == "sigmoid" else st * (1.0 + xt * (1.0 - st))
                for arm, f in (("default", dflt), ("maxtext_accurate", acc)):
                    g = jax.jit(jax.vmap(jax.grad(lambda v, f=f: f(v).astype(jnp.float32))))
                    out = np.asarray(g(jax.device_put(np.ascontiguousarray(x), self.dev)))
                    cases.append(dict(name=f"{fn}_bwd.{dt_name}.{arm}", out=out.astype(np.float32) if dt_name == "fp32" else out,
                                      fp64=t, extra={"input": "unif[-8,8]", "impl": arm}))
            self.add(f"L1.elem.{fn}_bwd", cases, None, hard=False)

    def _kd_shifted(self):
        """softmax 里 exp 的真实输入：KD 学生分数减行最大值（只取有效位置）。"""
        s, m = self.inp["kd_score"].astype(np.float64), self.inp["kd_mask"] == 0
        mx = np.where(m, s, -np.inf).max(axis=-1, keepdims=True)
        return (s - mx)[m].astype(np.float32)

    # ---- FP32 sqrt 与除法（优化器 Adam 的 m̂ / (√v̂ + eps)；JAX 文档给 TPU v5p：sqrt 3、reciprocal 40 ULP）
    def sqrt_div(self):
        jax, jnp = self.jax, self.jnp
        r = self.rng
        n = 1 << 22
        pos = self.logu(1e-30, 1e30, n)
        self.add("L1.elem.sqrt.fp32", [dict(name="sqrt.fp32.logu[1e-30,1e30]", out=self.run(jnp.sqrt, pos),
                                            fp64=np.sqrt(pos.astype(np.float64)), extra={"input": "logu[1e-30,1e30]"})],
                 GATES["sqrt"])
        a = (r.standard_normal(n) * np.exp(r.uniform(-20, 20, n))).astype(np.float32)
        b = self.logu(1e-20, 1e20, n) * np.where(r.random(n) < 0.5, -1, 1).astype(np.float32)
        # Adam 一步的形状：m̂ / (√v̂ + eps)，m̂ 与 √v̂ 量级相近
        m = (r.standard_normal(n) * 1e-4).astype(np.float32)
        den = ((np.abs(r.standard_normal(n)) * 1e-4 + 1e-12) + 1e-10).astype(np.float32)
        f = jax.jit(lambda x, y: x / y)
        cases = []
        for nm, x, y in (("div.fp32.random", a, b), ("div.fp32.adam_like", m, den)):
            out = np.asarray(f(jax.device_put(x, self.dev), jax.device_put(y, self.dev)))
            cases.append(dict(name=nm, out=out, fp64=x.astype(np.float64) / y.astype(np.float64), extra={"input": nm}))
        self.add("L1.elem.div.fp32", cases, GATES["div"])
        # Adam 的偏差修正：optax 在图里算 1 − β^t（t 是 int32 的步数）再除；GPU 在 Python 里用 FP64 算。只报
        bc = jax.jit(lambda t: (1.0 - 0.9 ** t, 1.0 - 0.95 ** t))
        tt = np.arange(1, 1001, dtype=np.int32)
        o1, o2 = (np.asarray(x) for x in bc(jax.device_put(tt, self.dev)))
        self.add("L1.elem.bias_correction.fp32", [
            dict(name="bias_correction.fp32.b0.9", out=o1, fp64=1.0 - 0.9 ** tt.astype(np.float64), extra={"input": "t=1..1000"}),
            dict(name="bias_correction.fp32.b0.95", out=o2, fp64=1.0 - 0.95 ** tt.astype(np.float64), extra={"input": "t=1..1000"}),
        ], None, hard=False)
        # MaxText 开关 muon_adam_accurate 的三样（maxtext.optimizers.muon_mcore_compat 的 _sqrt、_recip、_one_minus_pow）
        try:
            from maxtext.optimizers import muon_mcore_compat as mc
        except ImportError:
            return
        if not hasattr(mc, "_one_minus_pow"):
            return
        self.add("L1.elem.sqrt.fp32_mt", [dict(name="sqrt.fp32_mt.logu[1e-30,1e30]", out=self.run(mc._sqrt, pos),
                                               fp64=np.sqrt(pos.astype(np.float64)), extra={"input": "logu[1e-30,1e30]"})],
                 GATES["sqrt"])
        fm = jax.jit(mc._div) if hasattr(mc, "_div") else jax.jit(lambda x, y: x * mc._recip(y))
        cases = []
        for nm, x, y in (("div.fp32_mt.random", a, b), ("div.fp32_mt.adam_like", m, den)):
            out = np.asarray(fm(jax.device_put(x, self.dev), jax.device_put(y, self.dev)))
            cases.append(dict(name=nm, out=out, fp64=x.astype(np.float64) / y.astype(np.float64), extra={"input": nm}))
        self.add("L1.elem.div.fp32_mt", cases, GATES["div"])
        bm = jax.jit(lambda t: (mc._one_minus_pow(0.9, t), mc._one_minus_pow(0.95, t)))
        o1, o2 = (np.asarray(x) for x in bm(jax.device_put(tt, self.dev)))
        self.add("L1.elem.bias_correction.fp32_mt", [
            dict(name="bias_correction.fp32_mt.b0.9", out=o1, fp64=1.0 - 0.9 ** tt.astype(np.float64), extra={"input": "t=1..1000"}),
            dict(name="bias_correction.fp32_mt.b0.95", out=o2, fp64=1.0 - 0.95 ** tt.astype(np.float64), extra={"input": "t=1..1000"}),
        ], None, hard=False)

    # ---- BF16：全部有限 BF16 穷举，BF16 直接算 vs 先升 FP32 算再舍到 BF16（只报）
    def bf16(self):
        jnp = self.jnp
        allbf = np.arange(1 << 16, dtype=np.uint16).view(BF)
        allbf = allbf[np.isfinite(allbf.astype(np.float32))]
        for fn, (jf, f64) in self.fns.items():
            x = allbf[allbf.astype(np.float32) > 0] if fn in ("rsqrt", "log") else allbf
            # 输入或真值是次正规数的位置另计（TPU 与 XLA CPU 都把次正规数冲成 0），不进 ULP 统计
            with np.errstate(all="ignore"):
                t_all = f64(x.astype(np.float64))
            keep = _normal(x.astype(np.float64)) & _normal(t_all.astype(BF).astype(np.float64))
            x = x[keep]
            direct = self.run(jf, x)
            up = self.run(lambda v, jf=jf: jf(v.astype(jnp.float32)).astype(jnp.bfloat16), x)
            truth = t_all[keep]
            cases = [dict(name=f"{fn}.bf16_direct", out=direct, fp64=truth, bitref=up,
                          extra={"impl": "BF16 张量直接算，BF16 出；bitref = 先升 FP32 算再舍到 BF16",
                                 "excluded_subnormal": int((~keep).sum())}),
                     dict(name=f"{fn}.fp32_then_bf16", out=up, fp64=truth, extra={"impl": "先升 FP32 算再舍到 BF16"})]
            self.add(f"L1.elem.{fn}.bf16", cases, None, hard=False)

    # ---- 加、乘、类型转换逐位（同混元 3 L1.elem.basic）
    def basic(self):
        jnp = self.jnp
        r = self.rng
        n = 1 << 22
        a32 = r.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
        b32 = r.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
        a32 = np.concatenate([a32, r.standard_normal(n).astype(np.float32)])
        b32 = np.concatenate([b32, r.standard_normal(n).astype(np.float32)])
        # TPU 把次正规数冲成 0：输入、输出含次正规数的位置另计，不进逐位判定
        normal = _normal
        cases = []
        jax = self.jax
        with np.errstate(all="ignore"):
            for op, f in (("add", lambda a, b: a + b), ("mul", lambda a, b: a * b)):
                o = np.asarray(jax.jit(f)(jax.device_put(a32, self.dev), jax.device_put(b32, self.dev)))
                ref = f(a32, b32)
                keep = normal(a32) & normal(b32) & normal(ref)
                cases.append(dict(name=f"{op}.fp32", out=o[keep], bitref=ref[keep],
                                  fp64=f(a32.astype(np.float64), b32.astype(np.float64))[keep],
                                  extra={"excluded_subnormal": int((~keep).sum())}))
            keep = normal(a32)
            o = self.run(lambda v: v.astype(jnp.bfloat16), a32)
            cases.append(dict(name="cast.fp32_to_bf16", out=o[keep], bitref=a32.astype(BF)[keep],
                              fp64=a32.astype(np.float64)[keep], extra={"excluded_subnormal": int((~keep).sum())}))
        self.add("L1.elem.basic", cases, "exact")

    # ---- log_softmax / LSE：图里的写法与三步拆解（只报），对 FP64
    def lse(self):
        jax, jnp = self.jax, self.jnp
        rows = {}
        if "kd_score" in self.inp:
            rows["kd_real"] = (self.inp["kd_score"], self.inp["kd_mask"])
        r = self.rng
        S = 4096
        causal = np.where(np.tril(np.ones((1024, S), bool), k=S - 1024), 0.0, -np.inf).astype(np.float32)
        for nm, sd in (("normal_sd0.01", 0.01), ("normal_sd1", 1.0), ("normal_sd5", 5.0)):
            rows[nm] = (r.standard_normal((1024, S)).astype(np.float32) * np.float32(sd), causal)
        self.lse_rows = {}
        for nm, (s, mask) in rows.items():
            valid = mask == 0
            s64 = np.where(valid, s.astype(np.float64), -np.inf)
            mx = s64.max(axis=-1, keepdims=True)
            sum64 = np.exp(s64 - mx).sum(axis=-1)
            lse64 = mx[:, 0] + np.log(sum64)
            add = (s + mask).astype(np.float32)          # 图里：分数 + 加性掩码（KD 的掩码是 −2.38e38 / −inf）
            res = {}
            hi = jax_hi_fns(jax)
            for tag, dev in (("tpu", self.dev), ("cpu", self.cpu_dev), ("tpu_highest", self.dev)):
                if dev is None:
                    continue
                if tag == "tpu_highest":   # 同一个写法（x − max、exp、求和、log），exp 与 log 换 accuracy=HIGHEST
                    lsefn = lambda v: _lse_with(v, hi["exp"], hi["log"], jnp)  # noqa: E731
                    lsm = self.run(lambda v: v - lsefn(v)[..., None], add, dev)
                    lse = self.run(lsefn, add, dev)
                else:
                    lsm = self.run(lambda v: jax.nn.log_softmax(v, axis=-1), add, dev)
                    lse = self.run(lambda v: jax.scipy.special.logsumexp(v, axis=-1), add, dev)
                # 由 log_softmax 输出反推的 LSE：第一个有效位置上 s − log_softmax（同 t1-layer0-kd-attribution 的口径）
                j = valid.argmax(axis=-1)
                ar = np.arange(s.shape[0])
                implied = s[ar, j].astype(np.float64) - lsm[ar, j].astype(np.float64)
                sp = np.spacing(lse64.astype(np.float32)).astype(np.float64)
                # 三步拆解：exp(s − max) 逐元素、FP32 求和、log；每一步的输入都换成 FP64 正确舍入到 FP32 的值
                sh32 = np.where(valid, (s64 - mx), -np.inf).astype(np.float32)
                ex, lgf = (hi["exp"], hi["log"]) if tag == "tpu_highest" else (jnp.exp, jnp.log)
                e = self.run(ex, sh32, dev)
                e_ref = np.exp(sh32.astype(np.float64))
                e_cr = np.where(valid, e_ref, 0).astype(np.float32)
                ssum = self.run(lambda v: jnp.sum(v, axis=-1), e_cr, dev)
                ssum_ref = e_cr.astype(np.float64).sum(axis=-1)
                s32 = sum64.astype(np.float32)
                lg = self.run(lgf, s32, dev)
                lg_ref = np.log(s32.astype(np.float64))
                lgsp = np.spacing(lg_ref.astype(np.float32)).astype(np.float64)
                res[tag] = {
                    "lse_logsumexp_ulp_signed_mean": float(((lse.astype(np.float64) - lse64) / sp).mean()),
                    "lse_logsumexp_ulp_absmax": float(np.abs((lse.astype(np.float64) - lse64) / sp).max()),
                    "lse_implied_by_log_softmax_ulp_signed_mean": float(((implied - lse64) / sp).mean()),
                    "lse_implied_frac_rows_positive": float(((implied - lse64) > 0).mean()),
                    "log_softmax_rel_l2": M.rel_l2(lsm.astype(np.float64)[valid], (s64 - lse64[:, None])[valid]),
                    "step_exp_ulp_signed_mean": float(((e.astype(np.float64) - e_ref)[valid] /
                                                       np.spacing(e_cr[valid]).astype(np.float64)).mean()),
                    "step_sum_rel_err_signed_mean": float(((ssum.astype(np.float64) - ssum_ref) / ssum_ref).mean()),
                    "step_sum_err_in_lse_ulp_signed_mean": float((((ssum.astype(np.float64) - ssum_ref) / ssum_ref) / sp).mean()),
                    "step_log_ulp_signed_mean": float(((lg.astype(np.float64) - lg_ref) / lgsp).mean()),
                    "step_log_ulp_absmax": float(np.abs((lg.astype(np.float64) - lg_ref) / lgsp).max()),
                }
            res["rows"] = int(s.shape[0])
            res["lse_mean"] = float(lse64.mean())
            res["lse_ulp_mean"] = float(np.spacing(lse64.astype(np.float32)).mean())
            self.lse_rows[nm] = res
            print(nm, json.dumps(res, ensure_ascii=False), flush=True)
        # 也写成一条 op_replay 检查（只报）：logsumexp 输出对 FP64，便于在清单里引用
        cases = []
        for nm, (s, mask) in rows.items():
            valid = mask == 0
            s64 = np.where(valid, s.astype(np.float64), -np.inf)
            mx = s64.max(axis=-1, keepdims=True)
            lse64 = mx[:, 0] + np.log(np.exp(s64 - mx).sum(axis=-1))
            out = self.run(lambda v: jax.scipy.special.logsumexp(v, axis=-1), (s + mask).astype(np.float32))
            cases.append(dict(name=f"lse.fp32.{nm}", out=out, fp64=lse64, extra={"input": nm}))
        self.add("L1.elem.lse.fp32", cases, None, hard=False)


def jax_hi_fns(jax) -> dict:
    """lax 的 accuracy=HIGHEST 版本（exp、log、sigmoid = lax.logistic、silu = x · logistic）。"""
    from jax import lax
    hi = lax.AccuracyMode.HIGHEST
    return {"exp": lambda v: lax.exp(v, accuracy=hi), "log": lambda v: lax.log(v, accuracy=hi),
            "sigmoid": lambda v: lax.logistic(v, accuracy=hi), "silu": lambda v: v * lax.logistic(v, accuracy=hi)}


def maxtext_accurate_fns() -> dict:
    """MaxText 开关 transcendental_accuracy_highest 的实现（maxtext.utils.accurate_math，accurate=True）；装不上就空。"""
    try:
        from maxtext.utils import accurate_math as am
    except ImportError:
        return {}
    return {"exp": lambda v: am.exp(v, True), "log": lambda v: am.log(v, True),
            "sigmoid": lambda v: am.sigmoid(v, True), "silu": lambda v: am.silu(v, True)}


def _lse_with(v, exp, log, jnp):
    """logsumexp 的同一写法（jax.nn.logsumexp：减去行最大值、exp、求和、log、加回最大值），exp / log 可换。"""
    m = jnp.max(v, axis=-1, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    return (log(jnp.sum(exp(v - m), axis=-1, keepdims=True)) + m)[..., 0]


def summarize(checks) -> dict:
    """每个函数、每种 dtype：最差用例的 ulp_max、最大的平均 ULP、最小的正确舍入命中率。"""
    out = {}
    for r in checks:
        for d in r.details:
            fn, dt = d["name"].split(".")[0], ("bf16" if ".bf16" in d["name"] else "fp32")
            if "fp32_then_bf16" in d["name"]:
                dt = "fp32_then_bf16"
            if r.id == "L1.elem.tail":
                dt = "fp32_tail_underflow"
            if ".fp32_highest." in d["name"]:
                dt = "fp32_highest"
            if ".fp32_mt." in d["name"]:
                dt = "fp32_maxtext_accurate"
            s = out.setdefault(fn, {}).setdefault(dt, {"ulp_max": 0, "ulp_mean": 0.0, "cr_match_frac": 1.0, "cases": 0})
            s["cases"] += 1
            if d.get("ulp_max") is not None:
                s["ulp_max"] = max(s["ulp_max"], d["ulp_max"])
            if d.get("ulp_mean") is not None:
                s["ulp_mean"] = max(s["ulp_mean"], d["ulp_mean"])
            if d.get("cr_match_frac") is not None:
                s["cr_match_frac"] = min(s["cr_match_frac"], d["cr_match_frac"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--make-inputs", action="store_true")
    p.add_argument("--gpu-step", help="GPU golden 运行目录的 step_1（--make-inputs）")
    p.add_argument("--kd", help="TPU debug_sow_indexer 的 diag_kd_inputs.safetensors（--make-inputs）")
    p.add_argument("--inputs", help="--make-inputs 生成的 npz（本地或 gs://）")
    p.add_argument("--out", required=True)
    p.add_argument("--cpu", action="store_true", help="在 CPU 上跑：只验证脚本，不是 TPU 的数值")
    p.add_argument("--only", default="", help="逗号分隔：fp32,sqrt_div,bf16,basic,lse")
    a, _rest = p.parse_known_args(argv)   # run.sh 会附带 --stage / --load / --golden 等，这里不用
    if a.make_inputs:
        make_inputs(a.gpu_step, a.kd, a.out)
        return 0
    import jax
    if not a.cpu and not jax.distributed.is_initialized():
        try:
            jax.distributed.initialize()
        except Exception as e:  # noqa: BLE001 — 单机 TPU 不需要分布式初始化
            print("jax.distributed.initialize 跳过：", e)
    if jax.process_index() == 0:   # 先建运行目录：中途失败时 run.sh 也能认出交结果的 Pod、取回日志
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    dev = jax.local_devices(backend="cpu")[0] if a.cpu else jax.local_devices()[0]
    cpu_dev = None if a.cpu else jax.local_devices(backend="cpu")[0]
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    job = Job(dev, cpu_dev, load_inputs(a.inputs))
    only = set(filter(None, a.only.split(",")))
    for part in ("fp32", "sqrt_div", "bf16", "basic", "lse"):
        if not only or part in only:
            getattr(job, part)()
    checks = []
    for r in job.checks:
        d = r.to_json()
        d["details_inline"] = r.details
        checks.append(d)
    summ = {k: sum(1 for c in checks if c["status"] == k) for k in ("pass", "fail", "inconclusive", "error")}
    rec = {
        "what": "逐元素函数对 FP64 的 ULP（TPU 一侧，JAX / XLA）；混元 3 GPU 一侧 L1.elem.* 同一组用例，外加 log、relu、clamp、LSE",
        "platform": dev.platform, "device_kind": dev.device_kind, "jax": jax.__version__,
        "started_at": started, "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": a.inputs, "seed": SEED, "rule": "ulp_max ≤ 门槛（每条检查的 threshold.source）；relu / clamp / 加乘转换逐位；BF16、尾部、LSE 只报",
        "per_function": summarize(job.checks), "lse_breakdown": getattr(job, "lse_rows", {}),
        "summary": summ, "checks": checks,
        "note": "CPU 上跑的只验证脚本，不是 TPU 的数值" if a.cpu else "TPU 一侧",
    }
    if jax.process_index() == 0:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float))
        print(json.dumps(rec["per_function"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
