"""L1 单算子回放与合成用例（GPU 侧）。规格见 docs/dev/design/0003-l1-op-replay.md 与 docs/reference/checks/op_replay.md。

用第 1 步抓下的真实算子输入，在 GPU 上单独回放算子，对 FP64 真值、对训练时抓的输出比；外加合成用例。
产出与 `janus run` 同一格式：results.json（docs/reference/result-schema.md）、details/<检查>.jsonl、report.md。

用法（Pod 里，只用一张卡、小规模，训练任务同时在跑也不影响）：
    source /data/env.sh; export PYTHONPATH=/tmp/janus-ops:$PYTHONPATH
    CUDA_VISIBLE_DEVICES=0 python models/hunyuan3/frameworks/megatron/l1_ops.py --run /data/janus-runs2/d-a --out /data/janus-l1/d-a

各检查对应的底账条目写在每个方法开头的注释里（#25–#47）；门槛的来源写在每条的 source 里。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F

from janus import __version__
from janus.analysis import ops as O
from janus.catalog import Catalog
from janus.checks import REGISTRY, Threshold
from janus.checks.base import ERROR, FAIL, INCONCLUSIVE, PASS
from janus.run import _render_report
from janus.store import Run

BF = ml_dtypes.bfloat16
DEV = "cuda:0"
STEP = 1
OPS = REGISTRY["op_replay"]
SEED = 20261103


def T(a: np.ndarray) -> torch.Tensor:
    """numpy → GPU，BF16 保持位模式。"""
    a = np.ascontiguousarray(a)
    if a.dtype == BF:
        return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).to(DEV)
    return torch.from_numpy(a).to(DEV)


def N(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(BF)
    return t.numpy()


def num(values: dict, source: str) -> Threshold:
    return Threshold("numeric", values=values, source=source)


class tf32:
    """临时设定 TF32 开关（matmul 与 cuDNN 一起），退出时恢复。"""

    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        self.old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = self.on

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.old


def environment() -> dict:
    import transformer_engine as te
    return {
        "torch": torch.__version__, "transformer_engine": te.__version__, "cuda": torch.version.cuda,
        "cublas": os.environ.get("CUBLAS_VERSION"), "device": torch.cuda.get_device_name(0),
        # 进程启动时的默认值（未被本脚本改动前）：容器设了 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1，所以 matmul 默认就开 TF32
        "allow_tf32_matmul_default": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn_default": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision_default": torch.get_float32_matmul_precision(),
        "env": {k: os.environ.get(k) for k in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE",
                                               "CUBLAS_WORKSPACE_CONFIG", "NVTE_ALLOW_NONDETERMINISTIC_ALGO")},
    }


# ---------------------------------------------------------------- CPU 模拟并行（numpy，多进程按行切块）

_EMU: dict = {}


def _emu_chunk(args):
    key, rows, kw = args
    x, w = _EMU[key]
    return O.gemm_emulate(x, w, rows=rows, **kw)


def emulate_parallel(key: str, x, w, rows, kw: dict, workers: int) -> np.ndarray:
    _EMU[key] = (x, w)
    chunks = [rows[i:i + 8] for i in range(0, len(rows), 8)]
    with ProcessPoolExecutor(workers) as ex:                 # fork：子进程继承 _EMU，不碰 CUDA
        parts = list(ex.map(_emu_chunk, [(key, c, kw) for c in chunks]))
    return np.concatenate(parts, axis=0)


TC = {"mode": "tc", "group": 16, "frac_bits": 25, "rounding": "rz"}   # B200 上 BF16 GEMM 的累加模型（试出来的，见 docs/dev/design/0003-l1-op-replay.md）


def _bad_range(x, out, truth, ulp: int = 4) -> dict:
    """输入落在哪个区间时输出离正确舍入超过 ulp 格：报超线元素数与对应输入的最小、最大值。"""
    from janus import metrics as M
    d = M.ulp_distance(out, O.round_from_f64(truth, np.asarray(out).dtype))
    bad = d > ulp
    if not bad.any():
        return {f"n_ulp_gt{ulp}": 0}
    xs = np.asarray(x).astype(np.float64).ravel()[bad]
    return {f"n_ulp_gt{ulp}": int(bad.sum()), f"x_ulp_gt{ulp}_min": float(xs.min()), f"x_ulp_gt{ulp}_max": float(xs.max())}


# ---------------------------------------------------------------- 主流程

class L1:
    def __init__(self, run: Run, workers: int, emulate_rows: int):
        self.run, self.workers, self.emulate_rows = run, workers, emulate_rows
        self.checks = []
        self.evidence = [{"path": str(run.path), "sha256": run.meta_sha256()}]
        mc = run.meta.get("model_config") or {}
        self.eps = float(mc.get("layernorm_epsilon", 1e-5))
        self.topk = int(mc.get("moe_router_topk", 8))
        self.scale = float(mc.get("moe_router_topk_scaling_factor") or 1.0)
        self.layers = sorted({int(n.split(".")[1]) for n in run.names(STEP, "layers.*.out")})
        self.moe = sorted({int(n.split(".")[1]) for n in run.names(STEP, "layers.*.mlp.router.logits")})
        self.rng = np.random.default_rng(SEED)
        torch.manual_seed(SEED)

    def t(self, name):
        return self.run.tensor(STEP, name)

    def p(self, layer, name):
        return self.run.tensor(STEP, f"param.decoder.layers.{layer}.{name}")

    def add(self, cid, cases, threshold=None, hard=True, params=None):
        t0 = time.time()
        r = OPS(cid, "L1", cases, threshold, hard=hard, params=params)
        r.evidence = list(self.evidence)
        self.checks.append(r)
        print(f"{r.status:<13} {cid:<34} {time.time() - t0:6.1f}s  {r.message[:100]}", flush=True)
        return r

    # ---- #29 / #26：路由 GEMM 与注意力输出投影回放
    def gemm(self):
        from megatron.core.transformer.moe.moe_utils import router_gating_linear
        rep, var = [], []
        self.router_inputs = {}
        for L in self.moe:
            x = self.t(f"layers.{L}.mlp_norm.out").reshape(-1, self.t(f"layers.{L}.mlp_norm.out").shape[-1])
            w = self.p(L, "mlp.router.weight")
            cap = self.t(f"layers.{L}.mlp.router.logits").reshape(x.shape[0], -1)
            self.router_inputs[L] = (x, w, cap)
            xt, wt = T(x), T(w)
            f64 = N(xt.double() @ wt.double().t())
            with torch.no_grad():
                out = N(router_gating_linear(xt, wt, None, torch.float32))      # Megatron 训练时的同一函数
            rep.append(dict(name=f"layers.{L}.router.logits", out=out, fp64=f64, bitref=cap,
                            extra={"impl": "megatron router_gating_linear（moe_router_dtype=fp32 → TE general_gemm，BF16 入 FP32 出）"}))
            for on in (False, True):
                with tf32(on), torch.no_grad():
                    mm = N(torch.mm(xt.float(), wt.float().t()))
                var.append(dict(name=f"layers.{L}.router.logits.upcast_fp32_mm.tf32_{'on' if on else 'off'}",
                                out=mm, fp64=f64, bitref=cap, extra={"impl": "先升 FP32 再 torch.mm", "allow_tf32": on}))
        self.add("L1.gemm.router.replay", rep, Threshold("exact"))
        self.add("L1.gemm.router.upcast_fp32", var, None, hard=False)

        from transformer_engine.pytorch.cpp_extensions import general_gemm
        cases = []
        self.proj_inputs = {}
        for L in self.layers:
            x = self.t(f"layers.{L}.attn.core.out")
            x = x.reshape(-1, x.shape[-1])
            w = self.p(L, "self_attention.linear_proj.weight")
            cap = self.t(f"layers.{L}.attn.proj.out").reshape(x.shape[0], -1)
            self.proj_inputs[L] = (x, w, cap)
            xt, wt = T(x), T(w)
            f64 = N(xt.double() @ wt.double().t())
            with torch.no_grad():
                te_out = N(general_gemm(wt, xt, torch.bfloat16, layout="TN")[0])
                lin = N(F.linear(xt, wt))
            cases.append(dict(name=f"layers.{L}.attn.proj.te_gemm", out=te_out, fp64=f64, bitref=cap,
                              extra={"impl": "TE general_gemm（TE Linear 的同一调用），BF16 出"}))
            cases.append(dict(name=f"layers.{L}.attn.proj.torch_linear", out=lin, fp64=f64, bitref=cap,
                              extra={"impl": "torch F.linear，BF16 出"}))
        self.add("L1.gemm.attn_proj.replay", cases, Threshold("exact"))

    # ---- #26：qkv（RMSNorm 融合 GEMM）、q/k 归一化、MLP 前归一化回放
    def norms(self):
        import transformer_engine.pytorch as tep
        qkv_cases, norm_cases = [], []
        self.qkv_inputs = {}
        for L in self.layers:
            x = self.t(f"layers.{L}.in")
            h = x.shape[-1]
            x = x.reshape(-1, h)
            wn = self.p(L, "self_attention.linear_qkv.layer_norm_weight")
            wq = self.p(L, "self_attention.linear_qkv.weight")
            cap = self.t(f"layers.{L}.attn.qkv.out").reshape(x.shape[0], -1)
            lnl = tep.LayerNormLinear(h, wq.shape[0], eps=self.eps, bias=False, normalization="RMSNorm",
                                      params_dtype=torch.bfloat16)
            rn = tep.RMSNorm(h, eps=self.eps, params_dtype=torch.bfloat16)
            with torch.no_grad():
                lnl.weight.copy_(T(wq))
                lnl.layer_norm_weight.copy_(T(wn))
                rn.weight.copy_(T(wn))
                out = N(lnl(T(x)))
                ln = rn(T(x))
                f64 = N(ln.double() @ T(wq).double().t())
            self.qkv_inputs[L] = (x, wn, wq, N(ln), cap)
            qkv_cases.append(dict(name=f"layers.{L}.attn.qkv", out=out, fp64=f64, bitref=cap,
                                  extra={"impl": "TE LayerNormLinear（RMSNorm 融合 GEMM）",
                                         "fp64_of": "GEMM 部分：FP64(RMSNorm 的 BF16 输出 · Wᵀ)"}))
            # q / k 归一化：从抓下的 qkv 输出按 Megatron 的分组布局切出 q、k（每组 np/ng 个 q 头 + 1 k + 1 v）
            mc = self.run.meta.get("model_config") or {}
            nh, ng, hd = int(mc.get("num_attention_heads")), int(mc.get("num_query_groups")), int(mc.get("kv_channels"))
            g = cap.reshape(-1, ng, nh // ng + 2, hd)
            q = np.ascontiguousarray(g[:, :, : nh // ng, :].reshape(-1, nh, hd))
            k = np.ascontiguousarray(g[:, :, nh // ng, :])
            for nm, inp, pname in (("q_norm", q, "q_layernorm"), ("k_norm", k, "k_layernorm")):
                wn2 = self.p(L, f"self_attention.{pname}.weight")
                m = tep.RMSNorm(hd, eps=self.eps, params_dtype=torch.bfloat16)
                with torch.no_grad():
                    m.weight.copy_(T(wn2))
                    o = N(m(T(inp)))
                norm_cases.append(dict(name=f"layers.{L}.attn.{nm}", out=o, fp64=O.rmsnorm_f64(inp, wn2, self.eps),
                                       bitref=self.t(f"layers.{L}.attn.{nm}.out").reshape(o.shape),
                                       extra={"impl": "TE RMSNorm，BF16 入出"}))
            if L in self.moe:
                # MLP 前归一化的输入 = 层输入 + 注意力输出（BF16 相加）
                with torch.no_grad():
                    hs = T(x) + T(self.t(f"layers.{L}.attn.out").reshape(x.shape))
                    m = tep.RMSNorm(h, eps=self.eps, params_dtype=torch.bfloat16)
                    wm = self.p(L, "pre_mlp_layernorm.weight")
                    m.weight.copy_(T(wm))
                    o = N(m(hs))
                norm_cases.append(dict(name=f"layers.{L}.mlp_norm", out=o, fp64=O.rmsnorm_f64(N(hs), wm, self.eps),
                                       bitref=self.t(f"layers.{L}.mlp_norm.out").reshape(o.shape),
                                       extra={"impl": "TE RMSNorm，输入 = 层输入 + 注意力输出（BF16）"}))
        self.add("L1.gemm.qkv.replay", qkv_cases, Threshold("exact"))
        self.add("L1.norm.replay", norm_cases, Threshold("exact"))

    # ---- #28：初等函数 ULP（FP32；BF16 直接算 vs 先升 FP32）
    def elementwise(self):
        r = self.rng
        logits = np.concatenate([self.t(f"layers.{L}.mlp.router.logits").ravel() for L in self.moe]).astype(np.float32)
        lg = np.concatenate([self.t(f"layers.{L}.mlp.router.logits").reshape(-1, self.t(f"layers.{L}.mlp.router.logits").shape[-1])
                             for L in self.moe]).astype(np.float64)
        sm_in = (lg - lg.max(axis=-1, keepdims=True)).astype(np.float32).ravel()        # softmax 里 exp 的真实输入
        # 稠密层 MLP 的 fc1 输出 = [gate | up]，silu 作用在 gate 半边
        fc1n = self.run.names(STEP, "layers.*.mlp.fc1.out")
        fc1 = self.t(fc1n[0]) if fc1n else None
        gate = fc1.reshape(-1, fc1.shape[-1])[:, : fc1.shape[-1] // 2].astype(np.float32).ravel() if fc1 is not None else None
        rs_in = []
        for L in self.layers:
            x = self.t(f"layers.{L}.in").astype(np.float64)
            rs_in.append(((x * x).mean(axis=-1) + self.eps).ravel())
        rs_in = np.concatenate(rs_in).astype(np.float32)                                # RMSNorm 里 rsqrt 的真实输入

        def unif(lo, hi, n=1 << 22):
            return r.uniform(lo, hi, n).astype(np.float32)

        def logu(lo, hi, n=1 << 22):
            return np.exp(r.uniform(np.log(lo), np.log(hi), n)).astype(np.float32)
        binade = (np.arange(1 << 24, dtype=np.uint32) + np.float32(1.0).view(np.uint32)).view(np.float32)  # [1, 4) 全部 FP32
        fns = {
            "exp": (torch.exp, np.exp, [("real.softmax_in", sm_in), ("real.router_logits", logits),
                                        ("unif[-87,88]", unif(-87, 88)), ("unif[-10,10]", unif(-10, 10))]),
            "sigmoid": (torch.sigmoid, O.sigmoid_f64, [("real.router_logits", logits), ("unif[-30,30]", unif(-30, 30))]),
            "silu": (F.silu, O.silu_f64, ([("real.mlp_gate", gate)] if gate is not None else []) +
                     [("unif[-30,30]", unif(-30, 30))]),
            "rsqrt": (torch.rsqrt, lambda v: 1.0 / np.sqrt(np.asarray(v, np.float64)),
                      [("real.rmsnorm_ms", rs_in), ("all[1,4)", binade), ("logu[1e-37,1e37]", logu(1e-37, 1e37))]),
        }
        src = {
            "exp": ({"ulp_max": 2}, "CUDA Programming Guide 数学函数附录：expf 最大 2 ULP（非 fast-math）"),
            "rsqrt": ({"ulp_max": 2}, "CUDA Programming Guide 数学函数附录：rsqrtf 最大 2 ULP"),
            "sigmoid": ({"ulp_max": 4}, "推断：torch 的 sigmoid = 1/(1+exp(−x))，expf 2 ULP 经一次加、一次除（各 0 ULP）后的起点"),
            "silu": ({"ulp_max": 4}, "推断：silu = x·sigmoid(x)，同 sigmoid 再多一次正确舍入的乘法"),
        }
        for fn, (tf, f64, inputs) in fns.items():
            cases = []
            for nm, x in inputs:
                with torch.no_grad():
                    out = N(tf(T(x)))
                truth = f64(x.astype(np.float64))
                cases.append(dict(name=f"{fn}.fp32.{nm}", out=out, fp64=truth, extra={"input": nm, **_bad_range(x, out, truth)}))
            self.add(f"L1.elem.{fn}.fp32", cases, num(*src[fn]))
        # 尾部：x < −88.7 时 exp(−x) 在 FP32 里溢出，1/(1+exp(−x)) 得 0，而真值是次正规数。单列诊断，不并进上面的门槛
        cases = []
        for fn in ("sigmoid", "silu"):
            tf, f64, _ = fns[fn]
            x = unif(-110, 110)
            with torch.no_grad():
                out = N(tf(T(x)))
            truth = f64(x.astype(np.float64))
            cases.append(dict(name=f"{fn}.fp32.unif[-110,110]", out=out, fp64=truth, extra={"input": "unif[-110,110]", **_bad_range(x, out, truth)}))
        self.add("L1.elem.tail", cases, None, hard=False)

        # BF16 输入：全部有限 BF16 值穷举，直接算（BF16 出） vs 先升 FP32 算再舍到 BF16
        allbf = np.arange(1 << 16, dtype=np.uint16).view(BF)
        allbf = allbf[np.isfinite(allbf.astype(np.float32))]
        cases = []
        for fn, (tf, f64, _) in fns.items():
            x = allbf[allbf.astype(np.float32) > 0] if fn == "rsqrt" else allbf
            with torch.no_grad():
                direct = N(tf(T(x)))
                up32 = tf(T(x).float())
                up = N(up32.to(torch.bfloat16))
            truth = f64(x.astype(np.float64))
            cases.append(dict(name=f"{fn}.bf16_direct", out=direct, fp64=truth, bitref=up,
                              extra={"impl": "BF16 张量直接算，BF16 出；bitref = 先升 FP32 算再舍到 BF16", **_bad_range(x, direct, truth)}))
            cases.append(dict(name=f"{fn}.fp32_then_bf16", out=up, fp64=truth, extra={"impl": "先升 FP32 算再舍到 BF16"}))
            cases.append(dict(name=f"{fn}.fp32_out_on_bf16_grid", out=N(up32), fp64=truth, extra={"impl": "先升 FP32 算，FP32 出"}))
        self.add("L1.elem.bf16_vs_fp32", cases, None, hard=False)

    # ---- #27（= #57）：逐元素加、乘、类型转换，torch GPU ↔ numpy CPU 逐位
    def basic(self):
        r = self.rng
        n = 1 << 22
        a32 = r.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
        b32 = r.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
        # 再加一批「正常量级」的数，免得全随机位模式里大部分加法被量级悬殊吃掉
        a32 = np.concatenate([a32, r.standard_normal(n).astype(np.float32)])
        b32 = np.concatenate([b32, r.standard_normal(n).astype(np.float32)])
        a16 = r.integers(0, 1 << 16, 2 * n, dtype=np.uint32).astype(np.uint16).view(BF)
        b16 = r.integers(0, 1 << 16, 2 * n, dtype=np.uint32).astype(np.uint16).view(BF)
        cases = []
        with np.errstate(all="ignore"), torch.no_grad():
            for dt, a, b in (("fp32", a32, b32), ("bf16", a16, b16)):
                A, B = T(a), T(b)
                a64, b64 = a.astype(np.float64), b.astype(np.float64)
                cases.append(dict(name=f"add.{dt}", out=N(A + B), bitref=a + b, fp64=a64 + b64))
                cases.append(dict(name=f"mul.{dt}", out=N(A * B), bitref=a * b, fp64=a64 * b64))
            A = T(a32)
            cases.append(dict(name="cast.fp32_to_bf16", out=N(A.to(torch.bfloat16)), bitref=a32.astype(BF),
                              fp64=a32.astype(np.float64)))
            cases.append(dict(name="cast.fp32_to_fp16", out=N(A.to(torch.float16)), bitref=a32.astype(np.float16),
                              fp64=a32.astype(np.float64)))
            cases.append(dict(name="cast.bf16_to_fp32", out=N(T(a16).float()), bitref=a16.astype(np.float32),
                              fp64=a16.astype(np.float64)))
            d64 = r.standard_normal(n) * np.exp(r.uniform(-100, 100, n))
            cases.append(dict(name="cast.fp64_to_fp32", out=N(T(d64).float()), bitref=d64.astype(np.float32), fp64=d64))
        self.add("L1.elem.basic", cases, num({"bit_match_frac": 1.0, "cr_match_frac": 1.0, "nonfinite_mismatch": 0},
                                             "IEEE 754 与 CUDA Programming Guide：x+y、x*y、类型转换都是 RNE 正确舍入（0 ULP）"))

    # ---- #25：R-ratio 三方比，干净实现 vs 多舍一次的实现
    def rratio(self):
        import transformer_engine.pytorch as tep
        L = self.moe[0] if self.moe else self.layers[0]
        x = self.t(f"layers.{L}.in")
        h = x.shape[-1]
        x = x.reshape(-1, h)
        wn = self.p(L, "self_attention.linear_qkv.layer_norm_weight")
        truth = O.rmsnorm_f64(x, wn, self.eps)
        rn = tep.RMSNorm(h, eps=self.eps, params_dtype=torch.bfloat16)
        with torch.no_grad():
            rn.weight.copy_(T(wn))
            ref = N(rn(T(x)))
            xf, wf = T(x).float(), T(wn).float()
            r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            clean = N(((xf * r) * wf).to(torch.bfloat16))
            extra_round = N(((xf * r).to(torch.bfloat16).float() * wf).to(torch.bfloat16))   # 多舍一次
        lg = self.t(f"layers.{L}.mlp.router.logits").reshape(-1, self.t(f"layers.{L}.mlp.router.logits").shape[-1]) \
            if self.moe else None
        cases_c = [dict(name=f"layers.{L}.rmsnorm.torch_fp32_formula", out=clean, fp64=truth, ref=ref,
                        extra={"test": "torch 公式，FP32 里算、最后一次舍到 BF16", "ref_impl": "TE RMSNorm（训练用的内核）"})]
        cases_d = [dict(name=f"layers.{L}.rmsnorm.extra_bf16_round", out=extra_round, fp64=truth, ref=ref,
                        extra={"test": "x·r 先舍到 BF16 再乘权重（多舍一次）", "ref_impl": "TE RMSNorm"})]
        if lg is not None:
            with torch.no_grad():
                s_ref = N(torch.sigmoid(T(lg)))
                s_clean = N(torch.sigmoid(T(lg).double()).float())
                s_bad = N(torch.sigmoid(T(lg).to(torch.bfloat16).float()))                     # logits 先舍到 BF16
            st = O.sigmoid_f64(lg)
            cases_c.append(dict(name=f"layers.{L}.router.sigmoid.fp64_then_fp32", out=s_clean, fp64=st, ref=s_ref,
                                extra={"test": "FP64 里算 sigmoid 再舍到 FP32", "ref_impl": "torch.sigmoid（FP32）"}))
            cases_d.append(dict(name=f"layers.{L}.router.sigmoid.logits_bf16", out=s_bad, fp64=st, ref=s_ref,
                                extra={"test": "logits 先舍到 BF16 再算 sigmoid", "ref_impl": "torch.sigmoid（FP32）"}))
        th = num({"r_ratio": 1.2}, "推断：起点。只差舍入顺序时 R≈1；多一次同量级的独立舍入误差方差约翻倍、R≈√2≈1.41，取两者之间")
        self.add("L1.rratio.clean", cases_c, th, params={"worst_by": ["r_ratio"]})
        self.add("L1.rratio.drill", cases_d, th, params={"expect": "fail", "worst_by": ["r_ratio"]})

    # ---- #47：CPU 模拟 GPU 舍入方式，逐位复现
    def emulate(self):
        rows_all = None
        cases, naive = [], []
        for L in self.moe:
            x, w, cap = self.router_inputs[L]
            rows_all = np.arange(x.shape[0])
            t0 = time.time()
            e = emulate_parallel(f"router{L}", x, w, rows_all, TC, self.workers)
            cases.append(dict(name=f"layers.{L}.router.logits", out=cap, emulate=e,
                              extra={"model": TC, "rows": int(len(rows_all)), "seconds": round(time.time() - t0, 1)}))
            if L == self.moe[0]:
                for mode in ("fp64", "seq"):
                    naive.append(dict(name=f"layers.{L}.router.logits.{mode}", out=cap,
                                      emulate=O.gemm_emulate(x, w, mode), extra={"model": {"mode": mode}}))
                for kw in ({"mode": "tc", "group": 16, "frac_bits": 25, "rounding": "rne"},
                           {"mode": "tc", "group": 8, "frac_bits": 25, "rounding": "rz"},
                           {"mode": "tc", "group": 16, "frac_bits": 24, "rounding": "rz"}):
                    sub = rows_all[:: max(1, len(rows_all) // 256)]
                    naive.append(dict(name=f"layers.{L}.router.logits.tc_g{kw['group']}_b{kw['frac_bits']}_{kw['rounding']}",
                                      out=cap[sub], emulate=emulate_parallel(f"r{L}{kw}", x, w, sub, kw, self.workers),
                                      extra={"model": kw, "rows": int(len(sub))}))
        # BF16 出的 GEMM：张量核累加到 FP32 后 RNE 舍到 BF16。抽样若干行（整行全部输出列）
        for L in self.layers:
            x, w, cap = self.proj_inputs[L]
            sub = np.arange(0, x.shape[0], max(1, x.shape[0] // self.emulate_rows))
            e = O.round_from_f64(emulate_parallel(f"proj{L}", x, w, sub, TC, self.workers).astype(np.float64), BF)
            cases.append(dict(name=f"layers.{L}.attn.proj", out=cap[sub], emulate=e, extra={"model": TC, "rows": int(len(sub))}))
            x, wn, wq, ln_gpu, cap = self.qkv_inputs[L]
            e = O.round_from_f64(emulate_parallel(f"qkv{L}", ln_gpu, wq, sub, TC, self.workers).astype(np.float64), BF)
            cases.append(dict(name=f"layers.{L}.attn.qkv.gemm_part", out=cap[sub], emulate=e,
                              extra={"model": TC, "rows": int(len(sub)), "gemm_input": "GPU 上 TE RMSNorm 的 BF16 输出"}))
            # 整条 CPU 模拟（RMSNorm 也在 CPU 上模拟）与朴素猜法
            ln_cpu = O.rmsnorm_emulate(x, wn, self.eps, BF)
            naive.append(dict(name=f"layers.{L}.attn.qkv.rmsnorm_part", out=ln_gpu, emulate=ln_cpu,
                              extra={"model": "FP32 公式 r = 1/sqrt(mean(x²)+eps) 正确舍入，y = (x·r)·w → BF16"}))
            e2 = O.round_from_f64(emulate_parallel(f"qkvc{L}", ln_cpu, wq, sub, TC, self.workers).astype(np.float64), BF)
            naive.append(dict(name=f"layers.{L}.attn.qkv.full_cpu", out=cap[sub], emulate=e2,
                              extra={"model": "CPU RMSNorm 模拟 + 张量核模型", "rows": int(len(sub))}))
            naive.append(dict(name=f"layers.{L}.attn.qkv.fp64_round", out=cap[sub],
                              emulate=O.round_from_f64(ln_gpu[sub].astype(np.float64) @ wq.astype(np.float64).T, BF),
                              extra={"model": "FP64 点积一次舍到 BF16", "rows": int(len(sub))}))
        self.add("L1.emulate.gemm", cases, Threshold("exact"))
        self.add("L1.emulate.other_guesses", naive, None, hard=False)

    # ---- #42：极端输入对独立 FP64 公式
    def extreme(self):
        import transformer_engine.pytorch as tep
        h = 4096
        r = self.rng
        rows = {
            "normal": r.standard_normal(h),
            "zero": np.zeros(h),
            "tiny_1e-20": np.full(h, 1e-20) * np.sign(r.standard_normal(h)),       # x² = 1e-40 是 FP32 次正规数
            "subnormal_1e-40": np.full(h, 1e-40) * np.sign(r.standard_normal(h)),
            "large_1e18": r.standard_normal(h) * 1e18,                               # x² ≈ 1e36，mean 不溢出
            "overflow_1e20": r.standard_normal(h) * 1e20,                            # x² ≈ 1e40 > FP32 最大值
            "one_huge_3e38": np.concatenate([[3e38], np.ones(h - 1)]),
        }
        w = np.ones(h)
        cases, over = [], []
        # 平方和超过 FP32 最大值（RMS > sqrt(3.4e38 / 4096) ≈ 2.9e17）的行单列：所有实现都在这里静默出 0
        overflow = {"large_1e18", "overflow_1e20", "one_huge_3e38"}
        for nm, row in rows.items():
            x32 = row.astype(np.float32)[None, :]
            xb = row.astype(BF)[None, :]
            with torch.no_grad():
                xt = T(x32)
                manual = N(xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + self.eps) * T(w.astype(np.float32)))
                frms = N(F.rms_norm(xt, (h,), T(w.astype(np.float32)), self.eps))
                te32 = tep.RMSNorm(h, eps=self.eps, params_dtype=torch.float32)
                te16 = tep.RMSNorm(h, eps=self.eps, params_dtype=torch.bfloat16)
                te32.weight.fill_(1.0)
                te16.weight.fill_(1.0)
                t32 = N(te32(xt))
                t16 = N(te16(T(xb)))
            f32 = O.rmsnorm_f64(x32, w, self.eps)
            fb = O.rmsnorm_f64(xb, w, self.eps)
            for impl, out, truth in (("torch_formula_fp32", manual, f32), ("F.rms_norm_fp32", frms, f32),
                                     ("te_fp32", t32, f32), ("te_bf16", t16, fb)):
                (over if nm in overflow else cases).append(dict(
                    name=f"rmsnorm.{impl}.{nm}", out=out, fp64=truth,
                    extra={"input": nm, "impl": impl, "out_zero_frac": float(np.mean(out.astype(np.float64) == 0.0))}))
        th = num({"ulp_max": 2, "nonfinite_mismatch": 0},
                 "推断：FP32 里算 rsqrt（rsqrtf 2 ULP）再乘一次，输出离正确舍入不超过 2 格；全零行必须出 0 而不是 NaN")
        self.add("L1.extreme.rmsnorm", cases, th)
        self.add("L1.extreme.rmsnorm.overflow", over, None, hard=False)

        sig_in = np.array([-np.inf, -1e30, -104.0, -88.7, -20.0, -1e-30, -0.0, 0.0, 1e-45, 1e-38, 1e-30, 20.0,
                           88.7, 104.0, 1e30, np.inf, np.nan])
        cases = []
        for dt, np_dt in (("fp32", np.float32), ("bf16", BF)):
            x = sig_in.astype(np_dt)
            with torch.no_grad():
                out = N(torch.sigmoid(T(x)))
            cases.append(dict(name=f"sigmoid.{dt}", out=out, fp64=O.sigmoid_f64(x.astype(np.float64)),
                              extra={"inputs": [str(v) for v in sig_in]}))
        self.add("L1.extreme.sigmoid", cases, num({"ulp_max": 4, "nonfinite_mismatch": 0},
                                                  "推断：同 L1.elem.sigmoid.fp32 的起点"))

        e = 192
        sm_rows = {
            "all_zero": np.zeros(e),
            "all_1e30": np.full(e, 1e30),
            "one_3e38": np.concatenate([[3e38], np.zeros(e - 1)]),
            "spread_1e30": np.linspace(-1e30, 1e30, e),
            "half_neg_inf": np.concatenate([np.full(e // 2, -np.inf), r.standard_normal(e // 2)]),
            "all_neg_inf": np.full(e, -np.inf),
            "has_pos_inf": np.concatenate([[np.inf], r.standard_normal(e - 1)]),
            "subnormal": np.full(e, 1e-40),
            "normal": r.standard_normal(e) * 5,
        }
        cases = []
        for nm, row in sm_rows.items():
            x = row.astype(np.float32)[None, :]
            with torch.no_grad():
                out = N(torch.softmax(T(x), dim=-1))
            cases.append(dict(name=f"softmax.fp32.{nm}", out=out, fp64=O.softmax_f64(x), extra={"input": nm}))
        self.add("L1.extreme.softmax", cases, num({"rel_l2": 1e-5, "nonfinite_mismatch": 0},
                                                  "推断：x − max 在 FP32 里舍入一次，exp 把它放大 |x − max| 倍（输入到 −30 时约 2e-6），"
                                                  "再加 expf 2 ULP 与 192 项求和，取 1e-5；小概率项的 ULP 随 |x − max| 线性涨，不用 ULP 判"))

    # ---- #43：打平用例（torch.topk 对 CPU 稳定排序规则）
    def ties(self):
        r = self.rng
        k, e = self.topk, 192
        cases, cpu = [], []
        for nm, s in (("levels4", r.integers(0, 4, (4096, e)).astype(np.float32) * np.float32(0.25)),
                      ("levels32", r.integers(0, 32, (4096, e)).astype(np.float32) / np.float32(32)),
                      ("all_equal", np.full((256, e), 0.5, np.float32))):
            with torch.no_grad():
                gpu = N(torch.topk(T(s), k, dim=1, sorted=True).indices)
                cpu_t = torch.topk(torch.from_numpy(s), k, dim=1, sorted=True).indices.numpy()
            exp = O.topk_stable(s, k)
            ties = int(O.boundary_ties(s, k).size)
            cases.append(dict(name=f"{nm}.gpu_topk_vs_stable", out=gpu, expect=exp, extra={"boundary_tie_rows": ties}))
            cpu.append(dict(name=f"{nm}.cpu_topk_vs_stable", out=cpu_t, expect=exp, extra={"boundary_tie_rows": ties}))
            cpu.append(dict(name=f"{nm}.gpu_vs_cpu_topk", out=gpu, expect=cpu_t, extra={"boundary_tie_rows": ties}))
        self.add("L1.ties.synthetic", cases, Threshold("exact"))
        self.add("L1.ties.cpu_torch", cpu, None, hard=False)

        real = []
        for L in self.moe:
            lg = self.t(f"layers.{L}.mlp.router.logits")
            lg = lg.reshape(-1, lg.shape[-1])
            b = self.run.tensor(STEP, f"buffer.decoder.layers.{L}.mlp.router.expert_bias")
            with torch.no_grad():
                sc = torch.sigmoid(T(lg).float()) + T(b).float()          # Megatron：scores_for_routing
                idx = N(torch.topk(sc, k, dim=1, sorted=True).indices)
            scn = N(sc)
            ties = int(O.boundary_ties(scn, k).size)
            real.append(dict(name=f"layers.{L}.replay_vs_captured", out=np.sort(idx, -1).astype(np.int32),
                             expect=self.t(f"layers.{L}.mlp.router.topk_idx"), extra={"boundary_tie_rows": ties}))
            real.append(dict(name=f"layers.{L}.gpu_topk_vs_stable", out=idx, expect=O.topk_stable(scn, k),
                             extra={"boundary_tie_rows": ties}))
        self.add("L1.ties.real", real, Threshold("exact"))

    # ---- #39：随机 cotangent 的 VJP，GPU autograd 对 FP64 解析梯度
    def vjp(self):
        import transformer_engine.pytorch as tep
        from megatron.core.transformer.moe.moe_utils import router_gating_linear, topk_routing_with_score_function
        L = self.moe[0] if self.moe else self.layers[0]
        r = self.rng
        x = self.t(f"layers.{L}.in")
        h = x.shape[-1]
        x = x.reshape(-1, h)
        wn = self.p(L, "self_attention.linear_qkv.layer_norm_weight")
        u = r.standard_normal(x.shape).astype(np.float32)
        cases32, cases16 = [], []
        # 解析公式自身先对 FP64 中心差分（抽 8 行）
        sub = np.arange(0, x.shape[0], x.shape[0] // 8)
        v = r.standard_normal((len(sub), h))
        gx_s, _ = O.rmsnorm_vjp_f64(x[sub], wn, self.eps, u[sub])
        fd = O.directional_fd(lambda t: O.rmsnorm_f64(t, wn, self.eps), x[sub].astype(np.float64), v, u[sub])
        fd_err = abs(float(np.sum(gx_s * v)) - fd) / abs(fd)

        xt = T(x).float().requires_grad_(True)
        wt = T(wn).float().requires_grad_(True)
        y = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + self.eps) * wt
        y.backward(T(u))
        gx64, gw64 = O.rmsnorm_vjp_f64(x, wn, self.eps, u)
        cases32.append(dict(name=f"layers.{L}.rmsnorm.torch_fp32.dx", out=N(xt.grad), fp64=gx64, extra={"fd_check_rel_err": fd_err}))
        cases32.append(dict(name=f"layers.{L}.rmsnorm.torch_fp32.dw", out=N(wt.grad), fp64=gw64))

        m = tep.RMSNorm(h, eps=self.eps, params_dtype=torch.bfloat16)
        with torch.no_grad():
            m.weight.copy_(T(wn))
        xb = T(x).clone().requires_grad_(True)
        ub = T(u).to(torch.bfloat16)
        m(xb).backward(ub)
        gx64b, gw64b = O.rmsnorm_vjp_f64(x, wn, self.eps, N(ub))
        cases16.append(dict(name=f"layers.{L}.rmsnorm.te_bf16.dx", out=N(xb.grad), fp64=gx64b))
        cases16.append(dict(name=f"layers.{L}.rmsnorm.te_bf16.dw", out=N(m.weight.grad), fp64=gw64b))

        if self.moe:
            xr, wr, cap = self.router_inputs[L]
            bias = self.run.tensor(STEP, f"buffer.decoder.layers.{L}.mlp.router.expert_bias")
            ue = r.standard_normal(cap.shape).astype(np.float32)
            # 分数函数（FP32）：抓下的 logits → sigmoid → 加偏置选 top-k → 归一化 × 缩放
            z = T(cap).clone().requires_grad_(True)
            probs, rmap = topk_routing_with_score_function(z, self.topk, score_function="sigmoid",
                                                           expert_bias=T(bias), scaling_factor=self.scale)
            (probs * T(ue)).sum().backward()
            idx = np.argsort(~N(rmap), axis=-1, kind="stable")[:, : self.topk]
            u_sel = np.take_along_axis(ue, idx, -1)
            gz64 = O.router_probs_vjp_f64(cap, idx, self.scale, u_sel)
            sub = np.arange(0, cap.shape[0], cap.shape[0] // 8)
            vz = r.standard_normal((len(sub), cap.shape[1]))
            fdz = O.directional_fd(lambda t: O.router_probs_f64(t, bias, self.topk, self.scale, idx=idx[sub])[0],
                                   cap[sub].astype(np.float64), vz, u_sel[sub])
            fdz_err = abs(float(np.sum(gz64[sub] * vz)) - fdz) / abs(fdz)
            cases32.append(dict(name=f"layers.{L}.router.score_fn_fp32.dlogits", out=N(z.grad), fp64=gz64,
                                extra={"fd_check_rel_err": fdz_err}))
            # 整条路由（门控 GEMM 在内）：BF16 输入与权重 → logits（FP32）→ probs；梯度回到 BF16
            xg = T(xr).clone().requires_grad_(True)
            wg = T(wr).clone().requires_grad_(True)
            lg = router_gating_linear(xg, wg, None, torch.float32)
            probs, rmap = topk_routing_with_score_function(lg, self.topk, score_function="sigmoid",
                                                           expert_bias=T(bias), scaling_factor=self.scale)
            (probs * T(ue)).sum().backward()
            idx = np.argsort(~N(rmap), axis=-1, kind="stable")[:, : self.topk]
            l64 = xr.astype(np.float64) @ wr.astype(np.float64).T
            gz = O.router_probs_vjp_f64(l64, idx, self.scale, np.take_along_axis(ue, idx, -1))
            cases16.append(dict(name=f"layers.{L}.router.full.dx", out=N(xg.grad), fp64=gz @ wr.astype(np.float64)))
            cases16.append(dict(name=f"layers.{L}.router.full.dw", out=N(wg.grad), fp64=gz.T @ xr.astype(np.float64)))
        self.add("L1.vjp.fp32", cases32, num({"rel_l2": 1e-5},
                                             "推断：FP32 单位舍入 6e-8，反向含长 4096 的归约，误差按 √4096 倍放大约 4e-6，取 1e-5"),
                 params={"worst_by": ["rel_l2"]})
        self.add("L1.vjp.bf16", cases16, num({"rel_l2": 1e-2},
                                             "推断：梯度以 BF16 输出，一次舍入相对误差 2^-9≈2e-3，取 1e-2 宽松起点（同 phase1-parallel）"),
                 params={"worst_by": ["rel_l2"]})


# ---------------------------------------------------------------- 写结果

_COLS = ["rel_l2", "cr_match_frac", "ulp_max", "ulp_p999", "bit_match_frac", "emulate_match_frac", "r_ratio",
         "cosine_test", "expect_match_frac", "rows_differ", "boundary_tie_rows", "nonfinite_mismatch",
         "n_ulp_gt4", "x_ulp_gt4_min", "x_ulp_gt4_max", "out_zero_frac", "fd_check_rel_err"]


def _fmt(v):
    if isinstance(v, float):
        if v in (0.0, 1.0):
            return f"{v:g}"
        return f"{v:.6f}" if 0.99 < v < 1.0 else f"{v:.3e}"
    return str(v)


REF_LINE = "参照：FP64 真值、训练时抓的输出、CPU 上的同一运算与舍入模拟"
COL_NOTE = ("列：rel_l2 对 FP64；cr_match_frac 对 FP64 正确舍入到输出 dtype 的逐位相同率；ulp 对正确舍入；"
            "bit_match_frac 对应当逐位相同的结果（训练抓数 / CPU 同一运算）；emulate_match_frac 对 CPU 模拟。")


def render(cat: Catalog, results: dict, checks, columns=None, ref_line=REF_LINE, col_note=COL_NOTE) -> str:
    md = _render_report(cat, results).replace("数字全部来自 `results.json`，本报告不手写数字。\n", "")
    md = md.replace("；golden：\n", f"；{ref_line}\n")
    env = results["environment"]
    md += ("\n## 环境与开关（进程内实测）\n\n" + "\n".join(f"- `{k}`：{v}" for k, v in env.items()) + "\n")
    md += f"\n## 逐用例明细\n\n{col_note}\n"
    for r in checks:
        if not r.details:
            continue
        cols = [c for c in (columns or _COLS) if any(c in d for d in r.details)]
        md += f"\n### `{r.id}`（{r.status}）\n\n| 用例 | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
        for d in r.details:
            md += f"| `{d['name']}` | " + " | ".join(_fmt(d[c]) if c in d else "" for c in cols) + " |\n"
    md += "\n数字全部来自 `results.json` 与 `details/`，本报告不手写数字。\n"
    return md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    ap.add_argument("--emulate-rows", type=int, default=64, help="BF16 出 GEMM 的 CPU 模拟抽多少行")
    ap.add_argument("--only", default="", help="逗号分隔，只跑这些部分（调试用）")
    a = ap.parse_args(argv)
    torch.cuda.set_device(0)
    env = environment()
    run = Run.open(a.run)
    run.enable_cache()
    out = Path(a.out)
    (out / "details").mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    job = L1(run, a.workers, a.emulate_rows)
    parts = ["gemm", "norms", "elementwise", "basic", "rratio", "emulate", "extreme", "ties", "vjp"]
    only = set(filter(None, a.only.split(",")))
    for p in parts:
        if only and p not in only and not (p in ("gemm", "norms") and only & {"emulate", "vjp"}):
            continue
        getattr(job, p)()
    return write_outputs(out, job.checks, [run], env, Path(__file__), "phase1-l1", started)


def write_outputs(out: Path, check_objs, runs, env: dict, script: Path, stage: str, started: str, **render_kw) -> int:
    """写 results.json（result-schema）、details/<检查>.jsonl、report.md，返回退出码（同 janus run）。"""
    (out / "details").mkdir(parents=True, exist_ok=True)
    checks = []
    for r in check_objs:
        d = r.to_json()
        if r.details:
            rel = f"details/{r.id}.jsonl"
            with open(out / rel, "w") as f:
                for row in r.details:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            d["details"] = rel
        checks.append(d)
    summary = {k: sum(1 for c in checks if c["status"] == k) for k in (PASS, FAIL, INCONCLUSIVE, ERROR)}
    summary["hard_fail"] = any(c["status"] == FAIL and c.get("hard", True) for c in checks)
    rel_script = f"models/hunyuan3/frameworks/megatron/{script.name}"
    results = {
        "schema_version": 1, "janus": {"version": __version__, "commit": None},
        "catalog": {"path": rel_script, "sha256": hashlib.sha256(script.read_bytes()).hexdigest()},
        "runs": {r.name: {"path": str(r.path), "meta_sha256": r.meta_sha256()} for r in runs},
        "started_at": started, "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": env, "summary": summary, "checks": checks,
    }
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    model = runs[0].meta.get("model", "?") if runs else "?"
    cat = Catalog(Path(rel_script), {"model": model, "stage": stage,
                                     "runs": {"test": ", ".join(r.name for r in runs), "golden": []}})
    (out / "report.md").write_text(render(cat, results, check_objs, **render_kw))
    print(f"pass {summary['pass']} / fail {summary['fail']} / inconclusive {summary['inconclusive']} / error {summary['error']} → {out}")
    if any(c["status"] == ERROR and c.get("hard", True) for c in checks):
        return 2
    return 1 if summary["hard_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
