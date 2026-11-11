"""L1 kernel 回放（GPU 侧）：TF32 开关实测、专家分组 GEMM 与核心注意力的真实输入回放、合成 poison 与因果性用例。
规格见 docs/dev/design/0004-l1-gpu-kernels.md 与 docs/reference/checks/op_replay.md。

输入是带可选观测点的确定性运行（`run_check.sh g61-obs 2 det --full --dump-steps 1 --observe attn,moe,expw,gemm`）
第 1 步 rank 0 第一个 microbatch 抓下的张量：注意力 q / k / v / 输出及其梯度，专家输入 / GEMM1 / 激活 / GEMM2 及其梯度、本地 24 个专家的权重。
产出与 l1_ops.py 同一格式：results.json + details/ + report.md。

用法（Pod 里，一张卡；GPU 共享时包 flock）：
    source /data/env.sh; export PYTHONPATH=/tmp/janus-g61:$PYTHONPATH
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8      # 与训练时 Megatron 确定性模式设的一致
    CUDA_VISIBLE_DEVICES=0 flock /data/gpu.lock python models/hunyuan3/frameworks/megatron/l1_kernels.py \
        --run /data/janus-runs2/g61-obs --tf32-run /data/janus-runs2/g61-notf32 --out /data/janus-g61/kernels
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l1_ops import BF, DEV, OPS, SEED, STEP, N, T, environment, num, write_outputs  # noqa: E402

from janus.checks import Threshold  # noqa: E402
from janus.checks.base import ERROR, FAIL, PASS, CheckResult  # noqa: E402
from janus.store import Run  # noqa: E402

BF16 = torch.bfloat16


def env_more() -> dict:
    e = environment()
    e["env"].update({k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN")})
    e["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    return e


def scalar_check(cid: str, level: str, rows: list[dict], ok: bool | None, message: str, hard=True,
                 threshold: dict | None = None, metrics: dict | None = None) -> CheckResult:
    """非数组类结论（开关读回值、清单）也写成同一格式：ok=None 表示只诊断。"""
    r = CheckResult(cid, "op_replay", level, hard)
    r.details = rows
    r.threshold = threshold or {}
    r.metrics = dict(metrics or {}, cases=len(rows))
    r.status = PASS if ok in (True, None) else FAIL
    r.message = message
    return r


# ---------------------------------------------------------------- TF32 子进程（NVIDIA_TF32_OVERRIDE 只在 cuBLAS 初始化时读，每种设置一个进程）

def tf32_child(mode: str, run_path: str, layer: int, tmp: str):
    if "torch_false" in mode:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if "highest" in mode:
        torch.set_float32_matmul_precision("highest")
    from megatron.core.transformer.moe.moe_utils import router_gating_linear
    from transformer_engine.pytorch.cpp_extensions import general_gemm
    rng = np.random.default_rng(SEED)
    a = rng.standard_normal((1024, 4096)).astype(np.float32)
    b = rng.standard_normal((1024, 4096)).astype(np.float32)
    out = {}
    with torch.no_grad():
        out["torch_mm_fp32"] = N(T(a) @ T(b).t())
        out["te_general_gemm_fp32"] = N(general_gemm(T(b), T(a), torch.float32, layout="TN")[0])
    run = Run.open(run_path)
    x = run.tensor(STEP, f"layers.{layer}.mlp_norm.out")
    x = x.reshape(-1, x.shape[-1])
    w = run.tensor(STEP, f"param.decoder.layers.{layer}.mlp.router.weight")
    g = rng.standard_normal((x.shape[0], w.shape[0])).astype(np.float32)
    xt, wt = T(x).requires_grad_(True), T(w).requires_grad_(True)
    router_gating_linear(xt, wt, None, torch.float32).backward(T(g))   # Megatron 训练时的同一函数：反向两个 GEMM 输入都是 FP32
    out["router_bwd_dx"] = N(xt.grad)
    out["router_bwd_dw"] = N(wt.grad)
    np.savez(os.path.join(tmp, f"{mode}.npz"), **{k: (v.view(np.int16) if v.dtype == BF else v) for k, v in out.items()},
             _bf16=np.array([k for k, v in out.items() if v.dtype == BF]))
    state = {"allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
             "float32_matmul_precision": torch.get_float32_matmul_precision(),
             "env": {k: os.environ.get(k) for k in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")}}
    Path(tmp, f"{mode}.json").write_text(json.dumps(state))


TF32_VARIANTS = [
    # (名字, 去掉的环境变量, 设的环境变量)；名字里带 torch_false / highest 的，子进程里另设 torch 开关
    ("default", [], {}),
    ("torch_false", [], {}),
    ("highest", [], {}),
    ("no_cublas_override_env", ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"], {}),
    ("NVIDIA_TF32_OVERRIDE=0", [], {"NVIDIA_TF32_OVERRIDE": "0"}),
    ("NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env", ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"], {"NVIDIA_TF32_OVERRIDE": "0"}),
]


def _load_npz(path):
    z = np.load(path)
    bf = set(z["_bf16"].tolist())
    return {k: (z[k].view(BF) if k in bf else z[k]) for k in z.files if k != "_bf16"}


# ---------------------------------------------------------------- 主流程

class Kernels:
    def __init__(self, run: Run, args):
        self.run, self.a = run, args
        self.checks = []
        self.evidence = [{"path": str(run.path), "sha256": run.meta_sha256()}]
        mc = run.meta.get("model_config") or {}
        self.nh, self.ng, self.hd = int(mc["num_attention_heads"]), int(mc["num_query_groups"]), int(mc["kv_channels"])
        self.layers = sorted({int(n.split(".")[1]) for n in run.names(STEP, "layers.*.out")})
        self.moe = sorted({int(n.split(".")[1]) for n in run.names(STEP, "layers.*.mlp.router.logits")})
        p = run.path / f"observe_step{STEP}.json"
        self.info = json.loads(p.read_text()) if p.exists() else {}
        self.rng = np.random.default_rng(SEED)
        torch.manual_seed(SEED)

    def t(self, name):
        return self.run.tensor(STEP, name)

    def p(self, layer, name):
        return self.run.tensor(STEP, f"param.decoder.layers.{layer}.{name}")

    def add(self, cid, cases, threshold=None, hard=True, params=None, level="L1"):
        t0 = time.time()
        r = OPS(cid, level, cases, threshold, hard=hard, params=params)
        r.evidence = list(self.evidence)
        self.checks.append(r)
        print(f"{r.status:<13} {cid:<40} {time.time() - t0:6.1f}s  {r.message[:100]}", flush=True)
        return r

    def add_result(self, r: CheckResult):
        r.evidence = list(self.evidence)
        self.checks.append(r)
        print(f"{r.status:<13} {r.id:<40}         {r.message[:100]}", flush=True)

    # ---------------------------------------------------------------- 1. TF32
    def tf32(self):
        """开关矩阵（每种设置一个子进程）+ 训练进程里的实测 + 训练第 1 步 GEMM 清单里输入为 FP32 的调用。"""
        layer = self.moe[0]
        res = {}
        with tempfile.TemporaryDirectory() as tmp:
            for name, unset, setv in TF32_VARIANTS:
                env = {k: v for k, v in os.environ.items() if k not in unset}
                env.update(setv)
                cmd = [sys.executable, __file__, "--tf32-child", name, "--run", str(self.run.path), "--tmp", tmp,
                       "--layer", str(layer), "--out", tmp]
                t0 = time.time()
                p = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if p.returncode != 0:
                    raise RuntimeError(f"TF32 子进程 {name} 失败：{p.stderr[-2000:]}")
                res[name] = (_load_npz(os.path.join(tmp, f"{name}.npz")), json.loads(Path(tmp, f"{name}.json").read_text()))
                print(f"  tf32 子进程 {name} {time.time() - t0:.0f}s", flush=True)
        rng = np.random.default_rng(SEED)
        a = rng.standard_normal((1024, 4096)).astype(np.float32)
        b = rng.standard_normal((1024, 4096)).astype(np.float32)
        truth_mm = a.astype(np.float64) @ b.astype(np.float64).T
        x = self.t(f"layers.{layer}.mlp_norm.out")
        x = x.reshape(-1, x.shape[-1])
        w = self.p(layer, "mlp.router.weight")
        g = rng.standard_normal((x.shape[0], w.shape[0])).astype(np.float32)
        # router_gating_linear 反向：weight / inp 先升 FP32 再进 TE general_gemm（输入是 BF16 值的精确 FP32 表示）
        truth_dx = g.astype(np.float64) @ w.astype(np.float64)
        truth_dw = g.astype(np.float64).T @ x.astype(np.float64)
        base = res["default"][0]
        ieee = res["NVIDIA_TF32_OVERRIDE=0"][0]
        matrix, router = [], []
        for name, _, _ in TF32_VARIANTS:
            out, state = res[name]
            for op in ("torch_mm_fp32", "te_general_gemm_fp32"):
                matrix.append(dict(name=f"{name}.{op}", out=out[op], fp64=truth_mm, bitref=ieee[op],
                                   extra={"variant": name, "op": op, **{f"readback.{k}": v for k, v in state.items()}}))
            router.append(dict(name=f"{name}.router_bwd.dx", out=out["router_bwd_dx"], fp64=truth_dx, bitref=base["router_bwd_dx"],
                               extra={"variant": name, "bitref": "default（TF32 开）"}))
            router.append(dict(name=f"{name}.router_bwd.dw", out=out["router_bwd_dw"], fp64=truth_dw, bitref=base["router_bwd_dw"],
                               extra={"variant": name, "bitref": "default（TF32 开）"}))
        # 列 bit_match_frac：与 NVIDIA_TF32_OVERRIDE=0（IEEE FP32）逐位相同的比例；rel_l2 对 FP64
        self.add("TF32.switch.matrix", matrix, None, hard=False)
        off = [c for c in matrix if c["extra"]["variant"].startswith("NVIDIA_TF32_OVERRIDE=0")]
        self.add("TF32.switch.override_off", off,
                 num({"rel_l2": 1e-5}, "推断：IEEE FP32 累加 K=4096，误差约 √K·2^-24 ≈ 4e-6 的量级（实测约 1e-6）；TF32（10 位尾数）约 3e-4，取 1e-5 分开两者"))
        self.add("TF32.router_bwd", router, None, hard=False)

        # 训练进程里的实测（train_hy3.py 写进 meta.json 的 tf32 字段）
        rows, ok = [], True
        for path in filter(None, [str(self.run.path), self.a.tf32_run]):
            m = Run.open(path).meta.get("tf32") or {}
            row = {"name": Path(path).name, **{k: v for k, v in m.items() if k != "env"}, **{f"env.{k}": v for k, v in (m.get("env") or {}).items()}}
            rows.append(row)
            if path == self.a.tf32_run:
                ok = ok and all(m.get(k, 1.0) <= 1e-5 for k in ("probe_rel_l2_torch_mm_fp32", "probe_rel_l2_te_general_gemm_fp32"))
        self.add_result(scalar_check(
            "TF32.in_run", "L1", rows, ok if self.a.tf32_run else None,
            "关 TF32 的运行里，torch 与 TE 的 FP32 GEMM 实测都是 IEEE FP32" if ok else "关 TF32 的运行里仍有 FP32 GEMM 走 TF32",
            threshold={"kind": "numeric", "values": {"probe_rel_l2_*": 1e-5}, "source": "同 TF32.switch.override_off"}))

        inv = self.run.path / "gemm_inventory.json"
        if inv.exists():
            calls = json.loads(inv.read_text())["calls"]
            fp32 = [dict(c, name=f"{c['op']} @ {c['site']}") for c in calls if "float32" in (c["a_dtype"], c["b_dtype"])]
            self.add_result(scalar_check(
                "TF32.gemm_inventory", "L1", [dict(c, name=f"{c['op']} @ {c['site']}") for c in calls], None,
                f"第 1 步 rank 0 共 {sum(c['count'] for c in calls)} 次 GEMM 调用，输入含 FP32 的 {sum(c['count'] for c in fp32)} 次"
                f"（{len(fp32)} 处调用点）", hard=False,
                metrics={"calls": sum(c["count"] for c in calls), "fp32_input_calls": sum(c["count"] for c in fp32),
                         "fp32_input_sites": len(fp32)}))

    # ---------------------------------------------------------------- 2a. 专家分组 GEMM
    def _act(self, y1, probs, info):
        """Megatron TEGroupedMLP 的激活（bias_act_func）：本配置走非融合的 GLU 分支——
        silu(前半) * (后半 + glu_linear_offset)，BF16 里算；再乘 FP32 概率（升 FP32）后转回 BF16。"""
        if info.get("use_te_activation_func") or info.get("bias_activation_fusion"):
            raise RuntimeError(f"激活走的不是非融合 GLU 分支（{info}），回放写法要换")
        g, u = torch.chunk(y1, 2, dim=-1)
        a = F.silu(g) * (u + 0.0)
        return (a * probs.unsqueeze(-1)).to(a.dtype)

    def experts(self):
        import transformer_engine.pytorch as tep
        from transformer_engine.pytorch.cpp_extensions import general_gemm
        fwd, bwd, loop, wgrad = [], [], [], []
        for L in self.moe:
            info = self.info.get(f"layers.{L}.mlp.moe", {})
            X = self.t(f"layers.{L}.mlp.experts.in")
            m = [int(v) for v in self.t(f"layers.{L}.mlp.experts.tokens_per_expert")]
            P = self.t(f"layers.{L}.mlp.experts.probs").astype(np.float32)
            E = len(m)
            W1 = [self.p(L, f"mlp.experts.linear_fc1.weight{i}") for i in range(E)]
            W2 = [self.p(L, f"mlp.experts.linear_fc2.weight{i}") for i in range(E)]
            fuse = bool(info.get("gradient_accumulation_fusion", True))
            fc1 = tep.GroupedLinear(E, W1[0].shape[1], W1[0].shape[0], bias=False, params_dtype=BF16, device=DEV,
                                    fuse_wgrad_accumulation=fuse)
            fc2 = tep.GroupedLinear(E, W2[0].shape[1], W2[0].shape[0], bias=False, params_dtype=BF16, device=DEV,
                                    fuse_wgrad_accumulation=fuse)
            with torch.no_grad():
                for i in range(E):
                    getattr(fc1, f"weight{i}").copy_(T(W1[i]))
                    getattr(fc2, f"weight{i}").copy_(T(W2[i]))
            for mod in (fc1, fc2):
                for i in range(E):
                    wp = getattr(mod, f"weight{i}")
                    wp.main_grad = torch.zeros(wp.shape, dtype=torch.float32, device=DEV)   # 第 1 个 microbatch：主梯度从 0 起累加
            x = T(X).requires_grad_(True)
            probs = T(P)
            y1 = fc1(x, m)
            y1.retain_grad()
            a = self._act(y1, probs, info)
            a.retain_grad()
            y2 = fc2(a, m)
            y2.backward(T(self.t(f"layers.{L}.mlp.experts.out.grad")))
            dW1 = [N(getattr(fc1, f"weight{i}").main_grad if fuse else getattr(fc1, f"weight{i}").grad) for i in range(E)]
            dW2 = [N(getattr(fc2, f"weight{i}").main_grad if fuse else getattr(fc2, f"weight{i}").grad) for i in range(E)]
            off = np.concatenate([[0], np.cumsum(m)])
            # FP64 真值：各段对各自专家（段内逐行独立），输入取训练抓下的上一阶段 BF16 输出
            def seg_mm(inp, Ws, transpose=True):
                inp = torch.from_numpy(np.ascontiguousarray(inp).astype(np.float64)).to(DEV)
                outs = [inp[off[i]:off[i + 1]] @ (torch.from_numpy(Ws[i].astype(np.float64)).to(DEV).t() if transpose
                                                  else torch.from_numpy(Ws[i].astype(np.float64)).to(DEV)) for i in range(E)]
                return torch.cat(outs).cpu().numpy()
            cap1, capa, cap2 = (self.t(f"layers.{L}.mlp.experts.{n}") for n in ("fc1.out", "act.out", "out"))
            fwd.append(dict(name=f"layers.{L}.experts.gemm1", out=N(y1), fp64=seg_mm(X, W1), bitref=cap1,
                            extra={"impl": "TE GroupedLinear（训练同一模块类）", "experts": E, "rows": int(off[-1]),
                                   "m_min": min(m), "m_max": max(m), "m_mod128_nonzero": int(sum(1 for v in m if v % 128))}))
            c1 = cap1.astype(np.float64)
            g, u = c1[:, : c1.shape[1] // 2], c1[:, c1.shape[1] // 2:]
            act64 = g / (1 + np.exp(-g)) * u * P.astype(np.float64)[:, None]
            fwd.append(dict(name=f"layers.{L}.experts.act", out=N(a), fp64=act64, bitref=capa,
                            extra={"impl": "silu(前半)·(后半+0)（BF16）× FP32 概率 → BF16", "fp64_of": "训练抓下的 GEMM1 输出"}))
            fwd.append(dict(name=f"layers.{L}.experts.gemm2", out=N(y2), fp64=seg_mm(capa, W2), bitref=cap2,
                            extra={"impl": "TE GroupedLinear", "fp64_of": "训练抓下的激活输出"}))
            dY2 = self.t(f"layers.{L}.mlp.experts.out.grad")
            gA, g1, gX = (self.t(f"layers.{L}.mlp.experts.{n}.grad") for n in ("act.out", "fc1.out", "in"))
            bwd.append(dict(name=f"layers.{L}.experts.gemm2.dgrad", out=N(a.grad), fp64=seg_mm(dY2, W2, transpose=False), bitref=gA,
                            extra={"fp64_of": "训练抓下的 GEMM2 输出梯度 · W2"}))
            bwd.append(dict(name=f"layers.{L}.experts.act.bwd", out=N(y1.grad), bitref=g1,
                            extra={"impl": "autograd 过激活（与训练同一组 torch 算子）"}))
            bwd.append(dict(name=f"layers.{L}.experts.gemm1.dgrad", out=N(x.grad), fp64=seg_mm(g1, W1, transpose=False), bitref=gX,
                            extra={"fp64_of": "训练抓下的 GEMM1 输出梯度 · W1"}))
            # 逐专家循环参照：同一份输入逐个专家单独算（torch F.linear 前向与 dgrad；TE general_gemm 单个 GEMM 的 wgrad，FP32 出）
            with torch.no_grad():
                l1 = torch.cat([F.linear(x[off[i]:off[i + 1]], T(W1[i])) for i in range(E)])
                l2 = torch.cat([F.linear(T(capa)[off[i]:off[i + 1]], T(W2[i])) for i in range(E)])
                ld = torch.cat([T(g1)[off[i]:off[i + 1]] @ T(W1[i]) for i in range(E)])
            loop.append(dict(name=f"layers.{L}.experts.gemm1.fwd", out=N(y1), bitref=N(l1), fp64=seg_mm(X, W1),
                             extra={"loop_impl": "逐专家 torch F.linear"}))
            loop.append(dict(name=f"layers.{L}.experts.gemm2.fwd", out=N(y2), bitref=N(l2), fp64=seg_mm(capa, W2),
                             extra={"loop_impl": "逐专家 torch F.linear（输入为训练抓下的激活）"}))
            loop.append(dict(name=f"layers.{L}.experts.gemm1.dgrad", out=N(x.grad), bitref=N(ld), fp64=seg_mm(g1, W1, transpose=False),
                             extra={"loop_impl": "逐专家 torch matmul（dY1·W1）"}))
            for nm, dWs, dY, inp in (("gemm1", dW1, g1, X), ("gemm2", dW2, dY2, capa)):
                grouped = np.concatenate([d.ravel() for d in dWs])
                ref_loop, truth = [], []
                for i in range(E):
                    xs, ys = T(inp[off[i]:off[i + 1]]), T(dY[off[i]:off[i + 1]])
                    if off[i + 1] > off[i]:
                        r = general_gemm(xs, ys, torch.float32, layout="NT", grad=True)[0]
                    else:
                        r = torch.zeros(dWs[i].shape, dtype=torch.float32, device=DEV)
                    ref_loop.append(N(r).ravel())
                    yd = torch.from_numpy(dY[off[i]:off[i + 1]].astype(np.float64)).to(DEV)
                    xd = torch.from_numpy(inp[off[i]:off[i + 1]].astype(np.float64)).to(DEV)
                    truth.append((yd.t() @ xd).cpu().numpy().ravel())
                wgrad.append(dict(name=f"layers.{L}.experts.{nm}.wgrad", out=grouped.astype(np.float32),
                                  bitref=np.concatenate(ref_loop), fp64=np.concatenate(truth),
                                  extra={"grouped": f"TE GroupedLinear 反向（fuse_wgrad_accumulation={fuse}，FP32 主梯度从 0 累加）",
                                         "loop_impl": "逐专家 TE general_gemm（NT，grad=True，FP32 出）",
                                         "out_dtype": "float32", "experts": E}))
            del x, y1, y2, a
            torch.cuda.empty_cache()
        self.add("L1.kernel.experts.fwd.replay", fwd, Threshold("exact"))
        self.add("L1.kernel.experts.bwd.replay", bwd, Threshold("exact"))
        self.add("L1.kernel.experts.vs_loop", loop, None, hard=False)
        self.add("L1.kernel.experts.wgrad", wgrad,
                 num({"rel_l2": 1e-5}, "推断：FP32 主梯度（FP32 出、K = 该专家 token 数 ≤ 数千）的累加误差约 √K·2^-24，取 1e-5；"
                                       "BF16 输入是精确值，FP64 真值与它只差累加舍入"), params={"worst_by": ["rel_l2"]})

    def gemm_rows(self):
        """GEMM 的逐行结果是否与「同一批里其他行」无关：取真实专家输入的前 M 行算一次，再在后面接 d 行别的 token 重算，
        比前 M 行是否逐位相同。形状取第一个 MoE 层专家的 GEMM1（K=4096、N=3072）与 GEMM2（K=1536、N=4096），
        torch F.linear 与 TE GroupedLinear（两组，被测组在前）各一遍。只报不判：用来解释整网因果性里的差（causal_model.py）。"""
        import transformer_engine.pytorch as tep
        L = self.moe[0]
        X, A = self.t(f"layers.{L}.mlp.experts.in"), self.t(f"layers.{L}.mlp.experts.act.out")
        W1, W2 = self.p(L, "mlp.experts.linear_fc1.weight0"), self.p(L, "mlp.experts.linear_fc2.weight0")
        rng = np.random.default_rng(SEED)
        cases = []
        for gname, inp, W in (("gemm1", X, W1), ("gemm2", A, W2)):
            mod = tep.GroupedLinear(2, W.shape[1], W.shape[0], bias=False, params_dtype=BF16, device=DEV)
            with torch.no_grad():
                mod.weight0.copy_(T(W))
                mod.weight1.copy_(T(W))
            other = inp[-100:]
            for M in (1, 2, 4, 8, 16, 24, 32, 33, 48, 64, 128, 256, 1024):
                base = inp[:M]
                with torch.no_grad():
                    y0 = N(F.linear(T(base), T(W)))
                    g0 = N(mod(T(np.concatenate([base, other])), [M, len(other)]))[:M]
                for d in (1, 8, 64, 512):
                    ext = inp[rng.integers(0, inp.shape[0] - 100, d)]
                    with torch.no_grad():
                        y = N(F.linear(T(np.concatenate([base, ext])), T(W)))[:M]
                        g = N(mod(T(np.concatenate([base, ext, other])), [M + d, len(other)]))[:M]
                    ex = {"gemm": gname, "K": int(W.shape[1]), "N": int(W.shape[0]), "M": M, "extra_rows": d}
                    cases.append(dict(name=f"{gname}.torch.M{M}+{d}", out=y, bitref=y0,
                                      extra=dict(ex, impl="torch F.linear", rows_changed=int((y.view(np.int16) != y0.view(np.int16)).any(1).sum()))))
                    cases.append(dict(name=f"{gname}.grouped.M{M}+{d}", out=g, bitref=g0,
                                      extra=dict(ex, impl="TE GroupedLinear", rows_changed=int((g.view(np.int16) != g0.view(np.int16)).any(1).sum()))))
        self.add("L1.kernel.gemm_row_invariance", cases, None, hard=False)

    # ---------------------------------------------------------------- 2b. 核心注意力
    def _dpa(self, mask="causal", fmt="sbhd"):
        import transformer_engine.pytorch as tep
        info = self.info.get("layers.0.attn", {})
        return tep.DotProductAttention(self.nh, self.hd, num_gqa_groups=self.ng, attention_dropout=0.0, qkv_format=fmt,
                                       attn_mask_type=mask, softmax_scale=info.get("softmax_scale"), layer_number=1)

    def _attn_f64(self, q, k, v, dO, scale):
        """FP64 因果注意力及其 VJP。q [s, nh, hd]，k / v [s, ng, hd]（torch FP64，GPU）。头 h 用 KV 组 h // (nh/ng)。"""
        s, nh, _ = q.shape
        rep = nh // k.shape[1]
        O, dq = torch.empty_like(q), torch.empty_like(q)
        dk, dv = torch.zeros_like(k), torch.zeros_like(v)
        mask = torch.ones(s, s, dtype=torch.bool, device=q.device).tril()
        for h in range(nh):
            g = h // rep
            S = (q[:, h] @ k[:, g].t()) * scale
            S.masked_fill_(~mask, float("-inf"))
            Pm = torch.softmax(S, dim=-1)
            O[:, h] = Pm @ v[:, g]
            dP = dO[:, h] @ v[:, g].t()
            D = (dO[:, h] * O[:, h]).sum(-1, keepdim=True)
            dS = Pm * (dP - D)
            dq[:, h] = (dS @ k[:, g]) * scale
            dk[:, g] += (dS.t() @ q[:, h]) * scale
            dv[:, g] += Pm.t() @ dO[:, h]
        return O, dq, dk, dv

    def attention(self):
        import logging
        msgs = []

        class H(logging.Handler):
            def emit(self, rec):
                msgs.append(rec.getMessage())
        lg = logging.getLogger("DotProductAttention")
        lg.addHandler(H())
        cases_f, cases_b = [], []
        for L in self.layers:
            q, k, v = (self.t(f"layers.{L}.attn.core.{n}") for n in ("q", "k", "v"))
            dO = self.t(f"layers.{L}.attn.core.out.grad")
            cap = self.t(f"layers.{L}.attn.core.out")
            dpa = self._dpa()
            scale = dpa.softmax_scale
            qt, kt, vt = (T(z).requires_grad_(True) for z in (q, k, v))
            out = dpa(qt, kt, vt)
            out.backward(T(dO).reshape(out.shape))
            s = q.shape[0]
            qd, kd, vd = (torch.from_numpy(z.reshape(s, -1, self.hd).astype(np.float64)).to(DEV) for z in (q, k, v))
            dOd = torch.from_numpy(dO.reshape(s, self.nh, self.hd).astype(np.float64)).to(DEV)
            O64, dq64, dk64, dv64 = self._attn_f64(qd, kd, vd, dOd, scale)
            cases_f.append(dict(name=f"layers.{L}.attn.core.out", out=N(out).reshape(cap.shape), fp64=O64.cpu().numpy().reshape(cap.shape),
                                bitref=cap, extra={"impl": "TE DotProductAttention（sbhd、causal、GQA 64/8）", "softmax_scale": scale}))
            for nm, gt, g64 in (("dq", qt.grad, dq64), ("dk", kt.grad, dk64), ("dv", vt.grad, dv64)):
                capg = self.t(f"layers.{L}.attn.core.{nm[1]}.grad")
                cases_b.append(dict(name=f"layers.{L}.attn.core.{nm}", out=N(gt).reshape(capg.shape),
                                    fp64=g64.cpu().numpy().reshape(capg.shape), bitref=capg))
            del qd, kd, vd, dOd, O64, dq64, dk64, dv64
            torch.cuda.empty_cache()
        backends = sorted({m for m in msgs if "backend" in m.lower()})[:5]
        for c in cases_f + cases_b:
            c.setdefault("extra", {})["te_backend_log"] = backends
        self.add("L1.kernel.attn.fwd.replay", cases_f, Threshold("exact"))
        self.add("L1.kernel.attn.bwd.replay", cases_b, Threshold("exact"))

    # ---------------------------------------------------------------- 4. 合成 poison 与因果性
    def poison(self):
        self._poison_grouped()
        self._poison_attn_thd()
        self._causal_attn()

    def _grouped_run(self, X, dY, m, W, buf_rows=0, fill=np.nan):
        """跑一次 TE GroupedLinear 前向 + 反向。buf_rows > 0 时把输入、输出梯度放进更大的缓冲区（尾部填 fill），只把前面有效行的视图传进去。"""
        import transformer_engine.pytorch as tep
        E = len(m)
        mod = tep.GroupedLinear(E, W[0].shape[1], W[0].shape[0], bias=False, params_dtype=BF16, device=DEV, fuse_wgrad_accumulation=True)
        with torch.no_grad():
            for i in range(E):
                getattr(mod, f"weight{i}").copy_(T(W[i]))
                getattr(mod, f"weight{i}").main_grad = torch.zeros(W[i].shape, dtype=torch.float32, device=DEV)

        def place(a):
            t = T(a)
            if not buf_rows:
                return t
            big = torch.full((t.shape[0] + buf_rows, t.shape[1]), float(fill), dtype=t.dtype, device=DEV)
            big[: t.shape[0]] = t
            return big[: t.shape[0]]                      # 连续视图，尾部内存里是 NaN / Inf
        x = place(X).requires_grad_(True)
        y = mod(x, m)
        y.backward(place(dY))
        return N(y), N(x.grad), [N(getattr(mod, f"weight{i}").main_grad) for i in range(E)]

    def _poison_grouped(self):
        """分组 GEMM：① 尾部越界内存填 NaN / Inf（每段行数不是 128 的倍数），有效行输出、dgrad、wgrad 必须逐位不变；
        ② 某一个专家的行全换成 NaN，其余专家的输出行、dgrad 行、wgrad 必须逐位不变（段边界落在 tile 中间）。"""
        L = self.moe[0]
        X = self.t(f"layers.{L}.mlp.experts.in")
        m = [int(v) for v in self.t(f"layers.{L}.mlp.experts.tokens_per_expert")]
        dY = self.t(f"layers.{L}.mlp.experts.fc1.out.grad")
        W = [self.p(L, f"mlp.experts.linear_fc1.weight{i}") for i in range(len(m))]
        y0, dx0, dw0 = self._grouped_run(X, dY, m, W)
        cases = []
        for fill in (np.nan, np.inf):
            y, dx, dw = self._grouped_run(X, dY, m, W, buf_rows=256, fill=fill)
            tag = "nan" if np.isnan(fill) else "inf"
            ex = {"poison": f"输入与输出梯度缓冲区尾部 256 行填 {tag}", "m_mod128_nonzero": int(sum(1 for v in m if v % 128))}
            cases += [dict(name=f"tail_{tag}.y", out=y, bitref=y0, extra=ex), dict(name=f"tail_{tag}.dx", out=dx, bitref=dx0, extra=ex),
                      dict(name=f"tail_{tag}.dw", out=np.concatenate([d.ravel() for d in dw]),
                           bitref=np.concatenate([d.ravel() for d in dw0]), extra=ex)]
        off = np.concatenate([[0], np.cumsum(m)])
        e = int(np.argmax([v if v % 128 else 0 for v in m]))      # 取一个行数不整除 128、行数最多的专家下毒
        Xp, dYp = X.copy(), dY.copy()
        Xp[off[e]:off[e + 1]] = np.nan
        dYp[off[e]:off[e + 1]] = np.nan
        y, dx, dw = self._grouped_run(Xp, dYp, m, W)
        keep = np.ones(off[-1], bool)
        keep[off[e]:off[e + 1]] = False
        ex = {"poison": f"专家 {e} 的 {m[e]} 行输入与输出梯度全换成 NaN", "poisoned_expert": e,
              "neighbors_rows": [m[e - 1] if e > 0 else None, m[e + 1] if e + 1 < len(m) else None]}
        cases += [dict(name="group_isolation.y_other_rows", out=y[keep], bitref=y0[keep], extra=ex),
                  dict(name="group_isolation.dx_other_rows", out=dx[keep], bitref=dx0[keep], extra=ex),
                  dict(name="group_isolation.dw_other_experts", out=np.concatenate([d.ravel() for i, d in enumerate(dw) if i != e]),
                       bitref=np.concatenate([d.ravel() for i, d in enumerate(dw0) if i != e]), extra=ex)]
        self.add("L1.poison.grouped_gemm", cases, Threshold("exact"))

    def _poison_attn_thd(self):
        """注意力 padding：两条序列打包（THD），长度 1000 / 1500 不整除 128，各自补到 1024 / 1536；padding 行的 q / k / v / dO 填 NaN 或 Inf。
        有效行的输出与 dq / dk / dv 必须与干净输入（padding 填 0）逐位相同。"""
        q, k, v = (self.t(f"layers.0.attn.core.{n}") for n in ("q", "k", "v"))
        dO = self.t("layers.0.attn.core.out.grad").reshape(q.shape[0], self.nh, self.hd)
        lens, pads = [1000, 1500], [1024, 1536]
        cu = torch.tensor([0, lens[0], lens[0] + lens[1]], dtype=torch.int32, device=DEV)
        cup = torch.tensor([0, pads[0], pads[0] + pads[1]], dtype=torch.int32, device=DEV)
        valid = np.zeros(sum(pads), bool)
        valid[: lens[0]] = True
        valid[pads[0]: pads[0] + lens[1]] = True
        src = np.r_[0:lens[0], 1024:1024 + lens[1]]               # 两条序列取真实激活的两段

        def build(z, fill):
            z = z.reshape(z.shape[0], -1, self.hd)
            out = np.full((sum(pads),) + z.shape[1:], fill, dtype=z.dtype)
            out[valid] = z[src]
            return out

        def run(fill):
            dpa = self._dpa(mask="padding_causal", fmt="thd")
            qt, kt, vt = (T(build(z, fill)).requires_grad_(True) for z in (q, k, v))
            o = dpa(qt, kt, vt, cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cup, cu_seqlens_kv_padded=cup,
                    max_seqlen_q=max(pads), max_seqlen_kv=max(pads))
            o.backward(T(build(dO, fill)).reshape(o.shape))
            return [N(o).reshape(sum(pads), -1)[valid]] + [N(g.grad)[valid] for g in (qt, kt, vt)]
        try:
            clean = run(0.0)
            cases = []
            for fill in (np.nan, np.inf):
                tag = "nan" if np.isnan(fill) else "inf"
                pois = run(fill)
                for nm, a, b in zip(("out", "dq", "dk", "dv"), pois, clean):
                    cases.append(dict(name=f"thd_pad_{tag}.{nm}", out=a, bitref=b,
                                      extra={"lens": lens, "padded_to": pads, "poison": f"padding 行 q/k/v/dO 填 {tag}"}))
            self.add("L1.poison.attn_padding", cases, Threshold("exact"))
        except Exception as e:  # noqa: BLE001  这版 TE 不支持就如实记 error，不当作通过
            r = CheckResult("L1.poison.attn_padding", "op_replay", "L1", True, status=ERROR, message=f"THD padding 跑不起来：{e!r}"[:300])
            self.add_result(r)

    def _causal_attn(self):
        """因果性（真实注意力模块、真实 q/k/v）：只改位置 ≥ p 的 q / k / v（换成另一层的真实值），dO 在 ≥ p 处置 0，
        位置 < p 的输出与 dq / dk / dv 必须逐位不变。p 取 2048（整除 128）与 3001（不整除）。
        另做诊断：未来位置填 NaN（dO 仍置 0）时，过去位置会不会被污染——掩码掉的位置若被读进来乘 0，NaN 会漏过去。"""
        q, k, v = (self.t(f"layers.0.attn.core.{n}") for n in ("q", "k", "v"))
        q2, k2, v2 = (self.t(f"layers.1.attn.core.{n}") for n in ("q", "k", "v"))
        dO = self.t("layers.0.attn.core.out.grad").reshape(q.shape)
        s = q.shape[0]

        def run(qq, kk, vv, dd):
            dpa = self._dpa()
            qt, kt, vt = (T(z).requires_grad_(True) for z in (qq, kk, vv))
            o = dpa(qt, kt, vt)
            o.backward(T(dd).reshape(o.shape))
            return [N(o).reshape(s, -1)] + [N(g.grad).reshape(s, -1) for g in (qt, kt, vt)]
        hard, diag = [], []
        for p in (2048, 3001):
            d0 = dO.copy()
            d0[p:] = 0
            base = run(q, k, v, d0)
            qa, ka, va = q.copy(), k.copy(), v.copy()
            qa[p:], ka[p:], va[p:] = q2[p:], k2[p:], v2[p:]
            alt = run(qa, ka, va, d0)
            for nm, a, b in zip(("out", "dq", "dk", "dv"), alt, base):
                hard.append(dict(name=f"p{p}.future_changed.{nm}", out=a[:p], bitref=b[:p], extra={"p": p, "p_mod128": p % 128}))
            qn, kn, vn = q.copy(), k.copy(), v.copy()
            qn[p:], kn[p:], vn[p:] = np.nan, np.nan, np.nan
            nanr = run(qn, kn, vn, d0)
            for nm, a, b in zip(("out", "dq", "dk", "dv"), nanr, base):
                bad_rows = np.where(~np.isfinite(a[:p].astype(np.float32)).all(axis=1))[0]
                diag.append(dict(name=f"p{p}.future_nan.{nm}", out=a[:p], bitref=b[:p],
                                 extra={"p": p, "p_mod128": p % 128, "past_rows_nonfinite": int(len(bad_rows)),
                                        "first_bad_row": int(bad_rows[0]) if len(bad_rows) else None}))
        self.add("L1.causal.attn", hard, Threshold("exact"))
        self.add("L1.causal.attn_future_nan", diag, None, hard=False)


# ---------------------------------------------------------------- 写结果

COLS = ["rel_l2", "cr_match_frac", "ulp_max", "ulp_p999", "bit_match_frac", "nonfinite_mismatch", "variant", "op",
        "impl", "M", "extra_rows", "rows_changed",
        "readback.allow_tf32_matmul", "rows", "m_mod128_nonzero", "past_rows_nonfinite", "first_bad_row",
        "probe_rel_l2_torch_mm_fp32", "probe_rel_l2_te_general_gemm_fp32", "allow_tf32_matmul", "env.NVIDIA_TF32_OVERRIDE",
        "env.TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "a_dtype", "b_dtype", "out_dtype", "layout", "grad", "count"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="带 --observe 的确定性运行（g61-obs）")
    ap.add_argument("--tf32-run", default=None, help="关 TF32 的运行（g61-notf32），读它 meta 里的实测值")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="逗号分隔：tf32,experts,gemm_rows,attention,poison")
    ap.add_argument("--tf32-child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--tmp", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--layer", type=int, default=1, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.tf32_child:
        tf32_child(a.tf32_child, a.run, a.layer, a.tmp)
        return 0
    torch.cuda.set_device(0)
    env = env_more()
    run = Run.open(a.run)
    run.enable_cache()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    job = Kernels(run, a)
    only = set(filter(None, a.only.split(",")))
    for p in ("tf32", "experts", "gemm_rows", "attention", "poison"):
        if not only or p in only:
            getattr(job, p)()
    runs = [run] + ([Run.open(a.tf32_run)] if a.tf32_run else [])
    return write_outputs(out, job.checks, runs, env, Path(__file__), "phase1-kernels", started, columns=COLS,
                         ref_line="参照：训练时抓的输出（逐位）、FP64 真值、逐专家循环、干净输入（poison / 因果性）")


if __name__ == "__main__":
    sys.exit(main())
