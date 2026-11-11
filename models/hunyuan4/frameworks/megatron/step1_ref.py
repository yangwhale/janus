"""混元 4 第 1 步整网梯度的 CPU 参照：GPU golden 的同一套 Megatron-Core 代码，在 CPU 上以 FP64（或 FP32）跑第 1 步。

为什么：L3 第 1 步逐张量梯度的定责要一份和 GPU、TPU 都不同源的参照。这里用 GPU golden 的模型代码
（`hy4_mcore.py`，Megatron-Core 0.20.0 的 local 层，不用 TE）和 GPU 训练入口（`train_hy4mg.py`）的 loss 语义，
在 CPU 上整网 FP64 重算，TPU 一侧的代码一行不用。

精度：`--dtype float64` 时参数、激活、梯度全部 FP64。模型代码里写死的低精度（`.float()`、`dtype=torch.float32`、
`.to(torch.bfloat16)` 等，GPU 上是 FP32 / BF16 的舍入位置）由一个 TorchDispatchMode 在 ATen 一层改写成
FP64（前向、反向都在内），改写次数记进 summary；`--audit` 再挂一个 TorchDispatchMode，逐个算子核输出没有 FP64
以外的浮点类型（违例数记进 summary，应为 0）。

loss 语义（同 `train_hy4mg.py`，GPU g2 / g3 / T2 的设置）：
- 数据：第 1 步全局批 16 条 = tokens.npy 第 0–15 行（GPU 第 m 个 microbatch、DP rank r 取第 m·DP + r 行；
  第 1 步的 16 条与求和顺序无关）；tokens = 前 seq 个、labels = 后 seq 个；loss_mask = (输入 token ≠ eod)。
- 每个 microbatch：主 loss 逐 token 乘 loss_mask 求和（`calculate_per_token_loss`，调度器不除）；MTP loss 由
  Megatron-Core 的 `process_mtp_loss` 挂进反向（系数 mtp_loss_factor，按本条的原 / 滚动 token 数重标），
  反向缩放 1；索引器 KD 由 `compute_dsa_indexer_loss`（sparse、per-token：逐行之和 × 逐层系数 0.1 ÷ 产出层数）
  经 `DSAIndexerLossAutoScaler` 挂进反向，缩放 1，索引器的输入 detach。
- 16 个 microbatch 的梯度相加，再除以全局有效 token 数（`finalize_model_grads` 的 per-token 归一化）。
- 路由：sigmoid、偏置只用于选择（偏置取检查点里的初值）、归一化 top-k × 缩放系数；DSA 选择：full 层自己的
  索引器选 top-k，shared 层复用上一个 full 层，MTP 层自带 full。两种离散选择都由本侧自己的精度选。

输出（--out 目录）：
- `grads_hf.safetensors`：每个参数的梯度（HF checkpoint 名字与布局，与 `export_hf_grads.py` 同一张表；路由专家全部
  16 个），dtype 同 --dtype；
- `summary.json`：逐 microbatch 的 loss 和、有效 token 数、MTP 均值、逐层 KD、耗时、峰值 RSS；全局 lm_loss /
  mtp_loss / indexer_loss（与 GPU metrics.jsonl 同口径）、grad_norm；dtype 改写与审计计数；
- `choices.safetensors`：每个 microbatch 每个 MoE 层的路由选择与选择分数（sigmoid + 偏置），
  `--detail-rows` 指定的行另存 DSA top-k 与索引器的 q / k / w（翻转定责用）；`--dsa-topk-all` 每一行都存 DSA top-k；
- `partial.pt`：每个 microbatch 之后的累加梯度与记录（断点续跑，跑完删掉）。

`--inject-l1 <GPU l1 抓数 step_N_fwd.safetensors> --inject-row R`：第 R 行用 GPU 抓到的 DSA top-k 与路由
（`hf.model.layers.<L>.self_attn.topk`、`.mlp.gate.routing_map`，含 MTP 层）代替本侧的选择，只跑这一行
（灌 GPU 选择的灵敏度试验）。
`--inject-dir <目录>`：全局批每一行都灌 GPU 同一次运行抓到的选择（`step_<N>_row_<R>.safetensors`），输出整批梯度；
要和同一次 GPU 运行的梯度配对比（GPU 重跑时离散选择也会变）。

`--emulate-base2 kappa|full|q16 [--emulate-layers 0,1,mtp0]`（诊断，只前向时用）：在 FP64 里只加 TPU splash base2 模式的
那一处差（`install_base2_emulation`），看主 loss 移多少；不是 GPU 的语义，不能当参照。
`--expert-store` 时路由专家直接载成 BF16（不先过一份 FP64），D3（3.21e10 参数）只前向峰值 RSS 约 109 GiB。

用法（CPU；Megatron-Core 0.20.0 源码在 PYTHONPATH，内存上限用 systemd-run / cgroup 外加）：
    python step1_ref.py --hf <HF 目录> --tokens tokens.npy --dtype float64 --out <目录> --threads 40 --audit
同一脚本用于 hy4-mini / hy4-mid / T2 截层（层数、MTP 由 HF config 读出；--layers 可截层）。内存：hy4-mini FP64
峰值约 60 GB、每个 microbatch 约 1 分钟（40 核）。hy4-mid 的注意力分数（64 头 × 4096²）要加 `--recompute`。
T2 截层（约 121 亿参数）FP64 的参数加梯度约 194 GB，加上激活预计超过单进程 250 GB（估算，没有试）。
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import resource
import shutil
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LOW_FLOATS = (torch.float32, torch.bfloat16, torch.float16)


class ForceDtype(torch.utils._python_dispatch.TorchDispatchMode):
    """在 ATen 一层把关键字参数 dtype 里的低精度浮点类型改成目标类型（`.float()`、`.to(torch.float32)`、
    `torch.zeros(..., dtype=torch.float32)` 都落到带 dtype 关键字的 ATen 算子上；反向里的同样被改）。记改写次数。"""

    def __init__(self, target: torch.dtype):
        super().__init__()
        self.target = target
        self.low = tuple(d for d in LOW_FLOATS if torch.finfo(d).bits < torch.finfo(target).bits)
        self.rewrites: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if kwargs.get("dtype") in self.low:
            kwargs = {**kwargs, "dtype": self.target}
            k = str(func)
            self.rewrites[k] = self.rewrites.get(k, 0) + 1
        return func(*args, **kwargs)


class DtypeAudit(torch.utils._python_dispatch.TorchDispatchMode):
    """逐个算子核输出：浮点输出必须是目标类型。记违例的算子名与次数。"""

    def __init__(self, target: torch.dtype):
        super().__init__()
        self.target = target
        self.bad: dict[str, int] = {}
        self.n_ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.n_ops += 1
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != self.target:
                k = f"{func}:{t.dtype}"
                if k not in self.bad and os.environ.get("STEP1_AUDIT_TRACE"):
                    import traceback

                    print("AUDIT", k, "".join(traceback.format_stack(limit=12)[:-2]), flush=True)
                self.bad[k] = self.bad.get(k, 0) + 1
        return out


def cpu_megatron_setup():
    """Megatron-Core 在 CPU 上跑需要的补丁（同 equiv_hf.py）。"""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    torch.distributed.init_process_group("gloo", rank=0, world_size=1)
    torch.cuda.current_device = lambda: "cpu"
    import megatron.core.transformer.multi_latent_attention as _mla

    if _mla.QuantizedTensor is None:
        _mla.QuantizedTensor = ()
    import megatron.core.transformer.multi_token_prediction as _mtp

    _mtp.is_inference_column_parallel_linear = lambda m: False
    from megatron.core import parallel_state as ps

    ps.initialize_model_parallel(1, 1)


def _unforced(fn):
    """包一层：调用期间暂时退出 ForceDtype。"""

    def wrapped(*x, **k):
        with torch.utils._python_dispatch._pop_mode_temporarily():
            return fn(*x, **k)

    return wrapped


def relax_mask_dtype(dt):
    """DSA 的加性掩码（0 / -inf，任何浮点类型里都精确）原版断言是 FP32；目标类型的掩码也放行，其余逻辑原样。"""
    import megatron.core.transformer.experimental_attention_variant.dsa_masking as DM

    orig = DM.prepare_additive_mask

    def prepare_additive_mask(mask, *, sq, sk, b, device):
        if mask is None or mask.dtype != dt or dt == torch.float32:
            return orig(mask, sq=sq, sk=sk, b=b, device=device)
        assert mask.ndim in (2, 3) and mask.shape[-2:] == (sq, sk)
        if mask.ndim == 2:
            return mask, mask.view(1, 1, sq, sk), mask.unsqueeze(0), torch.isfinite(mask).unsqueeze(0).expand(b, sq, sk)
        return mask, mask.view(b, 1, sq, sk), mask, torch.isfinite(mask)

    DM.prepare_additive_mask = prepare_additive_mask


class RouterTap:
    """挂在 TopKRouter.router_replay 上：记每次选择（下标与含偏置的选择分数），可选灌给定的下标。"""

    def __init__(self):
        self.records = []
        self.inject = None  # [T, k] long

    def get_replay_topk(self, scores, topk, num_groups=None, group_topk=None, default_compute_topk=None):
        vals, idx = default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
        self.records.append({"idx": idx.detach().clone(), "scores": scores.detach().clone()})
        if self.inject is not None:
            inj = self.inject.to(device=scores.device, dtype=torch.long)
            return scores.gather(1, inj), inj
        return vals, idx


class ExpertGradStore:
    """路由专家的权重梯度累加器（T2 这类专家参数占大头的截层用）：权重本身只存 BF16，梯度单独在 FP64 里累加。

    path 为 None：累加在内存里（每个专家一块 FP64）。给了 path：累加在磁盘上的 FP64 文件里（每次加 pread、加、pwrite、
    fdatasync + POSIX_FADV_DONTNEED，省内存，盘慢时很慢）。断点时整份存一次（partial.pt.experts），续跑时读回。
    """

    def __init__(self, path=None):
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644) if path else None
        self.buf = {}
        self.off = {}
        self.size = 0
        self.only = None  # 给了集合就只累加这些名字（--expert-grads-kept-only），其余专家的权重梯度不算

    def register(self, name, shape):
        n = 1
        for d in shape:
            n *= d
        self.off[name] = (self.size, n, tuple(shape))
        self.size += n

    def finalize(self):
        if self.fd is not None:
            os.ftruncate(self.fd, self.size * 8)  # 稀疏文件，读出来是 0

    def _io(self, name):
        o, n, _ = self.off[name]
        return o * 8, n * 8

    def wants(self, name):
        return self.only is None or name in self.only

    def add(self, name, g):
        if not self.wants(name):
            return
        if self.fd is None:
            if name in self.buf:
                self.buf[name] += g.detach()
            else:
                self.buf[name] = g.detach().clone()
            return
        import numpy as np

        o, nb = self._io(name)
        cur = np.frombuffer(os.pread(self.fd, nb, o), dtype=np.float64).copy()
        cur += g.detach().reshape(-1).numpy()
        os.pwrite(self.fd, cur.tobytes(), o)
        os.fdatasync(self.fd)
        os.posix_fadvise(self.fd, o, nb, os.POSIX_FADV_DONTNEED)

    def get(self, name):
        shape = self.off[name][2]
        if self.fd is None:
            return self.buf.get(name, torch.zeros(shape, dtype=torch.float64))
        import numpy as np

        o, nb = self._io(name)
        x = np.frombuffer(os.pread(self.fd, nb, o), dtype=np.float64).copy()
        os.posix_fadvise(self.fd, o, nb, os.POSIX_FADV_DONTNEED)
        return torch.from_numpy(x).reshape(shape)

    def save(self, path):
        if self.fd is None:
            torch.save(self.buf, path)
        else:
            shutil.copyfile(self.path, path)

    def load(self, path):
        if self.fd is None:
            self.buf = torch.load(path, weights_only=False)
        else:
            shutil.copyfile(path, self.path)

    def close(self, remove=True):
        if self.fd is not None:
            os.close(self.fd)
            if remove:
                os.remove(self.path)


class _StoredExpertLinear(torch.autograd.Function):
    """y = x · Wᵀ，W 从 BF16 存储精确转成 x 的类型；dW = gᵀ · x 在 x 的类型里算，交给 ExpertGradStore。

    与 Megatron-Core local 线性层（LinearWithGradAccumulationAndAsyncCommunication）同一组矩阵乘。
    """

    @staticmethod
    def forward(ctx, x, w_store, name, store):
        ctx.save_for_backward(x)
        ctx.w_store, ctx.name, ctx.store = w_store, name, store
        return torch.matmul(x, w_store.to(x.dtype).t())

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        w = ctx.w_store.to(g.dtype)
        gx = torch.matmul(g, w)
        if ctx.store.wants(ctx.name):
            gw = torch.matmul(g.reshape(-1, g.shape[-1]).t(), x.reshape(-1, x.shape[-1]))
            ctx.store.add(ctx.name, gw)
        return gx, None, None, None


def install_expert_store(model, store):
    """把每个路由专家的 linear_fc1 / linear_fc2 权重换成 BF16 存储（核过与 FP64 参数逐位相等）+ _StoredExpertLinear。"""
    n = 0
    for name, mod in list(model.named_modules()):
        if ".experts.local_experts." not in name or not name.endswith(("linear_fc1", "linear_fc2")):
            continue
        w = mod.weight
        wb = _unforced(lambda w=w: w.detach().to(torch.bfloat16))()
        assert _unforced(lambda: torch.equal(wb.to(w.dtype), w.detach()))(), f"{name} 的权重不是 BF16 可精确表示的"
        pname = name + ".weight"
        store.register(pname, tuple(w.shape))
        del mod._parameters["weight"]
        mod.register_buffer("weight_bf16", wb, persistent=False)

        def fwd(x, *args, _m=mod, _n=pname, **kw):
            return _StoredExpertLinear.apply(x, _m.weight_bf16, _n, store), None

        mod.forward = fwd
        n += 1
    store.finalize()
    return n


def build(a, dt):
    import hy4_mcore as H
    from equiv_hf import build_mcore

    hf, _ = H.complete_hf_config(json.load(open(os.path.join(a.hf, "config.json"))))
    L = a.layers or hf["num_hidden_layers"]
    mtp = a.mtp if a.mtp is not None else int(hf.get("num_nextn_predict_layers", 0)) > 0
    coeff = H.indexer_loss_coeff_per_layer(a.indexer_loss, hf["indexer_types"][:L], int(mtp))
    over = dict(
        params_dtype=dt,
        hy4_train_indexer=a.indexer_loss > 0,
        dsa_indexer_loss_coeff=coeff,
        dsa_indexer_use_sparse_loss=True,
        calculate_per_token_loss=True,
        hy4_attn_scores_fp32=True,
        hy4_head_chunk=a.head_chunk,
        cross_entropy_loss_fusion=False,
        gradient_accumulation_fusion=False,
        moe_router_dtype="fp64" if dt == torch.float64 else "fp32",
    )
    if a.log_max_logit:  # 每层每头缩放后 q·k 的最大值（选中且因果，不含 sink），跨行取最大；QK-Clip 的统计量
        over["log_max_attention_logit"] = True
    if a.expert_store:  # 路由专家直接载成 BF16（HF 原件就是 BF16，逐位不变），不先过一份 FP64：峰值内存少一份专家
        import hy4_weights as W

        orig_load = W.load_hf_into_mcore

        def load_bf16_experts(model, *args, **kw):
            for name, mod in list(model.named_modules()):
                if ".experts.local_experts." in name and name.endswith(("linear_fc1", "linear_fc2")):
                    shape = tuple(mod.weight.shape)
                    del mod._parameters["weight"]
                    mod.weight = torch.nn.Parameter(
                        _unforced(lambda: torch.empty(shape, dtype=torch.bfloat16))(), requires_grad=False
                    )
            return orig_load(model, *args, **kw)

        W.load_hf_into_mcore = load_bf16_experts
        try:
            cfg, model = build_mcore(a.hf, L, torch.device("cpu"), False, mtp=mtp, **over)
        finally:
            W.load_hf_into_mcore = orig_load
        return hf, cfg, model, L, mtp, coeff
    cfg, model = build_mcore(a.hf, L, torch.device("cpu"), False, mtp=mtp, **over)
    return hf, cfg, model, L, mtp, coeff


LOG2E = 1.4426950408889634
LOG2E_BF16 = 1.4453125  # tokamax splash 的 base2 模式里 `q *= LOG2E`：q 是 BF16，Python 标量按 BF16 舍成这个值
KAPPA_BASE2 = LOG2E_BF16 / LOG2E  # 1.00181428…：logits（不含 sink）被整体放大的倍数


def install_base2_emulation(cores, mode: str, layers: set):
    """在 FP64 参照里只加 TPU splash base2 模式的一处差（诊断，不是 GPU 的语义）。

    - `kappa`：选中层的 q·k（不含 sink）乘 KAPPA_BASE2，其余照 FP64；
    - `full`：q 先舍成 BF16（TPU 进内核的 query 是 BF16，缩放 1/16 精确），再乘 BF16 的 LOG2E 舍回 BF16、除以精确的
      LOG2E（内核的 exp2 与换回自然底的 logsumexp 等价于这个 q），k / v / sink 照 FP64；
    - `q16`：只把 q 舍成 BF16（`full` 的对照：两者之差是 base2 那一乘一舍）。
    """
    import hy4_mcore as H

    orig = H.sparse_sink_attention
    cur = {"name": None}
    rnd = _unforced(lambda t: t.to(torch.bfloat16).to(t.dtype))

    def ssa(query, key, value, topk, sinks, scale, scores_fp32=False, return_max_logit=False):
        if cur["name"] not in layers:
            return orig(query, key, value, topk, sinks, scale, scores_fp32, return_max_logit)
        if mode == "kappa":
            return orig(query, key, value, topk, sinks, scale * KAPPA_BASE2, scores_fp32, return_max_logit)
        q = rnd(query * scale)
        if mode == "full":
            q = rnd(q * LOG2E_BF16) / LOG2E
        return orig(q, key, value, topk, sinks, 1.0, scores_fp32, return_max_logit)

    H.sparse_sink_attention = ssa
    for name, c in cores:
        f = c.forward

        def fwd(*x, _n=name, _f=f, **k):
            cur["name"] = _n
            try:
                return _f(*x, **k)
            finally:
                cur["name"] = None

        c.forward = fwd


def hf_name_table(cfg, mtp):
    """Megatron 参数名 → (HF 名, 变换)，同 export_hf_grads.mcore_to_hf，路由专家用 SequentialMLP 的名字。"""
    from hy4_weights import GLOBAL, expert_param_name, layer_plan, mtp_names

    out = {m: (h, None) for m, h in GLOBAL}
    for i in range(cfg.num_layers):
        pm, ph = f"decoder.layers.{i}.", f"model.layers.{i}."
        for m, h, tf in layer_plan(cfg, i):
            out[pm + m] = (tuple(ph + x for x in h), "cat") if tf == "cat" else (ph + h, None)
        if cfg.moe_layer_freq[i]:
            for e in range(cfg.num_moe_experts):
                out[expert_param_name(pm, "linear_fc1", e, False)] = (ph + "mlp.experts.gate_up_proj", ("expert", e))
                out[expert_param_name(pm, "linear_fc2", e, False)] = (ph + "mlp.experts.down_proj", ("expert", e))
    if mtp:
        for m, h, tf in mtp_names(cfg):
            if tf == "experts":
                for e in range(cfg.num_moe_experts):
                    out[expert_param_name(m, "linear_fc1", e, False)] = (h + "mlp.experts.gate_up_proj", ("expert", e))
                    out[expert_param_name(m, "linear_fc2", e, False)] = (h + "mlp.experts.down_proj", ("expert", e))
            elif tf == "cat":
                out[m] = (tuple(h), "cat")
            else:
                out[m] = (h, None)
    return out


def to_hf(grads: dict, cfg, mtp) -> dict:
    table = hf_name_table(cfg, mtp)
    out, experts = {}, {}
    for name, g in grads.items():
        if name.endswith("expert_bias"):
            continue
        h, tf = table[name]
        if tf == "cat":
            half = g.shape[0] // 2
            out[h[0]], out[h[1]] = g[:half], g[half:]
        elif isinstance(tf, tuple):
            experts.setdefault(h, {})[tf[1]] = g
        else:
            out[h] = g
    for h, by_e in experts.items():
        out[h] = torch.stack([by_e[e] for e in sorted(by_e)])
    return out


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--gbs", type=int, default=16)
    p.add_argument("--step", type=int, default=1, help="取第几步的全局批（行 (step-1)·gbs … step·gbs-1）")
    p.add_argument("--rows", default="", help="只跑这些行（逗号分隔，调试 / 估时用）；空 = 全局批全部")
    p.add_argument("--layers", type=int, default=0, help="截层；0 = config 的层数")
    p.add_argument("--mtp", type=lambda s: s.lower() in ("1", "true", "on"), default=None)
    p.add_argument("--indexer-loss", type=float, default=0.1)
    p.add_argument("--head-chunk", type=int, default=1024)
    p.add_argument("--eod-mask", default="on", choices=["on", "off"])
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--detail-rows", default="0", help="这些行另存 DSA top-k 与索引器 q / k / w")
    p.add_argument(
        "--dsa-topk-all",
        action="store_true",
        help="每一行都存各自带索引器层的 DSA top-k（r<行>.dsa.<层>.topk；路由一致率对真值口径用），不存 q / k / w",
    )
    p.add_argument("--inject-l1", default="", help="GPU l1 抓数的 step_N_fwd.safetensors")
    p.add_argument("--inject-row", type=int, default=0)
    p.add_argument(
        "--inject-dir",
        default="",
        help="整批灌 GPU 选择：目录里每行一个 step_<N>_row_<R>.safetensors（同 --inject-l1 的格式），全局批每行都灌",
    )
    p.add_argument("--audit", action="store_true")
    p.add_argument(
        "--checkpoint-every", type=int, default=1, help="每隔几行写一次断点（partial.pt，参数大小的两倍临时盘）"
    )
    p.add_argument(
        "--no-row-grads",
        dest="save_row_grads",
        action="store_false",
        help="不写第一行单独的梯度（只在和灌 GPU 选择的那一行比时才用得到）",
    )
    p.add_argument(
        "--recompute",
        action="store_true",
        help="主干每层与 MTP 层整层重算，省激活内存（hy4-mid、T2 用）；数学不变",
    )
    p.add_argument(
        "--expert-store",
        nargs="?",
        const="disk",
        default="",
        choices=["", "disk", "ram"],
        help="路由专家权重留 BF16，梯度单独在 FP64 里累加：disk（默认，磁盘文件，最省内存）或 ram（内存，快）",
    )
    p.add_argument("--keep-experts", default="", help="输出里只留这些路由专家（逗号分隔，如 0,1；空 = 全部）")
    p.add_argument(
        "--expert-grads-kept-only",
        action="store_true",
        help="和 --expert-store、--keep-experts 一起用：只累加留下的专家的权重梯度，其余专家不算权重梯度"
        "（输入梯度照算，其余参数的梯度不变）；省掉全部专家 FP64 累加器的内存。summary 的 grad_norm 不含没算的专家",
    )
    p.add_argument(
        "--forward-only",
        action="store_true",
        help="只跑前向（记 loss、选择、最大 logit 与 --detail-rows 的中间量），不算梯度",
    )
    p.add_argument(
        "--emulate-base2",
        default="off",
        choices=["off", "kappa", "full", "q16"],
        help="诊断：在 FP64 里只加 TPU splash base2 模式的那一处差（见 install_base2_emulation）；off = 原样参照",
    )
    p.add_argument("--emulate-layers", default="all", help="--emulate-base2 作用的层（0,1,…,mtp0；all = 全部注意力层）")
    p.add_argument(
        "--log-max-logit",
        action="store_true",
        help="记每层每头的最大注意力 logit（跨行取最大，写进 summary 的 max_logit）",
    )
    a = p.parse_args()
    if a.audit and a.emulate_base2 in ("full", "q16"):
        # _unforced 只退出最上面一层 dispatch mode：带 --audit 时退出的是审计，ForceDtype 还在，q 的 BF16 舍入被改回
        # FP64，full 静默退化成 kappa（D3 实测 lm_loss 与 kappa 逐位相同）。两者本来也矛盾：舍入就是要出 BF16
        raise SystemExit("--emulate-base2 full / q16 不能和 --audit 一起用")

    if a.threads:
        torch.set_num_threads(a.threads)
    dt = getattr(torch, a.dtype)
    torch.set_default_dtype(dt)
    os.makedirs(a.out, exist_ok=True)
    force = ForceDtype(dt)
    force.__enter__()  # 建模型、载入、前向反向全程在内
    cpu_megatron_setup()
    import hy4_weights as W

    for nm in ("get", "get_slice"):  # 读 safetensors 时不改写（它按原 dtype 解释字节）
        setattr(W.Safetensors, nm, _unforced(getattr(W.Safetensors, nm)))
    relax_mask_dtype(dt)
    import hy4_mcore as H
    import numpy as np
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper

    code_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    t0 = time.time()
    hf, cfg, model, L, mtp, coeff = build(a, dt)
    build_s = time.time() - t0
    model.train()
    if a.recompute:  # 主干每层整层重算（torch 的非重入 checkpoint；Megatron 的重算要 CUDA 的随机数状态，CPU 上用不了）
        import torch.utils.checkpoint as ckpt

        layers = list(model.decoder.layers) + ([model.mtp.layers[0].mtp_model_layer] if mtp else [])
        for layer in layers:  # 主干各层与 MTP 层内的解码层
            layer.forward = functools.partial(ckpt.checkpoint, layer.forward, use_reentrant=False)
    bad_dtype = [
        n
        for n, q in model.named_parameters()
        if q.dtype != dt and not (a.expert_store and ".experts.local_experts." in n and q.dtype == torch.bfloat16)
    ]
    assert not bad_dtype, bad_dtype[:5]
    store = None
    if a.expert_store:
        assert dt == torch.float64
        store = ExpertGradStore(os.path.join(a.out, "expert_grads.f64") if a.expert_store == "disk" else None)
        print(
            "EXPERT_STORE",
            install_expert_store(model, store),
            "linears",
            f"{store.size * 8 / 2**30:.1f}GiB",
            flush=True,
        )
        if a.expert_grads_kept_only:
            keep0 = {int(e) for e in a.keep_experts.split(",") if e}
            assert keep0, "--expert-grads-kept-only 要配 --keep-experts"
            table0 = hf_name_table(cfg, mtp)
            store.only = {p for p in store.off if table0[p][1][1] in keep0}
            print("EXPERT_GRADS_KEPT_ONLY", len(store.only), "linears", flush=True)

    # 选择的记录点
    cores = [(str(i), l.self_attention.core_attention) for i, l in enumerate(model.decoder.layers)]
    if mtp:
        cores.append(("mtp0", model.mtp.layers[0].mtp_model_layer.self_attention.core_attention))
    taps = {}
    moe_names = [str(i) for i, l in enumerate(model.decoder.layers) if isinstance(l.mlp, MoELayer)]
    for i, l in enumerate(model.decoder.layers):
        if isinstance(l.mlp, MoELayer):
            taps[str(i)] = l.mlp.router.router_replay = RouterTap()
    if mtp:
        taps["mtp0"] = model.mtp.layers[0].mtp_model_layer.mlp.router.router_replay = RouterTap()
        moe_names.append("mtp0")
    for _, c in cores:
        c.capture = True
    if a.emulate_base2 != "off":
        emul_layers = {n for n, _ in cores} if a.emulate_layers == "all" else set(a.emulate_layers.split(","))
        assert emul_layers <= {n for n, _ in cores}, (emul_layers, [n for n, _ in cores])
        install_base2_emulation(cores, a.emulate_base2, emul_layers)
        print("EMULATE_BASE2", a.emulate_base2, sorted(emul_layers), f"kappa={KAPPA_BASE2!r}", flush=True)
    idx_inputs = {}

    toks = np.load(a.tokens, mmap_mode="r")
    base = (a.step - 1) * a.gbs
    rows = [int(r) for r in a.rows.split(",") if r] if a.rows else list(range(base, base + a.gbs))
    if a.inject_l1:
        rows = [a.inject_row]
    detail = {int(r) for r in a.detail_rows.split(",") if r}
    eod = int(hf["eos_token_id"])
    pos = torch.arange(a.seq).unsqueeze(0)

    def load_inject(path, row):
        """GPU 抓到的这一行的 DSA top-k 与路由 → {("topk", 层): [1, S, k]，("route", 层): [T, k]}。"""
        from safetensors import safe_open

        out = {}
        with safe_open(path, "pt") as f:
            if "input.tokens" in f.keys():
                tok_gpu = f.get_tensor("input.tokens").reshape(-1)
                want = torch.from_numpy(np.asarray(toks[row, : a.seq]).astype(np.int64))
                assert torch.equal(tok_gpu, want), f"GPU 抓数的输入不是第 {row} 行"
            for name, c in cores:
                hn = f"hf.model.layers.{name}" if name != "mtp0" else "hf.model.mtp_layers.0"
                if c.is_full:
                    out[("topk", name)] = f.get_tensor(f"{hn}.self_attn.topk")
                if name in taps:
                    rm = f.get_tensor(f"{hn}.mlp.gate.routing_map")
                    k = cfg.moe_router_topk
                    assert int(rm.sum(1).min()) == k and int(rm.sum(1).max()) == k
                    out[("route", name)] = torch.argsort(rm.to(torch.int8), dim=1, descending=True, stable=True)[:, :k]
        return out

    inject = load_inject(a.inject_l1, a.inject_row) if a.inject_l1 else {}

    # 断点续跑
    part_path = os.path.join(a.out, "partial.pt")
    done, recs, choices = [], [], {}
    if os.path.exists(part_path) and not a.inject_l1:
        st = torch.load(part_path, weights_only=False)
        done, recs, choices = st["done"], st["recs"], st["choices"]
        for n, g in st["acc"].items():
            dict(model.named_parameters())[n].grad = g
        if store is not None:
            assert os.path.exists(part_path + ".experts"), "断点缺专家累加文件"
            store.load(part_path + ".experts")
        print(f"RESUME {len(done)} rows done", flush=True)
    if not done:
        choices = {}
    audit = DtypeAudit(dt) if a.audit else None
    params = dict(model.named_parameters())
    for r in rows:
        if r in done:
            continue
        for t in taps.values():  # 梯度不清零：各行直接在 .grad 里相加（省一份参数大小的内存）
            t.records.clear()
        if a.inject_dir:  # 整批灌 GPU 同一次运行的选择（每行一个文件）
            inject = load_inject(os.path.join(a.inject_dir, f"step_{a.step}_row_{r:02d}.safetensors"), r)
        if a.inject_l1 or a.inject_dir:
            for name, c in cores:
                if c.is_full:
                    c.inject_topk = inject[("topk", name)]
            for name, t in taps.items():
                t.inject = inject[("route", name)]
        row = np.asarray(toks[r, : a.seq + 1]).astype(np.int64)
        ids = torch.from_numpy(row[:-1])[None]
        labels = torch.from_numpy(row[1:])[None]
        lm = (ids != eod).to(dt) if a.eod_mask == "on" else torch.ones_like(ids, dtype=dt)
        if r in detail:  # 记索引器的 q / k / w（forward_before_topk 的输出）
            for name, c in cores:
                if c.is_full:
                    c.indexer.forward_before_topk_orig = c.indexer.forward_before_topk

                    def fbt(*args, _c=c, _n=name, **kw):
                        out = _c.indexer.forward_before_topk_orig(*args, **kw)
                        idx_inputs[_n] = tuple(t.detach().clone() for t in out)
                        return out

                    c.indexer.forward_before_topk = fbt
        MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0, dtype=dt))
        DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0, dtype=dt))
        for _, c in cores:
            c.indexer_loss_accum = None
        if mtp:
            MTPLossLoggingHelper.clean_metrics_in_tracker()
        t1 = time.time()
        if audit is not None:
            audit.__enter__()
        with torch.set_grad_enabled(not a.forward_only):
            out = model(ids, pos, attention_mask=None, labels=labels, loss_mask=lm)  # 逐 token 主 loss [1, S]
            s = (out.float() * lm).sum()
        tf = time.time() - t1
        if not a.forward_only:
            s.backward()
        if audit is not None:
            audit.__exit__(None, None, None)
        tb = time.time() - t1 - tf
        n_tok = float(lm.sum())
        rec = {"row": r, "loss_sum": float(s), "n_tokens": n_tok, "fwd_s": round(tf, 1), "bwd_s": round(tb, 1)}
        if mtp:
            rec["mtp_mean"] = float(MTPLossLoggingHelper.tracker["loss_values"][0])
            MTPLossLoggingHelper.clean_metrics_in_tracker()
        rec["kd"] = {n: float(c.last_indexer_loss) for n, c in cores if c.is_full and c.last_indexer_loss is not None}
        for name, t in taps.items():
            assert len(t.records) == (2 if a.recompute and not a.forward_only else 1), (
                name,
                len(t.records),
            )  # 重算时主干前向跑两遍
            rr = t.records[0]
            choices[f"r{r}.route.{name}.idx"] = rr["idx"].to(torch.int16)
            choices[f"r{r}.route.{name}.scores"] = rr["scores"]
        if r in detail:
            for name, c in cores:
                if c.is_full:
                    choices[f"r{r}.dsa.{name}.topk"] = c.last_topk.to(torch.int16)
                    q, k, w = idx_inputs[name]
                    choices[f"r{r}.dsa.{name}.q"], choices[f"r{r}.dsa.{name}.k"] = q, k
                    choices[f"r{r}.dsa.{name}.w"] = w
                    _, _, query, key = c.last_kd_inputs  # 教师（注意力）的 q / k：RoPE 之后、缩放之前 [S, B, 头, 维]
                    choices[f"r{r}.dsa.{name}.query"], choices[f"r{r}.dsa.{name}.key"] = query, key
                    c.indexer.forward_before_topk = c.indexer.forward_before_topk_orig
        elif a.dsa_topk_all:
            for name, c in cores:
                if c.is_full:
                    choices[f"r{r}.dsa.{name}.topk"] = c.last_topk.to(torch.int16)
        acc = {n: q.grad for n, q in params.items() if q.grad is not None}
        rec["peak_rss_gib"] = round(rss_gib(), 1)
        recs.append(rec)
        done.append(r)
        print("MICRO", json.dumps(rec), flush=True)
        last = len(done) == len(rows)
        if not a.inject_l1 and not last and len(done) % a.checkpoint_every == 0:
            need = 2 * sum(v.numel() * v.element_size() for v in acc.values()) + (20 << 30)
            need += 2 * store.size * 8 if store is not None else 0
            free = shutil.disk_usage(a.out).free
            if free > need:  # 盘不够就跳过这次断点（不写满盘），记一笔
                if store is not None:  # 专家累加器同一时刻的副本（行与行之间是一致的）
                    store.save(part_path + ".experts.tmp")
                torch.save({"done": done, "recs": recs, "acc": acc, "choices": choices}, part_path + ".tmp")
                if store is not None:
                    os.replace(part_path + ".experts.tmp", part_path + ".experts")
                os.replace(part_path + ".tmp", part_path)
            else:
                print(f"SKIP_CHECKPOINT free={free / 2**30:.0f}GiB need={need / 2**30:.0f}GiB", flush=True)
        if len(done) == 1 and (
            a.inject_l1 or a.save_row_grads
        ):  # 第一行单独的梯度（未除 token 数；灌 GPU 选择的灵敏度试验与它比）
            from safetensors.torch import save_file

            tag = "inject" if a.inject_l1 else "self"
            save_file(
                {k: v.contiguous() for k, v in to_hf(dict(acc), cfg, mtp).items()},
                os.path.join(a.out, f"grads_row{r}_{tag}_unscaled_hf.safetensors"),
            )

    acc = {n: q.grad for n, q in params.items() if q.grad is not None}
    # 归一化：全局有效 token 数（只在跑全局批全部 16 行时有意义）
    N = sum(x["n_tokens"] for x in recs)
    for n, q in params.items():
        if n not in acc and q.requires_grad and not a.forward_only:
            acc[n] = torch.zeros_like(q)
    grads = {n: g / N for n, g in acc.items()}
    keep = {int(e) for e in a.keep_experts.split(",") if e}
    expert_sq = {}
    if (
        store is not None and not a.forward_only
    ):  # 专家梯度从累加文件读回；平方和按 HF 名字记全部专家，输出只留 --keep-experts
        table = hf_name_table(cfg, mtp)
        for pname in store.off:
            if not store.wants(pname):
                continue
            g = store.get(pname) / N
            h, (_, e) = table[pname]
            expert_sq[h] = expert_sq.get(h, 0.0) + float(g.square().sum())
            if not keep or e in keep:
                grads[pname] = g
    elif keep:
        table = hf_name_table(cfg, mtp)
        for pname in list(grads):
            tf = table.get(pname, (None, None))[1]
            if isinstance(tf, tuple) and tf[1] not in keep:
                h = table[pname][0]
                expert_sq[h] = expert_sq.get(h, 0.0) + float(grads.pop(pname).double().square().sum())
    hfg = to_hf(grads, cfg, mtp)
    from safetensors.torch import save_file

    if not a.inject_l1 and not a.forward_only:
        save_file({k: v.contiguous() for k, v in sorted(hfg.items())}, os.path.join(a.out, "grads_hf.safetensors"))
    if choices:
        save_file({k: v.contiguous() for k, v in choices.items()}, os.path.join(a.out, "choices.safetensors"))
    gsq = {k: float(v.double().square().sum()) for k, v in hfg.items()}
    for h, v in expert_sq.items():  # 输出里没留的专家也计入 grad_norm
        if store is not None and h in gsq:
            gsq[h] = v  # 存储路径：expert_sq 已含留下的那几个
        else:
            gsq[h] = gsq.get(h, 0.0) + v
    if store is not None:
        store.close()
    S = a.seq
    n_mb = len(recs)
    summ = {
        "what": "第 1 步整网梯度的 CPU 参照（GPU golden 同一套 Megatron-Core 代码，local 层）",
        "dtype": a.dtype,
        "hf_dir": os.path.abspath(a.hf),
        "tensor_sha256_manifest": hashlib.sha256(
            open(os.path.join(a.hf, "tensor_sha256.json"), "rb").read()
        ).hexdigest(),
        "tokens": os.path.abspath(a.tokens),
        "rows": rows,
        "inject_l1": a.inject_l1 or None,
        "layers": L,
        "mtp": mtp,
        "indexer_coeff_per_layer": coeff,
        "code": {
            "hy4_mcore_sha256": hashlib.sha256(open(H.__file__, "rb").read()).hexdigest(),
            "step1_ref_sha256": code_sha,
            "megatron_core": __import__("megatron.core", fromlist=["__version__"]).__version__,
            "torch": torch.__version__,
        },
        "threads": torch.get_num_threads(),
        "build_seconds": round(build_s, 1),
        "global_tokens": N,
        "lm_loss": sum(x["loss_sum"] for x in recs) / N,
        "mtp_loss": (sum(x["mtp_mean"] for x in recs) / n_mb) if mtp else None,
        "indexer_loss": sum(sum(x["kd"].values()) for x in recs) / n_mb / S,
        "indexer_loss_per_layer": {
            k: sum(x["kd"].get(k, 0.0) for x in recs) / n_mb / S for k in (recs[0]["kd"] if recs else {})
        },
        "grad_norm": sum(gsq.values()) ** 0.5,
        "n_tensors": len(hfg),
        "kept_experts": sorted(keep) or "all",
        "expert_store": a.expert_store,
        "expert_grads_kept_only": bool(a.expert_grads_kept_only),
        "emulate_base2": None
        if a.emulate_base2 == "off"
        else {"mode": a.emulate_base2, "layers": a.emulate_layers, "kappa": KAPPA_BASE2},
        "dtype_rewrites": force.rewrites,
        "audit": None if audit is None else {"ops": audit.n_ops, "non_target_float_outputs": audit.bad},
        "max_logit": {
            n: [float(v) for v in c.current_max_attn_logits]
            for n, c in cores
            if getattr(c, "current_max_attn_logits", None) is not None
        },
        "micro": recs,
    }
    with open(os.path.join(a.out, "summary.json" if not a.inject_l1 else "summary_inject.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print("DONE", json.dumps({k: summ[k] for k in ("lm_loss", "mtp_loss", "indexer_loss", "grad_norm")}), flush=True)
    if os.path.exists(part_path) and not a.inject_l1 and len(done) == len(rows):
        os.remove(part_path)
        if os.path.exists(part_path + ".experts"):
            os.remove(part_path + ".experts")


if __name__ == "__main__":
    main()
