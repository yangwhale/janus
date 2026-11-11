"""整套参数导出 / 导入（GPU 侧，Megatron-Bridge 回调）：同一份状态换到另一侧或另一种精度上再算一步梯度用。

文件格式（两侧通用，TPU 侧 frameworks/maxtext/train_hy3.py 的 --export-params-at / --params-from 读写同一种）：
    <目录>/rank<r>.safetensors   Megatron 名字与布局、FP32；rank 0 存全部非专家参数 + 自己的专家，rank r>0 只存自己的专家
                                （专家用本地名 weight<k>，全局专家号 = 24r + k）；路由偏置等 buffer 名字前加 "buffer."
    <目录>/manifest.json         每个张量的 sha256（FP32 原始字节）、形状；导入后逐张量核对

导出：第 N 步优化器更新之后（on_train_step_end）写 FP32 主权重（就是下一步用的状态 θ_N；BF16 参数是它舍入来的）。
导入：开训前（on_train_start，在抓数回调之前）把 FP32 主权重、BF16 参数（= FP32 按 RNE 舍入）、buffer 全部换掉，
     换完逐张量读回算 sha256 对 manifest，不一致就停。优化器动量不动（只比一步梯度时用不到）。

数据跳步（--skip-steps K）：开训时先从每个 rank 的数据迭代器里丢掉 K 步的 microbatch，这一侧的第 1 步就吃原数据第 K+1 步的
batch（抓数的输入指纹可以和 golden 第 K+1 步逐位对）。配合 --import-params 在另一种设置（例如 FP32）下重算第 K+1 步的梯度。
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import torch
import torch.distributed as dist


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _sha_f32(t: torch.Tensor) -> str:
    a = t.detach().float().contiguous().cpu().numpy()
    return hashlib.sha256(a.reshape(-1).view(np.uint8)).hexdigest()


_EXPERT = re.compile(r"\.experts\.linear_fc[12]\.weight\d+$")


def _unwrap(model):
    model = getattr(model, "module", model)
    while hasattr(model, "module") and not hasattr(model, "decoder"):
        model = model.module
    return model


def _pairs(opt, params):
    """(名字, 模型参数, FP32 主权重)。与 janus/dump/megatron.py 的 JanusDump._pairs 相同的拆法；
    本身就是 FP32 的参数（FP32 训练时全部如此）参数即主权重。"""
    names = {id(p): n for n, p in params}
    out = []
    for sub in getattr(opt, "chained_optimizers", None) or [opt]:
        f16 = getattr(sub, "float16_groups", None) or []
        f32 = getattr(sub, "fp32_from_float16_groups", None) or []
        out += [(names[id(a)], a, b) for ga, gb in zip(f16, f32) for a, b in zip(ga, gb) if id(a) in names]
        out += [(names[id(a)], a, a) for g in (getattr(sub, "fp32_from_fp32_groups", None) or []) for a in g
                if id(a) in names]
        if not hasattr(sub, "float16_groups"):           # 全程 FP32（FP32Optimizer）：参数就是主权重
            inner = getattr(sub, "optimizer", sub)
            out += [(names[id(a)], a, a) for g in inner.param_groups for a in g["params"] if id(a) in names]
    return out


class ParamSwap:
    def __init__(self, import_dir: str | None = None, export_steps=(), export_dir: str | None = None,
                 round_bf16: bool = False):
        self.import_dir = import_dir
        self.round_bf16 = round_bf16          # 导入时先把 FP32 主权重舍到 BF16（FP32 参照用「BF16 程序实际用的那份权重」）
        self.export_steps = set(export_steps)
        self.export_dir = export_dir
        self.report: dict = {}

    def _setup(self, context):
        model = _unwrap(context.model[0])
        params = [(re.sub(r"^(module\.)+", "", n), p) for n, p in model.named_parameters()]
        buffers = [("buffer." + re.sub(r"^(module\.)+", "", n), b) for n, b in model.named_buffers()
                   if b is not None and b.is_floating_point()]
        pairs = _pairs(context.optimizer, params)
        if len(pairs) != len(params):
            missing = sorted(set(n for n, _ in params) - set(n for n, _, _ in pairs))
            raise RuntimeError(f"[param_swap] 有参数找不到 FP32 主权重：{missing[:5]}…（共 {len(missing)} 个）")
        return pairs, buffers

    def _mine(self, name: str) -> bool:
        return _rank() == 0 or bool(_EXPERT.search(name))

    # ---- 回调
    def on_train_start(self, context):
        if not self.import_dir:
            return
        from safetensors import safe_open
        pairs, buffers = self._setup(context)
        r = _rank()
        man = json.load(open(os.path.join(self.import_dir, "manifest.json")))
        mine = man["ranks"][str(r)]
        f = safe_open(os.path.join(self.import_dir, f"rank{r}.safetensors"), "pt")
        keys = set(f.keys())
        want = {n for n, _, _ in pairs if self._mine(n)} | {n for n, _ in buffers if r == 0}
        if r != 0:  # 非专家参数在每个 rank 上都有副本，只有 rank 0 的文件存了：从 rank 0 文件读
            f0 = safe_open(os.path.join(self.import_dir, "rank0.safetensors"), "pt")
        else:
            f0 = f
        need_all = {n for n, _, _ in pairs} | {n for n, _ in buffers}
        have = keys | set(f0.keys())
        if need_all - have:
            raise RuntimeError(f"[param_swap] rank {r} 导入文件缺 {sorted(need_all - have)[:5]}")
        bad, n = [], 0
        with torch.no_grad():
            for name, p, mp in pairs:
                src = f if name in keys else f0
                t = src.get_tensor(name)
                if tuple(t.shape) != tuple(mp.shape):
                    raise RuntimeError(f"[param_swap] {name} 形状 {tuple(t.shape)} ≠ {tuple(mp.shape)}")
                t = t.to(device=mp.device, dtype=torch.float32)
                ref = (man["ranks"][str(r)] if name in keys else man["ranks"]["0"])[name]["sha256"]
                if _sha_f32(t) != ref:
                    bad.append(name)
                if self.round_bf16:
                    t = t.to(torch.bfloat16).to(torch.float32)
                mp.copy_(t)
                if p is not mp:
                    p.copy_(t.to(p.dtype))
                if not torch.equal(mp, t):
                    bad.append(name + "（读回）")
                n += 1
            for name, b in buffers:
                src = f if name in keys else f0
                t = src.get_tensor(name).to(device=b.device, dtype=b.dtype)
                b.copy_(t)
                ref = (man["ranks"][str(r)] if name in keys else man["ranks"]["0"])[name]["sha256"]
                if b.dtype == torch.float32 and _sha_f32(b) != ref:
                    bad.append(name)
                n += 1
        if bad:
            raise RuntimeError(f"[param_swap] 导入后读回与 manifest 不一致：{bad[:5]}（共 {len(bad)} 个）")
        self.report = {"import_dir": self.import_dir, "tensors": n, "bitwise_vs_manifest": True,
                       "round_bf16": self.round_bf16, "source": man.get("source")}
        del mine, want
        if r == 0:
            print(f"[param_swap] 导入 {n} 个张量（rank 0），文件逐位对上 manifest，放回后读回一致"
                  f"{'（主权重先舍到 BF16）' if self.round_bf16 else ''}；来源 {man.get('source')}", flush=True)

    def on_train_step_start(self, context):
        pass

    def on_train_step_end(self, context):
        step = context.state.train_state.step + 1        # 刚执行完的迭代号（与 JanusDump 同一个口径）
        if step not in self.export_steps:
            return
        from safetensors.torch import save_file
        pairs, buffers = self._setup(context)
        r = _rank()
        d = os.path.join(self.export_dir, f"step_{step}")
        os.makedirs(d, exist_ok=True)
        out = {}
        for name, _p, mp in pairs:
            if self._mine(name):
                out[name] = mp.detach().float().contiguous().cpu()
        if r == 0:
            for name, b in buffers:
                out[name] = b.detach().float().contiguous().cpu()
        save_file(out, os.path.join(d, f"rank{r}.safetensors"))
        rec = {k: {"sha256": hashlib.sha256(v.numpy().reshape(-1).view(np.uint8)).hexdigest(), "shape": list(v.shape)}
               for k, v in out.items()}
        allrec = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
        if dist.is_initialized():
            dist.all_gather_object(allrec, rec)
        else:
            allrec = [rec]
        if r == 0:
            man = {"format": "megatron-per-rank-fp32", "step": step,
                   "state": f"θ_{step}（第 {step} 步更新之后的 FP32 主权重）",
                   "source": {"framework": "megatron", "export_dir": self.export_dir},
                   "ranks": {str(i): x for i, x in enumerate(allrec)}}
            json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=0)
            print(f"[param_swap] 第 {step} 步导出 → {d}", flush=True)
        if dist.is_initialized():
            dist.barrier()

    def on_train_end(self, context):
        pass


def wrap_skip(forward_step, skip_micro: int):
    """开训时先从数据迭代器里丢掉 skip_micro 个 microbatch（每个 rank 各自的迭代器）。"""
    left = {"n": skip_micro}

    def fs(state, data_iterator, model, return_schedule_plan=False):
        if left["n"] and data_iterator is not None:
            for _ in range(left["n"]):
                next(data_iterator)
            if _rank() == 0:
                print(f"[param_swap] 数据跳过 {left['n']} 个 microbatch", flush=True)
            left["n"] = 0
        return forward_step(state, data_iterator, model, return_schedule_plan)
    return fs


class Bf16WeightsFp32Compute:
    """（只作诊断参照，配 --fp32）「BF16 权重、全程 FP32 计算」：每步前向前把参数舍到 BF16（RNE），反向用这份权重，
    优化器更新前换回 FP32 主权重再更新。即生产 BF16 程序的权重语义（主权重 FP32、计算用它的 BF16 舍入），但前向反向的
    算术全部是 FP32——每一步的梯度就是逐步拆分里的「精确参照」，整条轨迹是 BF16 训练的精确版本。"""

    def __init__(self):
        self.params = []
        self.master = []
        self.steps = 0

    def on_train_start(self, context):
        seen = set()
        for chunk in context.model:
            for _n, p in _unwrap(chunk).named_parameters():
                if id(p) not in seen and p.dtype == torch.float32:
                    seen.add(id(p))
                    self.params.append(p)
        self.master = [torch.empty_like(p) for p in self.params]
        opt = context.optimizer
        orig = opt.step

        def step(*args, **kwargs):
            with torch.no_grad():
                for p, m in zip(self.params, self.master):
                    p.copy_(m)
            return orig(*args, **kwargs)

        opt.step = step
        if _rank() == 0:
            print(f"[bf16w] {len(self.params)} 个 FP32 参数：每步前向前舍 BF16，优化器更新前换回主权重", flush=True)

    def on_train_step_start(self, context):
        with torch.no_grad():
            for p, m in zip(self.params, self.master):
                m.copy_(p)
                p.copy_(p.to(torch.bfloat16).to(torch.float32))
        self.steps += 1

    def on_train_step_end(self, context):
        pass

    def on_train_end(self, context):
        pass
