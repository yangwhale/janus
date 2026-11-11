"""GPU 侧抓数：Megatron-Bridge 训练回调，不改 Megatron 源码。

用法（训练脚本里）：
    from janus.dump.megatron import JanusDump, wrap_forward_step
    cb = JanusDump(out_dir, meta, dump_steps={1}, full=True)
    pretrain(cfg, wrap_forward_step(forward_step, cb), callbacks=[cb])

每步（rank 0 写 metrics.jsonl）：loss 各项、lr、grad_norm、param_norm、num_zeros、每层路由偏置均值 / 标准差、
每层专家负载（rank 0 本地）的变异系数与最大 / 最小计数。
每步（每个 rank 写 data/rank<r>.jsonl）：每个 microbatch 的 tokens / labels / loss_mask / position_ids 的 sha256 与非 padding 数。
训练前后（每个 rank 写 params_sha/rank<r>.<init|final>.json）：本 rank 持有的每个参数的 sha256（全量，不抽样）。
dump_steps 里的步（rank 0、第一个 microbatch）：
- 前向：逐层各子模块输出（规范名见 names.py）、路由 logits / 选中专家 / 概率 / 专家计数、逐 token loss、输入 token；
- full=True 时另抓：各子模块输出的梯度（反向）、更新前的 bf16 参数与 fp32 主权重、更新后的 fp32 主权重与 Adam 两个动量；
- dump_grads=True 时抓每个参数的 main_grad（更新前，已归约）。
full / dump_grads 要求不开分布式优化器（每个 rank 有完整的主权重与梯度）。专家参数只有 rank 0 持有的那部分。
inject_from：逐层注入。dump_steps 的那一步，第 i 层（i ≥ 1）的输入换成参照运行同一步的 layers.{i-1}.out，
第 0 层输入换成 embed.out；只换 rank 0 第一个 microbatch。用来把误差隔离到单层。
observe：可选观测点（逗号分隔，默认不开，不开时与旧版逐位相同；只在 dump 步、rank 0、第一个 microbatch 抓）：
- attn：核心注意力的输入 q / k / v（RoPE 之后，[s, b, 头, 每头维度]）；full 时另抓三者的梯度；
- moe：MoE 层内各阶段——分发前置换（permute）后的 token 与概率、置换元数据（routing_map、逆置换下标、
  各 rank 收发数、每个本地专家从各 rank 收到的 token 数）、专家输入（all-to-all 之后按本地专家排好）、每专家 token 数、
  专家侧概率、GEMM1 输出、激活输出（GEMM2 输入）、combine 之后（回到本 rank、仍是置换顺序）、unpermute 之后（加共享专家之前）；
  full 时另抓专家输入、GEMM1 输出、激活输出的梯度；
- expw：rank 0 本地全部专家的权重（全量抓数默认只留每层前两个专家）；
- gemm：第一个 dump 步里每次 GEMM 调用的输入 / 输出 dtype 与调用处（TE general_gemm / general_grouped_gemm 与 torch 的
  mm / addmm / bmm / baddbmm），写 gemm_inventory.json。用来回答「哪些 GEMM 的输入是 FP32、会受 TF32 影响」。
special：存档前往指定参数里埋特殊值（−0.0、次正规数），续训时读回逐位核对（_special_poke / _special_dump）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from janus.dump.names import canonical
from janus.store import Run, RunWriter


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
        return x[0]
    return None


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach()
    if t.dtype == torch.bfloat16:
        import ml_dtypes
        return t.contiguous().view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16)
    return t.contiguous().cpu().numpy()


def _sha(t: torch.Tensor) -> str:
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()


def tf32_probe() -> dict:
    """实测 FP32 输入的 GEMM 有没有走 TF32：torch.mm 与 TE general_gemm 各算一次 256×4096 · 4096×512，对 FP64 的相对 L2。
    TF32（10 位尾数）约 3e-4，IEEE FP32 约 1e-6（models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-tf32）。TE 的 FP32 GEMM 不看 torch 的开关，所以两者分开测。"""
    out = {}
    try:
        dev = torch.device("cuda", torch.cuda.current_device())
        g = torch.Generator(device=dev).manual_seed(0)
        x = torch.randn(256, 4096, device=dev, generator=g)
        w = torch.randn(512, 4096, device=dev, generator=g)
        ref = x.double() @ w.double().t()

        def rel(y):
            return float((y.double() - ref).norm() / ref.norm())
        out["probe_rel_l2_torch_mm_fp32"] = rel(x @ w.t())
        from transformer_engine.pytorch.cpp_extensions import general_gemm
        out["probe_rel_l2_te_general_gemm_fp32"] = rel(general_gemm(w, x, torch.float32, layout="TN")[0])
    except Exception as e:  # noqa: BLE001  探针失败不影响训练，原样记下
        out["probe_error"] = repr(e)[:200]
    return out


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


class _RecordingIterator:
    """包住数据迭代器：记下每个 microbatch 的指纹；rank 0 在抓数步存第一个 microbatch 的原始输入。"""

    def __init__(self, it, dump):
        self.it, self.dump = it, dump

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.it)
        self.dump._record_batch(batch)
        return batch


def wrap_forward_step(forward_step, dump: JanusDump):
    def janus_forward_step(state, data_iterator, model, return_schedule_plan=False):
        it = _RecordingIterator(data_iterator, dump) if data_iterator is not None else None
        return forward_step(state, it, model, return_schedule_plan)
    return janus_forward_step


FULL_KINDS = ("grad", "param", "main_param", "main_param_after", "exp_avg", "exp_avg_sq", "momentum_buffer")
_EXPERT_RE = re.compile(r"^(.*\.experts\.linear_fc[12]\.weight)(\d+)$")


def global_expert_name(name: str, ep_rank: int, n_local: int) -> str:
    """本地专家参数名 → 全局专家号的名字：`...linear_fc1.weight<j>` → `...linear_fc1.weight<ep_rank·n_local + j>`。
    rank 0（专家并行 rank 0）上两者相同，所以收齐后的名字与原来 rank 0 抓数的名字一致。"""
    m = _EXPERT_RE.match(name)
    if not m:
        raise ValueError(f"不是路由专家参数：{name}")
    return f"{m.group(1)}{ep_rank * n_local + int(m.group(2))}"


class JanusDump:
    """Megatron-Bridge 回调（on_train_start / on_train_step_start / on_train_step_end / on_train_end）。"""

    # 全量抓数默认不抓的大张量：每层 rank 0 本地专家只留前两个（weight0 / weight1），其余专家权重太大（rank 0 约 13 亿参数）
    DEFAULT_SKIP = r"\.experts\.linear_fc[12]\.weight([2-9]|\d\d+)$"

    def __init__(self, out_dir: str, meta: dict, dump_steps=(1,), dump_grads: bool = False, full: bool = False,
                 inject_from: str | None = None, fault: dict | None = None, full_steps=None, skip: str | None = DEFAULT_SKIP,
                 only: str | None = None, recompute: bool = False, save_batches: bool = False,
                 observe: str | None = None, special: dict | None = None, group_norms: bool = False,
                 row_stats_every: int = 0, gather_experts: bool = False, full_kinds: str | None = None,
                 forward_acts: bool = True):
        """full_steps：做全量抓数（梯度、参数、优化器状态）的步，默认 dump_steps 里最小的那一步；其余 dump 步只抓前向。
        skip：全量抓数里跳过的参数名正则（清单 params_sha 仍然全量）。"""
        self.out_dir, self.meta, self.dump_steps = out_dir, meta, set(dump_steps)
        self.full_steps = set(full_steps) if full_steps else ({min(self.dump_steps)} if self.dump_steps else set())
        self.skip = re.compile(skip) if skip else None
        self.only = re.compile(only) if only else None
        self.recompute = recompute                   # 开激活重算时，同一步第二次前向另存为 <名>.recompute
        # 每 rank 每步每个 microbatch 的原始输入整批存下（data/batches/rank<r>.bin + .jsonl 索引），给另一侧原样回放同一批数据
        self.save_batches = save_batches
        self._bfile = self._bindex = None
        self.dump_grads, self.full = dump_grads or full, full
        self.inject_from, self.fault = inject_from, fault or {}
        self._orig_step = None
        self._params = []
        self._routers = {}
        self.writer = None
        self.capturing = False
        self.buf: dict[str, np.ndarray] = {}
        self.handles = []
        self.step_vals: dict = {}
        self.counts: dict[int, torch.Tensor] = {}
        self.batches: list[dict] = []
        self._ref = None
        self.observe = set(filter(None, (observe or "").split(",")))
        unknown = self.observe - {"attn", "moe", "expw", "gemm"}
        if unknown:
            raise ValueError(f"未知观测点 {sorted(unknown)}")
        self.obs_info: dict = {}
        self._gemm_log = None
        self.special = special or {}
        self.group_norms = group_norms            # 每步另记分组 FP32 主权重平方和（group_sq.*）与每个专家的平方和（expert_sq.*）
        self.row_stats_every = row_stats_every    # 每 N 步记输出层 / embedding 每行平方和与输出层每行 Adam 更新量（与 TPU 侧同名）
        # 全量抓数时把 rank 0 所在专家并行组里其他 rank 的本地专家也收到 rank 0（点对点），按全局专家号命名；默认关
        self.forward_acts = forward_acts          # False：抓数步不抓前向激活（只要参数 / 梯度 / 优化器状态时省体积）
        self.gather_experts = gather_experts
        if gather_experts:
            self.skip = None                          # 收齐全部专家时不再只留 weight0 / weight1
        # 全量抓数只写这些种类（逗号分隔：grad,param,main_param,main_param_after,exp_avg,exp_avg_sq,momentum_buffer）；空 = 全写
        self.full_kinds = set(filter(None, (full_kinds or "").split(","))) or None
        bad = (self.full_kinds or set()) - set(FULL_KINDS)
        if bad:
            raise ValueError(f"未知的全量抓数种类 {sorted(bad)}")

    # ---- 回调接口
    def on_train_start(self, context):
        self.rank = _rank()
        self.rank0 = self.rank == 0
        self.context = context
        if self.rank0:
            if isinstance(self.meta.get("tf32"), dict):
                self.meta["tf32"].update(tf32_probe())
            self.writer = RunWriter(self.out_dir, self.meta)
        if dist.is_initialized():
            dist.barrier()                                   # rank 0 先建目录
        os.makedirs(os.path.join(self.out_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "params_sha"), exist_ok=True)
        model = context.model[0]
        model = getattr(model, "module", model)          # 去掉 DDP 外壳
        while hasattr(model, "module") and not hasattr(model, "decoder"):
            model = model.module
        self.model = model
        self._params = [(re.sub(r"^(module\.)+", "", n), p) for n, p in model.named_parameters()]
        # buffer（如路由 expert_bias）不在 named_parameters() 里，单独列出，清单与抓数都带上
        self._buffers = [(re.sub(r"^(module\.)+", "", n), b) for n, b in model.named_buffers()
                         if b is not None and b.is_floating_point()]
        self._opt = context.optimizer
        self._apply_fault_weights()
        self._write_param_sha("init")
        if self.rank0:
            with open(os.path.join(self.out_dir, "modules.txt"), "w") as f:
                for name, mod in model.named_modules():
                    f.write(f"{name}\t{type(mod).__name__}\n")
                for name, p in self._params:
                    f.write(f"param\t{name}\t{tuple(p.shape)}\t{p.dtype}\n")
        if context.optimizer is not None and self._orig_step is None:
            self._wrap_optimizer(context.optimizer)
        for name, mod in model.named_modules():
            m = re.match(r"^decoder\.layers\.(\d+)\.mlp\.router$", name)
            if m:
                self._routers[int(m.group(1))] = mod
                self._wrap_router(mod, int(m.group(1)))
            if not self.rank0:
                continue
            cname = canonical(name)
            if cname:
                self.handles.append(mod.register_forward_hook(self._hook(cname)))
        for name, mod in model.named_modules():
            if name == "embedding":
                self.handles.append(mod.register_forward_pre_hook(self._embed_pre))
        self.handles.append(model.register_forward_hook(self._loss_hook))
        self._install_observe(model)
        if self.special.get("phase") == "load":
            self._special_dump("loaded")
        if self.rank0:
            if self.inject_from:
                self._ref = Run.open(self.inject_from)
                for name, mod in model.named_modules():
                    m = re.match(r"^decoder\.layers\.(\d+)$", name)
                    if m:
                        self.handles.append(mod.register_forward_pre_hook(self._inject(int(m.group(1))), with_kwargs=True))

    def on_train_step_start(self, context):
        step = context.state.train_state.step + 1            # 即将执行的迭代号（从 1 开始）
        self.step = step
        self.capturing = self.rank0 and step in self.dump_steps
        self.buf, self.step_vals, self.counts, self.batches = {}, {}, {}, []
        self._injected = set()
        import time
        self._t0 = time.perf_counter()
        f = self.fault
        if f.get("kind") == "lr_spike" and context.optimizer is not None:
            for g in context.optimizer.param_groups:
                if step == f["step"]:
                    g["_janus_lr"] = g["lr"]
                    g["lr"] = g["lr"] * f["factor"]
                elif "_janus_lr" in g:
                    g["lr"] = g.pop("_janus_lr")
        if self.capturing:
            for name, b in self._buffers:
                self.buf[f"buffer.{name}"] = _to_numpy(b)
            if "expw" in self.observe:
                for name, p in self._params:
                    if re.search(r"\.experts\.linear_fc[12]\.weight\d+$", name) and self.skip and self.skip.search(name):
                        self.buf[f"param.{name}"] = _to_numpy(p)
            if "gemm" in self.observe and self._gemm_log is None:
                self._gemm_log = _GemmInventory()
                self._gemm_log.start()

    def on_train_step_end(self, context):
        step = context.state.train_state.step + 1            # 刚执行完的迭代号（step 在回调之后才自增）
        self._write_batches(step)
        if self.special.get("phase") == "save" and step == int(self.special["step"]):
            self._special_poke()                             # 框架在本回调之后才存档（checkpoint_and_decide_exit）
            self._special_dump("saved")
        pn = self._params_norm(context)                      # 集合通信：所有 rank 都要调
        if self.group_norms:
            self.step_vals.update(self._group_sq(context))     # 集合通信：所有 rank 都要调
        if self.row_stats_every and step % self.row_stats_every == 0 and self.rank0:
            self.step_vals.update(self._row_stats(context, step))
        if not self.rank0:
            return
        vals = {}
        for k, v in (context.loss_dict or {}).items():
            key = {"lm loss": "lm_loss"}.get(k, k.replace(" ", "_"))
            vals[key] = float(v.item() if isinstance(v, torch.Tensor) else v)
        if context.grad_norm is not None:
            vals["grad_norm"] = float(context.grad_norm)
        if context.optimizer is not None:
            groups = context.optimizer.param_groups
            vals["lr"] = float(groups[0]["lr"])
            # 实际生效的权重衰减（调度器会覆盖 optimizer 配置里的值），取各参数组的最大值（不衰减的组是 0）
            vals["wd"] = float(max(g.get("weight_decay", 0.0) for g in groups))
        if pn is not None:
            vals["param_norm"] = pn
        vals["skipped_iter"] = int(bool(context.skipped_iter))
        import time
        vals["step_time"] = time.perf_counter() - self._t0
        vals.update(self.step_vals)
        if self.capturing:
            for name, b in self._buffers:
                self.buf[f"buffer_after.{name}"] = _to_numpy(b)
        for layer, router in sorted(self._routers.items()):
            b = getattr(router, "expert_bias", None)
            if b is not None:
                bf = b.detach().float()
                vals[f"router.bias_mean.{layer}"] = float(bf.mean())
                vals[f"router.bias_std.{layer}"] = float(bf.std())
            c = self.counts.get(layer)
            if c is not None:
                cf = c.float()
                vals[f"router.load_cv.{layer}"] = float(cf.std() / cf.mean()) if cf.mean() > 0 else 0.0
                vals[f"router.load_max.{layer}"] = float(cf.max())
                vals[f"router.load_min.{layer}"] = float(cf.min())
                if self.capturing:
                    self.buf[f"layers.{layer}.mlp.router.counts"] = c.cpu().numpy().astype(np.int64)
        self.writer.log_metrics(step, **vals)
        if self._gemm_log is not None and self._gemm_log.active:
            self._gemm_log.stop()
            with open(os.path.join(self.out_dir, "gemm_inventory.json"), "w") as f:
                json.dump({"step": step, "calls": self._gemm_log.summary()}, f, indent=1, ensure_ascii=False)
        if self.capturing and self.obs_info:
            with open(os.path.join(self.out_dir, f"observe_step{step}.json"), "w") as f:
                json.dump(self.obs_info, f, indent=1, ensure_ascii=False, default=str)
        if self.capturing and self.buf:
            self._flush(step)
        self.capturing = False
        self.buf = {}

    def on_train_end(self, context):
        for h in self.handles:
            h.remove()
        self._write_param_sha("final")
        if dist.is_initialized():
            dist.barrier()
        if self.rank0 and self.writer is not None:
            self.writer.done()

    # ---- 写出
    def _flush(self, step):
        by_scope: dict[str, dict] = {}
        for name, arr in self.buf.items():
            head = name.split(".")[0]
            if head in ("grad", "param", "main_param", "main_param_after", "exp_avg", "exp_avg_sq", "momentum_buffer",
                        "buffer", "buffer_after"):
                m = re.search(r"\.layers\.(\d+)\.", name)
                scope = f"{head}.layers.{m.group(1)}" if m else f"{head}.other"
            elif name.startswith("layers."):
                scope = ".".join(name.split(".")[:2]) + (".bwd" if name.endswith(".grad") else "")
            else:
                scope = head
            by_scope.setdefault(scope, {})[name] = arr
        for scope, tensors in sorted(by_scope.items()):
            head = scope.split(".")[0]
            kind, point = {"grad": ("grad", "backward"), "param": ("param", "before_update"),
                           "main_param": ("param", "before_update"), "main_param_after": ("param", "after_update"),
                           "buffer": ("param", "before_update"), "buffer_after": ("param", "after_update"),
                           "exp_avg": ("opt_state", "after_update"), "exp_avg_sq": ("opt_state", "after_update"),
                           "momentum_buffer": ("opt_state", "after_update"),
                           "input": ("tokens", "forward"), "loss": ("other", "forward")}.get(head, ("activation", "forward"))
            if scope.endswith(".bwd"):
                kind, point = "grad", "backward"
            self.writer.write_tensors(step, scope, tensors, kind=kind, point=point)

    def _write_batches(self, step):
        if not self.batches:
            return
        with open(os.path.join(self.out_dir, "data", f"rank{self.rank}.jsonl"), "a") as f:
            for i, b in enumerate(self.batches):
                f.write(json.dumps({"step": step, "micro": i, **b}) + "\n")

    def _write_param_sha(self, tag):
        out = {name: {"sha256": _sha(p), "shape": list(p.shape), "dtype": str(p.dtype).replace("torch.", "")}
               for name, p in self._params}
        out.update({f"buffer.{name}": {"sha256": _sha(b), "shape": list(b.shape), "dtype": str(b.dtype).replace("torch.", "")}
                    for name, b in self._buffers})
        with open(os.path.join(self.out_dir, "params_sha", f"rank{self.rank}.{tag}.json"), "w") as f:
            json.dump(out, f, indent=0, sort_keys=True)

    # 分组：和 models/hunyuan3/frameworks/maxtext/train_hy3.py 的 GROUP_RULES 同名，两边逐组对得上
    _GROUPS = [(r"^embedding\.word_embeddings\.weight$", "embedding"), (r"^output_layer\.weight$", "output"),
               (r"^decoder\.final_layernorm\.weight$", "final_norm"),
               (r"self_attention\.linear_qkv\.layer_norm_weight$", "input_norm"),
               (r"self_attention\.linear_qkv\.weight$", "attn_qkv"), (r"self_attention\.linear_proj\.weight$", "attn_proj"),
               (r"self_attention\.[qk]_layernorm\.weight$", "qk_norm"),
               (r"(pre_mlp_layernorm\.weight|mlp\.linear_fc1\.layer_norm_weight)$", "mlp_norm"),
               (r"shared_experts\.linear_fc1\.weight$", "shared_fc1"), (r"shared_experts\.linear_fc2\.weight$", "shared_fc2"),
               (r"mlp\.router\.weight$", "router"),
               (r"experts\.linear_fc1\.weight\d+$", "experts_fc1"), (r"experts\.linear_fc2\.weight\d+$", "experts_fc2"),
               (r"mlp\.linear_fc1\.weight$", "dense_fc1"), (r"mlp\.linear_fc2\.weight$", "dense_fc2")]

    def _group_sq(self, context) -> dict:
        """FP32 主权重按组的平方和（FP64 累加），全局（专家跨 EP 求和、其余只算一份）。"""
        from megatron.core import parallel_state as ps
        opt = context.optimizer
        names = {id(p): n for n, p in self._params}
        pairs = []
        for sub in getattr(opt, "chained_optimizers", None) or [opt]:
            f16, f32 = getattr(sub, "float16_groups", None), getattr(sub, "fp32_from_float16_groups", None)
            if f16 is not None and f32 is not None:
                pairs += [(names.get(id(a)), b) for ga, gb in zip(f16, f32) for a, b in zip(ga, gb) if id(a) in names]
            pairs += [(names[id(a)], a) for g in (getattr(sub, "fp32_from_fp32_groups", None) or []) for a in g if id(a) in names]
        ep_rank = ps.get_expert_model_parallel_rank()
        first_rank = self.rank == 0
        exp = {}
        acc: dict[str, torch.Tensor] = {}
        for name, mp in pairs:
            grp = next((g for pat, g in self._GROUPS if re.search(pat, name)), None)
            if grp is None:
                continue
            m = re.search(r"decoder\.layers\.(\d+)\.", name)
            key = f"group_sq.{grp}" + (f".{m.group(1)}" if m else "")
            sq = mp.detach().double().square().sum()
            if grp.startswith("experts_"):
                j = int(re.search(r"weight(\d+)$", name).group(1))
                n_local = len([1 for n2, _ in pairs if re.sub(r"weight\d+$", "", n2) == re.sub(r"weight\d+$", "", name)])
                e = ep_rank * n_local + j
                ek = f"expert_sq.{m.group(1)}"
                exp.setdefault(ek, torch.zeros(n_local * ps.get_expert_model_parallel_world_size(), dtype=torch.float64,
                                               device=mp.device))[e] += sq
            elif not first_rank:
                sq = sq * 0                                  # 非专家参数各 rank 都有一份，只算 rank 0 的
            acc[key] = acc.get(key, 0) + sq
        ks = sorted(acc)
        flat = torch.stack([acc[k] for k in ks]) if ks else torch.zeros(0, dtype=torch.float64, device="cuda")
        eks = sorted(exp)
        if dist.is_initialized():
            dist.all_reduce(flat)
            for k in eks:
                dist.all_reduce(exp[k])
        out = {k: float(v) for k, v in zip(ks, flat.tolist())}
        out.update({k: [float(x) for x in exp[k].tolist()] for k in eks})
        return out

    def _row_stats(self, context, step) -> dict:
        """输出层 [V, D] 与 embedding [V, D] 每行的 FP32 主权重平方和；输出层每行 |m̂/(√v̂+ε)| 的均值。
        TP = 1 时每个 rank 都有完整的一份，只在 rank 0 算。"""
        opt = context.optimizer
        names = {id(p): n for n, p in self._params}
        out = {}
        for sub in getattr(opt, "chained_optimizers", None) or [opt]:
            f16, f32 = getattr(sub, "float16_groups", None), getattr(sub, "fp32_from_float16_groups", None)
            inner = getattr(sub, "optimizer", None)
            if f16 is None or f32 is None:
                continue
            for ga, gb in zip(f16, f32):
                for a, mp in zip(ga, gb):
                    n = names.get(id(a))
                    if n == "output_layer.weight":
                        out["output_row_sq"] = mp.detach().double().square().sum(1)
                        st = inner.state.get(mp, {}) if inner is not None else {}
                        g = (inner.param_groups[0] if inner is not None else {})
                        b1, b2 = g.get("betas", (0.9, 0.95))
                        eps = g.get("eps", 1e-8)
                        if "exp_avg" in st:
                            m = st["exp_avg"].double() / (1 - b1 ** step)
                            v = st["exp_avg_sq"].double() / (1 - b2 ** step)
                            u = m / (v.sqrt() + eps)
                            out["output_row_u"] = u.abs().mean(1)
                            out["output_row_g"] = v.sqrt().mean(1)
                            out["output_row_wu"] = (mp.detach().double() * u).sum(1)
                    elif n == "embedding.word_embeddings.weight":
                        out["embedding_row_sq"] = mp.detach().double().square().sum(1)
        return {k: [float(x) for x in v.tolist()] for k, v in out.items()}

    def _params_norm(self, context):
        try:
            from megatron.bridge.training.utils.train_utils import calc_params_l2_norm
            from megatron.core.utils import get_model_config
            return float(calc_params_l2_norm(context.model, get_model_config(context.model[0])))
        except Exception as e:  # noqa: BLE001  统计项拿不到不影响训练，记一次
            if not getattr(self, "_pn_warned", False):
                print(f"[janus] param_norm 不可用：{e}")
                self._pn_warned = True
            return None

    # ---- 故障注入（只用于验证检查会报警）
    def _apply_fault_weights(self):
        f = self.fault
        if f.get("kind") != "weight_scale":
            return
        # 先改 FP32 主权重再拷回 BF16 参数：直接在 BF16 上乘 (1+eps)，eps 小于 BF16 半格（约 2e-3）时会被舍回原值，等于没埋
        pat = re.compile(f["param"])
        hit = 0
        with torch.no_grad():
            pairs = {name: mp for name, _, mp, _ in self._pairs()}
            for name, p in self._params:
                if pat.search(name):
                    mp = pairs.get(name)
                    if mp is not None:
                        mp.mul_(1 + float(f["eps"]))
                        p.copy_(mp.to(p.dtype))
                    else:
                        p.copy_((p.float() * (1 + float(f["eps"]))).to(p.dtype))
                    hit += 1
        if self.rank0:
            print(f"[janus] weight_scale 命中 {hit} 个参数")

    # ---- 钩子
    def _record_batch(self, batch):
        rec = {}
        for k in ("tokens", "labels", "loss_mask", "position_ids"):
            t = batch.get(k) if isinstance(batch, dict) else None
            if isinstance(t, torch.Tensor):
                if self.fault.get("kind") == "data_shift" and k == "tokens" and self.rank == 0 \
                        and self.step >= int(self.fault.get("from_step", 1)):
                    t.copy_(torch.roll(t, 1, dims=-1))
                rec[k] = _sha(t)
                if self.capturing and not self.batches:
                    self.buf[f"input.{k}"] = _to_numpy(t)
        if self.save_batches and isinstance(batch, dict):
            self._save_batch(batch)
        lm = batch.get("loss_mask") if isinstance(batch, dict) else None
        if isinstance(lm, torch.Tensor):
            rec["nonpad"] = int(lm.sum().item())
        self.batches.append(rec)

    def _save_batch(self, batch):
        if self._bfile is None:
            d = os.path.join(self.out_dir, "data", "batches")
            os.makedirs(d, exist_ok=True)
            self._bfile = open(os.path.join(d, f"rank{self.rank}.bin"), "ab")
            self._bindex = open(os.path.join(d, f"rank{self.rank}.jsonl"), "a")
        rec = {"step": self.step, "micro": len(self.batches), "fields": {}}
        for k in ("tokens", "labels", "loss_mask", "position_ids"):
            t = batch.get(k)
            if isinstance(t, torch.Tensor):
                a = t.detach().cpu().numpy()
                off = self._bfile.tell()
                self._bfile.write(np.ascontiguousarray(a).tobytes())
                rec["fields"][k] = {"offset": off, "dtype": str(a.dtype), "shape": list(a.shape),
                                    "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
        self._bindex.write(json.dumps(rec) + "\n")
        self._bindex.flush()
        self._bfile.flush()

    def _wrap_optimizer(self, opt):
        orig, dump = opt.step, self

        pairs = self._pairs

        def keep(name):
            if name is None or (dump.skip and dump.skip.search(name)):
                return False
            return not dump.only or bool(dump.only.search(name))

        def step_capture(*args, **kwargs):
            cap = dump.capturing and dump.step in dump.full_steps
            f = dump.fault
            if f.get("kind") == "grad_scale_seen" or (f.get("kind") == "grad_nan" and dump.step == f["step"]):
                pat = re.compile(f["param"])
                for name, p in dump._params:
                    g = getattr(p, "main_grad", None)
                    if g is not None and pat.search(name):
                        if f["kind"] == "grad_scale_seen":
                            g.mul_(float(f["factor"]))          # 在抓梯度之前缩放：抓到的就是被缩放后的梯度
                        else:
                            g.view(-1)[0] = float("nan")
            want = dump._want_kind
            gather = dump.gather_experts and dump.full and dump.step in dump.full_steps
            if cap and dump.dump_grads and want("grad"):
                for name, p in dump._params:
                    g = getattr(p, "main_grad", None)
                    if g is None:
                        g = p.grad
                    if g is not None and keep(name):
                        dump.buf[f"grad.{name}"] = _to_numpy(g)
            if gather and dump.dump_grads and want("grad"):
                dump._gather_experts("grad", lambda p, mp, inner: getattr(p, "main_grad", None) if getattr(p, "main_grad", None) is not None else p.grad)
            if dump.fault.get("kind") == "grad_scale":
                pat = re.compile(dump.fault["param"])
                for name, p in dump._params:
                    g = getattr(p, "main_grad", None)
                    if g is not None and pat.search(name):
                        g.mul_(float(dump.fault["factor"]))
            if cap and dump.full:
                for name, p, mp, _ in pairs():
                    if not keep(name):
                        continue
                    if want("param"):
                        dump.buf[f"param.{name}"] = _to_numpy(p)
                    if want("main_param"):
                        dump.buf[f"main_param.{name}"] = _to_numpy(mp)
            if gather:
                for kind, get in (("param", lambda p, mp, inner: p), ("main_param", lambda p, mp, inner: mp)):
                    if want(kind):
                        dump._gather_experts(kind, get)
            out = orig(*args, **kwargs)
            try:
                _, _, nz = out
                if nz is not None:
                    dump.step_vals["num_zeros"] = float(nz)
            except (TypeError, ValueError):
                pass
            if cap and dump.full:
                for name, _p, mp, inner in pairs():
                    if not keep(name):
                        continue
                    if want("main_param_after"):
                        dump.buf[f"main_param_after.{name}"] = _to_numpy(mp)
                    st = inner.state.get(mp, {}) if inner is not None else {}
                    # Adam 的两个动量；Muon（emerging_optimizers 的正交化优化器）的动量缓冲 momentum_buffer
                    for k in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                        if isinstance(st.get(k), torch.Tensor) and want(k):
                            dump.buf[f"{k}.{name}"] = _to_numpy(st[k])
            if gather:
                if want("main_param_after"):
                    dump._gather_experts("main_param_after", lambda p, mp, inner: mp)
                for k in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                    if want(k):
                        dump._gather_experts(k, lambda p, mp, inner, k=k: (inner.state.get(mp, {}) if inner is not None else {}).get(k))
            return out
        opt.step = step_capture
        self._orig_step = orig

    def _want_kind(self, kind: str) -> bool:
        return self.full_kinds is None or kind in self.full_kinds

    def _gather_experts(self, kind: str, get):
        """把 rank 0 所在专家并行组里 rank 1… 的本地路由专家张量点对点收到 rank 0，按全局专家号写进 buf。
        组里所有 rank 都要调（集合语义）；不在这个组里的 rank 直接返回。rank 0 自己的本地专家由原来的路径写。
        顺序：按源 rank，再按 rank 0 本地专家参数的顺序（各 rank 本地专家的名字、形状一一对应）；某个张量在源 rank 上
        不存在（如该参数没有这种优化器状态）时，源 rank 先发一个 0 维标志，两边都跳过。"""
        from megatron.core import parallel_state as ps

        if not dist.is_initialized():
            return
        grp = ps.get_expert_model_parallel_group()
        ranks = dist.get_process_group_ranks(grp)
        if 0 not in ranks or len(ranks) == 1:
            return
        ep_rank, ep_world = ps.get_expert_model_parallel_rank(), len(ranks)
        local = [(n, p, mp, inner) for n, p, mp, inner in self._pairs() if n is not None and _EXPERT_RE.match(n)]
        if not local:
            return
        n_local = {}
        for n, *_ in local:
            base = _EXPERT_RE.match(n).group(1)
            n_local[base] = n_local.get(base, 0) + 1
        dev = torch.device("cuda", torch.cuda.current_device())
        flag = torch.zeros((), dtype=torch.int32, device=dev)
        if ep_rank == 0:
            for src_ep in range(1, ep_world):
                src = ranks[src_ep]
                for n, p, mp, inner in local:
                    dist.recv(flag, src=src)
                    if int(flag) == 0:
                        continue
                    t = get(p, mp, inner)
                    buf = torch.empty_like(t.detach())
                    dist.recv(buf, src=src)
                    base = _EXPERT_RE.match(n).group(1)
                    gname = global_expert_name(n, src_ep, n_local[base])
                    self.buf[f"{kind}.{gname}"] = _to_numpy(buf)
        else:
            for _n, p, mp, inner in local:
                t = get(p, mp, inner)
                flag.fill_(1 if isinstance(t, torch.Tensor) else 0)
                dist.send(flag, dst=0)
                if isinstance(t, torch.Tensor):
                    dist.send(t.detach().contiguous(), dst=0)

    def _pairs(self):
        """(参数名, bf16 参数, fp32 主权重, 内层 torch 优化器)。ChainedOptimizer（稠密 + 专家各一个）逐个拆开；
        非分布式优化器下 float16_groups 与 fp32_from_float16_groups 一一对应。
        本身就是 FP32 的参数（如 --router-true-fp32 的路由权重）在 fp32_from_fp32_groups 里，参数即主权重，两处都给它自己。"""
        opt = getattr(self, "_opt", None)
        if opt is None:
            return []
        names = {id(p): n for n, p in self._params}
        out = []
        for sub in getattr(opt, "chained_optimizers", None) or [opt]:
            f16 = getattr(sub, "float16_groups", None)
            f32 = getattr(sub, "fp32_from_float16_groups", None)
            if f16 is None or f32 is None:
                continue
            inner = getattr(sub, "optimizer", None)
            out += [(names.get(id(a)), a, b, inner) for ga, gb in zip(f16, f32) for a, b in zip(ga, gb) if id(a) in names]
            out += [(names[id(a)], a, a, inner) for g in (getattr(sub, "fp32_from_fp32_groups", None) or [])
                    for a in g if id(a) in names]
        return out

    def _hook(self, cname):
        def fn(module, inputs, output):
            if not self.capturing or not self.forward_acts:
                return
            if cname in self.buf:                                 # 只抓第一个 microbatch
                # 开重算时同一模块在反向里会再前向一次：第一次重算另存，用来比「首次前向 vs 重算」
                rk = f"{cname}.recompute"
                if self.recompute and rk not in self.buf and self._in_backward():
                    t = _first_tensor(output)
                    if t is not None:
                        self.buf[rk] = _to_numpy(t)
                return
            t = _first_tensor(output)
            if t is None:
                return
            self.buf[cname] = _to_numpy(t)
            if self.full and self.step in self.full_steps and t.requires_grad:
                key = f"{cname}.grad"

                def on_grad(g):
                    if key not in self.buf:
                        self.buf[key] = _to_numpy(g)
                t.register_hook(on_grad)
        return fn

    def _in_backward(self):
        """是否在反向（激活重算）里：embedding 前向开始时进入 fwd，算出 loss 后进入 bwd；
        重算只重跑 decoder 层，不经过 embedding，所以 bwd 期间的层前向就是重算。"""
        return getattr(self, "_phase", "fwd") == "bwd"

    def _embed_pre(self, module, inputs):
        self._phase = "fwd"

    def _loss_hook(self, module, inputs, output):
        self._phase = "bwd"
        if not self.rank0:
            return
        t = _first_tensor(output)
        if self.capturing and t is not None and "loss.per_token" not in self.buf and t.dim() == 2:
            self.buf["loss.per_token"] = _to_numpy(t.float())

    def _inject(self, layer):
        def pre(module, args, kwargs):
            if not self.capturing or layer in self._injected:
                return None
            self._injected.add(layer)
            src = "embed.out" if layer == 0 else f"layers.{layer - 1}.out"
            ref = torch.from_numpy(np.ascontiguousarray(self._ref.tensor(self.step, src).view(np.int16))).view(torch.bfloat16)
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            # 值换成参照的层输入，但仍挂在原来的计算图上（ref + 0·hs 逐位等于 ref）：
            # 直接换成新的叶子张量会让前面各层在 rank 0 上拿不到梯度，梯度归约的钩子不触发，其他 rank 会一直等
            new = ref.to(hs.device).reshape(hs.shape) + hs * 0
            self.buf[f"layers.{layer}.in"] = _to_numpy(new)
            if "hidden_states" in kwargs:
                return args, dict(kwargs, hidden_states=new)
            return (new,) + tuple(args[1:]), kwargs
        return pre

    def _wrap_router(self, router, layer: int):
        gating = router.gating
        dump = self

        def gating_capture(inp):
            logits = gating(inp)
            key = f"layers.{layer}.mlp.router.logits"
            if dump.capturing and key not in dump.buf:
                dump.buf[key] = _to_numpy(logits)
            elif dump.capturing and dump.recompute and dump._in_backward() and f"{key}.recompute" not in dump.buf:
                dump.buf[f"{key}.recompute"] = _to_numpy(logits)      # 重算那次的 logits（比首次前向）
            return logits
        router.gating = gating_capture

        def after(module, inputs, output):
            probs, routing_map = output
            if dump._in_backward():                       # 重算的那次前向不重复计负载
                if dump.recompute and dump.capturing:
                    key = f"layers.{layer}.mlp.router.topk_idx"
                    if key in dump.buf and f"{key}.recompute" not in dump.buf:
                        k = int(routing_map[0].sum().item())
                        idx = torch.topk(routing_map.to(torch.int8), k, dim=-1).indices
                        dump.buf[f"{key}.recompute"] = np.sort(idx.cpu().numpy().astype(np.int32), axis=-1)
                        dump.buf[f"layers.{layer}.mlp.router.probs.recompute"] = _to_numpy(probs)
                return
            c = routing_map.sum(dim=0).to(torch.int64)
            dump.counts[layer] = dump.counts[layer] + c if layer in dump.counts else c
            key = f"layers.{layer}.mlp.router.topk_idx"
            if not dump.capturing or key in dump.buf:
                return
            k = int(routing_map[0].sum().item())
            idx = torch.topk(routing_map.to(torch.int8), k, dim=-1).indices  # 选中的专家下标
            dump.buf[key] = np.sort(idx.cpu().numpy().astype(np.int32), axis=-1)
            dump.buf[f"layers.{layer}.mlp.router.probs"] = _to_numpy(probs)
        self.handles.append(router.register_forward_hook(after))

    # ---- 可选观测点（observe）
    def _install_observe(self, model):
        if not self.observe or not self.rank0:
            return
        for name, mod in model.named_modules():
            m = re.match(r"^decoder\.layers\.(\d+)\.self_attention\.core_attention$", name)
            if m and "attn" in self.observe:
                self.handles.append(mod.register_forward_pre_hook(self._attn_pre(int(m.group(1))), with_kwargs=True))
            m = re.match(r"^decoder\.layers\.(\d+)\.mlp$", name)
            if m and "moe" in self.observe and hasattr(mod, "token_dispatcher") and hasattr(mod, "experts"):
                self._wrap_moe(mod, int(m.group(1)))

    def _keep(self, key, t, grad=False):
        """抓一个观测张量（第一个 microbatch）；grad=True 且 full 时另挂梯度钩子存 <key>.grad。"""
        if not self.capturing or key in self.buf or not isinstance(t, torch.Tensor):
            return False
        self.buf[key] = _to_numpy(t)
        if grad and self.full and self.step in self.full_steps and t.requires_grad:
            gk = f"{key}.grad"

            def on_grad(g):
                if gk not in self.buf:
                    self.buf[gk] = _to_numpy(g)
            t.register_hook(on_grad)
        return True

    def _attn_pre(self, layer):
        def pre(module, args, kwargs):
            if not self.capturing or self._in_backward():
                return None
            names = ("query", "key", "value")
            qkv = [args[i] if i < len(args) else kwargs.get(n) for i, n in enumerate(names)]
            for n, t in zip(("q", "k", "v"), qkv):
                self._keep(f"layers.{layer}.attn.core.{n}", t, grad=True)
            mask = args[3] if len(args) > 3 else kwargs.get("attention_mask")
            self.obs_info.setdefault(f"layers.{layer}.attn", {
                "attn_mask_type": str(kwargs.get("attn_mask_type")), "attention_mask_is_none": mask is None,
                "packed_seq_params_is_none": kwargs.get("packed_seq_params") is None,
                "softmax_scale": getattr(module, "softmax_scale", None),
                "qkv_format": getattr(module, "qkv_format", None),
                "q_shape": list(qkv[0].shape) if isinstance(qkv[0], torch.Tensor) else None})
            return None
        return pre

    def _wrap_moe(self, moe, layer):
        dump, disp = self, moe.token_dispatcher
        pre = f"layers.{layer}.mlp"

        orig_pre = moe.preprocess

        def preprocess(*a, **k):
            hs, probs = orig_pre(*a, **k)
            if dump.capturing and not dump._in_backward() and f"{pre}.moe.permuted" not in dump.buf:
                dump._keep(f"{pre}.moe.permuted", hs)
                dump._keep(f"{pre}.moe.permuted_probs", probs)
                for attr, key in (("routing_map", "routing_map"), ("reversed_local_input_permutation_mapping", "unpermute_index"),
                                  ("input_splits", "input_splits"), ("output_splits", "output_splits"),
                                  ("num_global_tokens_per_local_expert", "tokens_from_rank_per_local_expert")):
                    v = getattr(disp, attr, None)
                    if isinstance(v, torch.Tensor):
                        dump.buf[f"{pre}.moe.{key}"] = v.detach().cpu().numpy() if v.dtype == torch.bool \
                            else v.detach().cpu().numpy().astype(np.int64)
                dump.obs_info[f"{pre}.moe"] = {
                    "dispatcher": type(disp).__name__, "num_local_experts": getattr(disp, "num_local_experts", None),
                    "local_expert_indices": list(getattr(disp, "local_expert_indices", []) or []),
                    "ep_size": getattr(disp, "ep_size", None), "moe_permute_fusion": getattr(moe.config, "moe_permute_fusion", None),
                    "experts_type": type(moe.experts).__name__,
                    "experts_with_fused_impl": getattr(moe.experts, "_with_fused_impl", None),
                    "use_te_activation_func": getattr(moe.config, "use_te_activation_func", None),
                    "bias_activation_fusion": getattr(moe.config, "bias_activation_fusion", None),
                    "gated_linear_unit": getattr(moe.config, "gated_linear_unit", None),
                    "activation_func": getattr(getattr(moe.config, "activation_func", None), "__name__", None),
                    "moe_router_topk_scaling_factor": getattr(moe.config, "moe_router_topk_scaling_factor", None),
                    "gradient_accumulation_fusion": getattr(moe.config, "gradient_accumulation_fusion", None)}
            return hs, probs
        moe.preprocess = preprocess

        orig_comb = moe.combine

        def combine(output):
            out = orig_comb(output)
            if not dump._in_backward():
                dump._keep(f"{pre}.moe.combined", out)
            return out
        moe.combine = combine

        orig_post = disp.combine_postprocess

        def combine_postprocess(x):
            out = orig_post(x)
            if not dump._in_backward():
                dump._keep(f"{pre}.moe.unpermuted", out)
            return out
        disp.combine_postprocess = combine_postprocess

        def experts_pre(module, args, kwargs):
            if dump._in_backward():
                return None
            dump._keep(f"{pre}.experts.in", args[0], grad=True)
            tpe = args[1]
            if dump.capturing and f"{pre}.experts.tokens_per_expert" not in dump.buf:
                dump.buf[f"{pre}.experts.tokens_per_expert"] = np.asarray(
                    tpe.detach().cpu().numpy() if isinstance(tpe, torch.Tensor) else tpe, dtype=np.int64)
            dump._keep(f"{pre}.experts.probs", args[2])
            return None
        self.handles.append(moe.experts.register_forward_pre_hook(experts_pre, with_kwargs=True))

        def fc1_out(module, inputs, output):
            if not dump._in_backward():
                dump._keep(f"{pre}.experts.fc1.out", _first_tensor(output), grad=True)
        self.handles.append(moe.experts.linear_fc1.register_forward_hook(fc1_out))

        def fc2_in(module, args):
            if not dump._in_backward():
                dump._keep(f"{pre}.experts.act.out", args[0], grad=True)
            return None
        self.handles.append(moe.experts.linear_fc2.register_forward_pre_hook(fc2_in))

    # ---- 特殊值存 / 恢复（special）
    # 埋的值（FP32 位模式）：−0.0、+0.0、±1e-40（次正规）、±最小次正规 2^-149、最大次正规 0x007FFFFF、最小正规 2^-126
    SPECIAL_F32 = np.array([0x80000000, 0x00000000, 0x000116C2, 0x800116C2, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000],
                           dtype=np.uint32).view(np.float32)

    def _special_targets(self):
        """(规范名, 张量)：命中 special['params'] 正则的 BF16 参数、对应 FP32 主权重、Adam 两个动量，以及命中的 buffer。"""
        pat = re.compile(self.special["params"])
        pairs = {name: (mp, inner) for name, _, mp, inner in self._pairs()}
        out = []
        for name, p in self._params:
            if not pat.search(name):
                continue
            out.append((f"param.{name}", p))
            mp, inner = pairs.get(name, (None, None))
            if mp is not None:
                out.append((f"main_param.{name}", mp))
                st = inner.state.get(mp, {}) if inner is not None else {}
                for k in ("exp_avg", "exp_avg_sq"):
                    if isinstance(st.get(k), torch.Tensor):
                        out.append((f"{k}.{name}", st[k]))
        for name, b in self._buffers:
            if pat.search(name):
                out.append((f"buffer.{name}", b))
        return out

    def _special_poke(self):
        vals = self.SPECIAL_F32
        hit = []
        with torch.no_grad():
            for key, t in self._special_targets():
                flat = t.view(-1)
                n = min(len(vals), flat.numel())
                v = torch.from_numpy(vals[:n].copy()).to(flat.device)
                if key.startswith("exp_avg_sq."):
                    v = v.abs()                                   # 二阶动量非负：只埋 +0 与正的次正规数
                flat[:n] = v.to(flat.dtype)                       # BF16 参数：FP32 → BF16 的 RNE（次正规数保留为 BF16 次正规）
                hit.append(key)
        print(f"[janus] special 埋值 rank {self.rank}: {hit}")

    def _special_dump(self, tag):
        """写 special/rank<r>.<tag>.safetensors：命中张量的全量值（保存时埋值后 / 恢复后 on_train_start）。"""
        from janus.store import write_safetensors
        d = os.path.join(self.out_dir, "special")
        os.makedirs(d, exist_ok=True)
        tensors = {k: _to_numpy(t) for k, t in self._special_targets()}
        write_safetensors(Path(d) / f"rank{self.rank}.{tag}.safetensors", tensors)


class _GemmInventory:
    """记下一段时间里每次 GEMM 调用的输入 dtype 与调用处。

    TE 的 general_gemm / general_grouped_gemm 是 Python 函数，各模块按名字导入，所以在 sys.modules 里把所有指向原函数的名字换成包装；
    torch 的 mm / addmm / bmm / baddbmm 用 TorchDispatchMode 截（反向在 autograd 线程里，dispatch mode 随线程状态传过去）。
    """

    TORCH_OPS = ("mm", "addmm", "bmm", "baddbmm", "_scaled_mm", "convolution")

    def __init__(self):
        self.calls: dict = {}
        self.active = False
        self._patched = []
        self._mode = None

    @staticmethod
    def _dt(x):
        if isinstance(x, torch.Tensor):
            return str(x.dtype).replace("torch.", "")
        if isinstance(x, (list, tuple)) and x:
            return "list[" + ",".join(sorted({_GemmInventory._dt(e) for e in x})) + "]"
        return type(x).__name__

    @staticmethod
    def _site():
        import traceback
        frames = [f for f in traceback.extract_stack()[:-3]
                  if "janus/dump" not in f.filename and "torch/utils/_python_dispatch" not in f.filename
                  and "torch/autograd" not in f.filename and "torch/nn/modules/module.py" not in f.filename]
        return " < ".join(f"{os.path.basename(f.filename)}:{f.name}" for f in frames[-1:-4:-1])

    def _rec(self, key):
        self.calls[key] = self.calls.get(key, 0) + 1

    def start(self):
        import sys
        try:
            from transformer_engine.pytorch.cpp_extensions import gemm as tg
        except ImportError:
            tg = None
        inv = self
        if tg is not None:
            for fname in ("general_gemm", "general_grouped_gemm"):
                orig = getattr(tg, fname, None)
                if orig is None:
                    continue

                def make(orig, fname):
                    def wrapped(A, B, *a, **k):
                        out_dtype = k.get("out_dtype", a[0] if a and isinstance(a[0], torch.dtype) else None)
                        inv._rec((f"te.{fname}", inv._dt(A), inv._dt(B), str(out_dtype).replace("torch.", ""),
                                  str(k.get("layout", "TN")), bool(k.get("grad", False)), inv._site()))
                        return orig(A, B, *a, **k)
                    return wrapped
                w = make(orig, fname)
                for mod in list(sys.modules.values()):
                    d = getattr(mod, "__dict__", None) if mod is not None else None   # 只看模块字典：有的模块 __getattr__ 会抛错
                    if not isinstance(d, dict):
                        continue
                    for attr in (fname, "te_" + fname):
                        if d.get(attr) is orig:
                            setattr(mod, attr, w)
                            self._patched.append((mod, attr, orig))
        from torch.utils._python_dispatch import TorchDispatchMode

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                name = func.overloadpacket.__name__
                if name in inv.TORCH_OPS:
                    ins = [x for x in args if isinstance(x, torch.Tensor)]
                    inv._rec((f"torch.{name}", *[inv._dt(x) for x in ins[-2:]], "", "", False, inv._site()))
                return func(*args, **(kwargs or {}))
        self._mode = Mode()
        self._mode.__enter__()
        self.active = True

    def stop(self):
        if self._mode is not None:
            self._mode.__exit__(None, None, None)
        for mod, attr, orig in self._patched:
            setattr(mod, attr, orig)
        self.active = False

    def summary(self):
        cols = ("op", "a_dtype", "b_dtype", "out_dtype", "layout", "grad", "site")
        return [dict(zip(cols, k), count=v) for k, v in sorted(self.calls.items(), key=lambda kv: -kv[1])]
