"""反向复现诊断臂：在 GPU 上一次只把一处数值行为换成 TPU（MaxText）的写法，看哪一处能让 GPU 也出现 TPU 那样的漂移。

只用于诊断（案例 01 §6「反向复现」），不用于 golden。全部是运行时打补丁，不改 Megatron 源码。
train_hy3.py 的 `--rr-arm a,b,...` 选臂；`install_global` 在建模型之前装（类级补丁），`RRCallback.on_train_start` 在模型建好后装（实例级补丁）。

臂（MaxText 一侧的出处是 maxtext hunyuan 分支的源码路径）：
  ce-mt        交叉熵按 MaxText `max_utils.cross_entropy_with_logits` 的自定义 VJP 算：BF16 logits 转 FP32，减最大值、exp、求和、
               log；反向 deriv = exp_shifted / sum_exp − onehot，再乘逐 token 的上游梯度，FP32 出、经 .float() 的反向舍回 BF16
               （= MaxText `cast_logits_to_fp32: true` 时 astype 的转置）
  adam-optax   优化器一步按 optax `adamw` 逐式重写（`scale_by_adam` → `add_decayed_weights` → `scale_by_learning_rate`）：
               mu = (1−b1)·g + b1·mu；nu = (1−b2)·g² + b2·nu；mu_hat = mu / (1 − b1^t)；nu_hat = nu / (1 − b2^t)（t 与 b^t 都按 FP32）；
               u = mu_hat / (sqrt(nu_hat + eps_root) + eps)，eps_root = 0；u += wd·p；p += (−lr)·u。代替 TE FusedAdam 的 CUDA 核
  clip-optax   梯度裁剪按 optax `clip_by_global_norm`：范数 < max_norm 不动，否则 g = (g / norm) · max_norm（先除后乘，不加 1e-6）；
               代替 Megatron `clip_grad_by_total_norm_fp32` 的 g · max_norm / (norm + 1e-6)
  fnorm-mt     最后一层 RMSNorm 按 MaxText `normalizations.RMSNorm` 默认写法（rmsnorm_single_rounding=false，`t-a` 用的就是它）：
               FP32 归一化 → 舍 BF16 → 乘 BF16 scale → 再舍 BF16（两次舍入），反向走 autograd（对应 JAX 自动微分）
  fnorm-f32    最后一层 RMSNorm 在 FP32 里乘完 scale 只舍一次（MaxText rmsnorm_single_rounding=true），反向走 autograd
  wgrad-bf16   输出层权重梯度每个 microbatch 先舍成 BF16 再累加（MaxText logits_dense 的 dtype=bfloat16 时 dot 的转置出 BF16）
  attn-scale-bf16  注意力 softmax 缩放系数用 BF16 舍过的 1/sqrt(128) = 0.08837890625（比 0.0883883476 小 1.07e-4）。
               MaxText `qwen3.AttentionWithNorm` 设 query_pre_attn_scalar = head_dim**-0.5，`attentions.py` 里 query * 它；
               query 是 BF16，JAX 把 Python 标量按弱类型转成 BF16 常量（t-hlo 优化前 HLO：bf16[] constant(0.08838) 广播后乘
               bf16[16,4096,64,128]），splash 本身 sm_scale = 1。这一臂只换系数，缩放仍由 cuDNN 在 FP32 里做
  attn-qscale-mt   MaxText 原样：softmax_scale = 1，query 先在 BF16 里乘 BF16 常量 0.08837890625（乘积再舍一次 BF16）
  ogscale-<x> / egscale-<x>  敏感度探针：只把输出层 / embedding 的梯度在裁剪之后乘 x
  moe-combine-f32  路由权重在 fc2 之后乘、k 个专家输出在 FP32 里加权求和再舍 BF16，然后加共享专家（MaxText
               `float32_weight_sum: true` 的做法）；GPU 默认是先把 act·p 舍 BF16 再进 fc2、k 个输出在 BF16 里 scatter_add
  ogauge-<k>   规范方向探针：裁剪前把输出层 FP32 梯度的「跨行均值」（所有行共同的分量，softmax 不变方向）乘 k：
               G ← G + (k − 1)·mean_v(G)。k = 0 即把它去掉（精确算术下这一分量本来是 0）
  router-bwd-bf16  路由 logits 的梯度先舍 BF16 再进路由 GEMM 的两次反向（TPU matmul_precision=default 下 FP32 操作数
               先舍 BF16 单遍乘，见 01 results/phase2-l1-l2-replay.md L1）；GPU 默认这两次反向是 TF32
  dh-center    规范方向探针（隐状态一侧）：输出层对输入 h 的梯度减去 s_t·w̄（s_t = Σ_v dlogit_tv，w̄ = 输出层 BF16 权重的行均值），
               即去掉 dlogits 的规范残差经 dh 流回主干的那一份；输出层自己的权重梯度不动
  dhscale-<k>  同上，但改成 dh ← dh + (k − 1)·s_t·w̄：k = 0 等于 dh-center，k > 1 把这一份放大（对应 MaxText
               `lm_head_dh_gauge_scale`）
  dgrad-f32    输出层对输入 h 的梯度改成 FP32 精确矩阵乘：dh = float(dlogits) @ float(W)（dlogits 仍是 BF16 的值），
               去掉 BF16 GEMM 本身在 dh 上的误差（输出头回放里 GPU 这一份沿 ŵ̄ 有系统差，TPU 没有）。
               注意：容器里 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1、allow_tf32 = True，这个 FP32 乘法实际走 TF32 张量核，
               并不精确——第 1 步沿 û 的误差反而比 BF16 GEMM 还大（head_bwd_used.py --gpu-gemm：−8.1e-4 对 −6.5e-4）
  dgrad-exact  同上，但乘法时临时关掉 TF32（allow_tf32 = False，FP32 非张量核），第 1 步对 FP64 沿 û 的相对误差 −1.7e-6；
               前 4 次调用打印改前 / 改后 / GPU 上 FP64 精确值沿 û 的总和
  dgrad-chunk  同 dgrad-exact，但收缩维（词表 120832）按 1024 一段分开乘、段间 FP32 累加。B200 上 cuBLAS 的 FP32
               矩阵乘（关 TF32）一次乘完整个词表时，θ_299（第 300 步开始时的状态）下 dh 对 FP64 rel_l2 4.2e-5、沿 û −2.6e-4（第 1 步只有 −1.8e-6）；
               分段后 5.9e-7、+1e-7（案例 01「GPU 精确的输出头 dgrad」行）
  dgrad-chunk-bf16  同 dgrad-chunk，但不升 FP32：词表按 1024 一段，每段是 BF16 dlogits × BF16 W 的张量核乘法、输出留 FP32
               （torch.mm(..., out_dtype=float32)），段间 FP32 累加，最后舍回 BF16。消融里去掉偏差的是分段而不是 FP32
               （案例 01「输出头 dgrad 消融：分段还是 FP32」行）：θ_299 上沿 û +3.1e-7，与 dgrad-chunk 同量级，单次乘法快约 10 倍。
               与 dgrad-exact / dgrad-chunk 互斥
  dlogits-bf16 流进输出层反向的 dlogits 先按 RNE 舍到 BF16 再进 dgrad 与权重梯度（全程 FP32 程序里模仿生产程序的 BF16 dlogits；
               对应 TPU `train_hy3.py --head-round cotangent`）。要和 dgrad-chunk 一起用时，这个钩子先注册、先执行
  dhproj-<a>   dh ← dh − a·(dh·û)û，û = 输出层 BF16 权重行均值的单位方向（所有 logits 一起动的方向）；
               对应 MaxText `lm_head_dh_wbar_remove`
  gscale-<x>   敏感度探针（不是 TPU 行为）：裁剪之后把全部梯度再乘 x（等于把裁剪阈值从 1 改成 x），看 Adam ε 区的行对梯度尺度有多敏感
  resid-fp32   残差流用 FP32（Megatron fp32_residual_connection）；对照 TPU excess precision 下融合内残差加不舍 BF16 的情况
  wgrad-exact  输出层权重梯度（收缩长度 = token 数）改成 FP32 精确矩阵乘（临时关 TF32）：输出层反向开始前存一份 main_grad，
               反向写完后把这个 microbatch 的贡献换成 float(dlogits)ᵀ @ float(h)。前 4 次调用打印 BF16 贡献对精确值的整体尺度、
               沿词表均值（规范分量）与范数最小一半的行（冷门词）上的尺度
  head-log-<s1>.<s2>...  只记录、不改数值：在第 s1、s2… 步（rank 0、第一个 microbatch）记下输出层矩阵乘反向实际用到的张量，
               写到 <janus-run>/head_used/step_<s>.safetensors。挂在 Megatron `LinearWithGradAccumulationAndAsyncCommunication`
               的 forward / backward 上（TP = 1 时输出层走它，按权重行数 = 词表大小认）：head.h（输入）、head.logits（输出）、
               head.weight（ctx 里存的、dgrad 实际乘的权重）、head.dlogits_in（交给 backward 的 grad_output，原样 dtype）、
               head.dlogits_used（backward 里 `.to(input_dtype)` 之后真正进矩阵乘的那份）、head.dh（backward 返回的 grad_input）。
               离线分析见 models/hunyuan3/analysis/head_bwd_used.py
"""

import torch

ARMS = ("ce-mt", "adam-optax", "clip-optax", "fnorm-mt", "fnorm-f32", "wgrad-bf16", "attn-scale-bf16", "attn-qscale-mt",
        "resid-fp32", "moe-combine-f32", "dh-center", "router-bwd-bf16", "dgrad-f32", "dgrad-exact", "dgrad-chunk",
        "dgrad-chunk-bf16", "wgrad-exact", "dlogits-bf16")
MT_QSCALE = 0.08837890625          # = float(bf16(128 ** -0.5))


def parse(spec):
    arms = [x for x in (spec or "").split(",") if x]
    bad = [x for x in arms if x not in ARMS and not x.startswith(("gscale-", "ogscale-", "egscale-", "ogauge-", "dhscale-", "dhproj-", "head-log-"))]
    if bad:
        raise SystemExit(f"--rr-arm 不认识：{bad}；可选 {ARMS}")
    if "fnorm-mt" in arms and "fnorm-f32" in arms:
        raise SystemExit("--rr-arm：fnorm-mt 与 fnorm-f32 只能选一个")
    if "attn-scale-bf16" in arms and "attn-qscale-mt" in arms:
        raise SystemExit("--rr-arm：attn-scale-bf16 与 attn-qscale-mt 只能选一个")
    if sum(x in arms for x in ("dgrad-exact", "dgrad-chunk", "dgrad-chunk-bf16")) > 1:
        raise SystemExit("--rr-arm：dgrad-exact、dgrad-chunk、dgrad-chunk-bf16 只能选一个")
    return arms


def chunked_dgrad_bf16(dl, w, chunk=1024):
    """dgrad-chunk-bf16 的乘法：dl [..., V] BF16，w [V, D] BF16 → FP32 [..., D]。
    每段 BF16 张量核乘法、输出 FP32，段间 FP32 累加（离线消融 dgrad_accum_ablation.py 的 B 写法用的就是这个函数）。"""
    assert dl.dtype == torch.bfloat16 and w.dtype == torch.bfloat16, (dl.dtype, w.dtype)
    d2 = dl.reshape(-1, dl.shape[-1])
    acc = torch.zeros(d2.shape[0], w.shape[1], dtype=torch.float32, device=d2.device)
    for k0 in range(0, w.shape[0], chunk):
        acc += torch.mm(d2[:, k0:k0 + chunk], w[k0:k0 + chunk], out_dtype=torch.float32)
    return acc.reshape(dl.shape[:-1] + (w.shape[1],))


def apply_model_config(m, arms):
    """改 Megatron 模型配置的臂（建模型之前）。"""
    if "attn-scale-bf16" in arms:
        assert m.kv_channels == 128, m.kv_channels
        m.softmax_scale = MT_QSCALE
    if "attn-qscale-mt" in arms:
        assert m.kv_channels == 128, m.kv_channels
        m.softmax_scale = 1.0
    if "resid-fp32" in arms:
        m.fp32_residual_connection = True


# ---------------- ce-mt ----------------
class _MTXent(torch.autograd.Function):
    """MaxText cross_entropy_with_logits（z_loss = 0）。logits32: [..., V] FP32；labels: [...] int64。"""

    @staticmethod
    def forward(ctx, logits32, labels):
        max_logit = logits32.max(dim=-1, keepdim=True).values
        shifted = logits32 - max_logit
        exp_shifted = torch.exp(shifted)
        sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
        log_softmax_t = shifted.gather(-1, labels.unsqueeze(-1)) - torch.log(sum_exp)
        ctx.save_for_backward(exp_shifted, sum_exp, labels)
        return -log_softmax_t.squeeze(-1)

    @staticmethod
    def backward(ctx, g):
        exp_shifted, sum_exp, labels = ctx.saved_tensors
        deriv = exp_shifted / sum_exp
        deriv.scatter_add_(-1, labels.unsqueeze(-1), -torch.ones_like(labels, dtype=deriv.dtype).unsqueeze(-1))
        return g.unsqueeze(-1) * deriv, None


def _install_ce_mt():
    from megatron.core.models.common.language_module.language_module import LanguageModule

    def compute_language_model_loss(self, labels, logits):
        assert self.config.tensor_model_parallel_size == 1, "ce-mt 只支持 TP = 1（整词表在一张卡上）"
        labels = labels.transpose(0, 1).contiguous()          # [b s] -> [s b]，与 Megatron 原函数相同
        loss = _MTXent.apply(logits.float(), labels)          # .float() 的反向把 FP32 梯度舍回 BF16
        return loss.transpose(0, 1).contiguous()

    LanguageModule.compute_language_model_loss = compute_language_model_loss


# ---------------- adam-optax ----------------
def _install_adam_optax():
    import megatron.core.optimizer as mo
    cls = mo.Adam

    @torch.no_grad()
    def step(self, closure=None, grad_scaler=None):
        assert closure is None and grad_scaler is None
        for group in self.param_groups:
            group["_rr_t"] = group.get("_rr_t", 0) + 1
            t = group["_rr_t"]
            b1, b2 = group["betas"]
            eps, wd, lr = group["eps"], group["weight_decay"], group["lr"]
            assert group.get("bias_correction", True)
            f32 = torch.float32
            bc1 = 1 - torch.tensor(b1, dtype=f32) ** torch.tensor(t, dtype=f32)   # optax：decay**count，按 FP32
            bc2 = 1 - torch.tensor(b2, dtype=f32) ** torch.tensor(t, dtype=f32)
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.dtype == f32, "adam-optax 要求 FP32 主权重"
                st = self.state[p]
                if "exp_avg" not in st:
                    st["exp_avg"] = torch.zeros_like(p, dtype=f32)
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=f32)
                g = p.grad.float()
                mu, nu = st["exp_avg"], st["exp_avg_sq"]
                mu.copy_((1 - b1) * g + b1 * mu)
                nu.copy_((1 - b2) * (g * g) + b2 * nu)
                mu_hat = mu / bc1.to(p.device)
                nu_hat = nu / bc2.to(p.device)
                u = mu_hat / (torch.sqrt(nu_hat + 0.0) + eps)
                if wd:
                    u = u + wd * p
                p.add_((-lr) * u)
        return None

    cls.step = step


# ---------------- clip-optax ----------------
def _install_clip_optax():
    import megatron.core.optimizer.optimizer as moo

    def clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm, use_decoupled_grad=False):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        if not (total_norm >= max_norm):         # optax：jnp.where(g_norm < max_norm, t, (t / g_norm) * max_norm)
            return
        for p in parameters:
            g = p.decoupled_grad if use_decoupled_grad else p.grad
            if g is None:
                continue
            gn = torch.tensor(total_norm, dtype=g.dtype, device=g.device) if not torch.is_tensor(total_norm) \
                else total_norm.to(g.dtype)
            g.div_(gn).mul_(max_norm)

    moo.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32


SCALE_IDS = {}                     # id(FP32 主权重) -> 倍数；RRCallback.on_train_start 按参数名填
GAUGE_IDS = {}                     # id(输出层 FP32 主权重) -> k（ogauge-<k>）
GAUGE_LOG = []


def _install_ogauge():
    import megatron.core.optimizer.optimizer as moo
    orig = moo.clip_grad_by_total_norm_fp32

    def gauge_then_clip(parameters, max_norm, total_norm, use_decoupled_grad=False):
        for p in ([parameters] if isinstance(parameters, torch.Tensor) else parameters):
            if id(p) in GAUGE_IDS:
                g = p.decoupled_grad if use_decoupled_grad else p.grad
                k = GAUGE_IDS[id(p)]
                mean = g.double().mean(dim=0, keepdim=True)
                GAUGE_LOG.append(float(mean.norm()) * g.shape[0])
                g.add_(((k - 1.0) * mean).to(g.dtype))
        orig(parameters, max_norm, total_norm, use_decoupled_grad=use_decoupled_grad)

    moo.clip_grad_by_total_norm_fp32 = gauge_then_clip


def _install_gscale(factor, only=None):
    import megatron.core.optimizer.optimizer as moo
    orig = moo.clip_grad_by_total_norm_fp32

    def clip_then_scale(parameters, max_norm, total_norm, use_decoupled_grad=False):
        orig(parameters, max_norm, total_norm, use_decoupled_grad=use_decoupled_grad)
        for p in ([parameters] if isinstance(parameters, torch.Tensor) else parameters):
            g = p.decoupled_grad if use_decoupled_grad else p.grad
            if g is None:
                continue
            if only is None:
                g.mul_(factor)
            elif id(p) in SCALE_IDS:
                g.mul_(SCALE_IDS[id(p)])

    moo.clip_grad_by_total_norm_fp32 = clip_then_scale


def install_global(arms):
    """建模型之前装的类级补丁。"""
    for arm, fn in (("ce-mt", _install_ce_mt), ("adam-optax", _install_adam_optax), ("clip-optax", _install_clip_optax)):
        if arm in arms:
            fn()
    for arm in arms:
        if arm.startswith("gscale-"):
            _install_gscale(float(arm.split("-", 1)[1]))
        elif arm.startswith(("ogscale-", "egscale-")):
            _install_gscale(None, only=True)
        elif arm.startswith("ogauge-"):
            _install_ogauge()


# ---------------- 实例级：fnorm-*、wgrad-bf16 ----------------
def _mt_rmsnorm(x, w, eps, single):
    xf = x.float()
    mean2 = xf.square().mean(dim=-1, keepdim=True)
    normed = xf * torch.rsqrt(mean2 + eps)
    if single:
        return (normed * w.float()).to(x.dtype)
    return normed.to(x.dtype) * w.to(x.dtype)        # BF16 × BF16：PyTorch 在 FP32 里乘、舍一次 = 第二次舍入


# ---------------- head-log（只记录） ----------------
HEAD_LOG = {"steps": set(), "step": 0, "out": None, "seen": set(), "fwd": {}}


def _install_head_log():
    import os

    from megatron.core.tensor_parallel.layers import LinearWithGradAccumulationAndAsyncCommunication as Lin
    from safetensors.torch import save_file

    st = HEAD_LOG
    vocab = st["vocab"]
    orig_fwd, orig_bwd = Lin.forward, Lin.backward

    def active(weight):
        return (st["step"] in st["steps"] and weight.dim() == 2 and weight.shape[0] == vocab
                and torch.distributed.get_rank() == 0)

    def cpu(t):
        return t.detach().to("cpu", copy=True).contiguous()

    def forward(ctx, input, weight, *args):
        out = orig_fwd(ctx, input, weight, *args)
        if active(weight) and st["step"] not in st["fwd"] and st["step"] not in st["seen"]:
            st["fwd"][st["step"]] = {"head.h": cpu(input), "head.logits": cpu(out)}
        return out

    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        rec = active(weight) and st["step"] not in st["seen"] and st["step"] in st["fwd"]
        if rec:
            d = st["fwd"].pop(st["step"])
            d["head.dlogits_in"] = cpu(grad_output)
            d["head.dlogits_used"] = cpu(grad_output.to(ctx.input_dtype))     # 与原 backward 第一行同一个转换
            d["head.weight"] = cpu(weight)
        res = orig_bwd(ctx, grad_output)
        if rec:
            d["head.dh"] = cpu(res[0])
            st["seen"].add(st["step"])
            os.makedirs(os.path.join(st["out"], "head_used"), exist_ok=True)
            meta = {k: str(v.dtype) for k, v in d.items()}
            meta["input_dtype"] = str(ctx.input_dtype)
            save_file(d, os.path.join(st["out"], "head_used", f"step_{st['step']}.safetensors"), metadata=meta)
            print(f"[rr] head-log step {st['step']}: {meta}", flush=True)
        return res

    Lin.forward = staticmethod(forward)
    Lin.backward = staticmethod(backward)


class RRCallback:
    def __init__(self, arms, out=None):
        self.arms = arms
        self.out = out

    def on_train_step_start(self, context):
        HEAD_LOG["step"] = context.state.train_state.step + 1          # 即将执行的迭代号（从 1 开始），与 JanusDump 同口径

    def on_train_start(self, context):
        hl = next((x for x in self.arms if x.startswith("head-log-")), None)
        if hl is not None:
            assert self.out, "head-log 要有输出目录（--janus-run 或 --out）"
            models_ = context.model if isinstance(context.model, (list, tuple)) else [context.model]
            g = models_[0]
            while not hasattr(g, "decoder") and hasattr(g, "module"):
                g = g.module
            HEAD_LOG.update(steps={int(x) for x in hl.split("-", 2)[2].split(".") if x}, out=self.out,
                            vocab=g.output_layer.weight.shape[0])
            _install_head_log()
            if torch.distributed.get_rank() == 0:
                print(f"[rr] head-log steps {sorted(HEAD_LOG['steps'])} vocab {HEAD_LOG['vocab']} -> {self.out}/head_used", flush=True)
        targets = {}
        for arm in self.arms:
            if arm.startswith("ogscale-"):
                targets["output_layer.weight"] = float(arm.split("-", 1)[1])
            elif arm.startswith("egscale-"):
                targets["embedding.word_embeddings.weight"] = float(arm.split("-", 1)[1])
            elif arm.startswith("ogauge-"):
                targets["output_layer.weight"] = ("gauge", float(arm.split("-", 1)[1]))
        if targets:
            models0 = context.model if isinstance(context.model, (list, tuple)) else [context.model]
            names = {}
            for mdl in models0:
                for n, prm in mdl.named_parameters():
                    for t in targets:
                        if n.endswith(t):
                            names[id(prm)] = t
            opt = context.optimizer
            for sub in getattr(opt, "chained_optimizers", None) or [opt]:
                f16, f32 = getattr(sub, "float16_groups", None), getattr(sub, "fp32_from_float16_groups", None)
                if f16 is None:
                    continue
                for ga, gb in zip(f16, f32):
                    for mp16, mp32 in zip(ga, gb):
                        if id(mp16) in names:
                            t = targets[names[id(mp16)]]
                            if isinstance(t, tuple):
                                GAUGE_IDS[id(mp32)] = t[1]
                            else:
                                SCALE_IDS[id(mp32)] = t
            assert len(SCALE_IDS) + len(GAUGE_IDS) == len(targets), (SCALE_IDS, GAUGE_IDS, targets)
            if torch.distributed.get_rank() == 0:
                print(f"[rr] grad scale after clip: {targets}", flush=True)
        models = context.model if isinstance(context.model, (list, tuple)) else [context.model]
        for mdl in models:
            gpt = mdl
            while not hasattr(gpt, "decoder") and hasattr(gpt, "module"):
                gpt = gpt.module
            if "fnorm-mt" in self.arms or "fnorm-f32" in self.arms:
                fl = gpt.decoder.final_layernorm
                single = "fnorm-f32" in self.arms
                eps = getattr(fl, "eps", None)
                if eps is None:
                    eps = gpt.config.layernorm_epsilon
                assert not getattr(fl, "zero_centered_gamma", False)
                w = fl.weight

                def fwd(x, _w=w, _eps=eps, _single=single):
                    return _mt_rmsnorm(x, _w, _eps, _single)

                fl.forward = fwd
                if torch.distributed.get_rank() == 0:
                    print(f"[rr] final_layernorm -> MaxText RMSNorm single={single} eps={eps} type={type(fl).__name__}", flush=True)
            if "attn-scale-bf16" in self.arms or "attn-qscale-mt" in self.arms:
                for i, layer in enumerate(gpt.decoder.layers):
                    ca = layer.self_attention.core_attention
                    if "attn-qscale-mt" in self.arms:
                        orig = ca.forward

                        def fwd(query, key, value, *args, _orig=orig, **kw):
                            c = torch.tensor(MT_QSCALE, dtype=query.dtype, device=query.device)
                            return _orig(query * c, key, value, *args, **kw)

                        ca.forward = fwd
                    if torch.distributed.get_rank() == 0:
                        print(f"[rr] layer {i} core_attention {type(ca).__name__} softmax_scale="
                              f"{getattr(ca, 'softmax_scale', None)!r} qscale_mt={'attn-qscale-mt' in self.arms}", flush=True)
            if "moe-combine-f32" in self.arms:
                from megatron.core.transformer.moe.moe_utils import unpermute
                n = 0
                for layer in gpt.decoder.layers:
                    mlp = layer.mlp
                    if not hasattr(mlp, "token_dispatcher"):
                        continue
                    ex, disp = mlp.experts, mlp.token_dispatcher
                    ex_orig = ex.forward

                    def ex_fwd(x, tokens_per_expert, permuted_probs, *args, _orig=ex_orig, **kw):
                        return _orig(x, tokens_per_expert, torch.ones_like(permuted_probs), *args, **kw)

                    ex.forward = ex_fwd

                    def combine_postprocess(tokens, _d=disp):
                        out = unpermute(tokens, _d.reversed_local_input_permutation_mapping,
                                        restore_shape=_d.hidden_shape_before_permute, probs=_d.probs,
                                        routing_map=_d.routing_map, fused=_d.config.moe_permute_fusion,
                                        drop_and_pad=_d.drop_and_pad)
                        out = out.view(_d.hidden_shape)
                        if _d.shared_experts is not None:
                            out = out + _d.shared_experts.get_output()
                        return out

                    disp.combine_postprocess = combine_postprocess
                    n += 1
                if torch.distributed.get_rank() == 0:
                    print(f"[rr] moe-combine-f32 on {n} MoE layers, permute_fusion={gpt.config.moe_permute_fusion}", flush=True)
            if "router-bwd-bf16" in self.arms:
                n = 0
                for layer in gpt.decoder.layers:
                    router = getattr(layer.mlp, "router", None)
                    if router is None:
                        continue
                    g_orig = router.gating

                    def gating(inp, _g=g_orig):
                        lg = _g(inp)
                        if lg.requires_grad:
                            lg.register_hook(lambda g: g.to(torch.bfloat16).to(g.dtype))
                        return lg

                    router.gating = gating
                    n += 1
                if torch.distributed.get_rank() == 0:
                    print(f"[rr] router-bwd-bf16 on {n} routers", flush=True)
            dh_k = 0.0 if "dh-center" in self.arms else next(
                (float(x.split("-", 1)[1]) for x in self.arms if x.startswith("dhscale-")), None)
            if "dgrad-f32" in self.arms:
                olf = gpt.output_layer
                st32 = {}

                def fhook32(mod, inp, out, _st=st32):
                    x = inp[0]
                    y = out[0] if isinstance(out, (tuple, list)) else out
                    w = mod.weight
                    if not (torch.is_tensor(x) and x.requires_grad and y.requires_grad):
                        return

                    def on_dlogits(g):
                        _st["g"] = g
                        return g

                    def on_dh(g):
                        dl = _st.pop("g")
                        new = (dl.float() @ w.detach().float()).to(g.dtype)
                        if _st.get("n", 0) < 40 and torch.distributed.get_rank() == 0:
                            # 第二道钩子：比改前 / 改后的 dh（数据钩子挂在最后归一化输出上、先于这里执行，只看得到改前）
                            _st["n"] = _st.get("n", 0) + 1
                            u = w.detach().float().mean(dim=0)
                            u = u / u.norm()
                            go, gn = g.float(), new.float()
                            print(f"[rr] dgrad-f32 call {_st['n']}: dtype {g.dtype} rel_l2(new-old) "
                                  f"{float((gn - go).norm() / go.norm()):.3e}  sum(old.u) {float((go @ u).sum()):+.6e}  "
                                  f"sum(new.u) {float((gn @ u).sum()):+.6e}", flush=True)
                        return new

                    y.register_hook(on_dlogits)
                    x.register_hook(on_dh)

                olf.register_forward_hook(fhook32)
                if torch.distributed.get_rank() == 0:
                    print(f"[rr] dgrad-f32 hook on {type(olf).__name__}", flush=True)
            if "dlogits-bf16" in self.arms:
                old_ = gpt.output_layer
                stb = {}

                def fhookb(mod, inp, out, _st=stb):
                    y = out[0] if isinstance(out, (tuple, list)) else out
                    if not (torch.is_tensor(y) and y.requires_grad):
                        return

                    def on_dl(g):
                        r = g.to(torch.bfloat16).to(g.dtype)
                        if not _st.get("checked") and torch.distributed.get_rank() == 0:
                            _st["checked"] = True
                            print(f"[rr] dlogits-bf16：dlogits {g.dtype} {tuple(g.shape)} 舍 BF16，"
                                  f"改变的元素比例 {float((r != g).float().mean()):.3f}", flush=True)
                        return r

                    y.register_hook(on_dl)

                old_.register_forward_hook(fhookb)
                if torch.distributed.get_rank() == 0:
                    print(f"[rr] dlogits-bf16 hook on {type(old_).__name__}", flush=True)
            if "dgrad-exact" in self.arms or "dgrad-chunk" in self.arms or "dgrad-chunk-bf16" in self.arms:
                olx = gpt.output_layer
                stx = {}
                chunk = 1024 if ("dgrad-chunk" in self.arms or "dgrad-chunk-bf16" in self.arms) else 0
                bf16_tc = "dgrad-chunk-bf16" in self.arms

                def fhookx(mod, inp, out, _st=stx):
                    x = inp[0]
                    y = out[0] if isinstance(out, (tuple, list)) else out
                    w = mod.weight
                    if not (torch.is_tensor(x) and x.requires_grad and y.requires_grad):
                        return

                    def on_dlogits(g):
                        _st["g"] = g
                        return g

                    def on_dh(g):
                        dl = _st.pop("g")
                        mm = torch.backends.cuda.matmul
                        old_tf32 = mm.allow_tf32
                        mm.allow_tf32 = False
                        try:
                            tf32_in_gemm = mm.allow_tf32
                            if bf16_tc:
                                exact32 = chunked_dgrad_bf16(dl, w.detach(), chunk)
                            elif chunk:
                                dlf, wf = dl.float(), w.detach().float()
                                exact32 = torch.zeros(dlf.shape[:-1] + (wf.shape[1],), dtype=torch.float32, device=dlf.device)
                                for k0 in range(0, wf.shape[0], chunk):
                                    exact32 += dlf[..., k0:k0 + chunk] @ wf[k0:k0 + chunk]
                            else:
                                exact32 = dl.float() @ w.detach().float()
                        finally:
                            mm.allow_tf32 = old_tf32
                        if not _st.get("checked") and torch.distributed.get_rank() == 0:
                            # 自检（一次）：进乘法的是生产路径实际用的 BF16 dlogits 与 BF16 参数 W（不是 FP32 主权重），乘法时 TF32 关着
                            _st["checked"] = True
                            print(f"[rr] dgrad-exact 自检（分段 {chunk}，{'BF16 张量核、段输出 FP32' if bf16_tc else 'FP32 乘法'}）：dlogits {dl.dtype} {tuple(dl.shape)}，W {w.dtype}（模型参数，"
                                  f"非主权重：{not hasattr(w, 'main_param') or w.dtype != torch.float32}），乘法时 allow_tf32={tf32_in_gemm}，"
                                  f"输出舍回 {g.dtype}", flush=True)
                        new = exact32.to(g.dtype)
                        if _st.get("n", 0) < 4 and torch.distributed.get_rank() == 0:
                            _st["n"] = _st.get("n", 0) + 1
                            u = w.detach().double().mean(dim=0)
                            u = u / u.norm()
                            ref = dl.double() @ w.detach().double()
                            print(f"[rr] dgrad-exact call {_st['n']}: sum(old.u) {float((g.double() @ u).sum()):+.7e}  "
                                  f"sum(new.u) {float((new.double() @ u).sum()):+.7e}  sum(fp64.u) {float((ref @ u).sum()):+.7e}  "
                                  f"rel_l2(fp32-fp64) {float((exact32.double() - ref).norm() / ref.norm()):.2e}", flush=True)
                            del ref
                        return new

                    y.register_hook(on_dlogits)
                    x.register_hook(on_dh)

                olx.register_forward_hook(fhookx)
                if torch.distributed.get_rank() == 0:
                    print(f"[rr] dgrad-exact hook on {type(olx).__name__}", flush=True)
            if "wgrad-exact" in self.arms:
                olw = gpt.output_layer
                stw = {}

                def fhookw(mod, inp, out, _st=stw):
                    x = inp[0]
                    y = out[0] if isinstance(out, (tuple, list)) else out
                    w = mod.weight
                    if not (torch.is_tensor(x) and x.requires_grad and y.requires_grad):
                        return
                    xd = x.detach()

                    def on_dlogits(g):
                        mg = getattr(w, "main_grad", None)
                        assert mg is not None, "wgrad-exact 要求 gradient_accumulation_fusion（权重有 main_grad）"
                        _st["g"] = g
                        _st["saved"] = mg.detach().clone()
                        return g

                    def on_dh(gx):
                        dl = _st.pop("g")
                        saved = _st.pop("saved")
                        mg = w.main_grad
                        mm = torch.backends.cuda.matmul
                        old_tf32 = mm.allow_tf32
                        mm.allow_tf32 = False
                        try:
                            exact = dl.reshape(-1, dl.shape[-1]).float().t() @ xd.reshape(-1, xd.shape[-1]).float()
                        finally:
                            mm.allow_tf32 = old_tf32
                        if _st.get("n", 0) < 4 and torch.distributed.get_rank() == 0:
                            _st["n"] = _st.get("n", 0) + 1
                            c = (mg.detach() - saved).double()
                            e = exact.double()
                            rn = e.norm(dim=1)
                            low = rn <= rn.median()
                            scale = lambda a, b: float((a * b).sum() / (b * b).sum() - 1.0)  # noqa: E731
                            # 沿词表的均值（规范分量）在 FP64 里精确算：mean_v Σ_t dl_tv h_t = Σ_t s_t h_t / V，s_t 是 dlogits 的行和
                            dl2 = dl.reshape(-1, dl.shape[-1])
                            cm64 = (dl2.double().sum(1) @ xd.reshape(-1, xd.shape[-1]).double()) / dl2.shape[1]
                            print(f"[rr] wgrad-exact call {_st['n']}: rel_l2(bf16-exact) {float((c - e).norm() / e.norm()):.3e}  "
                                  f"scale-1 {scale(c, e):+.3e}  colmean scale-1 vs fp32 {scale(c.mean(0), e.mean(0)):+.3e} "
                                  f"vs fp64 {scale(c.mean(0), cm64):+.3e} (fp32 vs fp64 {scale(e.mean(0), cm64):+.3e}; "
                                  f"|colmean|/|G|·√V {float(cm64.norm() / e.norm() * e.shape[0] ** 0.5):.3e})  "
                                  f"low-half rows scale-1 {scale(c[low], e[low]):+.3e}", flush=True)
                        mg.copy_(saved + exact)
                        return gx

                    y.register_hook(on_dlogits)
                    x.register_hook(on_dh)

                olw.register_forward_hook(fhookw)
                if torch.distributed.get_rank() == 0:
                    print(f"[rr] wgrad-exact hook on {type(olw).__name__}", flush=True)
            dh_a = next((float(x.split("-", 1)[1]) for x in self.arms if x.startswith("dhproj-")), None)
            if dh_a is not None:
                olp = gpt.output_layer

                def phook(mod, inp, out, _a=dh_a):
                    x = inp[0]
                    if not (torch.is_tensor(x) and x.requires_grad):
                        return
                    w = mod.weight

                    def on_dh(g):
                        u = w.detach().float().mean(dim=0)
                        u = u / u.norm()
                        gf = g.float()
                        return (gf - _a * (gf @ u).unsqueeze(-1) * u).to(g.dtype)

                    x.register_hook(on_dh)

                olp.register_forward_hook(phook)
                if torch.distributed.get_rank() == 0:
                    print(f"[rr] dhproj hook (a = {dh_a}) on {type(olp).__name__}", flush=True)
            if dh_k is not None:
                ol = gpt.output_layer
                state = {}

                def fhook(mod, inp, out, _st=state):
                    x = inp[0]
                    y = out[0] if isinstance(out, (tuple, list)) else out
                    w = mod.weight
                    if not (torch.is_tensor(x) and x.requires_grad and y.requires_grad):
                        return

                    def on_dlogits(g):
                        _st["s"] = g.float().sum(dim=-1, keepdim=True)      # [s, b, 1]
                        return g

                    def on_dh(g, _k=dh_k):
                        wbar = w.detach().float().mean(dim=0)                # [D]
                        return (g.float() + (_k - 1.0) * _st.pop("s") * wbar).to(g.dtype)

                    y.register_hook(on_dlogits)
                    x.register_hook(on_dh)

                ol.register_forward_hook(fhook)
                if torch.distributed.get_rank() == 0:
                    print(f"[rr] dh gauge hook (k = {dh_k}) on {type(ol).__name__}", flush=True)
            if "wgrad-bf16" in self.arms:
                ol = gpt.output_layer
                # TE / Megatron 的线性层开 gradient_accumulation_fusion 时直接往 main_grad 累加 FP32 wgrad，拦不到；
                # 这里关掉该层的融合，让 wgrad 走 .grad，再用钩子舍成 BF16
                if hasattr(ol, "gradient_accumulation_fusion"):
                    ol.gradient_accumulation_fusion = False
                ol.weight.register_hook(lambda g: g.to(torch.bfloat16).to(g.dtype))
                if torch.distributed.get_rank() == 0:
                    print(f"[rr] output_layer wgrad -> BF16 rounding ({type(ol).__name__})", flush=True)
