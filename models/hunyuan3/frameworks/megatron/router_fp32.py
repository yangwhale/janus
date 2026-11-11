"""路由真 FP32（train_hy3.py --router-true-fp32）：路由权重留在 FP32，路由 logits 的 GEMM 前向、反向都按 IEEE FP32 算。

**只作诊断（定责对照），不用于 golden**：golden 仍是生产配置（`moe_router_dtype=fp32` 走 TE general_gemm、TF32 开）。
用法是对照一次运行开 / 不开这个开关，看某个差有多少来自路由 GEMM 的精度。

不改 Megatron 源码，只做两处 janus 侧的替换：

1. 模型包 Float16Module 之前（Megatron-Bridge 的 pre_wrap_hook），把每个 TopKRouter 的 weight 转成 FP32，
   并标上 Bridge 自带的 `_keep_in_float32_parameter_names = ["weight"]`，包装时不被降成 BF16。
   于是路由权重是一个 FP32 参数：优化器把它放进 fp32_from_fp32_groups，这个参数本身就是主权重，
   没有另一份 BF16 拷贝；梯度也是 FP32 直接进 main_grad，不先舍到 BF16。
2. 把 `megatron.core.transformer.moe.router` 模块里的 `router_gating_linear` 换成下面的
   `router_gating_linear_true_fp32`：输入（BF16 激活）升 FP32，与 FP32 权重做 `torch.mm`；
   反向 dgrad、wgrad 也是 FP32 × FP32 的 `torch.mm`。dgrad 输出照原实现舍回输入 dtype（BF16），
   wgrad 保持权重 dtype（FP32）。

为什么用 FP32 参数（即主权重本身）而不是「把 BF16 参数升 FP32」：
- TPU 侧（MaxText GateLogit，`float32_gate_logits: True`）的核就是 FP32 主权重；GPU 用 BF16 参数升上来，
  第 1 步两者相同（检查点是 BF16，主权重从它初始化），第 2 步起差一次 BF16 舍入（每个元素相对 2^-9 量级），
  这比要消掉的 GEMM 误差（4e-6）大两三个数量级，等于把差从 GEMM 挪到了权重上。
- 升 BF16 参数时 wgrad 仍要舍成 BF16（autograd 要求梯度 dtype 与参数相同），TPU 的路由梯度是 FP32。

FP32 GEMM 要真是 IEEE FP32，必须关 TF32：本镜像只有 NVIDIA_TF32_OVERRIDE=0 关得干净
（docs/dev/design/0004-l1-gpu-kernels.md「一」），train_hy3.py 在这个开关下检查它。

探针：每个 rank 第一次训练迭代里每次路由 GEMM（前向 3 层 × 每 rank 2 个 microbatch、反向 dgrad / wgrad 同样多），
对 FP64 参照算相对 L2，写进 PROBE；train_hy3.py 在训练结束时把 rank 0 的结果写进 meta.json 的 `router.probe`。
前向另算一次「原 golden 写法」（TE general_gemm，BF16 入、FP32 出）对 FP64 的误差作对照：
第 1 步权重恰是 BF16 可表示的值，这一份与原 golden 逐位同法。探针只读不写，不改训练数值。
"""
import torch

PROBE: dict = {"fwd": [], "dgrad": [], "wgrad": []}
_NAMES: dict[int, str] = {}
_LEFT = {"fwd": 0, "bwd": 0}


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y.double() - ref).norm() / ref.norm())


def _ulp_stats(y: torch.Tensor, ref: torch.Tensor) -> dict:
    """FP32 输出对 FP64 真值的 ULP（以 FP64 真值舍到 FP32 后的间距为单位）与逐位相同率（对正确舍入）。"""
    r32 = ref.float()
    spacing = (torch.nextafter(r32.abs(), torch.full_like(r32, float("inf"))) - r32.abs()).double()
    ulp = ((y.double() - ref).abs() / spacing).flatten()
    sub = ulp[::max(1, ulp.numel() >> 20)]          # 等距抽样（torch.quantile 有元素数上限）；不用随机数，不动 RNG 状态
    q = torch.quantile(sub.float(), torch.tensor([0.5, 0.999], device=ulp.device))
    return {"exact_frac": float((y == r32).float().mean()), "ulp_p50": float(q[0]), "ulp_p999": float(q[1]),
            "ulp_max": float(ulp.max())}


class _TrueFP32RouterGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, router_dtype):
        if router_dtype != torch.float32:
            raise RuntimeError(f"--router-true-fp32 需要 moe_router_dtype=fp32，收到 {router_dtype}")
        ctx.save_for_backward(inp, weight, bias)
        ctx.inp_dtype, ctx.weight_dtype = inp.dtype, weight.dtype
        shape = inp.shape
        x = inp.reshape(-1, shape[-1]).float()
        w = weight.float()
        out = torch.mm(x, w.t()) if bias is None else torch.addmm(bias.float(), x, w.t())
        if _LEFT["fwd"] > 0:
            _LEFT["fwd"] -= 1
            _probe_fwd(inp.reshape(-1, shape[-1]), x, w, bias, out, weight)
        return out.view(*shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors
        x = inp.reshape(-1, inp.shape[-1]).float()
        w = weight.float()
        g = grad_output.reshape(-1, grad_output.shape[-1]).float()
        grad_input = torch.mm(g, w)
        grad_weight = torch.mm(g.t(), x)
        if _LEFT["bwd"] > 0:
            _LEFT["bwd"] -= 1
            name = _NAMES.get(id(weight), "?")
            ref_i = g.double() @ w.double()
            ref_w = g.double().t() @ x.double()
            PROBE["dgrad"].append({"param": name, "rel_l2_vs_fp64": _rel(grad_input, ref_i), **_ulp_stats(grad_input, ref_i)})
            PROBE["wgrad"].append({"param": name, "rel_l2_vs_fp64": _rel(grad_weight, ref_w), **_ulp_stats(grad_weight, ref_w)})
        grad_bias = g.sum(dim=0).to(bias.dtype) if bias is not None else None
        return (grad_input.to(ctx.inp_dtype).view(inp.shape), grad_weight.to(ctx.weight_dtype), grad_bias, None)


def _probe_fwd(inp2d, x, w, bias, out, weight):
    ref = x.double() @ w.double().t()
    if bias is not None:
        ref = ref + bias.double()
    rec = {"param": _NAMES.get(id(weight), "?"), "rel_l2_vs_fp64": _rel(out, ref), **_ulp_stats(out, ref)}
    try:   # 原 golden 写法对照：TE general_gemm，BF16 权重 × BF16 输入，FP32 出
        from transformer_engine.pytorch.cpp_extensions import general_gemm
        w16 = weight.to(torch.bfloat16)
        rec["weight_is_bf16_exact"] = bool((w16.float() == w).all())
        old = general_gemm(w16, inp2d.to(torch.bfloat16), torch.float32, layout="TN")[0]
        rec["te_bf16_gemm_rel_l2_vs_fp64"] = _rel(old, ref)
    except Exception as e:  # noqa: BLE001  对照失败不影响训练，原样记下
        rec["te_bf16_gemm_error"] = repr(e)[:200]
    PROBE["fwd"].append(rec)


def router_gating_linear_true_fp32(inp, weight, bias, router_dtype):
    return _TrueFP32RouterGemm.apply(inp, weight, bias, router_dtype)


def install(provider, probe_calls: int = 6) -> None:
    """在构建模型之前调用：换掉 router_gating_linear，并注册 pre_wrap_hook 把路由权重留在 FP32。
    probe_calls：前向、反向各探多少次（每 rank 第 1 步 3 层 × 2 个 microbatch = 6）。"""
    import megatron.core.transformer.moe.router as router_mod

    router_mod.router_gating_linear = router_gating_linear_true_fp32
    _LEFT["fwd"] = _LEFT["bwd"] = probe_calls

    def keep_router_fp32(model):
        n = 0
        for chunk in model:
            for name, sub in chunk.named_modules():
                if isinstance(sub, router_mod.TopKRouter):
                    sub.weight.data = sub.weight.data.float()
                    sub._keep_in_float32_parameter_names = ["weight"]
                    _NAMES[id(sub.weight)] = f"{name}.weight"
                    n += 1
        if n == 0:
            raise RuntimeError("--router-true-fp32：模型里没找到 TopKRouter")
        return model
    provider.register_pre_wrap_hook(keep_router_fp32)


def check_after_setup(params) -> list[str]:
    """训练开始时核对：路由权重确实是 FP32 参数。返回不是 FP32 的参数名（应为空）。"""
    return [n for n, p in params if n.endswith("mlp.router.weight") and p.dtype != torch.float32]
