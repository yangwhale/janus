"""B200 上 sigmoid / silu / exp 的实测精度（对 FP64）：单卡、几十秒。

    NVIDIA_TF32_OVERRIDE=0 python3 sigmoid_silu_probe.py      # 训练镜像里跑，打印一行 PROBE_JSON {...}

- FP32：2000 万个 [-30, 30] 均匀随机输入，报最大 / 平均 ulp 与正确舍入的比例；
- BF16 进 BF16 出：[-30, 30] 内全部 BF16 值，报与 FP64 结果正确舍入到 BF16 的一致比例；
- 组合：门控 out × sigmoid(gate)、不融合的 SwiGLU silu(a) × b（两次 BF16 舍入），以及 TE 融合 SwiGLU 作参照
  （训练路径上没用到 TE 融合激活，见 frameworks/megatron/README.md「sigmoid / silu 各处的实际算法」）。
"""

import json

import torch
import torch.nn.functional as F

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
try:
    import transformer_engine as te

    res["te"] = te.__version__
except Exception as e:
    res["te"] = repr(e)


def ulp32(ref64):
    r = ref64.float()
    nx = torch.nextafter(r.abs(), torch.tensor(float("inf"), device=r.device))
    return (nx - r.abs()).double()


def stats32(y, ref):
    u = ulp32(ref)
    e = (y.double() - ref).abs() / u
    return {
        "max_ulp": float(e.max()),
        "mean_ulp": float(e.mean()),
        "frac_correctly_rounded": float((y == ref.float()).float().mean()),
    }


def stats_bf16(y, ref):
    cr = ref.to(torch.bfloat16)  # FP64 -> BF16 correctly rounded (RNE)
    d = (y.float() - cr.float()).abs()
    nxb = (cr.float().abs().to(torch.bfloat16).view(torch.int16) + 1).view(torch.bfloat16).float() - cr.float().abs()
    return {"frac_equal_correctly_rounded": float((y == cr).float().mean()), "max_bf16_ulp": float((d / nxb).max())}


g = torch.Generator(device=dev).manual_seed(0)
x32 = torch.rand(20_000_000, device=dev, generator=g) * 60 - 30
# all bf16 values in [-30, 30]
allb = torch.arange(-32768, 32768, device=dev, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
allb = allb[torch.isfinite(allb.float()) & (allb.float().abs() <= 30)]
res["n_fp32_inputs"] = x32.numel()
res["n_bf16_inputs"] = allb.numel()
x64 = x32.double()
b64 = allb.double()
sig64 = lambda t: 1 / (1 + torch.exp(-t))
res["sigmoid_fp32"] = stats32(torch.sigmoid(x32), sig64(x64))
res["silu_fp32"] = stats32(F.silu(x32), x64 * sig64(x64))
res["exp_fp32_[-30,30]"] = stats32(torch.exp(x32), torch.exp(x64))
res["sigmoid_bf16_in_out"] = stats_bf16(torch.sigmoid(allb), sig64(b64))
res["silu_bf16_in_out"] = stats_bf16(F.silu(allb), b64 * sig64(b64))
# 门控：BF16 out × BF16 sigmoid(gate)
o = torch.randn(4_000_000, device=dev, generator=g).to(torch.bfloat16)
gt = (torch.randn(4_000_000, device=dev, generator=g) * 3).to(torch.bfloat16)
res["gate_out_x_sigmoid_bf16"] = stats_bf16(o * torch.sigmoid(gt), o.double() * sig64(gt.double()))
# SwiGLU 不融合（稠密 / 共享）：silu(a) BF16 × b BF16
a = (torch.randn(4_000_000, device=dev, generator=g) * 3).to(torch.bfloat16)
b = (torch.randn(4_000_000, device=dev, generator=g) * 3).to(torch.bfloat16)
res["swiglu_unfused_bf16"] = stats_bf16(F.silu(a) * b, sig64(a.double()) * a.double() * b.double())
# TE 融合 swiglu（我们的路径上没用到，作参照）
try:
    import transformer_engine.pytorch as tep

    op = tep.ops.SwiGLU()
    y = op(torch.stack([a, b], 1).contiguous())
    # TE SwiGLU: 输入最后一维 [gate | linear]，输出 silu(gate) * linear
    res["te_ops_SwiGLU_bf16"] = stats_bf16(y.view(-1), sig64(a.double()) * a.double() * b.double())
except Exception as e:
    res["te_ops_SwiGLU_bf16"] = repr(e)[:300]
print("PROBE_JSON " + json.dumps(res))
