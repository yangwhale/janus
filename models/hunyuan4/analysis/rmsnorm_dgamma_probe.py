"""B200 上 TE RMSNorm 反向的权重梯度 dγ：累加精度与落地 dtype（单卡、几十秒）。

    NVIDIA_TF32_OVERRIDE=0 python3 rmsnorm_dgamma_probe.py      # 训练镜像里跑，打印一行 PROBE_JSON {...}

[4096, 1536]（q latent 的形状），权重分别取 BF16（训练里的情形）与 FP32，与 CPU 上三种写法逐位比：
FP64 精确和再舍到权重 dtype、FP32 按 token 依次累加、每 32 行 FP32 求和后块间 BF16 累加。
"""

import glob
import json
import os

import torch
import transformer_engine as te
import transformer_engine.pytorch as tep

res = {
    "gpu": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "te": te.__version__,
    "env": {k: v for k, v in os.environ.items() if k.startswith("NVTE_")},
}
root = os.path.dirname(te.__file__)
src = sorted(
    set(glob.glob("/opt/**/rmsnorm*bwd*", recursive=True) + glob.glob(root + "/**/*rmsnorm*", recursive=True))
)[:40]
res["rmsnorm_files"] = src
for f in src:
    if f.endswith((".cuh", ".cu", ".h")):
        txt = open(f, errors="ignore").read()
        res.setdefault("src_snippets", {})[f] = [
            l.strip() for l in txt.splitlines() if "dgamma" in l.lower() or "dz_sum" in l.lower()
        ][:60]
torch.manual_seed(0)
T, H = 4096, 1536
out = {}
for wdt in (torch.bfloat16, torch.float32):
    m = tep.RMSNorm(H, eps=1e-5, params_dtype=wdt).cuda()
    with torch.no_grad():
        m.weight.copy_((1 + 0.1 * torch.randn(H)).to(wdt))
    x = (torch.randn(T, H) * 2).to(torch.bfloat16).cuda().requires_grad_(True)
    dy = torch.randn(T, H).to(torch.bfloat16).cuda()
    y = m(x)
    y.backward(dy)
    g = m.weight.grad
    # FP64 参照：x̂ = x · rsqrt(mean(x²) + eps)（x 从 BF16 精确转），dγ = Σ_t dy·x̂
    xd, dyd = x.detach().double().cpu(), dy.double().cpu()
    rs = torch.rsqrt((xd**2).mean(-1, keepdim=True) + 1e-5)
    xhat = xd * rs
    exact = (dyd * xhat).sum(0)
    # FP32 写法：x̂ 与乘积在 FP32 里，按 token 顺序 FP32 累加
    xh32 = x.detach().float().cpu() * torch.rsqrt((x.detach().float().cpu() ** 2).mean(-1, keepdim=True) + 1e-5)
    prod32 = dy.float().cpu() * xh32
    seq32 = torch.zeros(H, dtype=torch.float32)
    for t in range(T):
        seq32 += prod32[t]
    # 逐块 BF16：每 32 行在 FP32 里求和，块间在 BF16 里累加
    blk = torch.zeros(H, dtype=torch.bfloat16)
    for s in range(0, T, 32):
        blk = (blk.float() + prod32[s : s + 32].sum(0)).to(torch.bfloat16)
    gc = g.detach().cpu()

    def eq(a):
        return float((gc == a.to(gc.dtype)).float().mean())

    def rel(a):
        return float((gc.double() - a.double()).norm() / a.double().norm())

    out[str(wdt)] = {
        "grad_dtype": str(g.dtype),
        "eq_round(FP64 exact)": eq(exact),
        "eq_round(FP32 seq)": eq(seq32),
        "eq_chunked_bf16": eq(blk),
        "relL2_vs_FP64": rel(exact),
        "relL2_bf16(FP64)_vs_FP64": float((exact.to(torch.bfloat16).double() - exact).norm() / exact.norm()),
        "relL2_FP32seq_vs_FP64": float((seq32.double() - exact).norm() / exact.norm()),
        "relL2_chunked_bf16_vs_FP64": float((blk.double() - exact).norm() / exact.norm()),
    }
res["dgamma"] = out
# 重跑是否确定
m = tep.RMSNorm(H, eps=1e-5, params_dtype=torch.bfloat16).cuda()
x = (torch.randn(T, H) * 2).to(torch.bfloat16).cuda().requires_grad_(True)
dy = torch.randn(T, H).to(torch.bfloat16).cuda()
m(x).backward(dy)
g1 = m.weight.grad.clone()
m.weight.grad = None
m(x).backward(dy)
res["rerun_bitwise"] = bool(torch.equal(g1, m.weight.grad))
print("PROBE_JSON " + json.dumps(res))
