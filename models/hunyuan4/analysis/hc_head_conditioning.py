"""hc_head_base 梯度的病态程度：跨 token 相消的倍数、逐 token Σ_k pre_k·g_k 被末层 RMSNorm 尺度不变性压低多少。

python models/hunyuan4/analysis/hc_head_conditioning.py <cap.pt>
"""

import sys

import torch

torch.set_num_threads(32)
caps = torch.load(sys.argv[1])
r = list(caps.values())[0]
xs = r.get("xs", [r.get("x")])
dys = r.get("dys", [r.get("dy")])
k = 4
tot_abs = 0
tot = 0
ratio = []
for x, dy in zip(xs, dys):
    x = x.squeeze(1).double().requires_grad_(False)
    dy = dy.squeeze(1).double()
    out = None
    fn = r["hc_fn"].double()
    base = r["hc_base"].double()
    scale = r["hc_scale"].double()
    s, kd = x.shape
    d = kd // k
    fl = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + r["inner_eps"].item())
    z = fl @ fn.T * scale + base
    sg = torch.sigmoid(z)
    pre = sg + r["hc_eps"].item()
    out = (pre.unsqueeze(-1) * x.view(s, k, d)).sum(1).requires_grad_(True)
    eps = r["norm_eps"].item()
    w = r["norm_w"].double()
    y = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + eps) * w
    (y * dy).sum().backward()
    dout = out.grad
    g = (dout.unsqueeze(1) * x.view(s, k, d)).sum(-1)  # [s,k]
    contrib = g * sg * (1 - sg)  # 每个 token 对 hc_base 的贡献
    tot = tot + contrib.sum(0)
    tot_abs = tot_abs + contrib.abs().sum(0)
    ratio.append(((pre * g).sum(1).abs() / (dout.norm(dim=1) * out.norm(dim=1))).median().item())
    ms = out.detach().square().mean(-1)
print("hc_base 梯度（本批）", tot.tolist())
print("Σ|逐 token 贡献| / |Σ 贡献| （跨 token 相消的倍数）", (tot_abs / tot.abs()).tolist())
print(
    "逐 token |Σ_k pre_k g_k| / (|dout||out|) 中位数（RMSNorm 尺度不变性的压低）",
    ratio[:4],
    "eps/(ms+eps) 中位数",
    (eps / (ms + eps)).median().item(),
)
