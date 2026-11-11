"""GPU（Megatron-Core autograd）里 iHC 的流梯度是怎么舍、按什么顺序加的：在 CPU 上用 GPU 同一段代码逐位核对。

hy4_mcore 的一层里，BF16 的流 x 有三处用到：`residual = x.view(...)`（残差，最先建）、iHC 门控的 `flat = x.float()`、
加权和里的 `pre.unsqueeze(-1) * x.view(...)`（FP32 × BF16，结果 FP32）。反向时三处各给 x 一份 BF16 梯度
（后两处是 FP32 梯度舍成 BF16），autograd 把它们逐个加进 x 的 BF16 梯度，每加一次舍一次。这里用 hook 取出三份梯度，
和 x.grad 逐位比几种加法顺序；另报分支输出（BF16）的梯度是否落在 BF16 上。

结论（torch 2.4.1、2.11.0 的 CPU autograd 相同；GPU 镜像是 2.14 的 NVIDIA 版，用同一套 autograd 引擎，推断相同）：
x.grad = r(r(d_sum + r(d_flat)) + d_res)，r 为舍成 BF16，100% 逐位相同；建 `residual` 的位置一变，分组就变。

    python ihc_stream_grad_order_probe.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def main():
    torch.manual_seed(0)
    s, b, n, H = 64, 2, 4, 256
    bf = torch.bfloat16

    def r(t):
        return t.to(bf).float()

    x = torch.randn(s, b, n * H).to(bf).requires_grad_(True)
    fn = torch.randn(2 * n, n * H) * 0.02
    base = torch.randn(2 * n) * 0.1
    scale = torch.tensor([0.5, 0.7])
    w = torch.randn(H).to(bf)
    g = torch.randn(s, b, n, H)
    grads = {}

    def hook(name):
        def f(t):
            grads[name] = t.detach().clone()

        return f

    # 与 Hy4TransformerLayer.forward + Hy4HyperConnection + hc_residual 相同的建图顺序
    residual = x.view(s, b, n, H)
    residual.register_hook(hook("res"))
    flat = x.float()
    flat.register_hook(hook("flat"))
    mixes = F.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    pre_b, post_b = base.split(n, -1)
    pre_s, post_s = scale.unbind(0)
    pre_l, post_l = mixes.split(n, -1)
    pre = torch.sigmoid(pre_l * pre_s + pre_b) + 1e-6
    post = 2 * torch.sigmoid(post_l * post_s + post_b) + 1e-6
    xv = x.view(s, b, n, H)
    xv.register_hook(hook("sum"))
    li = torch.sum(pre.unsqueeze(-1) * xv, dim=2).to(bf)
    lo = li * w  # 分支（BF16）
    lo.register_hook(hook("lo"))
    h = (post.float().unsqueeze(-1) * lo.float().unsqueeze(-2) + residual.float()).to(bf)
    (h.float() * g).sum().backward()
    gx = x.grad.float().view(s, b, n, H)
    a, bs, c = grads["res"].float(), grads["sum"].float(), r(grads["flat"].view(s, b, n, H))
    print("torch", torch.__version__, "梯度 dtype：res", grads["res"].dtype, "sum", grads["sum"].dtype,
          "flat", grads["flat"].dtype, "分支输出", grads["lo"].dtype)
    cands = {
        "r(r(d_sum + r(d_flat)) + d_res)": r(r(bs + c) + a),
        "r(r(d_res + d_sum) + r(d_flat))": r(r(a + bs) + c),
        "r(r(d_res + r(d_flat)) + d_sum)": r(r(a + c) + bs),
        "r(d_sum + d_flat + d_res)（FP32 加完舍一次）": r(bs + grads["flat"].view(s, b, n, H) + a),
    }
    for k, v in cands.items():
        print(f"{k:45s} 与 x.grad 逐位相同的比例 {float((v == gx).float().mean()):.6f}")


if __name__ == "__main__":
    main()
