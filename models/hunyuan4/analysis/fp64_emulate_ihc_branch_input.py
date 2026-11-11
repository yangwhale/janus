"""FP64 参照里只改 iHC「分支输入」那一处：EMU 环境变量选模式，看各层索引器 KD 怎么变（只跑前向、行 0）。
plain   不改
reimpl  重写那颗 RMSNorm、不舍（核对重写与原版一致）
round   分支输入舍成 BF16，均方与乘法都用舍过的（一致舍入，GPU / 强制舍入的样子）
ms_un   均方用没舍的、乘法用舍过的
mul_un  均方用舍过的、乘法用没舍的
"""

import os
import runpy
import sys

import torch

sys.path.insert(0, os.getcwd())
import hy4_mcore as H

MODE = os.environ["EMU"]
UNR = {}


class _BF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t):
        with torch.utils._python_dispatch._pop_mode_temporarily():
            return t.detach().to(torch.bfloat16).to(t.dtype)

    @staticmethod
    def backward(ctx, g):
        return g


def bf(t):
    return _BF.apply(t)


orig_hc = H.Hy4HyperConnection.forward


def hc_fwd(self, x):
    post, out = orig_hc(self, x)
    if MODE in ("plain", "reimpl"):
        return post, out
    r = bf(out)
    UNR[id(r)] = out
    return post, r


H.Hy4HyperConnection.forward = hc_fwd
orig_init = H.Hy4TransformerLayer.__init__


def init(self, *a, **k):
    orig_init(self, *a, **k)
    for nm in ("input_layernorm", "pre_mlp_layernorm"):
        mod = getattr(self, nm)

        def f(x, mod=mod):
            if MODE == "plain":
                return type(mod).forward(mod, x)
            u = UNR.pop(id(x), x)
            xm, xs = x, x
            if MODE == "ms_un":
                xs = u
            if MODE == "mul_un":
                xm = u
            return xm * torch.rsqrt(xs.square().mean(-1, keepdim=True) + mod.eps) * mod.weight

        mod.forward = f


H.Hy4TransformerLayer.__init__ = init
sys.argv = ["step1_ref.py"] + sys.argv[1:]
runpy.run_path("step1_ref.py", run_name="__main__")
