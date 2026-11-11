"""FP64 参照里模拟 TPU 强制 / 不强制 iHC 舍入的两种整体样子，看各层 KD（EMU=F 或 U）。
F（强制，GPU 的样子）：每处分支输入舍 BF16（均方与乘法一致），每次残差更新后的流舍 BF16。
U（TPU 不强制、按训练步 HLO 与 LLO 推断的样子）：第 0 层分支输入不舍；其余层均方用没舍的、乘法用舍过的；
   注意力那次残差更新后的流不舍（层内留 FP32），MLP 那次（层末，扫描的进位是 BF16）舍。"""

import os
import runpy
import sys

import torch

sys.path.insert(0, os.getcwd())
import hy4_mcore as H

MODE = os.environ["EMU"]
UNR = {}
CTX = {"first": False, "k": 0}


class _BF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t):
        with torch.utils._python_dispatch._pop_mode_temporarily():
            return t.detach().to(torch.bfloat16).to(t.dtype)

    @staticmethod
    def backward(ctx, g):
        return g


bf = _BF.apply
orig_layer = H.Hy4TransformerLayer.forward


def layer_fwd(self, *a, **k):
    CTX["first"] = self.layer_number == 1 and not self.is_mtp_layer
    CTX["k"] = 0
    return orig_layer(self, *a, **k)


H.Hy4TransformerLayer.forward = layer_fwd
orig_res = H.hc_residual


def res(post, h, residual):
    out = orig_res(post, h, residual)
    CTX["k"] += 1
    if MODE == "F" or (MODE == "U" and CTX["k"] == 2):
        return bf(out)
    return out


H.hc_residual = res
orig_hc = H.Hy4HyperConnection.forward


def hc_fwd(self, x):
    post, out = orig_hc(self, x)
    if MODE == "U" and CTX["first"]:
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
            u = UNR.pop(id(x), x)
            xs = u if MODE == "U" else x
            return x * torch.rsqrt(xs.square().mean(-1, keepdim=True) + mod.eps) * mod.weight

        mod.forward = f


H.Hy4TransformerLayer.__init__ = init
sys.argv = ["step1_ref.py"] + sys.argv[1:]
runpy.run_path("step1_ref.py", run_name="__main__")
