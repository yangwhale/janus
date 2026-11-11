"""在 CPU FP64 参照（frameworks/megatron/step1_ref.py）上抓 hc_head + 末层 RMSNorm 的输入与梯度，给 hc_head_replay.py 用。

    cd models/hunyuan4/frameworks/megatron
    CAP_OUT=<cap.pt> python ../../analysis/fp64_capture_final_norm.py --hf <hf-mini> --tokens <tokens.npy> --dtype float64 --out <目录>

每行前向记下 hc_head 输入流 xs、hc_head 输出 outs、RMSNorm 输出的梯度 dys；全部行跑完后存参数与参数梯度。"""

import os
import runpy
import sys

import torch

sys.path.insert(0, os.getcwd())
import hy4_mcore as H

SAVE = {}
orig = H.Hy4FinalNorm.forward


def fwd(self, x):
    name = f"fn{id(self)}"
    s, b, _ = x.shape
    with torch.autocast(device_type=x.device.type, enabled=False):
        flat = x.float()
        mixes = torch.nn.functional.linear(flat, self.hc_fn.float()) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.norm_eps
        )
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.hc_eps
        out = (pre.unsqueeze(-1) * x.view(s, b, self.n, self.H)).sum(dim=2).to(x.dtype)
    y = self.norm(out)
    rec = SAVE.setdefault(name, {})
    rec.setdefault("xs", []).append(x.detach().clone())
    rec.setdefault("outs", []).append(out.detach().clone())
    rec["norm_w"] = self.norm.weight.detach().clone()
    rec["norm_eps"] = torch.tensor(float(getattr(self.norm, "eps", self.norm_eps)))
    rec["hc_fn"] = self.hc_fn.detach().clone()
    rec["hc_base"] = self.hc_base.detach().clone()
    rec["hc_scale"] = self.hc_scale.detach().clone()
    rec["hc_eps"] = torch.tensor(float(self.hc_eps))
    rec["inner_eps"] = torch.tensor(float(self.norm_eps))
    mod = self

    def hook(g):
        rec.setdefault("dys", []).append(g.detach().clone())
        return g

    y.register_hook(hook)
    rec["_mod"] = mod
    return y


H.Hy4FinalNorm.forward = fwd
import atexit


def dump():
    out = {}
    for k, r in SAVE.items():
        m = r.pop("_mod")
        for p in ("hc_fn", "hc_base", "hc_scale"):
            g = getattr(m, p).grad
            if g is not None:
                r["grad_" + p] = g.detach().clone()
        out[k] = r
    torch.save(out, os.environ["CAP_OUT"])
    print("CAPTURED", {k: sorted(v) for k, v in out.items()})


atexit.register(dump)
sys.argv = ["step1_ref.py"] + sys.argv[1:]
runpy.run_path("step1_ref.py", run_name="__main__")
