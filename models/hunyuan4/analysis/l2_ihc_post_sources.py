"""单层里一处 FFN 前 iHC 参数梯度的来源拆分：流输入 x、分支输出 y 各自来自 GPU 还是 FP64 单层。

FFN 前 iHC 的 post 门控（hc_fn 后 n 行、hc_base 后 n 个）的梯度只经过
  d_post[t, k] = Σ_h y[t, h] · g[t, k, h]（g 为层输出的余切），
  d_post_logit = d_post · magnitude · σ(1 − σ)，d_fn_post = post_scale · Σ_t d_post_logit[t] ⊗ (x_flat[t] · rsqrt[t])，
pre 门控经过 FFN 分支输入的梯度。GPU 一侧的 x（`ffn_hc.in`）、y（`mlp.out`）、g（`out0.grad`）、分支输入梯度（`ffn_hc.out.grad`）
都在逐层抓数里（hy4_capture，同一行），用上面的式子在 FP64 里复算，等于 GPU 自己的参数梯度（复算的 post 门控与抓到的
`ffn_hc.post` 逐元素差 < 1e-6，可核）。FP64 单层的 x、y 由 `layer_ref.py` 跑时挂钩子存下（`--cap`，
每次 `hc_residual` 的 (post, 分支输出, 残差)；第 2 次是 FFN 那一处）。g 两边相同（单层回放吃同一份余切）。
把 x、y 逐一换成另一方的，看 post 门控梯度对 FP64 单层的 rel_L2 与沿真值方向的分量 par 跟着谁走。

    python l2_ihc_post_sources.py --gpu-l1 <step_1_fwd> --gpu-l1-bwd <step_1_bwd> --hf <HF 目录> --layer 3 \\
        --fp64 <layer_ref.safetensors> --cap <钩子存下的 safetensors> --cap-call 3 --out <json>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu-l1", required=True)
    p.add_argument("--gpu-l1-bwd", required=True)
    p.add_argument("--hf", required=True)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--fp64", required=True)
    p.add_argument("--cap", required=True)
    p.add_argument("--cap-call", type=int, required=True, help="钩子记录里这一层 FFN 那次 hc_residual 的序号")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    L = a.layer
    cfg = json.load(open(os.path.join(a.hf, "config.json")))
    n, eps, hce, mag = cfg["hc_mult"], cfg["rms_norm_eps"], cfg["hc_eps"], cfg["hc_magnitude"]
    f, b = safe_open(a.gpu_l1, "pt"), safe_open(a.gpu_l1_bwd, "pt")
    W = safe_open(os.path.join(a.hf, "model.safetensors"), "pt")
    P = f"model.layers.{L}.hc_mlp_layer.hc_pre."
    fn, base, scale = (W.get_tensor(P + x).double() for x in ("hc_fn", "hc_base", "hc_scale"))
    h = f"hf.model.layers.{L}"
    x_g = f.get_tensor(h + ".ffn_hc.in")[0].double()
    y_g = f.get_tensor(h + ".mlp.out")[0].double()
    g = b.get_tensor(h + ".out0.grad")[0].double()
    dout = b.get_tensor(h + ".ffn_hc.out.grad")[0].double()
    post_cap = f.get_tensor(h + ".ffn_hc.post")[0].double()
    C = safe_open(a.cap, "pt")
    x_f = C.get_tensor(f"c{a.cap_call}.res")[:, 0].double()
    y_f = C.get_tensor(f"c{a.cap_call}.y")[:, 0].double()
    Fr = safe_open(a.fp64, "pt")
    ref = {k: Fr.get_tensor(f"{L}.param.ffn_hc.{k}").double() for k in ("fn", "base", "scale")}

    def grads(x, y):
        s = x.shape[0]
        flat = x.reshape(s, -1)
        rs = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
        mix = (flat @ fn.T) * rs
        pl, ql = mix[:, :n], mix[:, n:]
        sp, sq = torch.sigmoid(pl * scale[0] + base[:n]), torch.sigmoid(ql * scale[1] + base[n:])
        dpost = (g * y[:, None, :]).sum(-1)
        dpre = (x * dout[:, None, :]).sum(-1)
        dqa, dpa = dpost * mag * sq * (1 - sq), dpre * sp * (1 - sp)
        dmix = torch.cat([dpa * scale[0], dqa * scale[1]], 1)
        return {"fn": dmix.T @ (flat * rs), "base": torch.cat([dpa.sum(0), dqa.sum(0)]),
                "scale": torch.stack([(dpa * pl).sum(), (dqa * ql).sum()])}, mag * sq + hce

    def rp(x, r):
        x, r = x.ravel().numpy(), r.ravel().numpy()
        e = x - r
        return {"rel": float(np.linalg.norm(e) / np.linalg.norm(r)), "par": float(e @ r / (r @ r))}

    _, post = grads(x_g, y_g)
    bf = lambda t: t.to(torch.bfloat16).double()  # noqa: E731
    combos = {"x_fp64,y_fp64": (x_f, y_f), "x_gpu,y_fp64": (x_g, y_f), "x_fp64,y_gpu": (x_f, y_g), "x_gpu,y_gpu": (x_g, y_g),
              "x_fp64,bf16(y_fp64)": (x_f, bf(y_f)), "bf16(x_fp64),y_fp64": (bf(x_f), y_f)}
    rows = {}
    for nm, (xx, yy) in combos.items():
        G, _ = grads(xx, yy)
        rows[nm] = {"hc_fn_post_rows": rp(G["fn"][n:], ref["fn"][n:]), "hc_base_post": rp(G["base"][n:], ref["base"][n:]),
                    "hc_fn": rp(G["fn"], ref["fn"]), "hc_base": rp(G["base"], ref["base"]), "hc_scale": rp(G["scale"], ref["scale"])}
        print(f"{nm:22s} hc_fn post rows rel {rows[nm]['hc_fn_post_rows']['rel']:.3e} par {rows[nm]['hc_fn_post_rows']['par']:+.3e}"
              f" | hc_base post rel {rows[nm]['hc_base_post']['rel']:.3e} par {rows[nm]['hc_base_post']['par']:+.3e}")
    out = {"what": f"单层第 {L} 层 FFN 前 iHC 参数梯度：流输入 x、分支输出 y 的来源拆分（GPU 抓数复算 vs FP64 单层）",
           "rule": "rel / par 都对 FP64 单层（layer_ref.py --param-grads）；par =（被测 − FP64）·FP64 ÷ ‖FP64‖²；g 两边同一份",
           "check": {"post_recompute_vs_capture_max_abs": float((post - post_cap).abs().max()),
                     "x_gpu_vs_fp64_rel": float((x_g - x_f).norm() / x_f.norm()),
                     "y_gpu_vs_fp64_rel": float((y_g - y_f).norm() / y_f.norm())},
           "rows": rows}
    print(json.dumps(out["check"]))
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
