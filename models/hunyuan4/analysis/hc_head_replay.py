"""hc_head + 末层 RMSNorm 的单算子回放：输入是 FP64 参照抓到的 x（4 路流）与 RMSNorm 输出的梯度 dy，
在 FP64 里算精确梯度，再按几种「TPU 可能的舍入位置」模拟，看 hc_head 三个参数的梯度离 FP64 多远。"""

import sys

import torch

torch.set_num_threads(32)
F64, F32, BF = torch.float64, torch.float32, torch.bfloat16


def rb(t):
    return t.to(BF).to(t.dtype)


class RoundGrad(torch.autograd.Function):  # 前向不变，反向把梯度舍成 bf16（JAX 里 astype(bf16) 的转置）
    @staticmethod
    def forward(ctx, t):
        return t.clone()

    @staticmethod
    def backward(ctx, g):
        return rb(g)


DY_EXACT = {}


class NormBwd(torch.autograd.Function):  # 末层 RMSNorm，反向按公式写出，可以单独把 S、r、c 舍成 bf16
    @staticmethod
    def forward(ctx, x, w, eps, how):
        r = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        ctx.save_for_backward(x, w, r)
        ctx.how = how
        ctx.eps_ = eps
        return x * r * w

    @staticmethod
    def backward(ctx, dy):
        x, w, r = ctx.saved_tensors
        how = ctx.how
        D = x.shape[-1]
        rr = rb(r) if "r_bf16" in how else r
        dn = dy * w
        S = (dn * x).sum(-1, keepdim=True)
        if "Sraw" in how:
            S = ((DY_EXACT["dy"].to(dy.dtype) * w) * x).sum(-1, keepdim=True)  # S 用没舍的余切（后端省掉那对 convert）
        if "S_bf16" in how:
            S = rb(S)
        c = -(r**3) * S / D
        if "mb_" in how:  # 反向里用的 (均方 + eps) 和前向不是同一个值
            m = x.square().mean(-1, keepdim=True) + ctx.eps_
            if "mb_rn" in how:
                mb = rb(m)
            elif "mb_tr" in how:
                mb = (m.view(torch.int32) & ~0xFFFF).view(torch.float32) if m.dtype == torch.float32 else m
            c = -r * S / (D * mb)
        if "c_noeps" in how:
            c = -r * S / x.square().sum(-1, keepdim=True)  # 反向里丢掉 eps：<dx, x> 恰好为 0
        if "c_bf16" in how:
            c = rb(c)
        dx = rr * dn + c * x
        return dx, None, None, None


class RoundFwd(torch.autograd.Function):  # 前向舍成 bf16，反向直通（MXU 默认精度对操作数的舍入，autodiff 看不见）
    @staticmethod
    def forward(ctx, t):
        return rb(t)

    @staticmethod
    def backward(ctx, g):
        return g


def run(r, mode):
    dt = F64 if mode.startswith("fp64") else F32
    x = r["x"].squeeze(1)
    dy = r["dy"].squeeze(1)
    DY_EXACT["dy"] = dy.clone()
    if mode != "fp64" and "xexact" not in mode:
        x = rb(x)
    if mode != "fp64" and "dyexact" not in mode:
        dy = rb(dy)
    x = x.to(dt)
    dy = dy.to(dt)
    fn = r["hc_fn"].to(dt).clone().requires_grad_(True)  # [k, k*d]
    base = r["hc_base"].to(dt).clone().requires_grad_(True)
    scale = r["hc_scale"].to(dt).clone().requires_grad_(True)
    w = r["norm_w"].to(dt)
    w = rb(w) if not mode.startswith("fp64") else w
    eps, ieps, heps = r["norm_eps"].item(), r["inner_eps"].item(), r["hc_eps"].item()
    s, kd = x.shape
    k = fn.shape[0]
    d = kd // k
    flat_norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + ieps)
    fl = RoundFwd.apply(flat_norm) if "mixbf16" in mode else flat_norm
    mixes = fl @ fn.T
    pre = torch.sigmoid(mixes * scale + base) + heps
    out = (pre.unsqueeze(-1) * x.view(s, k, d)).sum(1)
    if mode.startswith("fp64") or "castexact" in mode:
        xin = xms = out
    elif "noroundgrad" in mode:
        xin = xms = RoundFwd.apply(out)
    else:
        out_b = RoundFwd.apply(RoundGrad.apply(out))  # astype(bf16)：前向舍、梯度也舍
        if "mean_unrounded" in mode:  # 省掉来回：均方用没舍的 out，乘的是舍过的
            xin, xms = out_b, RoundGrad.apply(out)
        elif "mul_unrounded" in mode:
            xin, xms = RoundGrad.apply(out), out_b
        elif "nocast" in mode:  # 整段来回都省掉
            xin = xms = out
        else:
            xin = xms = out_b
    normed = xin * torch.rsqrt(xms.square().mean(-1, keepdim=True) + eps)
    if "manual" in mode:
        y = NormBwd.apply(xin, w, eps, mode)
    else:
        nm = RoundFwd.apply(normed) if "normbf16" in mode else normed
        y = nm * w
    (y * dy).sum().backward()
    return {"hc_head_base": base.grad.to(F64), "hc_head_scale": scale.grad.to(F64), "hc_head_fn": fn.grad.to(F64)}


if __name__ == "__main__":
    caps = torch.load(sys.argv[1])
    r0 = list(caps.values())[0]
    rows = list(zip(r0["xs"], r0["dys"])) if "xs" in r0 else [(r0["x"], r0["dy"])]
    modes = sys.argv[2].split(",")
    ref = None
    res = {}
    for m in modes:
        acc = None
        for x, dy in rows:
            g = run({**r0, "x": x, "dy": dy}, m)
            acc = g if acc is None else {kk: acc[kk] + g[kk] for kk in g}
        res[m] = acc
    ref = res["fp64"]
    for kk in ("grad_hc_base", "grad_hc_scale"):
        if kk in r0:
            print(
                "captured",
                kk,
                r0[kk].flatten()[:4].tolist(),
                "replay fp64",
                ref[kk.replace("grad_hc_", "hc_head_")].flatten()[:4].tolist(),
            )
    for m, g in res.items():
        print(
            f"{m:28s}",
            "  ".join(f"{kk.split('_')[-1]} rel {((g[kk] - ref[kk]).norm() / ref[kk].norm()).item():.3e}" for kk in g),
            " base",
            [f"{v:.4g}" for v in g["hc_head_base"].tolist()],
        )
    if len(sys.argv) > 3:
        torch.save(res, sys.argv[3])
