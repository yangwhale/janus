"""hc_head + 末层 RMSNorm 这一段反向：GPU（Hy4FinalNorm 的 autograd）与 MaxText（M52、M58 各种组合）在 CPU 上逐位对照。

同一份输入（BF16 的四路流 x、hc_head 三个参数、末层 norm 权重、输出头与 MTP 两路给 norm 输出的梯度 dy_head、dy_mtp），
三方各算一遍这一段的反向，逐级比：
- g_y：norm 输出处的梯度（GPU：两路在 BF16 里相加；MaxText 默认两次 norm 调用，没有这一级）；
- g_out：进 hc_head 收拢输出的梯度（norm 反向之后）；
- c_sum / c_gate：流的两份梯度（加权和那一路、门控那一路）；
- dx：流的梯度（两份相加之后）；以及 hc_head 三个参数与 norm 权重的梯度。

三方：
- gpu：PyTorch，Hy4FinalNorm.forward 的写法逐行照抄（`hy4_mcore.py`），末层 norm 用 TE 融合 RMSNorm 的语义
  （FP32 算、输出舍一次 BF16；反向从 BF16 的 dy 在 FP32 里算、dx 输出 BF16、dγ FP32）。CPU 上没有 TE，这一处是仿写，
  相加顺序与 TE kernel 不同属于「累加顺序」一类；
- maxtext：MaxText 真模块（`mhc.DeepSeek4HyperHead` + `normalizations.RMSNorm(single_rounding)`），CPU 上 jit；
  MTP 那一路照 MaxText 默认再调一次 decoder_norm，或照 `mtp_input_shared_final_norm` 共用一次；
- elide：同一段的 MaxText 写法，但把 TPU 在省舍入（excess precision）下可能省掉的 BF16 往返去掉（hc_head 输出不转 BF16
  就进 norm、norm 的 dx 与两路相加都留 FP32），用来推断 TPU 上的行为（推断：TPU 实际省掉哪几处要看 LLO）。
另给 FP64 参照（GPU 写法全 FP64）。

    python hc_head_order_probe.py inputs --out <dir>              # 任意环境（numpy）
    python hc_head_order_probe.py gpu --dir <dir>                 # 要 torch
    python hc_head_order_probe.py maxtext --dir <dir> --maxtext-src <maxtext>/src   # 要 jax + maxtext
    python hc_head_order_probe.py compare --dir <dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

S, B, K, D = 128, 2, 4, 256
EPS_NORM = 1e-5  # hy4-mini：rms_norm_eps、hc_head_norm_epsilon 都是 1e-5
EPS_HC = 1e-6  # hc_eps
MATERIAL = 1e-4  # 对 GPU 的 rel_L2 超过它算「实质不同」（一次 BF16 舍入的量级是 2^-9 ≈ 2e-3；只差 FP32 累加顺序时在 1e-5 量级）
STAGES = ("g_y", "g_out", "c_sum", "c_gate", "dx", "d_hc_fn", "d_hc_base", "d_hc_scale", "d_norm_w")


def _bf16_round(a):
    import ml_dtypes

    return np.asarray(np.asarray(a, np.float32).astype(ml_dtypes.bfloat16), np.float32)


def make_inputs(out: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    np.savez(
        os.path.join(out, "inputs.npz"),
        x=_bf16_round(rng.standard_normal((S, B, K, D))),
        hc_fn=(rng.standard_normal((K, K * D)) * 0.02).astype(np.float32),  # GPU 形状 [n, n·H]
        hc_base=np.full((K,), -np.log(K - 1), np.float32) + (rng.standard_normal(K) * 0.1).astype(np.float32),
        hc_scale=np.asarray([0.01 + 0.005 * rng.standard_normal()], np.float32),
        norm_w=_bf16_round(1.0 + 0.1 * rng.standard_normal(D)),  # 模型权重 BF16
        dy_head=_bf16_round(rng.standard_normal((S, B, D)) * 1e-4),
        dy_mtp=_bf16_round(rng.standard_normal((S, B, D)) * 1e-5),  # MTP loss 系数 0.1
    )


# ------------------------------------------------------------------ GPU（PyTorch）


def run_gpu(d: str):
    import torch

    inp = dict(np.load(os.path.join(d, "inputs.npz")))
    out = {}
    for dtype_name, dt in (("gpu", torch.bfloat16), ("fp64", torch.float64)):
        hi = torch.float64 if dtype_name == "fp64" else torch.float32

        class TENorm(torch.autograd.Function):
            """TE 融合 RMSNorm 的语义：FP32 算、输出舍一次；反向 dx 在 FP32 里算、输出原精度，dγ FP32。"""

            @staticmethod
            def forward(ctx, x, w):
                xf, wf = x.to(hi), w.to(hi)
                rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS_NORM)
                ctx.save_for_backward(xf, wf, rs)
                return (xf * rs * wf).to(x.dtype)

            @staticmethod
            def backward(ctx, dy):
                xf, wf, rs = ctx.saved_tensors
                dyf = dy.to(hi)
                xhat = xf * rs
                g = dyf * wf
                dx = rs * (g - xhat * (g * xhat).mean(-1, keepdim=True))
                dw = (dyf * xhat).sum(dim=(0, 1))
                return dx.to(dy.dtype), dw.to(hi)

        x = torch.tensor(inp["x"].reshape(S, B, K * D), dtype=dt, requires_grad=True)
        hc_fn = torch.tensor(inp["hc_fn"], dtype=hi, requires_grad=True)
        hc_base = torch.tensor(inp["hc_base"], dtype=hi, requires_grad=True)
        hc_scale = torch.tensor(inp["hc_scale"], dtype=hi, requires_grad=True)
        norm_w = torch.tensor(inp["norm_w"], dtype=dt, requires_grad=True)
        cap = {}

        def hook(name):
            def f(t):
                cap[name] = t.detach().clone()

            return f

        # ---- Hy4FinalNorm.forward（hy4_mcore.py，逐行）----
        flat = (
            x.to(hi, copy=True) if dtype_name == "fp64" else x.float()
        )  # FP64 时 .to 不拷贝，要单独一份才能取门控那一路
        flat.register_hook(hook("c_gate"))
        mixes = torch.nn.functional.linear(flat, hc_fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + EPS_NORM)
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + EPS_HC
        xv = x.view(S, B, K, D)
        xv.register_hook(hook("c_sum"))
        o = (pre.unsqueeze(-1) * xv).sum(dim=2)
        o = o.to(x.dtype)
        o.register_hook(hook("g_out"))
        y = TENorm.apply(o, norm_w)
        y.register_hook(hook("g_y"))
        # 输出头与 MTP 两路都用同一个 norm 输出，梯度在 y 的 dtype 里相加
        loss = (y.to(hi) * torch.tensor(inp["dy_head"], dtype=hi)).sum() + (
            y.to(hi) * torch.tensor(inp["dy_mtp"], dtype=hi)
        ).sum()
        loss.backward()
        out[dtype_name] = {
            "g_y": cap["g_y"],
            "g_out": cap["g_out"],
            "c_sum": cap["c_sum"].reshape(S, B, K, D),
            "c_gate": cap["c_gate"].reshape(S, B, K, D),
            "dx": x.grad.reshape(S, B, K, D),
            "d_hc_fn": hc_fn.grad.T,  # 存成 MaxText 的形状 [n·H, n]
            "d_hc_base": hc_base.grad,
            "d_hc_scale": hc_scale.grad,
            "d_norm_w": norm_w.grad,
        }
    for name, v in out.items():
        np.savez(os.path.join(d, f"{name}.npz"), **{k: t.to(torch.float64).numpy() for k, t in v.items()})
    print("gpu / fp64 done, torch", torch.__version__)


# ------------------------------------------------------------------ MaxText（JAX）


def run_maxtext(d: str, src: str):
    sys.path.insert(0, src)
    import types

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.layers import mhc, normalizations
    from maxtext.utils import accurate_math

    jax.config.update("jax_platforms", "cpu")
    # XLA 的省舍入开关：false 时 CPU 上逐个 convert 都照做（MaxText 写法的语义）；默认 true 时 XLA 的代数化简会把
    # FP32 → BF16 → FP32 这种往返省掉（与 TPU 同一类化简，融合方式不同），用来看编译器实际省掉了哪几处
    xla = "xoff" if "xla_allow_excess_precision=false" in os.environ.get("XLA_FLAGS", "") else "xon"
    inp = dict(np.load(os.path.join(d, "inputs.npz")))
    bf = jnp.bfloat16
    x = jnp.asarray(inp["x"].transpose(1, 0, 2, 3), bf)  # MaxText 的布局 [b, s, k, d]
    dy_head = jnp.asarray(inp["dy_head"].transpose(1, 0, 2), jnp.float32)
    dy_mtp = jnp.asarray(inp["dy_mtp"].transpose(1, 0, 2), jnp.float32)

    def cfg(**kw):
        c = types.SimpleNamespace(
            dtype=bf,
            weight_dtype=jnp.float32,
            mhc_expansion_rate=K,
            emb_dim=D,
            hc_head_norm_epsilon=EPS_NORM,
            hc_eps=EPS_HC,
            hc_matmul_precision="highest",
            matmul_precision="default",
            hc_head_base_init=0.0,
            transcendental_accuracy_highest=True,
            hc_head_force_bf16_rounding=False,
            hc_head_force_bf16_rounding_forward_only=False,
            ihc_stream_grad_gpu_order=False,
            ihc_stream_grad_gpu_order_parts="all",
            ihc_stream_grad_gpu_order_skip_hc_head=False,
        )
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def build(c):
        h = mhc.DeepSeek4HyperHead(c, None, nnx.Rngs(0))
        h.hc_fn[...] = jnp.asarray(inp["hc_fn"].T)
        h.hc_base[...] = jnp.asarray(inp["hc_base"])
        h.hc_scale[...] = jnp.asarray(inp["hc_scale"])
        n = normalizations.RMSNorm(
            D, epsilon=EPS_NORM, dtype=bf, weight_dtype=jnp.float32, single_rounding=True, rngs=nnx.Rngs(0)
        )
        n.scale[...] = jnp.asarray(inp["norm_w"])
        return h, n

    def collapse(c, p, x_sum, x_gate):
        """DeepSeek4HyperHead.__call__ 的数学（mhc.py），两路输入分开，用来取每一份的梯度；fanout 由调用方决定。"""
        b_, s_, k_, d_ = x_gate.shape
        flat_f32 = jnp.reshape(x_gate, (b_, s_, k_ * d_)).astype(jnp.float32)
        var = jnp.mean(jnp.square(flat_f32), axis=-1, keepdims=True)
        flat_norm = flat_f32 * jax.lax.rsqrt(var + EPS_NORM)
        mixes = jnp.einsum("bsm,mk->bsk", flat_norm, p["hc_fn"], precision=jax.lax.Precision.HIGHEST)
        pre = accurate_math.sigmoid(mixes * p["hc_scale"][None, None, :] + p["hc_base"][None, None, :], True) + EPS_HC
        return jnp.sum(pre[:, :, :, None] * x_sum.astype(jnp.float32), axis=2)

    def norm_math(o, w):
        """RMSNorm(single_rounding) 的数学（normalizations.py）：FP32 算，scale 先 reduce_precision 到 BF16。"""
        o = jnp.asarray(o, jnp.float32)
        m2 = jnp.mean(jax.lax.square(o), axis=-1, keepdims=True)
        fi = jnp.finfo(bf)
        return (
            o * jax.lax.rsqrt(m2 + EPS_NORM) * jax.lax.reduce_precision(jnp.asarray(w, jnp.float32), fi.nexp, fi.nmant)
        )

    variants = {
        "base": dict(),
        "m52": dict(hc_head_force_bf16_rounding=True),
        "m58": dict(ihc_stream_grad_gpu_order=True),
        "c": dict(hc_head_force_bf16_rounding=True, ihc_stream_grad_gpu_order=True),
        "c_fwdonly": dict(
            hc_head_force_bf16_rounding=True,
            hc_head_force_bf16_rounding_forward_only=True,
            ihc_stream_grad_gpu_order=True,
        ),
        "c_skiphead": dict(
            hc_head_force_bf16_rounding=True,
            ihc_stream_grad_gpu_order=True,
            ihc_stream_grad_gpu_order_skip_hc_head=True,
        ),
    }
    res = {}
    for vname, kw in variants.items():
        for shared in (False, True):
            for elide in (False, True) if xla == "xoff" else (False,):
                c = cfg(**kw)
                h, n = build(c)
                p = {
                    "hc_fn": jnp.asarray(h.hc_fn[...]),
                    "hc_base": jnp.asarray(h.hc_base[...]),
                    "hc_scale": jnp.asarray(h.hc_scale[...]),
                    "norm_w": jnp.asarray(n.scale[...]),
                }
                m52 = c.hc_head_force_bf16_rounding
                fwd_only = c.hc_head_force_bf16_rounding_forward_only
                fan = c.ihc_stream_grad_gpu_order and not c.ihc_stream_grad_gpu_order_skip_hc_head

                def head_out(p, xs, xg, m52=m52, fwd_only=fwd_only, elide=elide):
                    o = collapse(c, p, xs, xg)
                    if m52:
                        o = mhc.round_forward_only(o, bf) if fwd_only else mhc._round_to(o, bf)
                    return o if elide else o.astype(bf)  # elide：TPU 把 astype 往返省掉，norm 吃 FP32

                def loss(p, xs, xg):
                    o = head_out(p, xs, xg)
                    if shared:
                        yv = norm_math(o, p["norm_w"])
                        yv = yv if elide else yv.astype(bf)
                        return (
                            jnp.sum(yv.astype(jnp.float32) * (dy_head + dy_mtp))
                            if elide
                            else (jnp.sum(yv.astype(jnp.float32) * dy_head) + jnp.sum(yv.astype(jnp.float32) * dy_mtp))
                        )
                    y1 = norm_math(o, p["norm_w"])
                    y2 = norm_math(o, p["norm_w"])
                    if not elide:
                        y1, y2 = y1.astype(bf), y2.astype(bf)
                    return jnp.sum(y1.astype(jnp.float32) * dy_head) + jnp.sum(y2.astype(jnp.float32) * dy_mtp)

                def total(p, xv):
                    if fan:
                        xs, xg = mhc.stream_fanout(xv, 2)
                    else:
                        xs = xg = xv
                    return loss(p, xs, xg)

                # elide 且不开 M58：TPU 上流的两份梯度可能在 FP32 里加完才舍（对 FP32 的 x 取梯度，最后舍一次）
                x_total = x.astype(jnp.float32) if (elide and not fan) else x
                gp, gx = jax.jit(jax.grad(total, argnums=(0, 1)))(p, x_total)
                gx = mhc._round_to(jnp.asarray(gx, jnp.float32), bf)
                # 每一份流梯度：对两路分开的输入取梯度（输入是 BF16，梯度也是 BF16；elide 时按 FP32 输入取，再看是否舍）
                xin = x if not elide else x.astype(jnp.float32)  # elide：两份梯度不舍
                gxs, gxg = jax.jit(jax.grad(lambda a, b_: loss(p, a, b_), argnums=(0, 1)))(xin, xin)
                # g_out：进收拢输出的梯度
                o_val = jax.jit(lambda p: head_out(p, x, x))(p)

                def from_out(o):
                    if shared:
                        yv = norm_math(o, p["norm_w"])
                        yv = yv if elide else yv.astype(bf)
                        return jnp.sum(yv.astype(jnp.float32) * dy_head) + jnp.sum(yv.astype(jnp.float32) * dy_mtp)
                    y1, y2 = norm_math(o, p["norm_w"]), norm_math(o, p["norm_w"])
                    if not elide:
                        y1, y2 = y1.astype(bf), y2.astype(bf)
                    return jnp.sum(y1.astype(jnp.float32) * dy_head) + jnp.sum(y2.astype(jnp.float32) * dy_mtp)

                g_out = jax.jit(jax.grad(from_out))(o_val)
                if m52 and not fwd_only:
                    g_out = mhc._round_to(jnp.asarray(g_out, jnp.float32), bf)  # M52 的转置舍余切
                tr = lambda a: np.asarray(jnp.asarray(a, jnp.float32), np.float64)
                key = f"{xla}_{vname}{'_shared' if shared else ''}{'_elide' if elide else ''}"
                res[key] = {
                    "g_out": tr(g_out).transpose(1, 0, 2),
                    "c_sum": tr(gxs).transpose(1, 0, 2, 3),
                    "c_gate": tr(gxg).transpose(1, 0, 2, 3),
                    "dx": tr(gx).transpose(1, 0, 2, 3),
                    "d_hc_fn": tr(gp["hc_fn"]),
                    "d_hc_base": tr(gp["hc_base"]),
                    "d_hc_scale": tr(gp["hc_scale"]),
                    "d_norm_w": tr(gp["norm_w"]),
                }
                # 用真模块核一遍：semantic 写法的 dx 应与 MaxText 模块（norm 调两次或一次）逐位相同
                if not elide:

                    def mod_loss(xv, h=h, n=n):
                        o = h(xv)
                        if shared:
                            yv = n(o)
                            return jnp.sum(yv.astype(jnp.float32) * dy_head) + jnp.sum(yv.astype(jnp.float32) * dy_mtp)
                        return jnp.sum(n(o).astype(jnp.float32) * dy_head) + jnp.sum(n(o).astype(jnp.float32) * dy_mtp)

                    gm = np.asarray(jax.jit(jax.grad(mod_loss))(x).astype(jnp.float32), np.float64).transpose(
                        1, 0, 2, 3
                    )
                    res[key]["module_dx_equal"] = np.asarray(float(np.array_equal(gm, res[key]["dx"])))
    for key, v in res.items():
        np.savez(os.path.join(d, f"mt_{key}.npz"), **v)
    print("maxtext variants:", ", ".join(res), "jax", jax.__version__, "XLA_FLAGS", os.environ.get("XLA_FLAGS", ""))


# ------------------------------------------------------------------ 对照


def compare(d: str, out: str):
    gpu = dict(np.load(os.path.join(d, "gpu.npz")))
    ref = dict(np.load(os.path.join(d, "fp64.npz")))
    names = sorted(f[3:-4] for f in os.listdir(d) if f.startswith("mt_") and f.endswith(".npz"))
    rows = {}
    order = ("g_out", "c_sum", "c_gate", "dx", "d_hc_fn", "d_hc_base", "d_hc_scale", "d_norm_w")
    for nm in names:
        mt = dict(np.load(os.path.join(d, f"mt_{nm}.npz")))
        r = {}
        first = first_material = None
        manual_elide = nm.endswith("_elide")
        for st in order:
            a, g, f = mt[st], gpu[st], ref[st]
            if st == "c_gate" and not manual_elide:
                g = _bf16_round(g)  # GPU 的 d_flat 在 x.float() 的反向里转成 BF16 才进流的梯度
            eq = float(np.mean(a == g))
            rel_gpu = float(np.linalg.norm(a - g) / max(np.linalg.norm(g), 1e-300))
            r[st] = {
                "bitwise_equal_frac": eq,
                "rel_l2_vs_gpu": rel_gpu,
                "rel_l2_vs_fp64": float(np.linalg.norm(a - f) / max(np.linalg.norm(f), 1e-300)),
                "gpu_rel_l2_vs_fp64": float(np.linalg.norm(g - f) / max(np.linalg.norm(f), 1e-300)),
                "values_on_bf16_grid": bool(np.all(_bf16_round(a) == a)),
            }
            if first is None and eq < 1.0:
                first = st
            if first_material is None and rel_gpu > MATERIAL:
                first_material = st
        if "module_dx_equal" in mt:
            r["module_dx_equal"] = bool(mt["module_dx_equal"])
        r["first_differing_stage"] = first
        r["first_material_stage"] = first_material
        rows[nm] = r
    res = {
        "what": "hc_head + 末层 RMSNorm 反向：MaxText 各写法对 GPU（Hy4FinalNorm autograd，TE norm 语义仿写）逐级比；"
        "前缀 xoff：XLA 关省舍入（MaxText 写法的语义）；xon：XLA 默认（CPU 上的代数化简会省掉 FP32→BF16→FP32 往返）；"
        "后缀 _elide：xoff 下手写的推断写法，去掉 TPU 可能省掉的 BF16 往返；_shared：MTP 与输出头共用一次末层 norm；"
        "c_gate 在非 _elide 时与 GPU 舍成 BF16 后的门控梯度比；FP64 参照是 GPU 写法全 FP64",
        "shape": {"S": S, "B": B, "K": K, "D": D},
        "gpu_on_bf16_grid": {
            st: bool(np.all(_bf16_round(gpu[st]) == gpu[st])) for st in order if st.startswith(("g_", "c_", "dx"))
        },
        "material_threshold_rel_l2_vs_gpu": MATERIAL,
        "gpu_rel_l2_vs_fp64": {st: float(np.linalg.norm(gpu[st] - ref[st]) / np.linalg.norm(ref[st])) for st in order},
        "variants": rows,
    }
    with open(out, "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for nm, r in rows.items():
        print(
            f"{nm:28s} 首个实质不同 {str(r['first_material_stage']):10s} "
            + " ".join(
                f"{st}:{r[st]['bitwise_equal_frac']:.3f}/{r[st]['rel_l2_vs_gpu']:.1e}"
                for st in ("g_out", "c_sum", "c_gate", "dx", "d_hc_scale")
            )
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inputs", "gpu", "maxtext", "compare"])
    ap.add_argument("--dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--maxtext-src", default="")
    a = ap.parse_args()
    if a.cmd == "inputs":
        make_inputs(a.out)
    elif a.cmd == "gpu":
        run_gpu(a.dir)
    elif a.cmd == "maxtext":
        run_maxtext(a.dir, a.maxtext_src)
    else:
        compare(a.dir, a.out)


if __name__ == "__main__":
    main()
