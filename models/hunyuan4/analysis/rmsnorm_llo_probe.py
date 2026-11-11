"""RMSNorm 两种写法（MaxText 默认的两次舍入 / rmsnorm_single_rounding 的一次舍入）的单层最小复现：
CPU 上为 TPU AOT 编译看 HLO 与 LLO，TPU 上跑数值、对 FP64 比。

子命令
  aot   --variant V --ctx C --platform v5p|v7x --out DIR   编一格，写优化后 HLO；LLO dump 由 LIBTPU_INIT_ARGS 给
                                                            （--xla_jf_dump_to=DIR/jf --xla_jf_dump_llo_text=true）
  run   --out NPZ                                          在当前设备上跑全部变体（固定随机输入），存输出与梯度
  ref   --out NPZ                                          同样的输入，CPU 上 FP64 精确值
  cmp   REF RUN...                                         各变体离 FP64 的距离（相对 L2、带符号均值）

变体（形状 x bf16[1,4096,1024]，scale 为 FP32 主权重，和混元 4 mini 的 input_layernorm 相同）
  dr  MaxText 默认：y = bf16(x·rsqrt(mean(x²)+eps))，out = y ⊗ bf16(scale)（BF16 einsum，输出 BF16）
  sr  rmsnorm_single_rounding：out = bf16(x·rsqrt(·) ⊗ rp_bf16(scale))，乘完 scale 只舍一次
下游（--ctx）
  sum  loss = Σ out.f32 · w.f32
  mm   lo = out @ W（BF16 einsum），loss = Σ lo.f32 · w2.f32；对 x、scale、W 求导
  remat 同 mm，RMSNorm 包在 jax.checkpoint 里（反向重算）
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

B, S, D = 1, 4096, 1024
EPS = 1e-5
VARIANTS = ("dr", "sr")
CTXS = ("sum", "mm", "remat")


def rmsnorm(x, scale, variant):
    import jax.numpy as jnp
    from jax import lax

    bf, f32 = jnp.bfloat16, jnp.float32
    xf = jnp.asarray(x, f32)
    mean2 = jnp.mean(lax.square(xf), axis=-1, keepdims=True)
    if variant == "sr":
        normed = xf * lax.rsqrt(mean2 + EPS)
        fi = jnp.finfo(bf)
        s = lax.reduce_precision(jnp.asarray(scale, f32), fi.nexp, fi.nmant)
        return jnp.asarray(jnp.einsum("...k,k->...k", normed, s), bf)
    y = jnp.asarray(xf * lax.rsqrt(mean2 + EPS), bf)
    return jnp.einsum("...k,k->...k", y, jnp.asarray(jnp.asarray(scale, f32), bf))


def build(variant, ctx):
    import jax
    import jax.numpy as jnp

    bf, f32 = jnp.bfloat16, jnp.float32
    norm = (lambda x, s: rmsnorm(x, s, variant))
    if ctx == "remat":
        norm = jax.checkpoint(norm)

    if ctx == "sum":

        def f(x, scale, w):
            return jnp.sum(norm(x, scale).astype(f32) * w.astype(f32))

        shapes = [((B, S, D), bf), ((D,), f32), ((B, S, D), bf)]
        return jax.value_and_grad(f, argnums=(0, 1)), shapes

    def f(x, scale, W, w2):
        lo = jnp.einsum("bsd,de->bse", norm(x, scale), W)
        return jnp.sum(lo.astype(f32) * w2.astype(f32))

    shapes = [((B, S, D), bf), ((D,), f32), ((D, D), bf), ((B, S, D), bf)]
    return jax.value_and_grad(f, argnums=(0, 1, 2)), shapes


def inputs(ctx, seed=0):
    import numpy as np

    r = np.random.default_rng(seed)
    x = r.standard_normal((B, S, D)).astype(np.float32) * 0.7 + 0.05
    scale = (1.0 + 0.1 * r.standard_normal(D)).astype(np.float32)
    w = r.standard_normal((B, S, D)).astype(np.float32)
    if ctx == "sum":
        return [x, scale, w]
    W = (r.standard_normal((D, D)) / np.sqrt(D)).astype(np.float32)
    return [x, scale, W, w]


def to_bf16_np(a):
    import jax.numpy as jnp
    import numpy as np

    return np.asarray(jnp.asarray(a, jnp.bfloat16).astype(jnp.float32))


def cmd_aot(a):
    import jax
    from jax.experimental import topologies

    topo = {"v5p": "v5p:2x2x1", "v7x": "tpu7x:2x2x1"}[a.platform]
    dev = topologies.get_topology_desc(platform="tpu", topology_name=topo).devices[0]
    sh = jax.sharding.SingleDeviceSharding(dev)
    g, shapes = build(a.variant, a.ctx)
    args = [jax.ShapeDtypeStruct(s, d, sharding=sh) for s, d in shapes]
    c = jax.jit(g).lower(*args).compile()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "after_optimizations.hlo.txt").write_text(c.as_text())
    print("ok", a.variant, a.ctx, a.platform, len(c.as_text()))


def cmd_run(a):
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    if "," in os.environ.get("TPU_WORKER_HOSTNAMES", ""):  # 多主机切片：每台都要起，只用本机第一颗
        jax.distributed.initialize()
    dev = jax.local_devices()[0]
    res = {"backend": jax.default_backend(), "device": str(dev.device_kind)}
    for ctx in CTXS:
        xs = inputs(ctx)
        dt = [jnp.bfloat16, jnp.float32] + ([jnp.bfloat16] if ctx == "sum" else [jnp.bfloat16, jnp.bfloat16])
        args = [jax.device_put(jnp.asarray(v, d), dev) for v, d in zip(xs, dt)]
        for v in VARIANTS:
            g, _ = build(v, ctx)
            loss, grads = jax.jit(g)(*args)
            out = jax.jit(lambda x, s: rmsnorm(x, s, v))(args[0], args[1])
            res[f"{v}.{ctx}.loss"] = np.float64(loss)
            res[f"{v}.{ctx}.out"] = np.asarray(out.astype(jnp.float32))
            for i, gi in enumerate(grads):
                res[f"{v}.{ctx}.g{i}"] = np.asarray(jnp.asarray(gi, jnp.float32))
    if jax.process_index() == 0:
        np.savez_compressed(a.out, **res)
    print("saved", a.out, res["device"], jax.process_index())


def cmd_ref(a):
    import numpy as np

    res = {}
    for ctx in CTXS:
        xs = inputs(ctx)
        x = to_bf16_np(xs[0]).astype(np.float64)
        s = to_bf16_np(xs[1]).astype(np.float64)  # 两种写法都按 BF16 的 scale 乘（GPU 的 BF16 参数）
        inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
        n = x * inv
        out = n * s
        if ctx == "sum":
            w = to_bf16_np(xs[2]).astype(np.float64)
            dy = w
            res[f"{ctx}.loss"] = np.sum(out * w)
        else:
            W = to_bf16_np(xs[2]).astype(np.float64)
            w2 = to_bf16_np(xs[3]).astype(np.float64)
            res[f"{ctx}.loss"] = np.sum((out @ W) * w2)
            dy = w2 @ W.T
            res[f"{ctx}.g2"] = np.einsum("bsd,bse->de", out, w2)
        dn = dy * s
        res[f"{ctx}.g1"] = np.sum(dy * n, axis=(0, 1))
        res[f"{ctx}.g0"] = inv * (dn - n * np.mean(dn * n, axis=-1, keepdims=True))
        res[f"{ctx}.out"] = out
    np.savez_compressed(a.out, **res)
    print("saved", a.out)


def cmd_cmp(a):
    import numpy as np

    ref = np.load(a.ref)
    rows = []
    for p in a.runs:
        r = np.load(p)
        for k in sorted(r.files):
            m = re.match(r"(\w+)\.(\w+)\.(\w+)$", k)
            if not m:
                continue
            v, ctx, q = m.groups()
            e = np.asarray(r[k], np.float64)
            t = ref[f"{ctx}.{q}"]
            row = {"run": Path(p).stem, "variant": v, "ctx": ctx, "q": q}
            if q == "loss":
                row["err"] = float(e - t)
            else:
                d = e - t
                row["rel_l2"] = float(np.linalg.norm(d) / np.linalg.norm(t))
                row["signed_mean_rel"] = float(np.sum(d * np.sign(t)) / np.sum(np.abs(t)))
            rows.append(row)
    for row in rows:
        print(json.dumps(row))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")


OPS = (
    "vpack.c.bf16",
    "vpack.i.bf16",
    "vunpack.c.l.bf16",
    "vunpack.c.h.bf16",
    "vunpack.i.l.bf16",
    "vunpack.i.h.bf16",
    "vadd.f32",
    "vmul.f32",
    "vsub.f32",
    "vrsqrt.f32",
    "vmatmul",
)


def cmd_count(a):
    res = {}
    for d in a.dirs:
        r = {}
        for f in sorted(Path(d).glob("*final_bundles.txt")):
            if "schedule-analysis" in f.name:
                continue
            t = f.read_text(errors="ignore")
            name = re.sub(r"^[0-9]+-", "", f.name).replace("-final_bundles.txt", "")
            e = {o: len(re.findall(r"(?<![\w.])" + re.escape(o) + r"(?![\w])", t)) for o in OPS}
            r[name] = {k: v for k, v in e.items() if v}
        res[d] = r
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("aot")
    q.add_argument("--variant", choices=VARIANTS, required=True)
    q.add_argument("--ctx", choices=CTXS, default="mm")
    q.add_argument("--platform", choices=("v5p", "v7x"), default="v5p")
    q.add_argument("--out", required=True)
    q = sp.add_parser("run")
    q.add_argument("--out", required=True)
    q = sp.add_parser("ref")
    q.add_argument("--out", required=True)
    q = sp.add_parser("cmp")
    q.add_argument("ref")
    q.add_argument("runs", nargs="+")
    q.add_argument("--out")
    q = sp.add_parser("count")
    q.add_argument("dirs", nargs="+")
    q.add_argument("--out")
    a = p.parse_args()
    {"aot": cmd_aot, "run": cmd_run, "ref": cmd_ref, "cmp": cmd_cmp, "count": cmd_count}[a.cmd](a)


if __name__ == "__main__":
    main()
