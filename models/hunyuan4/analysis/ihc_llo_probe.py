"""iHC 残差更新（mhc.py _xla_call 末段）在 xla_allow_excess_precision 开/关下的 TPU v5p AOT 编译对比。

子命令
  aot --variant V --excess 0|1 --out DIR   编一格，写优化后 HLO；LLO dump 由环境变量 LIBTPU_INIT_ARGS 给
  count DIR...                             数每个 jf dump 里各 fusion 的 final_bundles 向量指令静态条数

变体（形状 x bf16[1,4096,4,1024]、post f32[1,4096,4]、layer_out bf16[1,4096,1024]）
  A   out = rp(xf + post*lo_f32).bf16；loss = sum(out.f32 * w.f32)，w bf16 随机；对 x/post/lo 求导
  B   同 A，但下游 loss = sum((out * w).f32)（BF16 乘法）
  C   同 A，round 用 astype(bf16) 来回代替 reduce_precision
  D   整层：gates(rms + 门控 sigmoid) + 展开的 k 路加权和 + rp + 分支 lo = li @ W(bf16) + 残差更新 + rp；
      下游再接下一层的 gates+加权和（与 mhc 相同写法）后 sum(·*w2)；对 x、W、alpha 求导
  E/F/G 只编 d layer_out 那颗前向函数（不求导）：E sum_k(post*g.f32).bf16；F 同 E 但 g 先 reduce_precision；
      G 沿 d 求和再 .bf16
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

B, S, K, D = 1, 4096, 4, 1024


def build(variant):
    import jax
    import jax.numpy as jnp

    bf, f32 = jnp.bfloat16, jnp.float32

    def rp(v):
        return jax.lax.reduce_precision(v, 8, 7)

    def rt(v):
        return v.astype(bf).astype(f32)

    def resid(x, post, lo, rnd):
        xf = x.astype(f32)
        out = xf + post[..., None] * jnp.expand_dims(lo.astype(f32), axis=2)
        return rnd(out).astype(bf)

    if variant in ("A", "B", "C"):
        rnd = rt if variant == "C" else rp

        def f(x, post, lo, w):
            out = resid(x, post, lo, rnd)
            if variant == "B":
                return jnp.sum((out * w).astype(f32))
            return jnp.sum(out.astype(f32) * w.astype(f32))

        g = jax.grad(f, argnums=(0, 1, 2))
        shapes = [((B, S, K, D), bf), ((B, S, K), f32), ((B, S, D), bf), ((B, S, K, D), bf)]
        return g, shapes

    if variant == "D":

        def gates(x, alpha):
            flat = jnp.reshape(x, (B, S, K * D)).astype(f32)
            inv = jax.lax.rsqrt(jnp.mean(jnp.square(flat), -1, keepdims=True) + 1e-6)
            mixes = jnp.einsum("bsm,mn->bsn", flat, alpha) * inv
            pre = jax.nn.sigmoid(mixes[..., :K]) + 1e-6
            post = 2.0 * jax.nn.sigmoid(mixes[..., K:]) + 1e-6
            return pre, post

        def ssum(x, pre):
            xf = x.astype(f32)
            li = pre[..., 0:1] * xf[:, :, 0]
            for i in range(1, K):
                li = li + pre[..., i : i + 1] * xf[:, :, i]
            return rp(li).astype(bf)

        def f(x, W, alpha, w2):
            pre, post = gates(x, alpha)
            li = ssum(x, pre)
            lo = jnp.einsum("bsd,de->bse", li, W)
            out = resid(x, post, lo, rp)
            pre2, _ = gates(out, alpha)
            li2 = ssum(out, pre2)
            return jnp.sum(li2.astype(f32) * w2.astype(f32))

        g = jax.grad(f, argnums=(0, 1, 2))
        shapes = [((B, S, K, D), bf), ((D, D), bf), ((K * D, 2 * K), f32), ((B, S, D), bf)]
        return g, shapes
    if variant in ("E", "F", "G"):
        # 只留 d layer_out 那条：E = sum_k(post * g.f32).astype(bf16)；F = 同 E 但 g 先过 reduce_precision；
        # G = 沿 d（长轴）求和后 astype(bf16)
        def f(post, g):
            gf = g.astype(f32)
            if variant == "F":
                gf = rp(gf)
            if variant == "G":
                return jnp.sum(gf * post[..., None], axis=3).astype(bf)
            return jnp.sum(post[..., None] * gf, axis=2).astype(bf)

        return f, [((B, S, K), f32), ((B, S, K, D), bf)]
    if variant in ("R0", "R1", "R0g", "R1g"):
        # 分支输入 -> RMSNorm（MaxText single_rounding 写法）。R0：加权和直接 astype(bf16)；R1：先 reduce_precision。
        # 带 g 后缀的：再对 sum(y*w) 求导（x、pre、scale）。
        def f(x, pre, scale, w):
            xf = x.astype(f32)
            li = pre[..., 0:1] * xf[:, :, 0]
            for i in range(1, K):
                li = li + pre[..., i : i + 1] * xf[:, :, i]
            if variant.startswith("R1"):
                li = rp(li)
            li = li.astype(bf)
            h = jnp.asarray(li, f32)
            mean2 = jnp.mean(jnp.square(h), axis=-1, keepdims=True)
            normed = h * jax.lax.rsqrt(mean2 + 1e-5)
            y = jnp.einsum("...k,k->...k", normed, scale).astype(bf)
            if variant.endswith("g"):
                return jnp.sum(y.astype(f32) * w.astype(f32))
            return y

        shapes = [((B, S, K, D), bf), ((B, S, K), f32), ((D,), f32), ((B, S, D), bf)]
        return (jax.grad(f, argnums=(0, 1, 2)) if variant.endswith("g") else f), shapes
    raise ValueError(variant)


def cmd_aot(a):
    import jax
    from jax.experimental import topologies

    dev = topologies.get_topology_desc(platform="tpu", topology_name="v5p:2x2x1").devices[0]
    sh = jax.sharding.SingleDeviceSharding(dev)
    g, shapes = build(a.variant)
    args = [jax.ShapeDtypeStruct(s, d, sharding=sh) for s, d in shapes]
    c = jax.jit(g).lower(*args).compile(compiler_options={"xla_allow_excess_precision": bool(a.excess)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "after_optimizations.hlo.txt").write_text(c.as_text())
    print("ok", a.variant, a.excess, len(c.as_text()))


OPS = (
    "vld",
    "vst",
    "vadd.f32",
    "vsub.f32",
    "vmul.f32",
    "vadd.bf16",
    "vmul.bf16",
    "vunpack.c.l.bf16",
    "vunpack.c.h.bf16",
    "vunpack.i.l.bf16",
    "vunpack.i.h.bf16",
    "vpack.c.bf16",
    "vpack.i.bf16",
)


def cmd_count(a):
    res = {}
    for d in a.dirs:
        r = {}
        for f in sorted(Path(d).glob("*final_bundles.txt")):
            if "schedule-analysis" in f.name:
                continue
            t = f.read_text()
            name = re.sub(r"^[0-9]+-", "", f.name).replace("-final_bundles.txt", "")
            e = {"bundles": len(re.findall(r"^\s*0x[0-9a-f]+", t, re.M)) or t.count("{")}
            e.update({o: len(re.findall(r"(?<![\w.])" + re.escape(o) + r"(?![\w])", t)) for o in OPS})
            r[name] = e
        res[d] = r
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("aot")
    q.add_argument("--variant", default="A")
    q.add_argument("--excess", type=int, default=1)
    q.add_argument("--out", required=True)
    q = sp.add_parser("count")
    q.add_argument("dirs", nargs="+")
    q.add_argument("--out")
    a = p.parse_args()
    {"aot": cmd_aot, "count": cmd_count}[a.cmd](a)


if __name__ == "__main__":
    main()
