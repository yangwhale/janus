"""hc_head + 末层 RMSNorm + 输出头这一段的 v5p 最小复现：CPU 上 AOT 编译，数反向里的 BF16 舍入点（HLO 与 LLO 两层）。

做法照 iHC 那次（`ihc_llo_probe.py`、04-tpu-self impacts/05）：同一段代码按几种开关组合各编一份，
看优化后 HLO 里每颗融合内的 convert / reduce_precision，和 LLO（`final_bundles`）里每颗 kernel 的 BF16 打包指令，
比较配置之间哪一处的舍入在、哪一处被省掉。

一段的写法用 MaxText 真模块（`mhc.DeepSeek4HyperHead`、`normalizations.RMSNorm(single_rounding)`），
输出头照对齐配置（FP32 权重、FP32 输入、HIGHEST 矩阵乘，M13）；MTP 那一路用「再调一次末层 norm + 一个 BF16 线性层」
代替整个 MTP 块（只为让末层 norm 输出有两处用到，与训练步的接法相同）；`shared` 时两路共用一次 norm
（`mtp_input_shared_final_norm`）。下游 loss = Σ logits · g + Σ mtp · g2（g、g2 是输入，代替交叉熵给的梯度）。
形状：x bf16[1, 4096, 4, 1024]（hy4-mini 的 seq、hc_mult、hidden），词表缩到 16384（训练步是 120832）。

子命令
  aot   --variant V --out DIR      编一格，写优化后 HLO；LLO 由环境变量
                                   LIBTPU_INIT_ARGS="--xla_jf_dump_to=DIR/jf --xla_jf_dump_llo_text=true" 给
  count DIR...  --out JSON         每格：按源文件（mhc.py / normalizations.py）归类的融合里 convert(→bf16)、
                                   reduce_precision 的条数（HLO），与对应 kernel 的 vpack / vunpack 条数（LLO）

变体：m58（只开 M58）、c（M52 + M58）、c_shared（再开共用 norm）、c_fwdonly（M52 只在前向舍）、base（都不开）。
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import types
from pathlib import Path

B, S, K, D, V = 1, 4096, 4, 1024, 16384
VARIANTS = {
    "base": dict(),
    "m58": dict(ihc_stream_grad_gpu_order=True),
    "c": dict(hc_head_force_bf16_rounding=True, ihc_stream_grad_gpu_order=True),
    "c_shared": dict(hc_head_force_bf16_rounding=True, ihc_stream_grad_gpu_order=True, shared=True),
    "c_fwdonly": dict(
        hc_head_force_bf16_rounding=True, hc_head_force_bf16_rounding_forward_only=True, ihc_stream_grad_gpu_order=True
    ),
}


def build(variant, src):
    sys.path.insert(0, src)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.layers import mhc, normalizations

    bf, f32 = jnp.bfloat16, jnp.float32
    kw = dict(VARIANTS[variant])
    shared = kw.pop("shared", False)
    c = types.SimpleNamespace(
        dtype=bf,
        weight_dtype=f32,
        mhc_expansion_rate=K,
        emb_dim=D,
        hc_head_norm_epsilon=1e-5,
        hc_eps=1e-6,
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
    head = mhc.DeepSeek4HyperHead(c, None, nnx.Rngs(0))
    norm = normalizations.RMSNorm(D, epsilon=1e-5, dtype=bf, weight_dtype=f32, single_rounding=True, rngs=nnx.Rngs(0))
    gdef, state, rest = nnx.split((head, norm), nnx.Param, ...)  # 只对参数求导（模块里另存了 rngs）

    def f(state, x, w_out, w_mtp, g, g2):
        h, n = nnx.merge(gdef, state, rest)
        o = h(x)
        y1 = n(o)
        y2 = y1 if shared else n(o)
        logits = jnp.einsum("bsd,dv->bsv", y1.astype(f32), w_out, precision=jax.lax.Precision.HIGHEST)
        mtp = jnp.einsum("bsd,de->bse", y2, w_mtp)
        return jnp.sum(logits * g) + jnp.sum(mtp.astype(f32) * g2)

    grad = jax.grad(f, argnums=(0, 1))
    shapes = [((B, S, K, D), bf), ((D, V), f32), ((D, D), bf), ((B, S, V), f32), ((B, S, D), f32)]
    return grad, state, shapes


def cmd_aot(a):
    import jax
    from jax.experimental import topologies

    dev = topologies.get_topology_desc(platform="tpu", topology_name="v5p:2x2x1").devices[0]
    sh = jax.sharding.SingleDeviceSharding(dev)
    g, state, shapes = build(a.variant, a.maxtext_src)
    st = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=sh), state)
    args = [jax.ShapeDtypeStruct(s, d, sharding=sh) for s, d in shapes]
    c = jax.jit(g).lower(st, *args).compile()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "after_optimizations.hlo.txt").write_text(c.as_text())
    print("ok", a.variant, len(c.as_text()))


PACK = re.compile(r"\bvpack\.[a-z]+\.bf16\b")
UNPACK = re.compile(r"\bvunpack\.[a-z]+\.[lh]\.bf16\b")


def _fusions(hlo: str):
    """融合名 → (调用的计算名, 源文件集合)；计算名 → 其中的指令行。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from llo_bf16_alu import kernel_files

    get = kernel_files(hlo)
    comps, cur = collections.defaultdict(list), None
    fus = {}
    for ln in hlo.splitlines():
        m = re.match(r"^(?:ENTRY )?%([\w.\-]+) ", ln)
        if m and ln.rstrip().endswith("{"):
            cur = m.group(1)
            continue
        if cur:
            comps[cur].append(ln)
        m = re.match(r"^\s*(?:ROOT )?%([\w.\-]+) = .*\bfusion\(.*calls=%([\w.\-]+)", ln)
        if m:
            fus[m.group(1)] = m.group(2)
    return fus, comps, get


def _hlo_counts(lines, comps, seen=None):
    """一个计算（含它调用的子计算）里 convert 到 bf16 与 reduce_precision 的条数。"""
    seen = seen or set()
    n = collections.Counter()
    for ln in lines:
        if re.search(r"= bf16\[[^\]]*\]\{?[^ ]* convert\(", ln) or re.search(r"= bf16\[[^ ]* convert\(", ln):
            n["convert_to_bf16"] += 1
        if re.search(r"= f32\[[^ ]* convert\(", ln):
            n["convert_to_f32"] += 1
        if "reduce-precision(" in ln:
            n["reduce_precision"] += 1
        for sub in re.findall(r"(?:calls|to_apply)=%([\w.\-]+)", ln):
            if sub not in seen and sub in comps:
                seen.add(sub)
                n.update(_hlo_counts(comps[sub], comps, seen))
    return n


def cmd_count(a):
    res = {}
    for d in map(Path, a.dirs):
        hlo = (d / "after_optimizations.hlo.txt").read_text()
        fus, comps, get = _fusions(hlo)
        llo = {}
        for f in sorted((d / "jf").glob("*-final_bundles.txt")):
            if "schedule-analysis" in f.name:
                continue
            m = re.match(r"\d+-(.+)-\d+-final_bundles\.txt", f.name)
            k = m.group(1) if m else f.name
            t = f.read_text(errors="ignore")
            llo[k] = {"vpack_bf16": len(PACK.findall(t)), "vunpack_bf16": len(UNPACK.findall(t))}
        rows = {}
        for name, comp in fus.items():
            files = get(name)
            tags = sorted(
                {"hc_head" for x in files if x == "mhc.py"} | {"norm" for x in files if x == "normalizations.py"}
            )
            if not tags:
                continue
            r = {"tags": tags, **_hlo_counts(comps[comp], comps)}
            r.update(llo.get(name, {}))
            rows[name] = r
        tot = collections.Counter()
        for r in rows.values():
            for k2, v in r.items():
                if isinstance(v, int):
                    tot[k2] += v
        res[d.name] = {"fusions": rows, "total": dict(tot), "n_fusions": len(rows), "n_llo_kernels": len(llo)}
        print(d.name, len(rows), dict(tot))
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("aot")
    q.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    q.add_argument("--out", required=True)
    q.add_argument("--maxtext-src", required=True)
    q = sp.add_parser("count")
    q.add_argument("dirs", nargs="+")
    q.add_argument("--out", required=True)
    a = p.parse_args()
    {"aot": cmd_aot, "count": cmd_count}[a.cmd](a)


if __name__ == "__main__":
    main()
