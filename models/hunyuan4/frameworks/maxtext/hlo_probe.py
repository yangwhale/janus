"""两个机制的 TPU 核实（结果与编译后的 HLO 一起存下，对 FP64 的比较在 CPU 上做：analysis/hlo_probe_compare.py）：
- M28 ihc_stream_sum_unrolled：iHC 四路流加权和 Σ_k pre_k·x_k 的三种写法——HIGHEST einsum（开关关）、逐路 FP32 乘加（开关开）、
  jnp.sum 归约——同一份输入（GPU 第 1 步第 1 层的 attn_hc.in 四路流；pre 取固定种子的 (0.05, 1.05) 均匀数，GPU 没抓 pre），
  输出留在 FP32（舍 BF16 之前）；
- M9 rmsnorm_single_rounding：MaxText 的 RMSNorm 模块单次舍入开 / 关，输入是 GPU 第 1 步第 0 层的 attn_hc.out，scale 取检查点的
  第 0 层 input_layernorm（不是全 1）。
XLA_FLAGS 由提交时给（例如 --xla_allow_excess_precision=false），记进结果。每个 JAX 进程都在自己的第一个设备上算，进程 0 写结果。

  python hlo_probe.py --gpu-fwd …/step_1_fwd.safetensors --hf-weights …/model.safetensors --out …/hlo_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu-fwd", required=True)
    p.add_argument("--hf-weights", required=True)
    p.add_argument("--out", required=True)
    a, _ = p.parse_known_args(argv)
    import jax
    if not jax.distributed.is_initialized():
        try:
            jax.distributed.initialize()
        except Exception as e:  # noqa: BLE001 — 单机不需要
            print("jax.distributed.initialize 跳过：", e)
    out_dir = Path(a.out).parent
    if jax.process_index() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.layers.normalizations import RMSNorm
    from op_replay_mini import load_st

    dev = jax.local_devices()[0]
    G = load_st(a.gpu_fwd, names={"hf.model.layers.1.attn_hc.in", "hf.model.layers.0.attn_hc.out"})
    W = load_st(a.hf_weights, names={"model.layers.0.input_layernorm.weight"})
    arrays, hlo, rec = {}, {}, {"xla_flags": os.environ.get("XLA_FLAGS", ""), "platform": dev.platform,
                                 "device_kind": dev.device_kind}
    # ---- M28
    xs = np.asarray(G["hf.model.layers.1.attn_hc.in"], np.float32)  # [1, S, 4, D]，BF16 值
    pre = np.random.default_rng(7).uniform(0.05, 1.05, xs.shape[:3]).astype(np.float32)
    xd = jax.device_put(jnp.asarray(xs, jnp.bfloat16), dev)
    pd = jax.device_put(jnp.asarray(pre), dev)

    def einsum_highest(p_, x_):
        return jnp.einsum("bsk,bskd->bsd", p_, x_.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)

    def unrolled(p_, x_):
        xf = x_.astype(jnp.float32)
        y = p_[..., 0:1] * xf[:, :, 0]
        for i in range(1, x_.shape[2]):
            y = y + p_[..., i : i + 1] * xf[:, :, i]
        return y

    def reduce_sum(p_, x_):
        return jnp.sum(p_[..., None] * x_.astype(jnp.float32), axis=2)

    for name, fn in (("einsum_highest", einsum_highest), ("unrolled", unrolled), ("reduce_sum", reduce_sum)):
        c = jax.jit(fn).lower(pd, xd).compile()
        hlo[f"m28_{name}"] = c.as_text()
        arrays[f"m28.{name}"] = np.asarray(c(pd, xd), np.float32)
    arrays["m28.x"], arrays["m28.pre"] = xs, pre
    # ---- M9
    x0 = np.asarray(G["hf.model.layers.0.attn_hc.out"], np.float32)
    scale = np.asarray(W["model.layers.0.input_layernorm.weight"], np.float32)
    x0d = jax.device_put(jnp.asarray(x0, jnp.bfloat16), dev)
    for single in (False, True):
        m = RMSNorm(num_features=x0.shape[-1], epsilon=1e-5, dtype=jnp.bfloat16, weight_dtype=jnp.float32,
                    single_rounding=single, rngs=nnx.Rngs(params=0))
        m.scale.set_value(jnp.asarray(scale))
        gd, st = nnx.split(m)
        f = jax.jit(lambda s_, x_, gd=gd: nnx.merge(gd, s_)(x_))
        c = f.lower(st, x0d).compile()
        tag = "single" if single else "default"
        hlo[f"m9_{tag}"] = c.as_text()
        arrays[f"m9.{tag}"] = np.asarray(c(st, x0d), np.float32)
    arrays["m9.x"], arrays["m9.scale"] = x0, scale
    rec["m9_bit_identical"] = bool(np.array_equal(arrays["m9.default"].view(np.uint32), arrays["m9.single"].view(np.uint32)))
    rec["m28_bit_identical"] = {f"{x}_vs_{y}": bool(np.array_equal(arrays[f"m28.{x}"].view(np.uint32), arrays[f"m28.{y}"].view(np.uint32)))
                                for x, y in (("einsum_highest", "unrolled"), ("unrolled", "reduce_sum"), ("einsum_highest", "reduce_sum"))}
    if jax.process_index() == 0:
        from safetensors.numpy import save_file
        save_file({k: np.ascontiguousarray(v) for k, v in arrays.items()}, str(Path(a.out).with_suffix(".safetensors")))
        for k, t in hlo.items():
            (out_dir / f"hlo_{k}.txt").write_text(t)
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        print("[hlo_probe] " + json.dumps(rec, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
