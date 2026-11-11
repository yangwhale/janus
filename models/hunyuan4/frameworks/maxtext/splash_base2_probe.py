"""CPU 上核 tokamax splash 的 base2 模式到底算了什么（Pallas interpret 模式跑 MaxText 分支里的同一份内核源码）。

内核前向在 base2 模式下做 `q *= LOG2E`：q 是 BF16，JAX 把 Python 标量按 q 的类型舍成 BF16 的 1.4453125，
比精确的 log2(e) = 1.442695… 大 1.81e-3；之后 exp2、换回自然底的 logsumexp 都按精确的 LOG2E 走，sink 在 FP32 里乘精确的
LOG2E。所以等价于：logits（不含 sink）整体乘 κ = 1.4453125 ÷ log2(e)，外加 q·1.4453125 再舍一次 BF16 的逐元素误差。

MaxText `attention_op.py` import 的是 pip 包 tokamax 的 `splash_attention_kernel`（默认 `--impl tokamax`）；树里
`kernels/tokamax_splash_attention/` 的副本同一写法。本脚本用同一组 BF16 输入跑内核（base2 开 / 关），与三种 FP64 参照比 logsumexp 与输出：
- exact：精确 softmax（带 sink）；
- kappa：logits 乘 κ；
- full：q_eff = bf16(q · 1.4453125) ÷ log2(e)（κ 加逐元素舍入，FP64 参照 `step1_ref.py --emulate-base2 full` 用的就是它）。

    JAX_PLATFORMS=cpu python splash_base2_probe.py --json out.json              # pip tokamax（MaxText 实际用的）
    JAX_PLATFORMS=cpu python splash_base2_probe.py --impl maxtext --maxtext-src <MaxText 仓库>/src --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxtext-src", default="", help="MaxText 仓库的 src；--impl maxtext 时用")
    ap.add_argument(
        "--impl",
        default="tokamax",
        choices=["tokamax", "maxtext"],
        help="tokamax：pip 包里的内核（MaxText attention_op 实际 import 的那份）；maxtext：MaxText 树里的副本",
    )
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--q-std", type=float, default=0.35)
    ap.add_argument("--k-std", type=float, default=1.4)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    if a.impl == "maxtext":
        sys.path.insert(0, a.maxtext_src)
        from maxtext.kernels.tokamax_splash_attention import splash_attention_kernel as sk
        from maxtext.kernels.tokamax_splash_attention import splash_attention_mask as ml
    else:
        from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as sk
        from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as ml

    H, S, D = a.heads, a.seq, a.head_dim
    log2e = math.log2(math.e)
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((H, S, D)) * a.q_std).astype(np.float32)
    k = (rng.standard_normal((H, S, D)) * a.k_std).astype(np.float32)
    v = rng.standard_normal((H, S, D)).astype(np.float32)
    sinks = np.linspace(0.3, -0.5, H).astype(np.float32)
    qb = jnp.asarray(q, jnp.bfloat16) * (1 / math.sqrt(D))  # MaxText：query × softmax_scale（1/16 精确）
    kb, vb = jnp.asarray(k, jnp.bfloat16), jnp.asarray(v, jnp.bfloat16)
    mask = ml.CausalMask((S, S))
    kern_out = {}
    for b2 in (True, False):
        cfg = sk.SplashConfig(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=128,
            block_kv_dkv=128,
            block_kv_dkv_compute=128,
            use_base2_exp=b2,
            interpret=True,
        )
        kern = sk.make_splash_mha_single_device(mask, config=cfg, save_residuals=True)
        o, st = kern(qb, kb, vb, sinks=jnp.asarray(sinks))
        kern_out[b2] = (np.asarray(o, np.float64), np.asarray(st["logsumexp"], np.float64))
    lse_const = float(jnp.bfloat16(log2e))
    kappa = lse_const / log2e
    K, V = np.asarray(kb, np.float64), np.asarray(vb, np.float64)
    causal = np.tril(np.ones((S, S), bool))[None]

    def ref(Q):
        s = np.where(causal, np.einsum("hqd,hkd->hqk", Q, K), -np.inf)
        s = np.concatenate([s, np.broadcast_to(sinks.astype(np.float64)[:, None, None], (H, S, 1))], -1)
        m = s.max(-1, keepdims=True)
        p = np.exp(s - m)
        l_ = p.sum(-1, keepdims=True)
        return np.einsum("hqk,hkd->hqd", p[..., :-1] / l_, V), (m + np.log(l_))[..., 0]

    Q = np.asarray(qb, np.float64)
    refs = {
        "exact": ref(Q),
        "kappa": ref(Q * kappa),
        "full": ref(np.asarray((qb * log2e).astype(jnp.float64)) / log2e),
    }
    res = {
        "what": "tokamax splash（MaxText 分支内核源码，Pallas interpret，CPU）base2 开 / 关对三种 FP64 参照",
        "impl": a.impl,
        "tokamax_version": __import__("importlib.metadata").metadata.version("tokamax"),
        "shape": {"heads": H, "seq": S, "head_dim": D, "q_std": a.q_std, "k_std": a.k_std},
        "log2e_after_bf16": lse_const,
        "kappa": kappa,
        "rows": [],
    }
    for b2 in (True, False):
        o, lse = kern_out[b2]
        for name, (o_r, l_r) in refs.items():
            res["rows"].append(
                {
                    "base2": b2,
                    "ref": name,
                    "lse_max_abs": float(np.abs(lse - l_r).max()),
                    "lse_mean_signed": float((lse - l_r).mean()),
                    "o_rel_l2": float(np.linalg.norm(o - o_r) / np.linalg.norm(o_r)),
                }
            )
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    for r in res["rows"]:
        print(r)


if __name__ == "__main__":
    main()
