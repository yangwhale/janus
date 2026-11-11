"""MaxText 开关 `indexer_kd_generalized_kl` 在 TPU 上的验收：真实输入上，记录值对 FP64、梯度开 / 关是否相同。

输入：`kd_norm_probe.py --make-inputs` 生成的 npz（TPU `debug_sow_indexer` 运行第 0 行、四个产出层的教师 q / k、
学生分数与掩码，即 `calculate_indexer_loss` 收到的原样输入）。每层在 TPU 上调 MaxText 的
`MLA.calculate_indexer_loss`（hy4-mini-t1.yml 的写法：indexer_kd_mcore_log、indexer_fp32_scores、
transcendental_accuracy_highest；sparse_loss、scaling_factor 1），开关关 / 开各一次：
- 记录值对 FP64（同一份输入，原版 Megatron-Core 写法的 KL）的相对差；
- 对学生分数的梯度：开 / 关是否逐位相同，最大差 ÷ 最大梯度。

    python kd_genkl_check.py --inputs gs://…/kd_probe_inputs.npz --out kd_genkl_check.json   # TPU 上（SCRIPT=kd_genkl_check.py）
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from kd_norm_probe import fp64_truth, load_inputs

S = 4096


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cpu", action="store_true", help="在 CPU 上跑：只验证脚本，不是 TPU 的数值")
    a, _rest = p.parse_known_args(argv)
    import jax
    import jax.numpy as jnp

    if a.cpu:
        jax.config.update("jax_platforms", "cpu")
    elif not jax.distributed.is_initialized():
        try:
            jax.distributed.initialize()
        except Exception as e:  # noqa: BLE001
            print("jax.distributed.initialize 跳过：", e)
    from maxtext.layers.attention_mla import MLA

    if jax.process_index() == 0:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    dev = jax.local_devices()[0]
    d = load_inputs(a.inputs)
    causal = np.tril(np.ones((S, S), bool))

    def loss(gen, q, k, z, m):
        cfg = SimpleNamespace(mla_qk_head_chunk_size=0, matmul_precision="highest", indexer_kd_mcore_log=True,
                              indexer_fp32_scores=True, transcendental_accuracy_highest=True,
                              indexer_kd_generalized_kl=gen)
        return MLA.calculate_indexer_loss(SimpleNamespace(config=cfg), z[None], q[None], k[None], None, m[None],
                                          True, 1.0)

    val = {g: jax.jit(lambda q, k, z, m, g=g: loss(g, q, k, z, m)) for g in (False, True)}
    grad = {g: jax.jit(jax.grad(lambda z, q, k, m, g=g: loss(g, q, k, z, m))) for g in (False, True)}
    out = {}
    for i, name in enumerate(d["layers"]):
        q, k, z, m = (jax.device_put(jnp.asarray(d[f"l{i}_{n}"]), dev) for n in "qkzm")
        ref = float(fp64_truth(d[f"l{i}_q"], d[f"l{i}_k"], d[f"l{i}_z"], d[f"l{i}_m"], causal).mean())
        off, on = float(val[False](q, k, z, m)), float(val[True](q, k, z, m))
        g0, g1 = np.asarray(grad[False](z, q, k, m)), np.asarray(grad[True](z, q, k, m))
        r = {"kl_fp64": ref, "off": off, "on": on, "off_rel_vs_fp64": off / ref - 1, "on_rel_vs_fp64": on / ref - 1,
             "grad_bitwise_equal": bool(np.array_equal(g0, g1)),
             "grad_max_abs_diff_over_max_abs": float(np.max(np.abs(g1 - g0)) / np.max(np.abs(g0)))}
        out[str(name)] = r
        print(name, json.dumps(r), flush=True)
    rec = {"what": __doc__.split("\n")[0], "platform": dev.platform, "device_kind": dev.device_kind, "jax": jax.__version__,
           "inputs": a.inputs, "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "layers": out, "note": "CPU 上跑的只验证脚本，不是 TPU 的数值" if a.cpu else "TPU 一侧"}
    if jax.process_index() == 0:
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
