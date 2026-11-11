"""tokamax splash 的精度探针（TPU）：用 GPU 第 1 步抓到的注意力输入（core.query / key / value，BF16）与检查点的 sink，
在 TPU 上跑 MaxText 用的 tokamax splash（因果掩码，块大小取 T1 配置），记下：
- 前向的 o 与 logsumexp：sa_use_base2_exp 开（MaxText 默认：内核里 q *= log2(e)，q 是 BF16）与关两组；
- sink 梯度：tokamax 原样的反向（BF16 逐项、BF16 求和）、MaxText 开关 splash_dsinks_fp32（FP32）与 splash_dsinks_gpu_order
  （M42b：内核外按 GPU 写法 −Σ_q p_sink·Σ_j p_j·bf16(dp_j) 重算，有这个模块时才算）；上游梯度取固定种子的
  BF16 随机数（GPU 没抓注意力输出的梯度）。
query 先乘 softmax_scale（1/16，2 的幂，BF16 里精确），同 MaxText 的 MLA。对 FP64 的比较在 CPU 上做（analysis/sink_probe_compare.py）。
每个 JAX 进程都在自己的第一个设备上算（单设备、单个序列），只有进程 0 写结果。

  python sink_probe.py --gpu-fwd gs://…/mini-l1/gpu/step_1_fwd.safetensors --hf-weights gs://…/model.safetensors --out …/sink_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LAYERS = ["1", "2", "4", "6", "mtp0"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu-fwd", required=True)
    p.add_argument("--hf-weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default=",".join(LAYERS))
    p.add_argument("--stage", default="mini")
    p.add_argument("--tokens", type=int, default=0, help="只用前 T 个 token（0 = 全部）")
    a, _ = p.parse_known_args(argv)
    import jax
    if not jax.distributed.is_initialized():
        try:
            jax.distributed.initialize()
        except Exception as e:  # noqa: BLE001 — 单机 TPU 不需要分布式初始化
            print("jax.distributed.initialize 跳过：", e)
    if jax.process_index() == 0:   # 先建运行目录：run.sh 按「有没有运行目录」认交结果的 Pod
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # 每个进程都在自己的第一个设备上算同样的东西（同 elementwise_ulp.py），只有进程 0 写结果
    import jax.numpy as jnp
    import mt_common
    from maxtext.kernels import splash_dsinks_fp32
    try:
        from maxtext.kernels.splash_dsinks_gpu_order import gpu_order_dsinks
    except ImportError:  # 早于 M42b 的 MaxText
        gpu_order_dsinks = None
    from op_replay_mini import load_st
    from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
    from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib

    cfg = mt_common.make_config(a.stage, ["run_name=sink_probe", "base_output_directory=/tmp/hy4out",
                                          "skip_jax_distributed_system=True"])
    layers = [x for x in a.layers.split(",") if x]
    hf = lambda L: "hf.model.mtp_layers.0" if L == "mtp0" else f"hf.model.layers.{L}"  # noqa: E731
    pre = lambda L: "model.mtp_layers.0" if L == "mtp0" else f"model.layers.{L}"  # noqa: E731
    names = {f"{hf(L)}.self_attn.core.{t}" for L in layers for t in ("query", "key", "value")}
    G = load_st(a.gpu_fwd, names=names)
    W = load_st(a.hf_weights, names={f"{pre(L)}.self_attn.learnable_sink_param" for L in layers})
    dev = jax.local_devices()[0]
    out, arrays = {"layers": {}}, {}
    for L in layers:
        q = np.asarray(G[f"{hf(L)}.self_attn.core.query"], np.float32)[0]  # [S, H, D]
        S = a.tokens or q.shape[0]
        q = q[:S] * (q.shape[-1] ** -0.5)  # softmax_scale = qk_head_dim^-0.5 = 1/16：BF16 里精确
        k = np.asarray(G[f"{hf(L)}.self_attn.core.key"], np.float32)[0][:S]
        v = np.asarray(G[f"{hf(L)}.self_attn.core.value"], np.float32)[0][:S]
        sinks = np.asarray(W[f"{pre(L)}.self_attn.learnable_sink_param"], np.float32)
        H = q.shape[1]
        do = np.random.default_rng(1234).standard_normal((H, S, v.shape[-1])).astype(np.float32)
        to = lambda x: jax.device_put(jnp.asarray(np.ascontiguousarray(np.swapaxes(x, 0, 1)), jnp.bfloat16), dev)  # noqa: E731
        qd, kd, vd = to(q), to(k), to(v)
        dod = jax.device_put(jnp.asarray(do, jnp.bfloat16), dev)
        sd = jax.device_put(jnp.asarray(sinks), dev)
        rec = {}
        for base2 in (True, False):
            sc = splash.SplashConfig(block_q=cfg.sa_block_q, block_kv=cfg.sa_block_kv, block_kv_compute=cfg.sa_block_kv_compute,
                                     block_q_dkv=cfg.sa_block_q_dkv, block_kv_dkv=cfg.sa_block_kv_dkv,
                                     block_kv_dkv_compute=cfg.sa_block_kv_dkv_compute, use_base2_exp=base2,
                                     fuse_reciprocal=cfg.sa_fuse_reciprocal, interpret=dev.platform == "cpu")
            mask = mask_lib.CausalMask((S, S))
            res_fn = splash.make_splash_mha_single_device(mask, config=sc, save_residuals=True)
            o, stats = jax.jit(lambda q_, k_, v_, s_: res_fn(q_, k_, v_, None, s_))(qd, kd, vd, sd)
            tag = "base2" if base2 else "natural"
            arrays[f"{L}.{tag}.o"] = np.asarray(o, np.float32)
            arrays[f"{L}.{tag}.lse"] = np.asarray(stats["logsumexp"], np.float32)
            for fp32 in (False, True):
                splash_dsinks_fp32.set_enabled(fp32)
                attn = splash.make_splash_mha_single_device(mask, config=sc)
                _, vjp = jax.vjp(lambda q_, k_, v_, s_: attn(q_, k_, v_, None, s_), qd, kd, vd, sd)
                g = jax.jit(vjp)(dod)
                arrays[f"{L}.{tag}.dsinks_{'fp32' if fp32 else 'tokamax'}"] = np.asarray(g[3], np.float32)
            splash_dsinks_fp32.set_enabled(False)
            rec[tag] = {"block_q": sc.block_q, "use_base2_exp": base2}
        # MaxText splash_dsinks_gpu_order（M42b）：内核外按 GPU 写法重算的 sink 梯度（与 base2 开关无关，只用 q、k、v、do、sink）
        if gpu_order_dsinks is not None:
            causal = jax.device_put(jnp.asarray(np.tril(np.ones((S, S), bool))), dev)
            g42b = jax.jit(gpu_order_dsinks, static_argnums=6)(sd, qd, kd, vd, causal, dod, 256)
            arrays[f"{L}.dsinks_gpuorder"] = np.asarray(g42b, np.float32)
        arrays[f"{L}.q"], arrays[f"{L}.k"], arrays[f"{L}.v"] = (np.asarray(x, np.float32) for x in (qd, kd, vd))
        arrays[f"{L}.do"] = np.asarray(dod, np.float32)
        arrays[f"{L}.sinks"] = sinks
        out["layers"][L] = {"tokens": S, "heads": H, **rec}
        print(f"[sink_probe] 第 {L} 层完成", flush=True)
    out.update({"platform": jax.devices()[0].platform, "device_kind": jax.devices()[0].device_kind,
                "note": "TPU 一侧的 splash 精度探针；FP64 比较见 analysis/sink_probe_compare.py"})
    if jax.process_index() == 0:
        from safetensors.numpy import save_file
        save_file(arrays, str(Path(a.out).with_suffix(".safetensors")))
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
