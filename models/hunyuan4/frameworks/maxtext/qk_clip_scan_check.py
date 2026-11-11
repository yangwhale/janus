"""QK-Clip 在层扫描（scan）下是不是每层各自一个缩放：小模型前向一次，按 MaxText 的 apply_qk_clip_nnx 改写 wq_b，逐层核。

    JAX_PLATFORMS=cpu python qk_clip_scan_check.py --out <结果.json> [--layers 11] [覆盖 …]

做法：hy4-tiny 结构，层数改成 --layers（默认 1 稠密 + 10 MoE：IndexCache 每 4 层一组，扫描长度 2，另有 2 层余数块），
threshold 设得很小，让每层每个头都被裁。前向一次取出 MaxText 记下的 max_logits，调用训练里同一个
`qk_clip_utils.apply_qk_clip_nnx`，再看每个 wq_b 的 nope 部分被乘了多少（= √scale）：
- 期望（Kimi K2 / MuonClip 的定义）：每层每个头一个 scale = min(1, τ / 该层该头的 max logit)；
- 实际：从改写前后的权重反推出每层每个头的 √scale，与按每层 max logit 算的期望逐个比，并看扫描维上各层是否相同。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=11)
    p.add_argument("--tau", type=float, default=1e-3)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from maxtext.configs import pyconfig
    from maxtext.layers.attention_mla import indexer_losses
    from maxtext.utils import maxtext_utils, model_creation_utils, qk_clip_utils
    from maxtext.utils.globals import MAXTEXT_PKG_DIR

    try:
        from maxtext.layers.multi_token_prediction import mtp_acceptance, mtp_losses
    except ImportError:  # 路径因版本而异
        mtp_losses = mtp_acceptance = None

    cfg = pyconfig.initialize([
        None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"), "model_name=hy4-tiny", "override_model_config=true",
        f"base_num_decoder_layers={a.layers}", "scan_layers=True", "attention=dot_product", "use_qk_clip=True",
        f"qk_clip_threshold={a.tau}", "dtype=float32", "weight_dtype=float32", "matmul_precision=highest",
        "per_device_batch_size=1", "max_target_length=64", "mtp_num_layers=0", "enable_dropout=false",
        "skip_jax_distributed_system=True", *a.overrides])
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        model = model_creation_utils.get_nnx_create_model_fn(cfg, mesh)()
        B, T = cfg.global_batch_size_to_train_on, cfg.max_target_length
        rng = np.random.default_rng(0)
        tok = jnp.asarray(rng.integers(0, cfg.vocab_size, (B, T)), jnp.int32)
        pos = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))
        seg = jnp.ones((B, T), jnp.int32)
        model(decoder_input_tokens=tok, decoder_positions=pos, decoder_segment_ids=seg, enable_dropout=False,
              decoder_target_tokens=tok, decoder_target_mask=seg)
        for col in (mtp_losses, mtp_acceptance, indexer_losses):
            if col is not None:
                nnx.pop(model, col)
        inter = nnx.pop(model, nnx.Intermediate).to_pure_dict()

        found = []

        def collect(path, x):
            keys = [str(getattr(k, "key", getattr(k, "idx", k))) for k in path]
            if "max_logits" in keys:
                found.append(("/".join(keys), np.asarray(x)))

        jax.tree_util.tree_map_with_path(collect, inter)
        _, params, _ = nnx.split(model, nnx.Param, ...)
        before = {"/".join(map(str, k)): np.asarray(v) for k, v in
                  jax.tree_util.tree_flatten_with_path(params.to_pure_dict())[0] for k in [tuple(getattr(e, "key", e) for e in k)]
                  if "wq_b" in "/".join(map(str, k)) and "indexer" not in "/".join(map(str, k))}
        qk_clip_utils.apply_qk_clip_nnx(SimpleNamespace(model=model), inter, cfg)
        _, params2, _ = nnx.split(model, nnx.Param, ...)
        after = {"/".join(map(str, k)): np.asarray(v) for k, v in
                 jax.tree_util.tree_flatten_with_path(params2.to_pure_dict())[0] for k in [tuple(getattr(e, "key", e) for e in k)]
                 if "wq_b" in "/".join(map(str, k)) and "indexer" not in "/".join(map(str, k))}

    nope = cfg.qk_nope_head_dim
    res = {"layers": a.layers, "tau": a.tau, "max_logits_leaves": [], "wq_b": []}
    ml_by_block = {}
    for path, x in found:
        res["max_logits_leaves"].append({"path": path, "shape": list(x.shape)})
        block = path.split("/max_logits")[0].split("/self_attention")[0]
        ml_by_block[block] = x
    for k in sorted(before):
        w0, w1 = before[k], after[k]
        # nope 部分的比值 = √scale；按每个头取中位数（权重元素间应完全一致）
        r = w1[..., :nope] / np.where(w0[..., :nope] == 0, np.nan, w0[..., :nope])
        heads_axis = w0.ndim - 2
        red = tuple(i for i in range(w0.ndim) if i not in (heads_axis,) and not (w0.ndim == 4 and i == 1))
        per = np.nanmedian(r, axis=tuple(i for i in red))  # 扫描时形状 [层, 头]，否则 [头]
        spread = float(np.nanmax(np.abs(r - np.expand_dims(np.expand_dims(per, 0), -1) if w0.ndim == 3 else
                                          r - per[None, :, :, None])))
        block = k.split("/self_attention")[0]
        ml = ml_by_block.get(block)
        entry = {"param": k, "shape": list(w0.shape), "sqrt_scale_per_layer_head": np.asarray(per).tolist(),
                 "within_head_spread": spread}
        if ml is not None:
            # 期望：每层每个头一个 scale；max_logits 形状若带扫描维，扫描维在最前
            if w0.ndim == 4:  # [rank, 扫描层, 头, 维]
                L = w0.shape[1]
                ml_l = ml.reshape(L, -1, ml.shape[-1]) if ml.shape[0] == L else None
                if ml_l is not None:
                    smax = ml_l.max(axis=1)
                    entry["expected_sqrt_scale_per_layer_head"] = np.sqrt(np.minimum(1.0, a.tau / (smax + 1e-6))).tolist()
                entry["max_logits_shape"] = list(ml.shape)
            else:
                smax = ml.reshape(-1, ml.shape[-1]).max(axis=0)
                entry["expected_sqrt_scale_per_layer_head"] = np.sqrt(np.minimum(1.0, a.tau / (smax + 1e-6))).tolist()
                entry["max_logits_shape"] = list(ml.shape)
        res["wq_b"].append(entry)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for e in res["wq_b"]:
        print(e["param"], e["shape"], "ml", e.get("max_logits_shape"))
        print("  actual  ", np.round(np.asarray(e["sqrt_scale_per_layer_head"]), 6).tolist())
        if "expected_sqrt_scale_per_layer_head" in e:
            print("  expected", np.round(np.asarray(e["expected_sqrt_scale_per_layer_head"]), 6).tolist())


if __name__ == "__main__":
    main()
