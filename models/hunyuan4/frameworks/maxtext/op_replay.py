"""L1 逐算子隔离回放（TPU 一侧）：每个算子的输入换成 GPU golden 第 1 步抓到的同一个输入，抓这个算子的输出，
和 GPU 的输出、FP64 参照比，算 R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64)（PROTOCOL.md 第 5 节，过线 R ≤ 1.2）。

和 GPU 一侧的 analysis/fp64_ops.py 同一种隔离：每个算子吃 GPU rank 0 第一个 microbatch 的输入，误差不沿层累积。
MaxText 不能跑 FP64（配置不收 float64），FP64 参照用 fp64_ops.py 存下的输出（--fp64-npz）；没有时只报 rel_L2(TPU, GPU)。

  python op_replay.py --stage A --load <检查点>/0/items --golden <ga1 目录> [--fp64-npz <fp64_ops 的输出>] --out op_replay.json
  # TPU 上：每个设备一行同样的输入（数据并行），只取第 0 行；SCRIPT=op_replay.py 用 run.sh 提交。CPU 上能跑通，但那不是 TPU 的数值

只支持阶段 A（第 0 层；扫描层里只有一层时整栈注入同一个输入才等于逐算子）。
iHC 的混合（attn_hc / ffn_hc）另跑一遍：MaxText 把 iHC 与分支合在一个模块里调，所以把 iHC 模块的输入（4 路流）换成 GPU 一侧的
同一个输入（attn_hc：embedding 输出复制 4 路；ffn_hc：attn_hc.post × self_attn.out + 4 路流，FP64 里算，与 fp64_ops.py 相同），
抓分支里第一个 RMSNorm 的输入（就是混合后送进分支的那一路），对 HF 的 attn_hc.out / ffn_hc.out。post 门控本身不单独比。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mt_common  # noqa: E402

# (算子名（与 fp64_ops.py 同名）, MaxText 模块路径, 注入的 GPU 输入, 比较的 GPU 输出)
OPS = [
    ("input_layernorm", "decoder/dense_layers/pre_self_attention_layer_norm", "hf.model.layers.0.attn_hc.out",
     "hf.model.layers.0.input_layernorm.out"),
    ("self_attn.gate_proj", "decoder/dense_layers/self_attention/gate_proj", "hf.model.layers.0.input_layernorm.out",
     "hf.model.layers.0.self_attn.gate_proj.out"),
    ("self_attn(own topk)", "decoder/dense_layers/self_attention", "hf.model.layers.0.input_layernorm.out",
     "hf.model.layers.0.self_attn.out"),
    ("post_attention_layernorm", "decoder/dense_layers/post_self_attention_layer_norm", "hf.model.layers.0.ffn_hc.out",
     "hf.model.layers.0.post_attention_layernorm.out"),
    ("mlp", "decoder/dense_layers/mlp", "hf.model.layers.0.post_attention_layernorm.out", "hf.model.layers.0.mlp.out"),
    ("hc_head", "decoder/hc_head", "hf.model.layers.0.out0", "hf.model.hc_head.out"),
    ("final_norm", "decoder/decoder_norm", "hf.model.hc_head.out", "hf.model.norm.out"),
    ("lm_head", "decoder/logits_dense", "hf.model.norm.out", "hf.lm_head.out"),
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="A", choices=["A"])
    p.add_argument("--load", required=True)
    p.add_argument("--golden", required=True, help="有 step_1/ 全量抓数的 GPU golden（ga1-*）")
    p.add_argument("--fp64-npz", default=None, help="fp64_ops.py 存下的 FP64 输出（键 = 算子名）")
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--out", default=None)
    p.add_argument("--cpu", action="store_true", help="在 CPU 上跑（dot_product 注意力）：只验证脚本，不是 TPU 的数值")
    p.add_argument("maxtext_args", nargs="*")
    a = p.parse_args(argv)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    if not a.cpu and not jax.distributed.is_initialized():
        jax.distributed.initialize()
    from maxtext.utils import model_creation_utils

    from janus import metrics as M
    from janus.dump import maxtext as jd
    from janus.store import Run

    T = a.tokens
    r = Run.open(a.golden)
    toks = np.asarray(r.tensor(1, "input.tokens")).reshape(1, -1)[:, :T].astype(np.int32)
    pos = np.arange(T, dtype=np.int32)[None]
    # 每个设备一行、整行相同（数据并行铺满全部设备），只取第 0 行：单机单卡、v5p 8 片都能跑
    n = jax.device_count()
    extra = ["run_name=op_replay", f"max_target_length={T}", "per_device_batch_size=1", "gradient_accumulation_steps=1",
             "ici_fsdp_parallelism=1", "ici_data_parallelism=-1", f"load_parameters_path={a.load}"]
    if a.cpu:
        extra += ["skip_jax_distributed_system=True", "hardware=cpu", "attention=dot_product", "sparse_matmul=False", "megablox=False", "use_tokamax_splash=False"]
    config = mt_common.make_config(a.stage, extra + list(a.maxtext_args))
    mesh = mt_common.make_mesh(config)
    model = model_creation_utils.from_pretrained(config, mesh)
    tagged = jd.tag_model(model, jd.HY4_TAG_RULES)
    missing = [o[1] for o in OPS if o[1] not in tagged]
    if missing:
        raise SystemExit(f"这些模块没打上标记：{missing}")
    inject = {path: np.repeat(np.asarray(r.tensor(1, gin))[:1, :T], n, axis=0) for _, path, gin, _ in OPS}
    toks = np.repeat(toks, n, axis=0)
    pos = np.repeat(pos, n, axis=0)

    @nnx.jit
    def fwd(m, x, ps, inj):
        with jd.capturing(rows=1, xent=False, inject=inj):
            m(decoder_input_tokens=x, decoder_positions=ps, decoder_segment_ids=jnp.ones_like(x), enable_dropout=False)
            inter = nnx.pop(m, nnx.Intermediate)
        return inter.to_pure_dict()

    t0 = time.time()
    with jax.set_mesh(mesh):
        inter = fwd(model, jnp.asarray(toks), jnp.asarray(pos), {k: jnp.asarray(v) for k, v in inject.items()})
    sown = {k: np.asarray(jax.device_get(v)) for k, v in jd.collect_sown(inter).items()}
    # 第二遍：iHC 混合。注入 iHC 模块的 4 路流输入，抓分支里第一个 RMSNorm 的输入
    G = lambda k: np.asarray(r.tensor(1, k), np.float64)[:1, :T]  # noqa: E731
    streams = np.repeat(G("hf.model.embed_tokens.out")[:, :, None, :], 4, axis=2)
    resid = G("hf.model.layers.0.attn_hc.post")[..., None] * G("hf.model.layers.0.self_attn.out")[:, :, None, :] + streams
    inject2 = {"decoder/dense_layers/mhc_attention": np.repeat(streams, n, axis=0),
               "decoder/dense_layers/mhc_mlp": np.repeat(resid, n, axis=0)}
    with jax.set_mesh(mesh):
        inter2 = fwd(model, jnp.asarray(toks), jnp.asarray(pos), {k: jnp.asarray(v, jnp.float32) for k, v in inject2.items()})
    sown2 = {k: np.asarray(jax.device_get(v)) for k, v in jd.collect_sown(inter2).items()}
    ihc_ops = [("attn_hc.out", "decoder/dense_layers/pre_self_attention_layer_norm/janus__in", "hf.model.layers.0.attn_hc.out"),
               ("ffn_hc.out", "decoder/dense_layers/post_self_attention_layer_norm/janus__in", "hf.model.layers.0.ffn_hc.out")]
    ref64 = np.load(a.fp64_npz) if a.fp64_npz else None
    rows = []
    for name, key, gout in ihc_ops:
        x = np.asarray(sown2[key][0][0], np.float64)
        g = np.asarray(r.tensor(1, gout), np.float64)[0][:T]
        x = x.reshape(g.shape)
        row = {"op": name, "maxtext": key, "rel_l2_vs_gpu": M.rel_l2(x, g)}
        if ref64 is not None and name in ref64:
            f = np.asarray(ref64[name], np.float64).reshape(g.shape)
            row["rel_l2_gpu_vs_fp64"] = M.rel_l2(g, f)
            row["rel_l2_maxtext_vs_fp64"] = M.rel_l2(x, f)
            row["R"] = row["rel_l2_maxtext_vs_fp64"] / max(row["rel_l2_gpu_vs_fp64"], 1e-300)
        rows.append(row)
    for name, path, _gin, gout in OPS:
        x = sown[f"{path}/janus__out"]
        if "/dense_layers/" in path + "/":
            x = x[0]
        x = np.asarray(x[0], np.float64)
        g = np.asarray(r.tensor(1, gout), np.float64)[0][:T]
        x = x.reshape(g.shape)
        row = {"op": name, "maxtext": path, "rel_l2_vs_gpu": M.rel_l2(x, g)}
        if ref64 is not None and name in ref64:
            f = np.asarray(ref64[name], np.float64).reshape(g.shape)
            row["rel_l2_gpu_vs_fp64"] = M.rel_l2(g, f)
            row["rel_l2_maxtext_vs_fp64"] = M.rel_l2(x, f)
            row["R"] = row["rel_l2_maxtext_vs_fp64"] / max(row["rel_l2_gpu_vs_fp64"], 1e-300)
        rows.append(row)
    rec = {"stage": a.stage, "tokens": T, "platform": jax.devices()[0].platform, "device_kind": jax.devices()[0].device_kind,
           "seconds": time.time() - t0, "fp64_reference": a.fp64_npz, "ops": rows,
           "note": "CPU 上跑的只验证脚本，不是 TPU 的数值" if a.cpu else "TPU 一侧的逐算子隔离回放"}
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
