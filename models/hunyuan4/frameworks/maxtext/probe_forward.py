"""诊断用的单独前向（接线核对）：用转换好的 MaxText 检查点，对 golden 第 1 步 rank 0 第 0 个 microbatch 的前 T 个 token
跑一次前向，算逐 token 交叉熵，和 GPU golden 第 1 步的 loss.per_token（有全量抓数的 golden，如 ga1-a）比。

和混元 3 的 probe_forward.py 一样只当「转换与读法接对了没有」的核对（错一处 RoPE 列序、eps、转置，逐 token loss 差 O(1)；
只差平台舍入时是 1e-3 量级），不当作 TPU 对 GPU 的精度结论。在 CPU 上跑（dot_product 注意力）时也不是 TPU 的数值。
T ≤ 2048 时 indexer 的 top-k 覆盖全部可见位置（稀疏注意力退化成稠密）；T = 4096 才走到真稀疏的那一段。

  JAX_PLATFORMS=cpu python probe_forward.py --stage A --load <检查点>/0/items --golden <golden 目录> --tokens 4096 --out probe.json
  # golden 目录要有 data/batches（取输入）；有 step_1/ 的 loss.per_token 时一并比较
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
import replay  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=sorted(mt_common.STAGES))
    p.add_argument("--load", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--ref", default=None, help="有 step_1/loss.per_token 的 golden 目录（默认同 --golden）")
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--out", default=None)
    p.add_argument("maxtext_args", nargs="*")
    a = p.parse_args(argv)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.utils import model_creation_utils

    T = a.tokens
    gb = replay.GoldenBatches(a.golden)
    g1 = gb.global_batch(gb.steps[0])
    toks = g1["tokens"][0:1, :T].astype(np.int32)                 # 第 0 行 = rank 0 的 micro 0
    labels = g1["labels"][0:1, :T].astype(np.int64)
    pos = np.arange(T, dtype=np.int32)[None]
    config = mt_common.make_config(a.stage, ["hardware=cpu", "skip_jax_distributed_system=True", "run_name=probe",
                                             "attention=dot_product", "sparse_matmul=False", "megablox=False",
                                             "use_tokamax_splash=False", f"max_target_length={T}", "per_device_batch_size=1",
                                             "gradient_accumulation_steps=1", "ici_fsdp_parallelism=1",
                                             f"load_parameters_path={a.load}", *a.maxtext_args])
    mesh = mt_common.make_mesh(config)
    t0 = time.time()
    model = model_creation_utils.from_pretrained(config, mesh)
    print(f"载入 {time.time() - t0:.0f} s", flush=True)

    from janus.dump import maxtext as jd
    tagged = jd.tag_model(model, jd.HY4_TAG_RULES)

    @nnx.jit
    def fwd(m, x, ps):
        with jd.capturing(rows=1, xent=False):
            lg = m(decoder_input_tokens=x, decoder_positions=ps, decoder_segment_ids=jnp.ones_like(x), enable_dropout=False)
            inter = nnx.pop(m, nnx.Intermediate)
        return lg, inter.to_pure_dict()

    t0 = time.time()
    with jax.set_mesh(mesh):
        logits, inter = fwd(model, jnp.asarray(toks), jnp.asarray(pos))
    sown = {k: np.asarray(jax.device_get(v)) for k, v in jd.collect_sown(inter).items()}
    logits = np.asarray(jax.device_get(logits)).astype(np.float64)[0]
    lse = np.log(np.sum(np.exp(logits - logits.max(-1, keepdims=True)), -1)) + logits.max(-1)
    tok_loss = lse - logits[np.arange(T), labels[0]]
    rec = {"stage": a.stage, "tokens": T, "seconds": time.time() - t0, "loss_mean": float(tok_loss.mean()),
           "tagged_modules": len(tagged), "sown": {k: list(v.shape) for k, v in sorted(sown.items())},
           "loss_first8": [float(x) for x in tok_loss[:8]],
           "platform": jax.devices()[0].platform, "config_dtype": str(config.dtype)}
    ref_dir = Path(a.ref or a.golden)
    if (ref_dir / "step_1").is_dir():
        from janus import metrics as M
        from janus.store import Run
        r = Run.open(ref_dir)
        have = set(r.names(1))
        if "input.tokens" in have:
            rec["ref_tokens_equal"] = bool(np.array_equal(np.asarray(r.tensor(1, "input.tokens")).reshape(-1)[:T], toks[0]))
        if "hf.lm_head.out" in have:
            ref_l = np.asarray(r.tensor(1, "hf.lm_head.out"), np.float64).reshape(-1, logits.shape[-1])[:T]
            rec["ref_logits"] = {"name": "hf.lm_head.out", "rel_l2": M.rel_l2(logits, ref_l), "cosine": M.cosine(logits, ref_l),
                                 "max_abs": M.max_abs(logits, ref_l)[0],
                                 "argmax_agree": float(np.mean(logits.argmax(-1) == ref_l.argmax(-1)))}
            del ref_l
        # 模块边界逐个比（第 0 层；扫描层的值多一个前导层轴）
        pairs = [("token_embedder/janus__out", "hf.model.embed_tokens.out"),
                 ("decoder/dense_layers/pre_self_attention_layer_norm/janus__out", "hf.model.layers.0.input_layernorm.out"),
                 ("decoder/dense_layers/self_attention/gate_proj/janus__out", "hf.model.layers.0.self_attn.gate_proj.out"),
                 ("decoder/dense_layers/self_attention/out/janus__out", "hf.model.layers.0.self_attn.o_proj.out"),
                 ("decoder/dense_layers/self_attention/janus__out", "hf.model.layers.0.self_attn.out"),
                 ("decoder/dense_layers/post_self_attention_layer_norm/janus__out", "hf.model.layers.0.post_attention_layernorm.out"),
                 ("decoder/dense_layers/mlp/janus__out", "hf.model.layers.0.mlp.out"),
                 ("decoder/dense_layers/janus__out", "hf.model.layers.0.out0"),
                 ("decoder/hc_head/janus__out", "hf.model.hc_head.out"),
                 ("decoder/decoder_norm/janus__out", "hf.model.norm.out")]
        mods = []
        for mk, gn in pairs:
            if mk not in sown or gn not in have:
                mods.append({"maxtext": mk, "gpu": gn, "missing": mk not in sown and "maxtext" or "gpu"})
                continue
            ref = np.asarray(r.tensor(1, gn), np.float64)[0][:T]
            x = sown[mk]
            if "/dense_layers/" in mk + "/":
                x = x[0]                                              # 扫描层：第 0 层
            x = np.asarray(x[0], np.float64)[:T]                      # 第 0 行
            if x.size != ref.size:
                mods.append({"maxtext": mk, "gpu": gn, "shape_maxtext": list(x.shape), "shape_gpu": list(ref.shape)})
                continue
            x = x.reshape(ref.shape)
            mods.append({"maxtext": mk, "gpu": gn, "rel_l2": M.rel_l2(x, ref), "cosine": M.cosine(x, ref)})
        rec["modules"] = mods
        names = [n for n in have if n.startswith("loss.per_token")]
        if names:
            ref = np.asarray(r.tensor(1, names[0]), np.float64).reshape(-1)[:T]
            rec["ref"] = {"name": names[0], "loss_mean": float(ref.mean()),
                          "rel_l2": M.rel_l2(tok_loss, ref), "max_abs": M.max_abs(tok_loss, ref)[0],
                          "mean_diff": float((tok_loss - ref).mean())}
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
