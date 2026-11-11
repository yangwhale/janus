"""L1 逐算子隔离回放，hy4-mini（op_replay.py 的阶段 mini；阶段 A 仍在 op_replay.py）。

每个算子的输入换成 GPU（Megatron-Core + TE，`g7-l1-a` 第 1 步 rank 0 第 0 个 microbatch，`--l1-capture-step`）抓到的同一个输入，
抓这个算子的 TPU 输出；和 GPU 的输出、FP64 参照（GPU 一侧 `analysis/fp64_l1_gpu.py` 用同一份输入在 CPU 上的重算）比，
R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64)（PROTOCOL.md 第 5 节，≤ 1.2 过）。

一次整网前向里同时注入全部算子（每个算子吃的都是 GPU 的输入，误差不沿层累积），另一遍注入 iHC 的 4 路流与分支输入，抓：
- 每层（0–6 与 MTP）：输入 RMSNorm、输出门投影、注意力整块（自选 top-k；IndexCache 共享层用块首层在 GPU 输入上自选的掩码）、
  FFN 前 RMSNorm、稠密 MLP / MoE 整块（自选路由）、路由 logits（sigmoid 之前，FP32）、共享专家、路由专家（MoE 整块减共享专家）；
- iHC：两处混合（4 路流 → 分支输入）、两处写回（4 路流 + post × 分支输出，分支的输出换成 GPU 的：janus 抓数的 inject_out）；
- hc_head、最后的 RMSNorm、FP32 输出头、交叉熵（GPU 的 logits 进 MaxText 的交叉熵）、embedding；MTP 的 enorm / hnorm / eh_proj / final norm。
TPU 一侧（`--gpu-fwd`）只写 TPU 输出与对 GPU 的 rel_L2；对 FP64 的 R 由 `--compare` 在 CPU 上算（FP64 参照 14 GB，不进 Pod）。

  # TPU（SCRIPT=op_replay_mini.py，run.sh 的其余脚本分支）：
  python op_replay_mini.py --load <检查点> --gpu-fwd gs://…/mini-l1/gpu/step_1_fwd.safetensors --out op_replay_mini.json
  # CPU：
  python op_replay_mini.py --compare <TPU 运行目录> --gpu-fwd <本地 step_1_fwd> --fp64 <本地 fp64_l1_step1.safetensors> \\
      --fp64-json <fp64_l1_step1.json> --out results/t1-l1-op-tpu.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

R_LIMIT = 1.2
MTP = "mtp_block/mtp_layer_1"


def layer_path(L) -> str:
    if L == "mtp0":
        return f"{MTP}/mtp_1_transformer_layer"
    if L == 0:
        return "decoder/dense_layers"
    if L <= 4:
        return f"decoder/moe_layers/layers_{L - 1}"
    return f"decoder/moe_layers_remainder/layers_{L - 5}"


def hf(L) -> str:
    return "hf.model.mtp_layers.0" if L == "mtp0" else f"hf.model.layers.{L}"


LAYERS = [0, 1, 2, 3, 4, 5, 6, "mtp0"]


def rules() -> list[tuple[str, str]]:
    out = [(r"^token_embedder$", "embed"), (r"^decoder/hc_head$", "norm_out"), (r"^decoder/decoder_norm$", "final_norm"),
           (r"^decoder/logits_dense$", "logits")]
    for n in ("embedding_norm", "hidden_state_norm", "projection", "final_norm"):
        out.append((rf"^{MTP}/mtp_1_{n}$", "norm_out"))
    for L in LAYERS:
        P = re.escape(layer_path(L))
        out += [(rf"^{P}$", "layer"), (rf"^{P}/(pre|post)_self_attention_layer_norm$", "norm_io"),
                (rf"^{P}/mhc_(attention|mlp)$", "ihc_out"), (rf"^{P}/self_attention$", "attn"),
                (rf"^{P}/self_attention/gate_proj$", "norm_out"), (rf"^{P}/mlp$", "mlp"),
                (rf"^{P}/self_attention/indexer/weights_proj$", "norm_out"),
                (rf"^{P}/DeepSeekMoeBlock_0$", "moe"), (rf"^{P}/DeepSeekMoeBlock_0/MoeBlock_0$", "routed"),
                (rf"^{P}/DeepSeekMoeBlock_0/shared_experts$", "shared"), (rf"^{P}/DeepSeekMoeBlock_0/MoeBlock_0/gate$", "gate")]
    return out


_FETCHED: dict[str, str] = {}


def fetch(path: str) -> str:
    """gs:// 路径下载到本地临时目录；同一路径只下载一次（--sweep 的每组都读同一份 GPU 抓数，7 GB，不缓存会撑满 Pod 的盘）。"""
    if not path.startswith("gs://"):
        return path
    if path not in _FETCHED:
        loc = Path(tempfile.mkdtemp()) / Path(path).name
        subprocess.run(["gcloud", "storage", "cp", "-q", path, str(loc)], check=True)
        _FETCHED[path] = str(loc)
    return _FETCHED[path]


def load_st(path: str, names=None) -> dict:
    """safetensors → {名: numpy}，浮点一律升 FP32（BF16 → FP32 精确）；janus.store 的读法，不要 torch。"""
    from janus.store import ST_TO_NP, read_safetensors_header, read_safetensors_tensor
    local = Path(fetch(path))
    header, _ = read_safetensors_header(local)
    out = {}
    for k, ent in header.items():
        if k == "__metadata__" or (names is not None and k not in names) or ent["dtype"] not in ST_TO_NP:
            continue
        x, _ = read_safetensors_tensor(local, k)
        out[k] = x.astype(np.float32) if x.dtype.kind == "f" or str(x.dtype) == "bfloat16" else np.array(x)
    return out


# (算子名, GPU 输出名, FP64 键, 在哪一遍抓、抓哪个点) —— 抓的点是 TPU 一侧 collect_sown 的键
def op_table() -> list[dict]:
    ops = [{"op": "embed", "gpu": "hf.model.embed_tokens.out", "fp64": "embed", "pass": 1, "key": "token_embedder/janus__out"},
           {"op": "hc_head", "gpu": "hf.model.hc_head.out", "fp64": "hc_head", "pass": 1, "key": "decoder/hc_head/janus__out"},
           {"op": "final_norm", "gpu": "hf.model.norm.out", "fp64": "final_norm", "pass": 1, "key": "decoder/decoder_norm/janus__out"},
           {"op": "lm_head", "gpu": "hf.lm_head.out", "fp64": "lm_head", "pass": 1, "key": "decoder/logits_dense/janus__out"},
           {"op": "cross_entropy", "gpu": "loss.per_token", "fp64": "cross_entropy(gpu logits)", "pass": "xent", "key": "xent"}]
    for n, g in (("embedding_norm", "enorm"), ("hidden_state_norm", "hnorm"), ("projection", "eh_proj"),
                 ("final_norm", "final_layernorm")):
        ops.append({"op": f"mtp0.{g}", "gpu": f"hf.model.mtp_layers.0.{g}.out", "fp64": f"hf.model.mtp_layers.0.{g}",
                    "pass": 1, "key": f"{MTP}/mtp_1_{n}/janus__out"})
    for L in LAYERS:
        P, H = layer_path(L), hf(L)
        moe = L != 0
        shared_layer = L in (2, 3, 4, 6)
        attn_fp64 = f"{H}.self_attn(gpu topk)" if shared_layer else f"{H}.self_attn(own topk)"
        ops += [
            {"op": f"{L}.input_layernorm", "gpu": f"{H}.input_layernorm.out", "fp64": f"{H}.input_layernorm", "pass": 1,
             "key": f"{P}/pre_self_attention_layer_norm/janus__out"},
            {"op": f"{L}.self_attn.gate_proj", "gpu": f"{H}.self_attn.gate_proj.out", "fp64": f"{H}.self_attn.gate_proj",
             "pass": 1, "key": f"{P}/self_attention/gate_proj/janus__out"},
            {"op": f"{L}.self_attn", "gpu": f"{H}.self_attn.out", "fp64": attn_fp64, "pass": 1,
             "key": f"{P}/self_attention/janus__out",
             "note": "共享层：top-k 掩码是块首层在 GPU 输入上自选的，FP64 用的是 GPU 的 top-k" if shared_layer else "自选 top-k"},
            {"op": f"{L}.self_attn.indexer.weights_proj", "gpu": None, "fp64": None, "pass": 1, "optional": True,
             "key": f"{P}/self_attention/indexer/weights_proj/janus__out", "fp64_local": "weights_proj",
             "note": "GPU 没抓；FP64 用 GPU 的 input_layernorm 输出与 HF 检查点（或 --fp64-override）的权重在本机算"},
            {"op": f"{L}.post_attention_layernorm", "gpu": f"{H}.post_attention_layernorm.out",
             "fp64": f"{H}.post_attention_layernorm", "pass": 1, "key": f"{P}/post_self_attention_layer_norm/janus__out"},
        ]
        if moe:
            ops += [
                {"op": f"{L}.mlp(own routing)", "gpu": f"{H}.mlp.out", "fp64": f"{H}.mlp(own routing)", "pass": 1,
                 "key": f"{P}/DeepSeekMoeBlock_0/janus__out"},
                {"op": f"{L}.mlp.gate.logits", "gpu": f"{H}.mlp.gate.logits", "fp64": f"{H}.mlp.gate.logits", "pass": 1,
                 "key": f"{P}/DeepSeekMoeBlock_0/MoeBlock_0/gate/janus__logits"},
                {"op": f"{L}.mlp.shared_experts", "gpu": f"{H}.mlp.shared_experts.out", "fp64": None, "pass": 1,
                 "key": f"{P}/DeepSeekMoeBlock_0/shared_experts/janus__out", "fp64_local": "shared"},
                {"op": f"{L}.mlp.routed_experts", "gpu": None, "fp64": None, "pass": 1,
                 "key": f"{P}/DeepSeekMoeBlock_0/MoeBlock_0/janus__out", "fp64_local": "routed"},
            ]
        else:
            ops.append({"op": f"{L}.mlp", "gpu": f"{H}.mlp.out", "fp64": f"{H}.mlp(block)", "pass": 1,
                        "key": f"{P}/mlp/janus__out"})
        if L != "mtp0":
            ops += [
                {"op": f"{L}.attn_hc.out", "gpu": f"{H}.attn_hc.out", "fp64": f"{H}.attn_hc.out", "pass": 2,
                 "key": f"{P}/pre_self_attention_layer_norm/janus__in"},
                {"op": f"{L}.ffn_hc.out", "gpu": f"{H}.ffn_hc.out", "fp64": f"{H}.ffn_hc.out", "pass": 2,
                 "key": f"{P}/post_self_attention_layer_norm/janus__in"},
                {"op": f"{L}.attn_residual", "gpu": f"{H}.ffn_hc.in", "fp64": f"{H}.attn_residual", "pass": 2,
                 "key": f"{P}/mhc_attention/janus__out", "note": "4 路流 + post × 注意力输出；注意力的输出换成 GPU 的（只测写回）"},
                {"op": f"{L}.ffn_residual", "gpu": f"{H}.out0", "fp64": f"{H}.ffn_residual", "pass": 2,
                 "key": f"{P}/mhc_mlp/janus__out", "note": "4 路流 + post × FFN 输出；FFN 的输出换成 GPU 的（只测写回）"},
            ]
        else:
            ops.append({"op": "mtp0.ffn_residual", "gpu": f"{H}.out0", "fp64": f"{H}.ffn_residual", "pass": 2,
                        "key": f"{P}/janus__out", "note": "层输入 + 注意力 + FFN；两个分支的输出换成 GPU 的（只测残差相加）"})
    return ops


def output_injections(G: dict) -> dict:
    """第 2 遍换输出：注意力与 FFN 两个分支的输出换成 GPU 的，iHC 写回（MTP：两次残差相加）只剩写回本身的算术，
    与 FP64 参照（用 GPU 的分支输出算写回）同一个输入。"""
    inj = {}
    for L in LAYERS:
        P, H = layer_path(L), hf(L)
        inj[f"{P}/self_attention"] = G[f"{H}.self_attn.out"]
        inj[f"{P}/mlp" if L == 0 else f"{P}/DeepSeekMoeBlock_0"] = G[f"{H}.mlp.out"]
    return inj


def injections(G: dict, which: int) -> dict:
    """第 which 遍的注入：{模块路径: GPU 数组 [1, S, …]}。"""
    inj = {}
    if which == 1:
        inj["decoder/hc_head"] = G["hf.model.layers.6.out0"]
        inj["decoder/decoder_norm"] = G["hf.model.hc_head.out"]
        inj["decoder/logits_dense"] = G["hf.model.norm.out"]
        for n, g in (("embedding_norm", "enorm"), ("hidden_state_norm", "hnorm"), ("projection", "eh_proj"),
                     ("final_norm", "final_layernorm")):
            inj[f"{MTP}/mtp_1_{n}"] = G[f"hf.model.mtp_layers.0.{g}.in"]
    for L in LAYERS:
        P, H = layer_path(L), hf(L)
        branch = f"{P}/mlp" if L == 0 else f"{P}/DeepSeekMoeBlock_0"
        if which == 1:
            pre_in = G[f"{H}.in0"] if L == "mtp0" else G[f"{H}.attn_hc.out"]
            inj[f"{P}/pre_self_attention_layer_norm"] = pre_in
            if L == "mtp0":   # MTP 没有 iHC：FFN 前 RMSNorm 的输入 = 层输入 + 注意力输出（FP64 里加，与 FP64 参照同一个输入）
                inj[f"{P}/post_self_attention_layer_norm"] = (G[f"{H}.in0"].astype(np.float64) +
                                                               G[f"{H}.self_attn.out"].astype(np.float64)).astype(np.float32)
            else:
                inj[f"{P}/post_self_attention_layer_norm"] = G[f"{H}.ffn_hc.out"]
        else:
            if L != "mtp0":
                inj[f"{P}/mhc_attention"] = G[f"{H}.attn_hc.in"]   # 第 0 层的 in0 是展开前的一路，4 路流看 attn_hc.in
                inj[f"{P}/mhc_mlp"] = G[f"{H}.ffn_hc.in"]
            else:
                inj[P] = G[f"{H}.in0"]
        if which == 1:
            inj[f"{P}/self_attention"] = G[f"{H}.input_layernorm.out"]
            inj[branch] = G[f"{H}.post_attention_layernorm.out"]
    return inj


def tpu_main(a) -> int:
    import jax
    if not a.cpu and not jax.distributed.is_initialized():
        jax.distributed.initialize()
    if jax.process_index() == 0:   # 先建运行目录：中途失败时 run.sh 也能认出交结果的 Pod、取回日志
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    import jax.numpy as jnp
    import mt_common
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from maxtext.utils import max_utils, model_creation_utils

    from janus import metrics as M
    from janus.dump import maxtext as jd

    t0 = time.time()
    G = load_st(a.gpu_fwd)
    S = int(G["input.tokens"].shape[1])
    n = jax.device_count()
    extra = ["run_name=op_replay_mini", f"max_target_length={S}", "per_device_batch_size=1", "gradient_accumulation_steps=1",
             "ici_fsdp_parallelism=1", "ici_data_parallelism=-1", f"load_parameters_path={a.load}",
             "base_output_directory=/tmp/hy4out"]
    if a.cpu:
        extra += ["skip_jax_distributed_system=True", "hardware=cpu", "attention=dot_product", "sparse_matmul=False",
                  "megablox=False", "use_tokamax_splash=False"]
    config = mt_common.make_config(a.stage, extra + a.mt_args)
    mesh = mt_common.make_mesh(config)
    with nn_partitioning.axis_rules(config.logical_axis_rules):
        model = model_creation_utils.from_pretrained(config, mesh)
    nb = set_router_bias(model, G, mesh)
    n_override = override_params(model, a) if getattr(a, "param_override", None) else 0
    tagged = jd.tag_model(model, rules())
    table = op_table()
    want = {o["key"].rsplit("/janus__", 1)[0] for o in table if o["pass"] in (1, 2) and not o.get("optional")}
    missing = sorted(p for p in want if p not in tagged)
    if missing:
        raise SystemExit(f"这些模块没打上标记：{missing}")
    rep = lambda x: np.repeat(np.asarray(x)[:1], n, axis=0)  # noqa: E731
    toks = rep(G["input.tokens"].astype(np.int32))
    pos = rep(np.arange(S, dtype=np.int32)[None])

    @nnx.jit
    def fwd(m, x, ps, inj, inj_out):
        with jd.capturing(rows=1, xent=False, inject=inj, inject_out=inj_out):
            m(decoder_input_tokens=x, decoder_positions=ps, decoder_segment_ids=jnp.ones_like(x), enable_dropout=False)
            inter = nnx.pop(m, nnx.Intermediate)
        return inter.to_pure_dict()

    sown = {}
    for which in (1, 2):
        inj = {k: jnp.asarray(rep(v)) for k, v in injections(G, which).items()}
        inj_out = {k: jnp.asarray(rep(v)) for k, v in output_injections(G).items()} if which == 2 else {}
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            inter = fwd(model, jnp.asarray(toks), jnp.asarray(pos), inj, inj_out)
        sown[which] = {k: np.asarray(jax.device_get(v)) for k, v in jd.collect_sown(inter).items()}
        if jax.process_index() == 0:
            print(f"第 {which} 遍：抓到 {len(sown[which])} 个点（{time.time() - t0:.0f} s）", flush=True)
    # 交叉熵：GPU 的 logits 进 MaxText 的交叉熵（FP32），与训练一样按 transcendental_accuracy_highest 选实现
    V = int(config.vocab_size)
    acc = bool(getattr(config, "transcendental_accuracy_highest", False))
    xent = jax.jit(lambda lg, lb: max_utils.cross_entropy_with_logits(
        lg, jax.nn.one_hot(lb, V, dtype=jnp.float32), 0.0, accurate=acc)[0])
    lab = G["input.labels"].astype(np.int32)
    out_x = np.asarray(jax.device_get(xent(jnp.asarray(G["hf.lm_head.out"], jnp.float32), jnp.asarray(lab))))

    def take(o):
        if o["pass"] == "xent":
            return out_x
        v = sown[o["pass"]].get(o["key"])
        if v is None:
            if o.get("optional"):
                return None
            raise SystemExit(f"没抓到 {o['key']}")
        v = np.asarray(v)
        # 扫描栈（层轴长 1）抓到的值多一维在最前
        while v.ndim > 2 and v.shape[0] == 1 and o["key"].startswith(("decoder/dense_layers", "decoder/moe_layers/")):
            v = v[0]
        return v[0] if v.shape[0] == n or v.shape[0] == 1 else v

    rows, save = [], {}
    for o in table:
        t = take(o)
        if t is None:
            continue
        t = t.astype(np.float32)
        if o["gpu"] is not None:
            g = G[o["gpu"]]
            if o["gpu"].endswith(("shared_experts.out", "shared_experts.in")):
                g = np.swapaxes(g, 0, 1)                  # Megatron 的 s,b → b,s
            g = np.asarray(g, np.float64).reshape(-1)
            t = t.reshape(-1)[: g.size] if t.size >= g.size else t.reshape(-1)
            r = {"op": o["op"], "maxtext": o["key"], "rel_l2_tpu_vs_gpu": M.rel_l2(t.astype(np.float64).reshape(g.shape), g)}
        else:
            r = {"op": o["op"], "maxtext": o["key"]}
        for k in ("fp64", "note", "fp64_local"):
            if o.get(k):
                r[k] = o[k]
        rows.append(r)
        save[o["op"]] = np.ascontiguousarray(t)
    if getattr(a, "save_ops", None):  # 只留名字匹配的算子（--sweep 里每组只存它碰到的算子，省得每组都传回 4 GB）
        keep = re.compile(a.save_ops)
        rows = [r for r in rows if keep.search(r["op"])]
        save = {k: v for k, v in save.items() if keep.search(k)}
    rec = {"stage": a.stage, "tokens": S, "platform": jax.devices()[0].platform, "device_kind": jax.devices()[0].device_kind,
           "router_bias_set": nb, "gpu_fwd": a.gpu_fwd, "overrides": list(a.mt_args),
           "param_override": {"file": a.param_override, "only": a.override_only, "n_leaves": n_override} if n_override else None, "transcendental_accuracy_highest": acc,
           "ihc_force_bf16_rounding": bool(getattr(config, "ihc_force_bf16_rounding", False)), "seconds": time.time() - t0, "ops": rows,
           "note": "CPU 上跑的只验证脚本，不是 TPU 的数值" if a.cpu else "TPU 一侧的逐算子隔离回放（hy4-mini）"}
    if jax.process_index() == 0:
        from safetensors.numpy import save_file
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        save_file(save, str(Path(a.out).with_suffix(".safetensors")))
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float))
        print(json.dumps([(r["op"], round(r.get("rel_l2_tpu_vs_gpu", -1), 6)) for r in rows], ensure_ascii=False), flush=True)
    return 0


def override_params(model, a) -> int:
    """--param-override：把 HF 名字的 FP32 张量（如 GPU 训练若干步后的主权重 main_param_after）换进模型参数，只换
    --override-only 匹配到的那几个 HF 张量所在的 MaxText 叶子（HF → MaxText 布局与 convert_hf_to_maxtext 同一段代码）。
    第 1 步的检查点权重都在 BF16 网格上，只作用在权重一侧的精度开关（M13、M27、M8）第 1 步量不到，换成离开网格的权重才量得到。"""
    import jax
    import mt_common
    from flax import nnx
    from make_tiny_hf import hf_shapes
    from optim_replay import build, hf_to_mt

    keep = re.compile(a.override_only or ".")
    P = {k: v for k, v in load_st(a.param_override).items() if keep.search(k)}
    for pre in ("main_param_after.", "main_param.", "param."):  # janus 抓数的前缀
        P = {(k[len(pre):] if k.startswith(pre) else k): v for k, v in P.items()}
    extra = ["hardware=cpu", "skip_jax_distributed_system=True", "run_name=override", "attention=dot_product"]
    cfg_c, abstract, mapping, hooks, hf_cfg, slash_of = build(a.hf_config, a.stage, extra)
    shapes = hf_shapes(hf_cfg, *mt_common.STAGES[a.stage])
    names = lambda m: [m] if isinstance(m, str) else [x for x in m if isinstance(x, str)]  # noqa: E731
    touched = {k for k in abstract if slash_of[k].startswith("params/") and any(n in P for n in names(mapping[k]))}
    vals = hf_to_mt(cfg_c, {k: abstract[k] for k in touched}, mapping, hooks, slash_of, P, shapes)
    st = nnx.state(model, nnx.Param)
    n = 0
    for path, v in nnx.to_flat_state(st):
        key = "params/" + "/".join(str(x) for x in path)
        if key not in vals:
            continue
        old = v.get_value()
        x = np.ascontiguousarray(vals[key].reshape(old.shape)).astype(old.dtype)
        v.set_value(jax.make_array_from_callback(x.shape, old.sharding, lambda idx, x=x: x[idx]))
        n += 1
    nnx.update(model, st)
    if jax.process_index() == 0:
        print(f"[override] {len(P)} 个 HF 张量 → {n} 个 MaxText 叶子：{sorted(vals)}", flush=True)
    return n


def set_router_bias(model, G: dict, mesh) -> int:
    """from_pretrained 只载入 Param，路由偏置（MoEBiasVar）留成已删除的数组；换成 GPU 抓到的 expert_bias（第 1 步更新前 = 检查点初值）。"""
    import jax
    from flax import nnx
    rsh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())   # 多主机：复制到每个设备的全局数组
    st = nnx.state(model)
    n = 0
    for path, v in nnx.to_flat_state(st):
        if type(v).__name__ != "MoEBiasVar":
            continue
        p = "/".join(str(x) for x in path)
        L = next((L for L in LAYERS if L != 0 and p.startswith(layer_path(L) + "/")), None)
        if L is None:
            continue
        b = np.asarray(G[f"{hf(L)}.mlp.gate.expert_bias"], np.float32)
        val = v.get_value()
        x = np.ascontiguousarray(np.broadcast_to(b, val.shape)).astype(val.dtype)
        v.set_value(jax.make_array_from_callback(x.shape, rsh, lambda idx, x=x: x[idx]))
        n += 1
    nnx.update(model, st)
    return n


# ---------------------------------------------------------------- CPU：对 FP64 算 R

def compare_main(a) -> int:
    from safetensors import safe_open

    from janus import metrics as M
    src = Path(a.compare)  # 运行目录（读 op_replay_mini.json），或 --sweep 写的某一组的 json
    src = src if src.suffix == ".json" else src / "op_replay_mini.json"
    rec = json.loads(src.read_text())
    T = load_st(str(src.with_suffix(".safetensors")))
    fj = json.loads(Path(a.fp64_json).read_text())["ops"]
    G = load_st(a.gpu_fwd)
    f64 = safe_open(a.fp64, "np")
    f64_keys = set(f64.keys())
    W = load_st(a.hf_weights) if a.hf_weights else {}
    Pov = {}
    if a.fp64_override:  # 权重被换过：受影响的算子 FP64 用换上的权重在本机算，GPU（用的是第 1 步权重）不比
        for k, v in load_st(a.fp64_override).items():
            for pre in ("main_param_after.", "main_param.", "param."):
                k = k[len(pre):] if k.startswith(pre) else k
            Pov[k] = v
    rows = []
    for r in rec["ops"]:
        t = np.asarray(T[r["op"]], np.float64)
        out = dict(r)
        ref = None
        ovr = override_fp64(r["op"], G, Pov) if Pov else None
        if ovr is not None:
            ref = ovr
            out["fp64_override"] = True
        elif r.get("fp64_local") == "weights_proj" and W:
            ref = override_fp64(r["op"], G, W)
        elif r.get("fp64") and r["fp64"] in f64_keys:
            ref = np.asarray(f64.get_tensor(r["fp64"]), np.float64)
        elif r.get("fp64_local") and W:
            ref = local_fp64(r, G, W, f64, fj)
        gname = None if out.get("fp64_override") else next((o["gpu"] for o in op_table() if o["op"] == r["op"]), None)
        g = None
        if gname:
            g = np.asarray(G[gname], np.float64)
            if gname.endswith("shared_experts.out"):
                g = np.swapaxes(g, 0, 1)
        elif r.get("fp64_local") == "routed" and not out.get("fp64_override"):
            H = hf(r["op"].split(".")[0] if r["op"].split(".")[0] == "mtp0" else int(r["op"].split(".")[0]))
            g = np.asarray(G[f"{H}.mlp.out"], np.float64) - np.swapaxes(np.asarray(G[f"{H}.mlp.shared_experts.out"], np.float64), 0, 1)
            out["gpu_note"] = "GPU 的路由专家输出没单独抓：用 mlp.out − shared_experts.out（含 mlp.out 的 BF16 舍入）"
        if ref is not None and ("mlp(own routing)" in r["op"] or "routed_experts" in r["op"]) and g is not None:
            out.update(routing_flips(r["op"], t, g, ref, G))
        if ref is not None:
            ref = ref.reshape(-1)
            tt = t.reshape(-1)[: ref.size]
            out["rel_l2_tpu_vs_fp64"] = M.rel_l2(tt, ref)
            if g is not None:
                gg = g.reshape(-1)[: ref.size]
                out["rel_l2_gpu_vs_fp64"] = M.rel_l2(gg, ref)
                out["R"] = out["rel_l2_tpu_vs_fp64"] / max(out["rel_l2_gpu_vs_fp64"], 1e-300)
        rows.append(out)
        print(f"{r['op']:<34} " + "  ".join(f"{k}={out[k]:.3e}" for k in ("rel_l2_tpu_vs_gpu", "rel_l2_tpu_vs_fp64",
                                                                            "rel_l2_gpu_vs_fp64", "R") if k in out), flush=True)
    rs = [r for r in rows if "R" in r]
    fails = [r["op"] for r in rs if r["R"] > R_LIMIT]
    worst = max(rs, key=lambda r: r["R"]) if rs else None
    res = {
        "what": "L1 逐算子隔离回放（hy4-mini，第 1 步 rank 0 第 0 个 microbatch）：每个算子吃 GPU（Megatron-Core + TE）抓到的输入，"
                "TPU 输出对 FP64 参照，GPU 输出对同一个参照",
        "gpu_run": "g7-l1-a（golden-t1/g7-l1-a，分析目录 mini-l1/gpu）", "fp64": "mini-l1/fp64（GPU 一侧 fp64_l1_gpu.py，CPU FP64）",
        "tpu": {"run": src.parent.name, "file": src.name, "overrides": rec.get("overrides"), **{k: rec[k] for k in ("platform", "device_kind", "tokens", "seconds", "note")}},
        "routing_flip_rule": "MoE 整块 / 路由专家里 TPU 对 FP64 逐 token 相对差 > 0.1 的 token 记成路由翻转，另报去掉它们的 R（诊断，不改判定）",
        "rule": "R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64) ≤ 1.2 过（PROTOCOL.md 第 5 节）；逐算子取最差",
        "n_ops_with_R": len(rs), "worst": {"op": worst["op"], "R": worst["R"]} if worst else None,
        "verdict": ("fail" if fails else "pass") if rs else "inconclusive", "fail_ops": fails,
        "no_fp64": [r["op"] for r in rows if "R" not in r], "ops": rows,
    }
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n")
    print(json.dumps({k: res[k] for k in ("n_ops_with_R", "worst", "verdict", "fail_ops", "no_fp64")}, ensure_ascii=False))
    return 0


def routing_flips(op, t, g, ref, G) -> dict:
    """自选路由的 MoE：逐 token 看 TPU 对 FP64 的相对差，> 0.1 的 token 记成路由翻转（选了别的专家），报它们在 FP64
    分数（sigmoid(GPU logits) + 偏置）里第 4 / 第 5 名的分差；另报去掉这些 token 以后的 R（诊断，不改判定）。"""
    from janus import metrics as M
    Lk = op.split(".")[0]
    H = hf(Lk if Lk == "mtp0" else int(Lk))
    D = ref.reshape(-1).size // 4096
    tt, gg, ff = (np.asarray(x, np.float64).reshape(-1)[: 4096 * D].reshape(4096, D) for x in (t, g, ref))
    rel_tok = np.linalg.norm(tt - ff, axis=1) / np.maximum(np.linalg.norm(ff, axis=1), 1e-300)
    flips = np.nonzero(rel_tok > 0.1)[0]
    lg = np.asarray(G[f"{H}.mlp.gate.logits"], np.float64).reshape(4096, -1)
    sc = np.sort(1.0 / (1.0 + np.exp(-lg)) + np.asarray(G[f"{H}.mlp.gate.expert_bias"], np.float64), axis=1)[:, ::-1]
    keep = np.ones(4096, bool)
    keep[flips] = False
    out = {"routing_flip_tokens": [int(i) for i in flips],
           "routing_flip_gap_4th_5th_fp64": [float(sc[i, 3] - sc[i, 4]) for i in flips]}
    if flips.size:
        et, eg = M.rel_l2(tt[keep], ff[keep]), M.rel_l2(gg[keep], ff[keep])
        out["R_without_flip_tokens"] = et / max(eg, 1e-300)
    return out


def override_fp64(op: str, G: dict, P: dict):
    """输出头、路由 logits、索引器 weights_proj 三个纯矩阵乘的 FP64：GPU 抓到的输入 × 权重 P（HF 名字），P 里没有就返回 None。"""
    if op == "lm_head":
        w = P.get("lm_head.weight")
        return None if w is None else np.asarray(G["hf.model.norm.out"], np.float64) @ np.asarray(w, np.float64).T
    m = re.fullmatch(r"(\d|mtp0)\.(mlp\.gate\.logits|self_attn\.indexer\.weights_proj)", op)
    if not m:
        return None
    L = m.group(1) if m.group(1) == "mtp0" else int(m.group(1))
    pre = "model.mtp_layers.0" if L == "mtp0" else f"model.layers.{L}"
    if m.group(2) == "mlp.gate.logits":
        w, x = P.get(f"{pre}.mlp.gate.weight"), G[f"{hf(L)}.mlp.gate.in"]
    else:
        w, x = P.get(f"{pre}.self_attn.indexer.weights_proj.weight"), G[f"{hf(L)}.input_layernorm.out"]
    return None if w is None else np.asarray(x, np.float64) @ np.asarray(w, np.float64).T


def local_fp64(r, G, W, f64, fj):
    """GPU 一侧没给整块 FP64 的两处，在这里用同一份输入与权重（HF 检查点，BF16）在 FP64 里算：
    共享专家整块 = down(silu(gate x) · up x)；路由专家 = MoE 整块（FP64，自选路由）− 共享专家整块。"""
    Lk = r["op"].split(".")[0]
    L = Lk if Lk == "mtp0" else int(Lk)
    H = hf(L)
    pre = "model.mtp_layers.0" if L == "mtp0" else f"model.layers.{L}"
    x = np.asarray(G[f"{H}.post_attention_layernorm.out"], np.float64)[0]          # [S, D]
    wg = np.asarray(W[f"{pre}.mlp.shared_experts.gate_proj.weight"], np.float64)
    wu = np.asarray(W[f"{pre}.mlp.shared_experts.up_proj.weight"], np.float64)
    wd = np.asarray(W[f"{pre}.mlp.shared_experts.down_proj.weight"], np.float64)
    a_ = x @ wg.T
    sh = ((a_ / (1.0 + np.exp(-a_))) * (x @ wu.T)) @ wd.T
    if r["fp64_local"] == "shared":
        return sh[None]
    moe = np.asarray(f64.get_tensor(f"{H}.mlp(own routing)"), np.float64).reshape(sh[None].shape)
    return moe - sh[None]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="mini")
    p.add_argument("--load")
    p.add_argument("--gpu-fwd", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--compare", default=None, help="TPU 运行目录（含 op_replay_mini.json / .safetensors）：在 CPU 上对 FP64 算 R")
    p.add_argument("--fp64", default=None)
    p.add_argument("--fp64-json", default=None)
    p.add_argument("--hf-weights", default=None, help="HF 检查点 model.safetensors（共享专家整块的 FP64 用）")
    p.add_argument("--save-ops", default=None, help="只存、只比名字匹配这个正则的算子")
    p.add_argument("--hf-config", default="/data/config.json")
    p.add_argument("--param-override", default=None, help="HF 名字的 FP32 张量 safetensors（本地或 gs://），换进模型参数")
    p.add_argument("--override-only", default=None, help="--param-override 只换 HF 名字匹配这个正则的张量")
    p.add_argument("--fp64-override", default=None, help="（--compare）权重被换过的算子，FP64 用这份权重在本机算，GPU 一列不比")
    p.add_argument("--sweep", default=None,
                   help="一个作业里依次跑几组配置的 JSON 文件（相对本目录或绝对路径）：[{\"name\", \"overrides\": [k=v…], "
                        "\"save_ops\": 正则}]；每组在命令行配置之上再加 overrides，输出写到 <out 去掉 .json>.<name>.json / .safetensors。"
                        "用文件而不是命令行字符串：run.sh 把参数拼进 Pod 的 shell 命令，正则里的 | ; $ 会被 shell 吃掉")
    a, rest = p.parse_known_args(argv)
    a.mt_args = [x for x in rest if "=" in x and not x.startswith("-")]
    if a.compare:
        return compare_main(a)
    if a.sweep:
        import copy
        rc = 0
        sp = Path(a.sweep)
        sp = sp if sp.is_absolute() else Path(__file__).resolve().parent / sp
        for item in json.loads(sp.read_text()):
            b = copy.copy(a)
            b.mt_args = a.mt_args + list(item.get("overrides", []))
            b.save_ops = item.get("save_ops") or a.save_ops
            b.param_override = item.get("param_override", a.param_override)  # 正则放 JSON 里，免得被 Pod 的 shell 吃掉
            b.override_only = item.get("override_only", a.override_only)
            b.out = str(Path(a.out).with_suffix("")) + f".{item['name']}.json"
            try:
                rc = tpu_main(b) or rc
            except Exception as e:  # noqa: BLE001  一组失败不拖累其余各组；失败记在结果旁边
                rc = 1
                import jax
                if jax.process_index() == 0:
                    Path(b.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(b.out).with_suffix(".error.txt").write_text(f"{type(e).__name__}: {e}\n")
                print(f"[sweep] {item['name']} 失败：{type(e).__name__}: {e}", flush=True)
            import gc
            gc.collect()
        return rc
    return tpu_main(a)


if __name__ == "__main__":
    sys.exit(main())
