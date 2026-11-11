"""逐层隔离回放（L2 主方法）：把 GPU golden 第 1 步第 k 层的输入（rank 0 第一个 microbatch，`layers.k.in`）
喂进 TPU 上单独编译的「只有第 k 层」的前向，与 GPU 第 k 层的输出及层内各观测点比；可选再用 golden 的
`layers.k.out.grad` 做一次反向，比 `layers.k.in.grad`。

每一层的输入都是 GPU 的，层与层之间的误差不再串起来；回放程序本身就是测量对象，不存在「加钩子改了生产图」的问题
（docs/reference/dump.md 原则 2 管的是训练步；这里的结论是「TPU 上这一层的这种写法」对 GPU 的差）。

  python layer_replay.py --load <MaxText 检查点 …/0/items> --golden <golden 运行目录（要 step_1 的 layers.*）> \\
      --out <输出目录> [--layers 0,1,2,3] [--grad] [MaxText key=value …]

输出目录：results.json（每层每个观测点的 rel_l2 / cosine / ULP 分布、路由一致比例与翻转 token 的分差）
与 step_1/ 下的张量（规范名，GPU 布局，可直接用 step1_compare.py / janus 读）。多主机时只有进程 0 写。

做法：MaxText 建模型并载入检查点；从扫描堆叠的层里切出第 j 层的参数与状态（与 nnx_decoders._apply_layers_sequentially
同样的切法），合成一个单层模块，在 jit 里调用它的 __call__（层参数与 MaxText 解码器调用时相同：
segment ids 全 1、positions = golden 的 position_ids、deterministic、train 模式）。8 个设备上 batch 维要能均分，
所以把这一条序列复制成 global_batch 行（每行相同），只取第 0 行比较。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l1_probes  # noqa: E402
import mt_common  # noqa: E402
import step1_compare  # noqa: E402

from janus.dump import maxtext as jd  # noqa: E402
from janus.store import Run, RunWriter  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--grad", action="store_true", help="另做反向：golden 的 layers.k.out.grad 作余切，比 layers.k.in.grad")
    p.add_argument("--l1", action="store_true", help="另做 L1 单算子回放（l1_probes.py：路由 GEMM、MLP 前 RMSNorm、稠密层门控激活）")
    p.add_argument("--l1-only", action="store_true", help="只做 L1，不做逐层回放")
    p.add_argument("--head", action="store_true",
                   help="另做输出头回放：golden 第 1 步的 final_norm.out 进 MaxText 输出头与交叉熵，记 dlogits 与 dh（隐状态梯度）")
    p.add_argument("--full-forward", action="store_true",
                   help="整网前向回放：golden 第 1 步 rank 0 第一个 microbatch 的 tokens 进整个模型（单独编译的只前向程序，不加任何抓数输出），"
                        "记 logits（BF16）与逐 token loss（FP32），名字同 golden（L3 第 1 步 logits / 逐 token loss 用）")
    p.add_argument("--head-from", default=None,
                   help="输出头回放改用这个运行目录（本地）第 --head-step 步的 final_norm.out、input.labels 和全量抓数里的输出层主权重"
                        "（例：TPU 加钩子运行第 100 步），不用检查点里的初始权重")
    p.add_argument("--head-step", type=int, default=1)
    p.add_argument("--head-total", type=float, default=65536.0, help="交叉熵求和后除以的 token 数（与训练的 total_weights 一致）")
    p.add_argument("--tokens", type=int, default=0, help="只用前 T 个 token（0 = 全部；CPU 上调试用）")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--side", default="tpu-replay")
    p.add_argument("--params-from", default=None,
                   help="载入检查点后再把整套参数换成这份 Megatron 布局的参数（param_io 格式，同 train_hy3.py --params-from），用来在训练中途的状态上回放")
    p.add_argument("--round-weights-bf16", action="store_true",
                   help="载入后把全部 FP32 参数用 reduce_precision 舍到 BF16 值（对 GPU `--fp32 --bf16-weights` 的 golden 回放）")
    p.add_argument("--jax-default-matmul-precision", default=None,
                   help="jax.config 的 jax_default_matmul_precision（同 train_hy3.py；没给 precision 的 dot，例如 ragged_dot，用它）")
    p.add_argument("--router-jitter", default=None,
                   help="（只作诊断，敏感度臂）<σ>:<seed>：路由选专家时给分数加固定种子的高斯抖动，只改选谁、不改权重（router_jitter.py）")
    p.add_argument("maxtext_args", nargs="*")
    a = p.parse_args(argv)
    if "://" not in a.load:
        a.load = str(Path(a.load).resolve())
    if a.cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from maxtext.utils import maxtext_utils_nnx, model_creation_utils

    if a.jax_default_matmul_precision:
        jax.config.update("jax_default_matmul_precision", a.jax_default_matmul_precision)
    router_jitter_info = None
    if a.router_jitter:
        import router_jitter
        router_jitter_info = router_jitter.install(a.router_jitter)
    if not a.cpu and not jax.distributed.is_initialized():
        jax.distributed.initialize()

    g = Run.open(a.golden)
    x0 = g.tensor(1, "layers.0.in")                                   # [S, 1, D]
    S = x0.shape[0] if not a.tokens else a.tokens
    ndev = jax.device_count()
    extra = [f"max_target_length={S}", "per_device_batch_size=1", f"load_parameters_path={a.load}",
             "run_name=layer-replay", "base_output_directory=/tmp/hy3out"]
    if a.cpu:
        extra += ["hardware=cpu", "skip_jax_distributed_system=True", "attention=dot_product", "sparse_matmul=False",
                  "megablox=False"]
    config = mt_common.make_config(extra + list(a.maxtext_args))
    nm = mt_common.name_map_for(config)
    mesh = mt_common.make_mesh(config)
    with nn_partitioning.axis_rules(config.logical_axis_rules):
        model = model_creation_utils.from_pretrained(config, mesh)
    jd.tag_model(model)
    params_from_info = None
    if a.params_from:
        import param_io

        class _Holder(nnx.Module):  # param_io.set_state 按训练状态的键（{col}/model/…）找叶子
            def __init__(self, m):
                self.model = m

        holder = _Holder(model)
        leaves, info = param_io.load_leaves(a.params_from, nm, nm.leaf_shapes())
        st = nnx.state(holder)
        nset = param_io.set_state(st, leaves)
        nnx.update(holder, st)
        del leaves
        params_from_info = {"path": a.params_from, "source": info.get("source"), "step": info.get("step"),
                            "tensors": info.get("tensors"), "leaves_set": nset}
        if jax.process_index() == 0:
            print(f"[replay] --params-from：{json.dumps(params_from_info, ensure_ascii=False, default=str)}", flush=True)
    if a.round_weights_bf16:
        # reduce_precision 而不是 astype 往返：TPU 上 XLA 允许 excess precision，往返会被消掉
        _, ps, _ = nnx.split(model, nnx.Param, ...)
        rnd = jax.jit(lambda t: jax.tree.map(
            lambda x: jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7) if x.dtype == jnp.float32 else x, t))
        nnx.update(model, rnd(ps))
    B = config.global_batch_size_to_train_on
    fd = int(config.first_num_dense_layers)
    scan_axis = int(config.param_scan_axis)
    p0 = g.tensor(1, "input.position_ids")[0, :S] if "input.position_ids" in g.names(1) else np.arange(S)
    pos = np.broadcast_to(p0.astype(np.int32), (B, S))
    seg = np.ones((B, S), np.int32)

    def layer_module(i):
        stack_name, j = ("dense_layers", i) if i < fd else ("moe_layers", i - fd)
        layers = getattr(model.decoder, stack_name)
        length = fd if i < fd else config.num_decoder_layers - fd
        graphdef, params, state = nnx.split(layers, nnx.Param, ...)
        params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0) if x.ndim > scan_axis else x, params)
        params = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(params, length)
        state = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(state, length)
        cur = jax.tree.map(lambda v: v[j], (params, state))
        cur = maxtext_utils_nnx.nnx_remove_scan_axis(cur, "layers")
        return stack_name, graphdef, cur

    results, tensors = {"layers": {}}, {}
    l1_save = {}                                                       # L1 各算子在生产开关下的 TPU 输出（l1_probes.run 的 save）
    t_all = time.time()
    dsh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "fsdp")))
    put = lambda v: jax.make_array_from_callback(v.shape, dsh, lambda idx, v=v: v[idx])  # noqa: E731
    if a.l1 or a.l1_only:
        results["l1"] = []
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            l1_probes.run(lambda i: (lambda m: nnx.merge(m[1], *m[2]))(layer_module(i)), g, S, B, put, jd.to_host,
                          int(config.num_experts_per_tok), config, mesh, results["l1"], save=l1_save)
        if jax.process_index() == 0:
            for r in results["l1"]:
                print("[l1] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
                      flush=True)
    if a.head:
        from maxtext.utils import max_utils
        V = int(config.vocab_size)
        if a.head_from:
            hsrc, hs = a.head_from, a.head_step
            if hsrc.startswith("gs://"):                                 # Pod 上没有这份运行目录：只拉要用的几个文件
                import subprocess
                loc = Path("/tmp/head_from")
                (loc / f"step_{hs}").mkdir(parents=True, exist_ok=True)
                subprocess.run(["gcloud", "storage", "cp", f"{hsrc}/meta.json", str(loc) + "/"], check=True)
                subprocess.run(["gcloud", "storage", "cp"] + [f"{hsrc}/step_{hs}/{f}" for f in
                               ("index.json", "final_norm.safetensors", "input.safetensors", "main_param.other.safetensors")]
                               + [str(loc / f"step_{hs}") + "/"], check=True)
                (loc / "DONE").touch()                                   # 只拉了一部分文件，Run.open 要 DONE
                hsrc = str(loc)
            hr = Run.open(hsrc)
            yh = hr.tensor(hs, "final_norm.out")[:S, 0]
            lab = hr.tensor(hs, "input.labels")[0, :S].astype(np.int32)
            wo = hr.tensor(hs, next(n for n in hr.names(hs) if n.startswith("main_param.") and n.endswith("output_layer.weight")))
            ker = model.decoder.logits_dense.kernel
            kt = np.ascontiguousarray(wo.T.astype(np.float32))           # Megatron [V, D] → MaxText kernel [D, V]
            ker.set_value(jax.make_array_from_callback(kt.shape, ker.get_value().sharding, lambda idx: kt[idx]))
        else:
            yh = g.tensor(1, "final_norm.out")[:S, 0]                   # [S, D] bf16
            lab = g.tensor(1, "input.labels")[0, :S].astype(np.int32)
        dgd, dst = nnx.split(model.decoder)

        def head_logits(st_, y_):
            dec = nnx.merge(dgd, st_)
            return dec.apply_output_head(None, y_, True, "train", normalize_y=False)

        def head_loss(st_, y_, lab_):
            lg = head_logits(st_, y_)
            xent, _ = max_utils.cross_entropy_with_logits(lg.astype(jnp.float32), jax.nn.one_hot(lab_, V, dtype=jnp.float32), 0.0)
            return jnp.sum(xent) / a.head_total

        def ce_only(lg_bf16, lab_):
            xent, _ = max_utils.cross_entropy_with_logits(lg_bf16.astype(jnp.float32), jax.nn.one_hot(lab_, V, dtype=jnp.float32), 0.0)
            return jnp.sum(xent) / a.head_total

        yd = put(np.ascontiguousarray(np.broadcast_to(np.asarray(yh)[None], (B,) + yh.shape)))
        ld = put(np.ascontiguousarray(np.broadcast_to(lab[None], (B, S))))
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            dh = jd.to_host(jax.jit(jax.grad(head_loss, argnums=1))(dst, yd, ld))[0]
            lg = jax.jit(head_logits)(dst, yd)
            dlg = jd.to_host(jax.jit(jax.grad(ce_only))(lg, ld))[0]
            lgh = jd.to_host(lg)[0]
        # 行数复制了 B 份，每份的 dlogits / dh 都按 1/head_total 归一，取第 0 行
        tensors["head.dh"] = np.ascontiguousarray(dh[:, None])
        tensors["head.dlogits"] = np.ascontiguousarray(dlg[:, None])
        tensors["head.logits"] = np.ascontiguousarray(lgh[:, None])
        results["head"] = {"dh_dtype": str(dh.dtype), "dlogits_dtype": str(dlg.dtype), "logits_dtype": str(lgh.dtype),
                           "rowsum_dlogits_mean": float(dlg.astype(np.float64).sum(-1).mean()), "total": a.head_total}
        if jax.process_index() == 0:
            print("[head] " + json.dumps(results["head"]), flush=True)
    if a.full_forward:
        import ml_dtypes
        import replay as rp
        from maxtext.utils import max_utils
        gb = rp.GoldenBatches(a.golden).global_batch(1)                  # 第 0 行 = GPU rank 0 的第一个 microbatch
        tok = np.asarray(gb["tokens"][0, :S], np.int32)
        lab = np.asarray(gb["labels"][0, :S], np.int32)
        fpos = np.broadcast_to(np.asarray(gb["position_ids"][0, :S], np.int32), (B, S))
        V = int(config.vocab_size)
        mgd, mst = nnx.split(model)

        def full_fwd(st_, t_, p_, s_):
            return nnx.merge(mgd, st_)(decoder_input_tokens=t_, decoder_positions=p_, decoder_segment_ids=s_, enable_dropout=False)

        def xent(lg_, lab_):
            x_, _ = max_utils.cross_entropy_with_logits(lg_.astype(jnp.float32), jax.nn.one_hot(lab_, V, dtype=jnp.float32), 0.0)
            return x_
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            td = put(np.ascontiguousarray(np.broadcast_to(tok[None], (B, S))))
            lg = jax.jit(full_fwd)(mst, td, put(np.ascontiguousarray(fpos)), put(seg))
            lg = lg[0] if isinstance(lg, tuple) else lg
            xt = jd.to_host(jax.jit(xent)(lg, put(np.ascontiguousarray(np.broadcast_to(lab[None], (B, S))))))[0]
            lgh = jd.to_host(lg)[0]
        # cast_logits_to_fp32：logits 先在 BF16 出再转 FP32，转回 BF16 不丢位
        tensors["logits"] = np.ascontiguousarray(lgh.astype(ml_dtypes.bfloat16)[:, None])
        tensors["loss.per_token"] = np.ascontiguousarray(xt.astype(np.float32)[None])
        results["full_forward"] = {"logits_dtype_from_model": str(lgh.dtype), "loss_mean": float(np.mean(xt))}
        if jax.process_index() == 0:
            print("[full_forward] " + json.dumps(results["full_forward"]), flush=True)
    for i in ([] if a.l1_only else [int(x) for x in a.layers.split(",") if x]):
        stack, graphdef, (cp, cs) = layer_module(i)
        xin = g.tensor(1, f"layers.{i}.in")[:S, 0]                      # [S, D] bf16
        xb = np.broadcast_to(np.asarray(xin)[None], (B,) + xin.shape)

        def fwd(params, state, x, seg_ids, positions):
            layer = nnx.merge(graphdef, params, state)
            with jd.capturing(rows=1):
                out = layer(x, seg_ids, positions, True, "train")
                y = out[0] if isinstance(out, tuple) else out
                inter = nnx.pop(layer, nnx.Intermediate)
            return y, inter.to_pure_dict()

        xd, segd, posd = put(np.ascontiguousarray(xb)), put(seg), put(np.ascontiguousarray(pos))
        t0 = time.time()
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            y, inter = jax.jit(fwd)(cp, cs, xd, segd, posd)
            y = jd.to_host(y)
            act = {f"decoder/{stack}/" + k: jd.to_host(v)[None] for k, v in jd.collect_sown(inter).items()}
            gin = None
            if a.grad and f"layers.{i}.out.grad" in g.names(1):
                ct = np.zeros_like(xb)
                ct[0] = g.tensor(1, f"layers.{i}.out.grad")[:S, 0]

                # 参数要作为参数传进 jit（多主机时不能闭包跨进程的全局数组）
                jf = jax.jit(lambda p_, s_, x_, g_, q_: fwd(p_, s_, x_, g_, q_)[0])
                _, vjp = jax.vjp(lambda x_: jf(cp, cs, x_, segd, posd), xd)
                gin = jd.to_host(vjp(put(ct))[0])[0]
        dt = time.time() - t0
        got = jd.assemble_activations(act, dict(nm.dims, routed_scaling=float(config.routed_scaling_factor)), rows=1)
        base = 0 if stack == "dense_layers" else fd                  # 单层模块抓到的值层轴只有 1，assemble 会记成栈的第一层
        got = {(f"layers.{i}." + k[len(f"layers.{base}."):]): v for k, v in got.items() if k.startswith(f"layers.{base}.")}
        got[f"layers.{i}.out"] = np.ascontiguousarray(y[0:1].swapaxes(0, 1))
        if gin is not None:
            got[f"layers.{i}.in.grad"] = np.ascontiguousarray(gin[None].swapaxes(0, 1))
        rows = step1_compare.compare_dict(got, g, 1, tokens=S, k=int(config.num_experts_per_tok))
        results["layers"][str(i)] = {"seconds": dt, "rows": rows,
                                     "first_beyond_bf16": step1_compare.first_beyond_bf16(rows)}
        tensors.update(got)
        if jax.process_index() == 0:
            for r in rows:
                print("[replay] " + "  ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in r.items() if k != "shape"), flush=True)
    if jax.process_index() == 0:
        import flax
        meta = {"schema_version": 1, "run_name": Path(a.out).name, "side": a.side, "framework": "maxtext",
                "model": "hunyuan3-4l", "code": [{"repo": "maxtext", "commit": os.environ.get("MAXTEXT_COMMIT", "unknown")},
                                                 {"repo": "janus", "commit": os.environ.get("JANUS_COMMIT", "unknown")}],
                "image": os.environ.get("JANUS_IMAGE", "unknown"),
                "versions": {"jax": jax.__version__, "flax": flax.__version__},
                "hardware": {"accelerator": jax.devices()[0].device_kind, "count": ndev, "processes": jax.process_count(),
                             "topology": os.environ.get("TPU_TOPOLOGY", "unknown")},
                "parallelism": {"dp": 1, "fsdp": ndev, "tp": 1, "ep": 1, "cp": 1, "pp": 1},
                "config": {"path": str(mt_common.HY3_YML), "overrides": extra + list(a.maxtext_args)},
                "checkpoint": {"path": a.load, "manifest_sha256": None},
                "data": {"manifest": f"{a.golden} step 1 layers.*.in（rank 0 micro 0）", "manifest_sha256": None, "seed": None},
                "steps": {"first": 1, "last": 1, "convention": "golden_step = step"},
                "determinism": jd.env_snapshot(), "dtypes": {"compute": str(config.dtype)},
                "replay": {"kind": "逐层隔离回放：每层输入来自 golden，单层单独编译", "layers": a.layers, "grad": a.grad,
                           "tokens": S, "rows_replicated": B, "round_weights_bf16": a.round_weights_bf16, "router_jitter": router_jitter_info,
                           "params_from": params_from_info,
                           "jax_default_matmul_precision": a.jax_default_matmul_precision},
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        w = RunWriter(a.out, meta)
        w.log_metrics(1, wall_seconds=time.time() - t_all)
        if "head" in results:
            w.write_tensors(1, "head", {k: v for k, v in tensors.items() if k.startswith("head.")}, kind="activation",
                            point="backward")
        if l1_save:
            w.write_tensors(1, "l1", l1_save, kind="activation", point="forward")
        if "full_forward" in results:
            w.write_tensors(1, "logits", {"logits": tensors["logits"]}, kind="activation", point="forward")
            w.write_tensors(1, "loss", {"loss.per_token": tensors["loss.per_token"]}, kind="other", point="forward")
        for i in results["layers"]:
            w.write_tensors(1, f"layers.{i}", {k: v for k, v in tensors.items() if k.startswith(f"layers.{i}.")},
                            kind="activation", point="forward")
        (Path(a.out) / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float))
        w.done()
        print("[replay] 各层第一个超过 BF16 末位量级的：",
              {k: v["first_beyond_bf16"] for k, v in results["layers"].items()}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
