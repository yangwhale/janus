"""逐层隔离回放（L2，TPU 一侧）：混元 3 layer_replay.py 的做法搬到混元 4 hy4-mini。
把 GPU golden 第 1 步第 k 层的输入（rank 0 第一个 microbatch，iHC 的 4 路流）喂进 TPU 上单独编译的「只有第 k 层」的前向，
比第 k 层的输出；再用 GPU 的 `layers.k.out.grad` 作余切做一次反向，比第 k 层输入的梯度。

参照（MaxText 不能跑 FP64，同 t1-layer0-kd-attribution 的做法）：同一段程序换成 `dtype=float32 matmul_precision=highest
sparse_matmul=false` 再跑一遍，作 FP32 参照（误差约 1e-7 量级，远小于被比的 BF16 量级）。
R = rel_L2(TPU, 参照) ÷ rel_L2(GPU, 参照)（PROTOCOL.md 第 5 节，≤ 1.2 过）。

输入从 GPU 的 janus 运行目录取（`--gpu-run`，g2s-a 这样的全量抓数）：
- 第 k ≥ 1 层的输入 = `layers.(k−1).out`（4 路流拍平的 [S, 1, 4·H]），第 0 层 = `embed.out` 复制 4 路（iHC 展开）；
- 余切 = `layers.k.out.grad`；比的梯度 = `layers.(k−1).out.grad`（第 0 层：输入梯度 4 路求和，对 `embed.out.grad`）。
索引器共享层（IndexCache：第 2–4 层用第 1 层的 top-k，第 6 层用第 5 层的）：top-k 掩码由 TPU 在 GPU 给的第 1 / 5 层输入上
用这一层自己的索引器算出（各自的程序：被测用被测的、参照用参照的）。MTP 层要 MTP 的输入，GPU 的 g2s-a 没抓，不在这里做。

  python layer_replay.py --load <检查点 …/0/items> --gpu-run gs://…/golden-t1/g2s-a --out layer_replay.json [--layers 0,1,…]
  # SCRIPT=layer_replay.py 用 run.sh 提交（STAGE=mini、YML=hy4-mini-t1.yml）；--cpu 在 CPU 上验证脚本（--tokens 截短）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mt_common  # noqa: E402

R_LIMIT = 1.2
REF_OVERRIDES = ["dtype=float32", "matmul_precision=highest", "sparse_matmul=false"]


def fetch_gpu(run: str, layers: list[int]) -> str:
    """只拉要用的文件（index.json、embed、各层的前向 / 反向）。返回本地运行目录（Run.open 能读）。"""
    if not run.startswith("gs://"):
        return run
    loc = Path(tempfile.mkdtemp()) / Path(run.rstrip("/")).name
    (loc / "step_1").mkdir(parents=True)
    subprocess.run(["gcloud", "storage", "cp", "-q", f"{run}/meta.json", str(loc) + "/"], check=True)
    files = ["index.json", "embed.safetensors"] + [f"{f}.layers.{k}.safetensors" if f == "buffer" else f"layers.{k}{f}.safetensors"
                                                    for k in range(max(layers) + 1) for f in ("", ".bwd", "buffer")]
    subprocess.run(["gcloud", "storage", "cp", "-q"] + [f"{run}/step_1/{f}" for f in files] + [str(loc / "step_1") + "/"],
                   check=True)
    (loc / "DONE").touch()
    return str(loc)


class RunSource:
    """GPU janus 全量抓数（g2s-a）：第 k 层输入 = layers.(k−1).out，第 0 层 = embed.out 复制 4 路；梯度同理。"""

    layer0_single_stream_grad = True

    def __init__(self, g, tokens: int):
        self.g = g
        self.S = tokens or int(g.tensor(1, "embed.out").shape[0])
        self.n = int(g.tensor(1, "layers.0.out").shape[-1]) // int(g.tensor(1, "embed.out").shape[-1])

    def _t(self, name, dt=np.float64):
        return np.asarray(self.g.tensor(1, name), dt)[: self.S, 0]

    def inp(self, k):
        if k == 0:
            e = self._t("embed.out", np.float32)
            return np.repeat(e[:, None, :], self.n, axis=1)
        return self._t(f"layers.{k - 1}.out", np.float32).reshape(self.S, self.n, -1)

    def out(self, k):
        return self._t(f"layers.{k}.out").reshape(self.S, self.n, -1)

    def in_grad(self, k):
        name = "embed.out.grad" if k == 0 else f"layers.{k - 1}.out.grad"
        if name not in self.g.names(1):
            return None
        x = self._t(name)
        return x if k == 0 else x.reshape(self.S, self.n, -1)

    def out_grad(self, k):
        name = f"layers.{k}.out.grad"
        return None if name not in self.g.names(1) else self._t(name, np.float32).reshape(self.S, self.n, -1)

    def bias(self, k):
        name = f"buffer.decoder.layers.{k}.mlp.router.expert_bias"
        return np.asarray(self.g.tensor(1, name), np.float32) if name in self.g.names(1) else None


class L1Source:
    """GPU 逐层抓数（hy4_capture.py，`g7-l1-a`）：每层 in0 / out0（4 路流 [b, s, n, H]；MTP 层是一路）与它们的梯度。
    第 0 层的 in0 是展开前的一路，TPU 单层程序的 4 路流输入取 attn_hc.in；它的梯度 in0.grad 也是一路（= 4 路梯度之和）。"""

    layer0_single_stream_grad = True

    def __init__(self, d: str, tokens: int):
        from op_replay_mini import load_st
        self.f = load_st(f"{d.rstrip('/')}/step_1_fwd.safetensors")
        self.b = load_st(f"{d.rstrip('/')}/step_1_bwd.safetensors")
        self.S = tokens or int(self.f["input.tokens"].shape[1])

    @staticmethod
    def h(k):
        return "hf.model.mtp_layers.0" if k == "mtp0" else f"hf.model.layers.{k}"

    def inp(self, k):
        name = f"{self.h(k)}.attn_hc.in" if k == 0 else f"{self.h(k)}.in0"
        return np.asarray(self.f[name], np.float32)[0, : self.S]

    def out(self, k):
        return np.asarray(self.f[f"{self.h(k)}.out0"], np.float64)[0, : self.S]

    def in_grad(self, k):
        x = self.b.get(f"{self.h(k)}.in0.grad")
        return None if x is None else np.asarray(x, np.float64)[0, : self.S]

    def out_grad(self, k):
        x = self.b.get(f"{self.h(k)}.out0.grad")
        return None if x is None else np.asarray(x, np.float32)[0, : self.S]

    def bias(self, k):
        x = self.f.get(f"{self.h(k)}.mlp.gate.expert_bias")
        return None if x is None else np.asarray(x, np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="mini")
    p.add_argument("--load", required=True)
    p.add_argument("--gpu-run", default=None, help="GPU 的 janus 全量抓数运行目录（g2s-a 这样的，7 层，无 MTP）")
    p.add_argument("--gpu-l1", default=None,
                   help="GPU 逐层抓数（hy4_capture.py：step_1_fwd / step_1_bwd.safetensors 所在目录，本地或 gs://），7 层 + MTP")
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="0,1,2,3,4,5,6")
    p.add_argument("--tokens", type=int, default=0, help="只用前 T 个 token（0 = 全部；CPU 调试用）")
    p.add_argument("--no-grad", action="store_true")
    p.add_argument("--no-ref", action="store_true", help="不跑 FP32 参照（调试）")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--param-grads", action="store_true",
                   help="另求这一层参数的梯度（同一个余切），换成 HF 名字存到 <out>.param_grads.safetensors（被测与参照各一份）")
    p.add_argument("--hf-config", default="/data/config.json", help="HF config.json（--param-grads 换 HF 名字用）")
    p.add_argument("--save", action="store_true", help="另存每层的输出与输入梯度（被测与参照，FP32）到 <out>.safetensors，供跨运行逐位核对")
    a, rest = p.parse_known_args(argv)
    mt_args = [x for x in rest if "=" in x and not x.startswith("-")]
    import jax
    if not a.cpu and not jax.distributed.is_initialized():
        jax.distributed.initialize()
    if jax.process_index() == 0:   # 先建运行目录：中途失败时 run.sh 也能认出交结果的 Pod、取回日志
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    import jax.numpy as jnp
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from maxtext.utils import maxtext_utils_nnx, model_creation_utils

    from janus import metrics as M
    from janus.dump import maxtext as jd
    from janus.store import Run

    layers = [x if x == "mtp0" else int(x) for x in a.layers.split(",") if x]
    if a.gpu_l1:
        src = L1Source(a.gpu_l1, a.tokens)
    else:
        if "mtp0" in layers:
            raise SystemExit("MTP 层要 GPU 的逐层抓数（--gpu-l1）；janus 全量抓数里没有 MTP 层的输入")
        src = RunSource(Run.open(fetch_gpu(a.gpu_run, layers)), a.tokens)
    S = src.S
    gpu_in, gpu_out, gpu_in_grad, gpu_out_grad = src.inp, src.out, src.in_grad, src.out_grad

    def build(extra_over):
        extra = [f"max_target_length={S}", "per_device_batch_size=1", "gradient_accumulation_steps=1",
                 f"load_parameters_path={a.load}", "run_name=layer-replay", "base_output_directory=/tmp/hy4out"]
        if a.cpu:
            extra += ["hardware=cpu", "skip_jax_distributed_system=True", "attention=dot_product", "sparse_matmul=False",
                      "megablox=False", "use_tokamax_splash=False"]
        config = mt_common.make_config(a.stage, extra + mt_args + extra_over)
        mesh = mt_common.make_mesh(config)
        with nn_partitioning.axis_rules(config.logical_axis_rules):
            model = model_creation_utils.from_pretrained(config, mesh)
        fix_router_bias(model, config, mesh)
        return config, mesh, model

    def fix_router_bias(model, config, mesh):
        """from_pretrained 只载入 Param，路由偏置（MoEBiasVar）留成已删除的数组；换成 GPU 第 1 步更新前的偏置（= 检查点里的初值）。
        MTP 层的偏置这里用不到，置 0。"""
        fd = int(config.first_num_dense_layers)
        num_moe = int(config.num_decoder_layers) - fd
        blk = int(config.indexer_share_interval) if config.use_indexer and config.indexer_share_interval > 1 else 0
        full = (num_moe // blk) * blk if blk else num_moe
        st = nnx.state(model)
        n = 0
        for path, v in nnx.to_flat_state(st):
            if type(v).__name__ != "MoEBiasVar":
                continue
            ps = [str(x) for x in path]
            val = v.get_value()
            shape, dtype = val.shape, val.dtype
            new = np.zeros(shape, np.float32)
            if ps[:2] == ["decoder", "moe_layers"]:
                i = int(ps[2].split("_")[1]) if blk else 0
                for j in range(shape[0] if len(shape) > 1 else 1):
                    k = fd + (j * blk + i if blk else j)
                    if len(shape) > 1:
                        new[j] = src.bias(k)
                    else:
                        new[:] = src.bias(k)
            elif ps[:2] == ["decoder", "moe_layers_remainder"]:
                new[:] = src.bias(fd + full + int(ps[2].split("_")[1]))
            elif ps[0] == "mtp_block":
                b = src.bias("mtp0")
                new[:] = 0.0 if b is None else b
            # 多主机：要全局（复制到每个设备）的数组，不能是只在本机一个设备上的 jnp 数组
            rsh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            new = new.astype(dtype)
            v.set_value(jax.make_array_from_callback(new.shape, rsh, lambda idx, x=new: x[idx]))
            n += 1
        nnx.update(model, st)
        return n

    def layer_of(model, config, k):
        """HF 第 k 层 → (单层模块, 它所在 IndexCache 块里第一层的 HF 号或 None)。扫描的栈切出第 j 份（同混元 3）。"""
        fd = int(config.first_num_dense_layers)
        scan_axis = int(config.param_scan_axis)

        def from_stack(stack, length, j):
            gd, prm, st = nnx.split(stack, nnx.Param, ...)
            prm = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0) if x.ndim > scan_axis else x, prm)
            prm = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(prm, length)
            st = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(st, length)
            cur = jax.tree.map(lambda v: v[j], (prm, st))
            cur = maxtext_utils_nnx.nnx_remove_scan_axis(cur, "layers")
            return nnx.merge(gd, *cur)

        if k == "mtp0":
            return model.mtp_block.mtp_layer_1.mtp_1_transformer_layer, None
        if k < fd:
            return from_stack(model.decoder.dense_layers, fd, k), None
        m = k - fd
        num_moe = int(config.num_decoder_layers) - fd
        blk = int(config.indexer_share_interval) if config.use_indexer and config.indexer_share_interval > 1 else 0
        if not blk:
            return from_stack(model.decoder.moe_layers, num_moe, m), None
        full = (num_moe // blk) * blk
        if m < full:
            block = from_stack(model.decoder.moe_layers, num_moe // blk, m // blk)
            i = m % blk
        else:
            block = model.decoder.moe_layers_remainder
            i = m - full
        return getattr(block, f"layers_{i}"), (None if i == 0 else k - i)

    def run_side(tag, extra_over):
        config, mesh, model = build(extra_over)
        B = int(config.global_batch_size_to_train_on)
        dsh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "fsdp")))
        put = lambda v: jax.make_array_from_callback(v.shape, dsh, lambda idx, v=v: v[idx])  # noqa: E731
        rep = lambda x: np.ascontiguousarray(np.broadcast_to(x[None], (B,) + x.shape))  # noqa: E731
        seg = put(np.ones((B, S), np.int32))
        pos = put(np.ascontiguousarray(np.broadcast_to(np.arange(S, dtype=np.int32), (B, S))))
        masks, out = {}, {}

        # seg / pos 是跨主机的全局数组：多主机时不能被 jit 的函数闭包捕获，一律当参数传
        def call(state, gd, x, mask, want_mask, seg, pos):
            layer = nnx.merge(gd, *state) if isinstance(state, tuple) else nnx.merge(gd, state)
            kw = {"shared_indexer_mask": mask} if mask is not None else {}
            if want_mask:
                kw["return_indexer_mask"] = True
            o = layer(x, seg, pos, True, "train", **kw)
            return (o[0], o[-1]) if want_mask else (o[0] if isinstance(o, tuple) else o)

        with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            for k in layers:
                layer, first = layer_of(model, config, k)
                gd, state = nnx.split(layer)
                if first is not None and first not in masks:      # 共享层：先在块里第一层的 GPU 输入上算出 top-k 掩码
                    fl, _ = layer_of(model, config, first)
                    fgd, fst = nnx.split(fl)
                    _, masks[first] = jax.jit(lambda s_, x_, sg, ps: call(s_, fgd, x_, None, True, sg, ps))(
                        fst, put(rep(gpu_in(first))), seg, pos)
                mask = masks.get(first) if first is not None else None
                x = put(rep(gpu_in(k)))
                t0 = time.time()
                f = jax.jit(lambda s_, x_, m_, sg, ps: call(s_, gd, x_, m_, False, sg, ps))
                y = f(state, x, mask, seg, pos)
                yh = np.asarray(jd.to_host(y))
                rec = {"out": yh.astype(np.float64)[0]}
                ct = gpu_out_grad(k)
                if a.param_grads and ct is not None:
                    pgd, prm, rest = nnx.split(layer, nnx.Param, ...)
                    fp = jax.jit(lambda p_, x_, m_, sg, ps: call((p_, rest), pgd, x_, m_, False, sg, ps))
                    ctb = np.zeros((B,) + ct.shape, np.float32)
                    ctb[0] = ct
                    _, vjp2 = jax.vjp(lambda p_: fp(p_, x, mask, seg, pos), prm)
                    gp = vjp2(put(ctb.astype(yh.dtype)))[0]
                    rec["param_grads"] = {"/".join(str(q) for q in path): np.asarray(jd.to_host(v.get_value() if hasattr(v, "get_value") else v), np.float32)
                                          for path, v in nnx.to_flat_state(gp)}
                if not a.no_grad and ct is not None:
                    ctb = np.zeros((B,) + ct.shape, np.float32)
                    ctb[0] = ct
                    _, vjp = jax.vjp(lambda x_: f(state, x_, mask, seg, pos), x)
                    gin = np.asarray(jd.to_host(vjp(put(ctb.astype(yh.dtype)))[0]), np.float64)[0]
                    rec["in_grad"] = gin.sum(axis=1) if (k == 0 and src.layer0_single_stream_grad) else gin
                rec["seconds"] = time.time() - t0
                out[k] = rec
                if jax.process_index() == 0:
                    print(f"[{tag}] 第 {k} 层 {rec['seconds']:.1f} s", flush=True)
        return out, {"dtype": str(config.dtype), "matmul_precision": str(config.matmul_precision),
                     "sparse_matmul": bool(config.sparse_matmul), "attention": str(config.attention), "global_batch": B}

    def layer_prefix(config, k):
        """HF 第 k 层在整模型状态里的前缀与扫描栈里的位置（与 layer_of 同一套规则）：(前缀, 栈里第几份或 None)。"""
        fd = int(config.first_num_dense_layers)
        if k == "mtp0":
            return "params/decoder/../mtp_block/mtp_layer_1/mtp_1_transformer_layer", None
        if k < fd:
            return "params/decoder/dense_layers", k
        m = k - fd
        num_moe = int(config.num_decoder_layers) - fd
        blk = int(config.indexer_share_interval) if config.use_indexer and config.indexer_share_interval > 1 else 0
        if not blk:
            return "params/decoder/moe_layers", m
        full = (num_moe // blk) * blk
        if m < full:
            return f"params/decoder/moe_layers/layers_{m % blk}", m // blk
        return f"params/decoder/moe_layers_remainder/layers_{m - full}", None

    def to_hf(config, k, pg: dict) -> dict:
        """单层的参数梯度（相对这一层的路径）→ HF 名字（HFView：与 L0 往返、全量抓数同一套反向变换）。"""
        from hf_view import HFView
        view = HFView(config, json.loads(Path(a.hf_config).read_text()), a.stage)
        pre, _slot = layer_prefix(config, k)
        if k == "mtp0":
            pre = "params/mtp_block/mtp_layer_1/mtp_1_transformer_layer"
        hpre = "model.mtp_layers.0." if k == "mtp0" else f"model.layers.{k}."
        flat = {}
        ax = int(config.param_scan_axis)
        for h, srcs in view.by_hf.items():
            if not h.startswith(hpre) or view.is_buffer(h):  # 路由偏置不是参数（MoEBiasVar）
                continue
            for mt_key, i in srcs:
                path = view.leaf_path(mt_key)
                if not path.startswith(pre + "/"):
                    raise ValueError(f"{h}: {path} 不在 {pre} 下")
                g = pg[path[len(pre) + 1:]]
                if i is not None:
                    shp = list(g.shape)
                    shp.insert(ax if g.ndim >= ax else 0, i + 1)
                    full = np.zeros(shp, np.float32)
                    idx = [slice(None)] * full.ndim
                    idx[ax if g.ndim >= ax else 0] = i
                    full[tuple(idx)] = g
                    g = full
                flat["params/model/" + path.split("/", 1)[1]] = g
        return {h: x for h, x in view.tensors(flat, lambda v: v, "param", keep=lambda h: h.startswith(hpre) and not view.is_buffer(h))}

    t_all = time.time()
    test, test_cfg = run_side("tpu", [])
    ref, ref_cfg = ({}, {}) if a.no_ref else run_side("ref", REF_OVERRIDES)
    rows = []
    for k in layers:
        go, gg = gpu_out(k), gpu_in_grad(k)
        r = {"layer": k, "tokens": S}
        for q, gpu in (("out", go), ("in_grad", gg)):
            if gpu is None or q not in test[k]:
                continue
            t = test[k][q]
            r[f"{q}.rel_l2_tpu_vs_gpu"] = M.rel_l2(t, gpu)
            r[f"{q}.cosine_tpu_vs_gpu"] = M.cosine(t, gpu)
            if k in ref and q in ref[k]:
                f = ref[k][q]
                et, eg = M.rel_l2(t, f), M.rel_l2(gpu, f)
                r[f"{q}.rel_l2_tpu_vs_ref"], r[f"{q}.rel_l2_gpu_vs_ref"] = et, eg
                r[f"{q}.R"] = et / max(eg, 1e-300)
        rows.append(r)
        if jax.process_index() == 0:
            print("[layer] " + json.dumps(r, ensure_ascii=False, default=float), flush=True)
    rs = [(r["layer"], q, r[f"{q}.R"]) for r in rows for q in ("out", "in_grad") if f"{q}.R" in r]
    fails = [f"{k}.{q}" for k, q, v in rs if v > R_LIMIT]
    rec = {
        "what": "L2 逐层隔离回放：GPU 第 1 步每层的输入（4 路流）进 TPU 单层程序，比输出与输入梯度",
        "gpu_run": a.gpu_l1 or a.gpu_run, "gpu_source": "逐层抓数（hy4_capture）" if a.gpu_l1 else "janus 全量抓数", "platform": jax.devices()[0].platform, "device_kind": jax.devices()[0].device_kind,
        "reference": "TPU 同一程序 dtype=float32、matmul_precision=highest、sparse_matmul=false（FP64 跑不了时的 FP32 参照）",
        "test_config": test_cfg, "ref_config": ref_cfg, "maxtext_overrides": mt_args,
        "indexer_share": "共享层的 top-k 掩码由同一侧程序在 GPU 给的块首层输入上算出",
        "rule": "R = rel_L2(TPU, 参照) ÷ rel_L2(GPU, 参照) ≤ 1.2 过；逐层逐量取最差",
        "worst": (max(rs, key=lambda t: t[2]) if rs else None), "verdict": ("fail" if fails else "pass") if rs else "inconclusive",
        "fail_items": fails, "seconds": time.time() - t_all, "rows": rows,
    }
    if jax.process_index() == 0:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float))
        if a.param_grads:
            from safetensors.numpy import save_file as _sf
            cfg_t = mt_common.make_config(a.stage, [f"max_target_length={S}", "per_device_batch_size=1", "gradient_accumulation_steps=1",
                                                    "run_name=layer-replay", "base_output_directory=/tmp/hy4out"] + mt_args)
            arrs = {}
            for side, res in (("tpu", test), ("ref", ref)):
                for k, d in res.items():
                    if "param_grads" in d:
                        for h, x in to_hf(cfg_t, k, d["param_grads"]).items():
                            arrs[f"{side}.{h}"] = np.ascontiguousarray(np.asarray(x, np.float32))
            _sf(arrs, str(Path(a.out).with_suffix(".param_grads.safetensors")))
            print(f"参数梯度 {len(arrs)} 个张量 → {Path(a.out).with_suffix('.param_grads.safetensors')}", flush=True)
        if a.save:
            from safetensors.numpy import save_file
            arrs = {f"{side}.{k}.{q}": np.ascontiguousarray(np.asarray(v, np.float32))
                    for side, res in (("tpu", test), ("ref", ref)) for k, d in res.items() for q, v in d.items() if q in ("out", "in_grad")}
            save_file(arrs, str(Path(a.out).with_suffix(".safetensors")))
        print(json.dumps({k: rec[k] for k in ("worst", "verdict", "fail_items")}, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
