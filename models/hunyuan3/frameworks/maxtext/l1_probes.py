"""第 1 步的 L1 单算子回放：把 GPU golden 的算子输入喂给 TPU 上同一个 MaxText 子模块，单独比这一个算子。
由 layer_replay.py 的 --l1 调用（同一次作业、同一份已载入的模型），结果写进 results.json 的 "l1"。

三类算子：

- 路由 GEMM（MoE 层）：输入 golden `layers.k.mlp_norm.out`（BF16），核是 TPU 检查点里的路由权重（第 1 步 = HF 的 BF16 值，
  已由 L0 逐位核对）。GateLogit 去掉 sigmoid 和偏置，输出原始 logits，对 golden `layers.k.mlp.router.logits` 比
  rel_l2、逐位相同率、FP32 ULP，并按 golden 的 expert_bias 重新选 top-8，数翻转与分差。
  `routed_gate_bf16_rounding` 关 / 开各跑一次。第 1 步核本来就是 BF16 值，开关在这里不该改变结果；
  要检验「DEFAULT 精度下 FP32 操作数按单遍 BF16 乘」这条推断，另用一份不能用 BF16 表示的核（BF16 核加一个
  固定种子的小扰动）：比 DEFAULT 的 FP32 点积、两个操作数先舍 BF16 再点积（FP32 累加）、HIGHEST、FP64 四者。
- RMSNorm（MoE 层）：输入 bf16(layers.k.in + layers.k.attn.out)（两边的残差加都是 BF16），过 TPU 的 MLP 前归一化，
  对 golden `layers.k.mlp_norm.out`；`single_rounding` 关 / 开各跑一次（要 MaxText 有 rmsnorm_single_rounding）。
- 门控激活（第 0 层稠密 MLP）：golden `layers.0.mlp.fc1.out` 按 Megatron 的 chunk 切成 gate | up，在 TPU 上按三种写法算
  silu(gate)·up：BF16 直接算（MaxText 默认）、FP32 算完舍一次（activations_in_float32）、每个算子舍一次
  （mlp_activation_round_per_op）。golden 没抓激活输出，参照是主机上的「GPU 写法」：bf16(bf16(silu(gate))·up)，
  silu 在 FP64 里算再舍到 BF16（第一阶段 L1：torch 的 BF16 silu 与这样舍入 99.974% 逐位相同，差在 x < −89 的尾部）。
"""
from __future__ import annotations

import numpy as np

from janus import metrics as M


def _bf16():
    import ml_dtypes
    return ml_dtypes.bfloat16


def _cmp(name, x, ref, fp64=None) -> dict:
    x, ref = np.asarray(x), np.asarray(ref)
    row = {"name": name, "rel_l2": M.rel_l2(x, ref), "exact_frac": float(np.mean(x.astype(np.float64) == ref.astype(np.float64)))}
    u = M.ulp_summary(x, ref)
    row.update(ulp_p99=u.p99, ulp_max=u.max, ulp_le1=u.buckets.get("0", 0.0) + u.buckets.get("1", 0.0))
    if fp64 is not None:
        f = np.asarray(fp64, np.float64)
        row["rel_l2_vs_fp64"] = M.rel_l2(x.astype(np.float64), f)
        row["ref_rel_l2_vs_fp64"] = M.rel_l2(ref.astype(np.float64), f)
        if x.dtype == _bf16():
            cr = f.astype(np.float32).astype(_bf16())  # FP64 → FP32 → BF16：FP32 那一步对 BF16 结果几乎无影响
            row["correctly_rounded_frac"] = float(np.mean(x.astype(np.float32) == cr.astype(np.float32)))
            row["ref_correctly_rounded_frac"] = float(np.mean(ref.astype(np.float32) == cr.astype(np.float32)))
    return row


def _route(logits, bias, k):
    s = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64))) + np.asarray(bias, np.float64)
    return np.argsort(-s, axis=-1, kind="stable")[:, :k]


def run(layer_of, g, S, B, put, to_host, k, config, mesh, out_rows, save=None):
    """layer_of(i) → (合成好的单层 nnx 模块, 层号)；g 是 golden Run；put 把 [B, …] 主机数组按 batch 切到设备上。
    save：给一个字典时，把生产配置（yml 里开关的取值）下各算子的 TPU 输出原样存进去（名字 l1.<层>.<观测点>），
    另做两个算子回放：注意力（golden 的 q_norm / k_norm / v 进 TPU 的 RoPE + attention_op）与 MoE 块（golden 的 mlp_norm.out
    进 TPU 的 RoutedAndSharedMoE，含路由、megablox 分组 GEMM、合并、共享专家）。FP64 真值与判定在 CPU 上做
    （models/hunyuan3/analysis/l1_replay_results.py），这里只存 TPU 的输出。"""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    bf16 = _bf16()
    rows = []

    def call(mod, *xs):
        gd, st = nnx.split(mod)
        return jax.jit(lambda st_, *a: nnx.merge(gd, st_)(*a))(st, *xs)

    def glob(a):
        """主机上的完整数组 → 全网格复制的全局数组（多主机时各进程给同一份）。"""
        a = np.ascontiguousarray(a)
        sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        return jax.make_array_from_callback(a.shape, sh, lambda idx: a[idx])

    def rep(x):
        x = np.ascontiguousarray(np.broadcast_to(np.asarray(x)[None], (B,) + np.shape(x)))
        return put(x)

    for i in range(int(config.num_decoder_layers)):
        layer = layer_of(i)
        L = f"layers.{i}"
        names = set(g.names(1))
        # ---- RMSNorm（MLP 前）
        # 稠密层的 mlp_norm.out 是残差和（Megatron 第 0 层的归一化融进 fc1，name_map），不是归一化输出，跳过
        if hasattr(layer, "Hunyuan3MoeBlock_0") and f"{L}.attn.out" in names and f"{L}.mlp_norm.out" in names:
            xin = g.tensor(1, f"{L}.in")[:S, 0]
            res = (xin.astype(np.float32) + g.tensor(1, f"{L}.attn.out")[:S, 0].astype(np.float32)).astype(bf16)
            ref = g.tensor(1, f"{L}.mlp_norm.out")[:S, 0]
            norm = layer.post_self_attention_layer_norm
            scale = to_host(norm.scale.get_value()).astype(bf16).astype(np.float64)
            r64 = res.astype(np.float64)
            f64 = r64 / np.sqrt(np.mean(r64 * r64, -1, keepdims=True) + float(norm.epsilon)) * scale
            prod = bool(getattr(config, "rmsnorm_single_rounding", False))
            for flag in ([False, True] if hasattr(norm, "single_rounding") else [False]):
                norm.single_rounding = flag
                y = to_host(call(norm, rep(res)))[0]
                rows.append(dict(_cmp(f"{L}.mlp_norm.out", y, ref, f64), op="rmsnorm", single_rounding=flag))
                if save is not None and flag == prod:
                    save[f"l1.{L}.mlp_norm.out"] = np.ascontiguousarray(y)
            if hasattr(norm, "single_rounding"):
                norm.single_rounding = bool(getattr(config, "rmsnorm_single_rounding", False))
        # ---- 门控激活（稠密层）
        if f"{L}.mlp.fc1.out" in names and not hasattr(layer, "Hunyuan3MoeBlock_0"):
            fc1 = g.tensor(1, f"{L}.mlp.fc1.out")[:S, 0]
            F = fc1.shape[-1] // 2
            gate, up = fc1[:, :F], fc1[:, F:]
            g64 = gate.astype(np.float64)
            silu64 = g64 / (1.0 + np.exp(-g64))
            gpu = (silu64.astype(np.float32).astype(bf16).astype(np.float32) * up.astype(np.float32)).astype(bf16)
            once64 = silu64 * up.astype(np.float64)
            fi = jnp.finfo(jnp.bfloat16)
            modes = {
                "bf16_default": lambda a, b: jax.nn.silu(a) * b,
                "f32_once": lambda a, b: (jax.nn.silu(a.astype(jnp.float32)) * b.astype(jnp.float32)).astype(jnp.bfloat16),
                "per_op": lambda a, b: (jax.lax.reduce_precision(jax.nn.silu(a.astype(jnp.float32)), fi.nexp, fi.nmant)
                                        * b.astype(jnp.float32)).astype(jnp.bfloat16),
            }
            prod_mode = ("per_op" if getattr(config, "mlp_activation_round_per_op", False) else
                         "f32_once" if getattr(config, "activations_in_float32", False) else "bf16_default")
            for mode, fn in modes.items():
                y = to_host(jax.jit(fn)(rep(gate), rep(up)))[0]
                if save is not None and mode == prod_mode:
                    save[f"l1.{L}.mlp.act.out"] = np.ascontiguousarray(y)
                rows.append(dict(_cmp(f"{L}.mlp.act.out", y, gpu, once64), op="silu_mul", mode=mode,
                                 reference="主机 GPU 写法 bf16(bf16(silu(gate))·up)"))
            ys = to_host(jax.jit(jax.nn.silu)(rep(gate)))[0]
            rows.append(dict(_cmp(f"{L}.mlp.silu.out", ys, silu64.astype(np.float32).astype(bf16), silu64),
                             op="silu", mode="bf16_default", reference="FP64 silu 舍到 BF16"))
        # ---- 路由 GEMM
        if hasattr(layer, "Hunyuan3MoeBlock_0") and f"{L}.mlp.router.logits" in names:
            gate_mod = layer.Hunyuan3MoeBlock_0.MoeBlock_0.gate
            x = g.tensor(1, f"{L}.mlp_norm.out")[:S, 0]
            ref = g.tensor(1, f"{L}.mlp.router.logits")[:S, 0].astype(np.float32)
            idx_ref = g.tensor(1, f"{L}.mlp.router.topk_idx")[:S]
            bias = g.tensor(1, f"buffer.decoder.{L}.mlp.router.expert_bias")
            W = to_host(gate_mod.kernel.get_value())                         # [D, E] FP32
            f64 = x.astype(np.float64) @ W.astype(np.float64)
            saved = (gate_mod.score_func, gate_mod.use_bias, getattr(gate_mod, "bf16_round_kernel", False))
            gate_mod.score_func, gate_mod.use_bias = "", False
            scores_ref = 1.0 / (1.0 + np.exp(-ref.astype(np.float64))) + bias.astype(np.float64)
            prod = bool(getattr(config, "routed_gate_bf16_rounding", False))
            for flag in ([False, True] if hasattr(gate_mod, "bf16_round_kernel") else [False]):
                gate_mod.bf16_round_kernel = flag
                lg = to_host(call(gate_mod, rep(x))[0])[0]
                if save is not None and flag == prod:
                    save[f"l1.{L}.mlp.router.logits"] = np.ascontiguousarray(lg)
                row = dict(_cmp(f"{L}.mlp.router.logits", lg, ref, f64), op="router_gemm", bf16_round_kernel=flag,
                           matmul_precision=str(gate_mod.matmul_precision))
                idx = _route(lg, bias, k)
                a = M.topk_agreement(idx, idx_ref)
                row.update(set_frac=a.set_frac, flipped=int(len(a.rows_differ)))
                allm = M.topk_margin(scores_ref, k)
                row["margin_all_p50"] = float(np.percentile(allm, 50))
                if len(a.rows_differ):
                    fm = M.topk_margin(scores_ref, k, rows=a.rows_differ)
                    row.update(margin_flipped_max=float(fm.max()), margin_flipped_p50=float(np.median(fm)),
                               margin_flipped_pctile_max=float((allm <= fm.max()).mean() * 100))
                rows.append(row)
            # 推断检验：不能用 BF16 表示的核
            rng = np.random.default_rng(1000 + i)
            Wp = (W.astype(np.float64) * (1 + 2.0 ** -10 * rng.standard_normal(W.shape))).astype(np.float32)
            gate_mod.kernel.set_value(glob(Wp))
            gate_mod.bf16_round_kernel = False
            d_default = to_host(call(gate_mod, rep(x))[0])[0]
            gate_mod.bf16_round_kernel = True
            d_round = to_host(call(gate_mod, rep(x))[0])[0]
            xb = rep(x.astype(np.float32))
            Wd = glob(Wp)
            d_bf16 = to_host(jax.jit(lambda a, w: jnp.einsum("bsd,de->bse", a.astype(jnp.bfloat16), w.astype(jnp.bfloat16),
                                                           preferred_element_type=jnp.float32))(xb, Wd))[0]
            d_high = to_host(jax.jit(lambda a, w: jnp.einsum("bsd,de->bse", a, w, precision=jax.lax.Precision.HIGHEST))(xb, Wd))[0]
            p64 = x.astype(np.float64) @ Wp.astype(np.float64)
            pb64 = x.astype(np.float64) @ Wp.astype(bf16).astype(np.float64)
            if save is not None:
                save[f"l1.{L}.router.probe.kernel"] = np.ascontiguousarray(Wp)
                save[f"l1.{L}.router.probe.default"] = np.ascontiguousarray(d_default)
                save[f"l1.{L}.router.probe.bf16_operands"] = np.ascontiguousarray(d_bf16)
            rows.append({"name": f"{L}.router.precision_probe", "op": "router_precision_probe",
                         "kernel": "BF16 核 × (1 + 2^-10·N(0,1))，FP32，不能用 BF16 表示",
                         "default_eq_bf16_operands_frac": float(np.mean(d_default == d_bf16)),
                         "default_eq_rounded_kernel_frac": float(np.mean(d_default == d_round)),
                         "default_rel_l2_vs_fp64": M.rel_l2(d_default.astype(np.float64), p64),
                         "default_rel_l2_vs_fp64_bf16_kernel": M.rel_l2(d_default.astype(np.float64), pb64),
                         "highest_rel_l2_vs_fp64": M.rel_l2(d_high.astype(np.float64), p64),
                         "bf16_operands_rel_l2_vs_fp64": M.rel_l2(d_bf16.astype(np.float64), p64)})
            gate_mod.kernel.set_value(glob(W))
            gate_mod.score_func, gate_mod.use_bias, gate_mod.bf16_round_kernel = saved
        # ---- MoE 块：golden 的 mlp_norm.out 进 TPU 的 RoutedAndSharedMoE（路由 + 分组 GEMM + 合并 + 共享专家）
        if save is not None and hasattr(layer, "Hunyuan3MoeBlock_0") and f"{L}.mlp_norm.out" in names:
            x = g.tensor(1, f"{L}.mlp_norm.out")[:S, 0]
            y = call(layer.Hunyuan3MoeBlock_0, rep(x))
            save[f"l1.{L}.mlp.out"] = np.ascontiguousarray(to_host(y[0] if isinstance(y, tuple) else y)[0])
        # ---- 注意力：golden 的 q_norm / k_norm（RoPE 之前）与 v（qkv.out 里按 KV 组交错的那一段）进 TPU 的 RoPE + attention_op
        if save is not None and all(f"{L}.attn.{n}.out" in names for n in ("q_norm", "k_norm", "qkv", "core")):
            att = layer.self_attention
            qn = g.tensor(1, f"{L}.attn.q_norm.out")[:S, 0]                 # [S, H, d]
            kn = g.tensor(1, f"{L}.attn.k_norm.out")[:S, 0]                 # [S, KV, d]
            H, KV, d = qn.shape[1], kn.shape[1], qn.shape[2]
            qkv = g.tensor(1, f"{L}.attn.qkv.out")[:S, 0].reshape(S, KV, (H // KV + 2) * d)
            v = np.ascontiguousarray(qkv[:, :, (H // KV + 1) * d:])         # 每组最后一个头是 v
            pos = g.tensor(1, "input.position_ids")[0, :S].astype(np.int32) if "input.position_ids" in names else np.arange(S, dtype=np.int32)
            gd_a, st_a = nnx.split(att)

            def att_core(st_, q_, k_, v_, seg_, p_):
                a_ = nnx.merge(gd_a, st_)
                q_ = a_.apply_rotary_embedding(q_, inputs_positions=p_)
                k_ = a_.apply_rotary_embedding(k_, inputs_positions=p_)
                if a_.query_pre_attn_scalar and a_.query_pre_attn_scalar != 1.0:   # 同 Attention.__call__：RoPE 之后乘 1/√d
                    q_ = q_ * a_.query_pre_attn_scalar
                return a_.attention_op(q_, k_, v_, seg_, p_, "train")
            o = jax.jit(att_core)(st_a, rep(qn), rep(kn), rep(v), rep(np.ones(S, np.int32)), rep(pos))
            o = to_host(o[0] if isinstance(o, tuple) else o)[0]
            save[f"l1.{L}.attn.core.out"] = np.ascontiguousarray(o.reshape(S, H * d))
    out_rows.extend(rows)
    return rows
