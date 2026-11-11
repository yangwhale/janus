"""诊断臂 `--grad-fp32 <部件,…>`：让指定参数的梯度不经过 BF16，直接在 FP32 里求和（对齐报告台账 #20 的定责）。

现象：第 1 步一维 norm 权重、输出层与 embedding 的梯度，TPU 离 FP64 是 GPU 的约 √2 倍（多一次同量级的独立舍入）。
推断的机制：MaxText 把这几类参数先转成计算精度 BF16 再参与计算，梯度于是先以 BF16 落地，再转回 FP32 主权重：
  norm    RMSNorm（rmsnorm_single_rounding）用 lax.reduce_precision 把 scale 舍到 BF16；JAX 把 reduce_precision 当线性算子，
          反向对余切做同样的舍入，整批 token 求和后的 scale 梯度被舍成 BF16。
  head    输出层 DenseGeneral 在 all-gather 之前把 kernel 转成 BF16，wgrad 以 BF16 输出（及 BF16 的 reduce-scatter）。
  embed   Embed 先把整张表转成 BF16 再按 token 取行，反向的 scatter-add 在 BF16 里累加。
GPU（Megatron）把 wgrad 直接累加进 FP32 的 main_grad，norm 的 dγ 在 FP32 kernel 里归约，都没有这一次舍入。

每个部件的改法都保证前向逐位不变（只改反向的精度）：
  norm    scale 的舍入改成直通（前向同一个 reduce_precision，反向梯度原样传回 FP32）；
  head    kernel 以 FP32 进 dot_general（DEFAULT 精度下 TPU 先把 FP32 操作数舍成 BF16 单遍乘，v5p 实测与 BF16 操作数逐位相同），
          FP32 累加、输出转 BF16；kernel 的余切于是是 FP32；
  embed   从 FP32 表取行再转 BF16（与先转后取逐位相同），scatter-add 在 FP32 里做。

用法（train_hy3.py 在建模型之前调 install）：--grad-fp32 norm,head,embed

正式开关在 maxtext 的 hunyuan 分支：rmsnorm_scale_grad_fp32、logits_dense_wgrad_fp32、
embedding_grad_fp32；v5p 上三个一起开与本诊断臂的第 1 步梯度逐位相同（t-mts-all 对 t-g32-all）。这个诊断臂留给还没有这三个
开关的 MaxText 提交（如 0622774）复现定责用。
"""
from __future__ import annotations

PARTS = ("norm", "head", "embed")


def _ste_reduce_precision():
    import functools

    import jax
    from jax import lax

    @functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
    def rp(x, exponent_bits, mantissa_bits):
        return lax.reduce_precision(x, exponent_bits, mantissa_bits)

    def fwd(x, e, m):
        return lax.reduce_precision(x, e, m), None

    def bwd(e, m, _, g):
        return (g,)

    rp.defvjp(fwd, bwd)
    return rp


def install(spec: str, vocab: int) -> dict:
    """vocab：词表大小，用来认出输出层（out_features_shape == (vocab,) 的 DenseGeneral）。"""
    parts = [p for p in spec.split(",") if p]
    bad = [p for p in parts if p not in PARTS]
    if bad:
        raise SystemExit(f"--grad-fp32 不认识：{bad}（可选 {PARTS}）")
    import jax
    import jax.numpy as jnp
    from jax import lax

    done = {}
    if "norm" in parts:
        from maxtext.layers import normalizations

        rp = _ste_reduce_precision()

        class _Lax:
            """normalizations 模块里的 lax：只换 reduce_precision（scale 的舍入）为直通版，其余照旧。"""

            def __getattr__(self, name):
                return getattr(lax, name)

            @staticmethod
            def reduce_precision(x, exponent_bits, mantissa_bits):
                return rp(x, exponent_bits, mantissa_bits)

        normalizations.lax = _Lax()
        done["norm"] = "RMSNorm 的 scale 舍入改直通：前向不变，scale 梯度不再舍成 BF16"
    if "head" in parts:
        from maxtext.layers import linears

        orig = linears.DenseGeneral.__call__

        def call(self, inputs, _initializing=False, out_sharding=None, slice_bounds=None):
            if tuple(self.out_features_shape) != (vocab,) or _initializing or slice_bounds is not None:
                return orig(self, inputs, _initializing, out_sharding, slice_bounds)
            kernel = self.kernel.get_value()
            x = jnp.asarray(inputs, self.dtype)
            axis = tuple(a % x.ndim for a in (self.axis if isinstance(self.axis, tuple) else (self.axis,)))
            out = lax.dot_general(x, jnp.asarray(kernel, jnp.float32), ((axis, (0,)), ((), ())),
                                  precision=lax.Precision.DEFAULT, preferred_element_type=jnp.float32)
            return out.astype(self.dtype)

        linears.DenseGeneral.__call__ = call
        done["head"] = "输出层 kernel 以 FP32 进乘法（DEFAULT 精度单遍 BF16 乘、FP32 累加），wgrad 以 FP32 落地"
    if "embed" in parts:
        from maxtext.layers import embeddings

        def ecall(self, inputs, model_mode=embeddings.MODEL_MODE_TRAIN):
            if self.config.use_iota_embed or self.cast_input_dtype:
                raise SystemExit("--grad-fp32 embed 只支持 use_iota_embed=false、不转输入类型")
            table = self.embedding.get_value()
            return jnp.asarray(table.at[inputs].get(), self.dtype)

        embeddings.Embed.__call__ = ecall
        done["embed"] = "embedding 从 FP32 表取行再转 BF16，反向 scatter-add 在 FP32"
    del jax
    return done
