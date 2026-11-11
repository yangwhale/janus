"""诊断臂：路由选专家时给「分数 + 偏置」加一份固定种子的高斯抖动（只影响 top-k 选谁，不影响路由权重）。

用途：第 1 步张量（L2 逐层反向、L3 梯度与优化器状态）的跨平台差主要来自路由近似平局在两种硬件上各落一边。
两侧只改执行方式的重跑都不翻转路由，量不出这一部分。这个臂造出「路由分数与本侧差一个给定量」的另一份实现：
抖动的标准差按「翻转的 token 数与 TPU 对 GPU 实测的翻转数相当」来定，不按张量差来定，所以不循环。
它改了数学（是敏感度臂，不是执行扰动），只作参照，不进对齐配置。

做法：替换 maxtext.layers.moe.RoutedMoE.deepseek_routing，把 gate_logits（选择用的、已加偏置的分数）换成
gate_logits + σ·N(0, 1)；噪声由 PRNGKey(seed) 按追踪顺序 fold_in 生成，同一个 seed 每次编译都相同。
"""
from __future__ import annotations


def install(spec: str):
    """spec = "<σ>:<seed>"，例如 "1.5e-5:3"。"""
    import itertools

    import jax
    import jax.numpy as jnp
    from maxtext.layers import moe

    sigma, seed = spec.split(":")
    sigma, seed = float(sigma), int(seed)
    orig = moe.RoutedMoE.deepseek_routing
    counter = itertools.count()

    def deepseek_routing(self, gate_logits, pre_bias_logits):
        key = jax.random.fold_in(jax.random.PRNGKey(seed), next(counter))
        noise = jax.random.normal(key, gate_logits.shape, jnp.float32).astype(gate_logits.dtype)
        return orig(self, gate_logits + jax.lax.stop_gradient(noise * sigma), pre_bias_logits)

    moe.RoutedMoE.deepseek_routing = deepseek_routing
    return {"sigma": sigma, "seed": seed, "applies_to": "top-k 选择用的分数（gate_logits），路由权重不变"}
