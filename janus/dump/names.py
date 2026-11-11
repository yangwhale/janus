"""框架模块名 → 规范名（docs/reference/dump-format.md）。不依赖 torch，便于单测。"""
import re

# Megatron 模块名 → 规范名。层内子模块按数据流：输入归一化 → 注意力（qkv、q/k 归一化、核心注意力、输出投影）
# → MLP 前归一化 → MLP（稠密：fc1 / fc2；MoE：路由在 megatron.py 单独抓、专家、共享专家）→ 层输出。
_L = r"^decoder\.layers\.(\d+)"
_RULES = [
    (re.compile(r"^embedding$"), "embed.out"),
    (re.compile(_L + r"\.input_layernorm$"), r"layers.\1.in"),          # 混元 3 这里是恒等（归一化融进 qkv），输出＝层输入
    (re.compile(_L + r"\.self_attention\.linear_qkv$"), r"layers.\1.attn.qkv.out"),
    (re.compile(_L + r"\.self_attention\.q_layernorm$"), r"layers.\1.attn.q_norm.out"),
    (re.compile(_L + r"\.self_attention\.k_layernorm$"), r"layers.\1.attn.k_norm.out"),
    (re.compile(_L + r"\.self_attention\.core_attention$"), r"layers.\1.attn.core.out"),
    (re.compile(_L + r"\.self_attention\.linear_proj$"), r"layers.\1.attn.proj.out"),
    (re.compile(_L + r"\.self_attention$"), r"layers.\1.attn.out"),
    (re.compile(_L + r"\.pre_mlp_layernorm$"), r"layers.\1.mlp_norm.out"),
    (re.compile(_L + r"\.mlp\.linear_fc1$"), r"layers.\1.mlp.fc1.out"),
    (re.compile(_L + r"\.mlp\.linear_fc2$"), r"layers.\1.mlp.fc2.out"),
    (re.compile(_L + r"\.mlp\.experts$"), r"layers.\1.mlp.experts.out"),
    (re.compile(_L + r"\.mlp\.shared_experts$"), r"layers.\1.mlp.shared.out"),
    (re.compile(_L + r"\.mlp$"), r"layers.\1.mlp.out"),
    (re.compile(_L + r"$"), r"layers.\1.out"),
    (re.compile(r"^decoder\.final_layernorm$"), "final_norm.out"),
    (re.compile(r"^output_layer$"), "logits"),
]


def canonical(name: str) -> str | None:
    for pat, rep in _RULES:
        if pat.match(name):
            return pat.sub(rep, name)
    return None
