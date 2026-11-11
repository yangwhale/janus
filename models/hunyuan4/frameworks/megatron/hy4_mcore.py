"""混元 4（Hy4-preview）的 Megatron-Core 模型：janus 一侧的模块，不 fork Megatron。

基于 Megatron-Core 0.20.0（NeMo 26.10 镜像里的版本）。能直接用的部件原样用：MLA（`MLASelfAttention`）、
DSA 索引器（`DSAIndexer`）与 top-k（`fused_qk_topk_naive`）、MoE（路由、分发、专家）、GPTModel 的嵌入 /
输出头 / loss、TransformerBlock 的逐层循环。Megatron-Core 没有、要在这里补的（语义以 HF transformers
`hy_v4` v5.22.1 `modeling_hy_v4.py` 加公开 config 为准，再加 项目定下的两处读法）：

1. 门控 MLA（`Hy4MLASelfAttention`）：`out × sigmoid(gate_proj(x))`，逐元素、每头 v_head_dim 个门；
   Megatron-Core 在 MLA 上打开 `attention_output_gate` 直接报错，这里自己建一个门控投影。
2. 稀疏注意力里每头一个可学习 sink（`Hy4DSAttention` + `sparse_sink_attention`）：与 HF eager 同一写法，
   分数在计算精度里算，与 FP32 的 sink 拼起来在 FP32 里做 softmax，丢掉 sink 那一列再乘 V。
3. iHC 超连接（`Hy4HyperConnection`、`Hy4FinalNorm`）：hc_mult 路流、没有 Sinkhorn / comb，强制 FP32；
   每层两处（注意力前、FFN 前），最后 hc_head 收成一路再过 RMSNorm。Megatron-Core 的 mHC 要求 Sinkhorn
   迭代 ≥ 1、把混合系数转回激活精度，语义不同，不用。
4. 逐层的索引器排布（`indexer_types`）：Megatron-Core 只支持「每 N 层一次」的规则排布
   （`dsa_indexer_topk_freq`），这里按显式列表：full 层自己算 top-k，shared 层复用上一个 full 层的。
5. 索引器的 RoPE 放在每头的后 qk_rope_head_dim 维（HF：「Flipped RoPE position」）；Megatron-Core 的
   `DSAIndexer` 按 DeepSeek-V3.2 放在前面，这里改写 `_apply_rope`。
6. 只有路由专家做 SwiGLU 截断（`swiglu_limit`）；稠密层与共享专家不截（Megatron-Core 的
   `activation_func_clamp_value` 对所有 MLP 生效，这里给稠密层与共享专家换一份不截断的 config）。
7. FP32 输出头（`enable_lm_head_fp32`）：输出头权重留 FP32，隐状态转 FP32 再乘；可选 dgrad 分段累加，
   与 HF golden 一侧 `ChunkedHead` 同一写法（`hy4_head_chunk`）。
8. 索引器：`hy4_train_indexer=False` 时不训练（AdamW 时期的 golden：HF 在 `no_grad` 里算），参数设成
   `requires_grad=False`；为 True 时由 Megatron-Core 的 DSA 索引器 KD loss 训练（见 Hy4DSAttention），
   输入照样 detach，主干不受它影响。

RoPE：HF 原样代码是拼接式（NeoX），项目定下的读法是交错式（与 vLLM / SGLang 一致，HF golden 一侧打了同样的
补丁）。Megatron-Core 的 MLA 一律按 `mla_rotary_interleaved` 先去交错再按拼接式转；索引器用
`dsa_indexer_rope_interleaved=True` 走同一条路。去交错是对 q、k 同一个置换，q·k 与交错式逐项相同。

只支持 TP = 1、CP = 1、不打包序列，不符合时构造时就报错；数据并行、专家并行、流水线并行可以任意组合。

流水线并行（PP）：层间传的是 n 路流 [s, b, n·H]，比 Megatron-Core 按 hidden_size 算的 P2P 形状宽 n 倍。
Megatron-Core 自己的 mHC（`enable_mhc_connections`）会在 TransformerBlock 里另做一次展开与收拢，且 GPTModel
不许它与 PP + MTP 同开，所以不用它；训练入口打开 `variable_seq_lengths`，P2P 先交换形状再收发（见
`train_hy4mg.py`）。shared 层复用的 top-k 只在本段里找，切分点要落在 full 层上（`check_pp_split`）。
"""

from __future__ import annotations

import copy
import functools
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerLossAutoScaler,
    DSAIndexerSubmodules,
    DSAttentionSubmodules,
    _compute_index_scores,
    compute_dsa_indexer_loss,
    fused_qk_topk_naive,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.multi_latent_attention import MLASelfAttention, MLASelfAttentionSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

# ------------------------------------------------------------------ config


@dataclass
class Hy4Fields:
    """Megatron-Core 没有的混元 4 字段。`Hy4Config` 与 Megatron-Bridge 的 provider 都继承它。"""

    hc_mult: int = 4
    hc_magnitude: float = 2.0
    hc_eps: float = 1e-6
    hy4_indexer_types: list = field(default_factory=list)  # 逐层 "full" / "shared"，长度 = num_layers
    hy4_lm_head_fp32: bool = True
    hy4_head_chunk: int = 0  # >0：输出头 dgrad 按词表每 chunk 一段累加（段内输出头精度、段间 FP32）
    hy4_attn_scores_fp32: bool = False  # False：q·k 在计算精度里算（HF eager 原样）；True：FP32
    # Muon 不管 iHC 的混合矩阵（fn / hc_head 的 hc_fn），交给标量优化器（Adam）；项目负责人定，同生产的
    # muon_exclude_ihc。用 Megatron-Core 现成的参数属性 use_muon=False（emerging_optimizers._is_muon_excluded 认它）
    hy4_muon_exclude_ihc: bool = True
    # 索引器是否训练：生产用 KD loss 训练索引器（MaxText indexer_loss_scaling_factor > 0）；False 时冻结（AdamW 时期的 golden）
    hy4_train_indexer: bool = False
    # 输出门（linear_gate [H·v, D]）在 Muon 里怎么正交化："whole" 整个矩阵（项目负责人 13:02 定）；
    # "per_head" 每头一个 [v, D]（MaxText 原来的写法）。per_head 用 Megatron-Core 现成的 QKV 拆分：
    # is_qkv=True、qkv_split_shapes=[v]*H（一组、H 段，每段单独正交化与缩放）
    hy4_muon_gate: str = "whole"


@dataclass
class Hy4Config(Hy4Fields, MLATransformerConfig):
    """只用 Megatron-Core（不经 Megatron-Bridge）建模型时的 config，如 CPU 等价性检查。"""


# 发布的 config.json（与 MaxText hy4-preview.yml 同值）里、和维度无关的键。随机初始化的小模型（hy4-tiny / hy4-mini）
# 的 config.json 只写了维度，缺这些键时 HF 的 HYV4Config 默认值会读错（rope_theta 默认 1e4、门控 / sink / iHC 默认 None），
# 所以按发布的取值补上，并在 meta 里记下补了哪些。
HY4_RELEASED_NON_DIM = {
    "attention_bias": False,
    "bos_token_id": 120000,
    "eos_token_id": 120025,
    "enable_ihc": True,
    "enable_lm_head_fp32": True,
    "gated_mla": True,
    "gating_type": "elementwise",
    "hc_eps": 1e-6,
    "hc_magnitude": 2.0,
    "hidden_act": "silu",
    "initializer_range": 0.006,
    "learnable_sink": True,
    "learnable_sink_init": 0.0,
    "mtp_loss_factor": 0.1,
    "n_group": 1,
    "norm_topk_prob": True,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    "routed_scaling_factor": 2.827,
    "swiglu_limit": 10.0,
    "topk_group": 1,
    "use_dsa": True,
    "use_mla": True,
}


def complete_hf_config(hf: dict) -> tuple[dict, list]:
    """缺的非维度键按发布的取值补；head_dim / qk_head_dim / layer_types 由维度推出。返回 (补全后的 config, 补了的键)。"""
    out, filled = dict(hf), []
    for k, v in HY4_RELEASED_NON_DIM.items():
        if k not in out:
            out[k] = v
            filled.append(k)
    derived = {
        "head_dim": out["qk_rope_head_dim"],
        "qk_head_dim": out["qk_nope_head_dim"] + out["qk_rope_head_dim"],
        "layer_types": ["deepseek_sparse_attention"] * out["num_hidden_layers"],
    }
    for k, v in derived.items():
        if k not in out:
            out[k] = v
            filled.append(k)
    return out, filled


def indexer_loss_coeff_per_layer(maxtext_coeff: float, indexer_types, mtp_num_layers: int) -> float:
    """MaxText 的 indexer_loss_scaling_factor → Megatron-Core 的逐层 dsa_indexer_loss_coeff。

    MaxText train.py 把各层（full 索引器层 + MTP 层；shared 层不产出）的 KD loss 取平均再乘系数；
    Megatron-Core 每层各自加进目标。所以逐层系数 = 系数 ÷ 产出 loss 的层数。
    """
    n = sum(t == "full" for t in indexer_types) + int(mtp_num_layers or 0)
    return float(maxtext_coeff) / max(n, 1)


def hy4_config_kwargs(hf: dict, num_layers: int | None = None) -> dict:
    """公开 config.json（dict）→ `Hy4Config` / provider 的字段。截层时三张逐层表一起截。"""
    L = num_layers if num_layers is not None else hf["num_hidden_layers"]
    mlp_types = list(hf["mlp_layer_types"])[:L]
    idx_types = list(hf["indexer_types"])[:L]
    assert len(mlp_types) == L and len(idx_types) == L, "逐层表比层数短"
    assert set(hf["layer_types"][:L]) == {"deepseek_sparse_attention"}, hf["layer_types"][:L]
    assert hf.get("use_mla", True) and hf.get("use_dsa", True) and hf.get("gated_mla", True)
    assert hf.get("gating_type", "elementwise") == "elementwise" and hf.get("enable_ihc", True)
    assert hf.get("learnable_sink", True) and hf["rope_parameters"]["rope_type"] == "default"
    assert hf["n_group"] == 1 and hf["topk_group"] == 1 and hf["norm_topk_prob"]
    assert hf["qk_nope_head_dim"] + hf["qk_rope_head_dim"] == hf["qk_head_dim"]
    assert hf["head_dim"] == hf["qk_rope_head_dim"], "RoPE 维数 = head_dim（HF 用它算频率）"
    assert not hf.get("attention_bias", False) and not hf.get("tie_word_embeddings", False)
    assert hf["hidden_act"] == "silu"
    return dict(
        num_layers=L,
        hidden_size=hf["hidden_size"],
        num_attention_heads=hf["num_attention_heads"],
        num_query_groups=hf["num_attention_heads"],
        kv_channels=hf["v_head_dim"],
        ffn_hidden_size=hf["intermediate_size"],
        normalization="RMSNorm",
        layernorm_epsilon=hf["rms_norm_eps"],
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method_std=hf["initializer_range"],
        # MLA（Megatron 的 qk_head_dim 是不带 RoPE 的那段）
        multi_latent_attention=True,
        q_lora_rank=hf["q_lora_rank"],
        kv_lora_rank=hf["kv_lora_rank"],
        qk_head_dim=hf["qk_nope_head_dim"],
        qk_pos_emb_head_dim=hf["qk_rope_head_dim"],
        v_head_dim=hf["v_head_dim"],
        qk_layernorm=True,
        # 项目定下的读法：MLA 两个 latent norm 用 rms_norm_eps（HF 原样是 1e-6）
        attention_latent_norm_epsilon=hf["rms_norm_eps"],
        rope_type="rope",
        rotary_base=float(hf["rope_parameters"]["rope_theta"]),
        rotary_percent=1.0,
        rotary_scaling_factor=1.0,
        mscale=1.0,
        mscale_all_dim=0.0,
        rotary_interleaved=False,
        apply_rope_fusion=False,
        # DSA 索引器
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=hf["index_n_heads"],
        dsa_indexer_head_dim=hf["index_head_dim"],
        dsa_indexer_topk=hf["index_topk"],
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_rope_interleaved=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_k_norm_epsilon=1e-6,  # 项目定下的读法：索引器 k_norm 1e-6（HF 原样是 rms_norm_eps）
        dsa_indexer_k_norm_fp32=True,
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype="fp32",
        dsa_kernel_backend="none",
        # MoE
        num_moe_experts=hf["n_routed_experts"],
        moe_router_topk=hf["num_experts_per_tok"],
        moe_ffn_hidden_size=hf["moe_intermediate_size"],
        moe_shared_expert_intermediate_size=hf["moe_intermediate_size"] * hf["n_shared_experts"],
        moe_layer_freq=[int(t == "sparse") for t in mlp_types],
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_topk_scaling_factor=float(hf["routed_scaling_factor"]),
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_router_pre_softmax=False,
        moe_token_dispatcher_type="alltoall",
        activation_func_clamp_value=float(hf["swiglu_limit"]),  # 只给路由专家；见 Hy4TransformerLayer
        # 混元 4 自己的
        hc_mult=hf["hc_mult"],
        hc_magnitude=float(hf["hc_magnitude"]),
        hc_eps=float(hf["hc_eps"]),
        hy4_indexer_types=idx_types,
        hy4_lm_head_fp32=bool(hf.get("enable_lm_head_fp32", True)),
    )


def _no_clamp(config):
    c = copy.copy(config)
    c.activation_func_clamp_value = None
    return c


def _check_parallel(config):
    if config.tensor_model_parallel_size != 1 or config.context_parallel_size != 1:
        raise NotImplementedError("混元 4 的 Megatron 模型只支持 TP = 1、CP = 1（golden 用 DP + EP）")


# ------------------------------------------------------------------ iHC


class Hy4HyperConnection(MegatronModule):
    """HF `HYV4HyperConnection`：一处 iHC。输入 n 路流 [s, b, n·H]，返回 (post [s, b, n] FP32, 聚合后的一路 [s, b, H])。

    全程 FP32：混合系数、聚合（逐路加权求和）都在 FP32 里，最后一路转回流的精度。参数（fn、base、scale）留 FP32。
    """

    def __init__(self, config):
        super().__init__(config)
        n, H = config.hc_mult, config.hidden_size
        self.n, self.H = n, H
        self.norm_eps = config.layernorm_epsilon  # HF：HYV4UnweightedRMSNorm(eps=rms_norm_eps)
        self.hc_eps = config.hc_eps
        self.magnitude = config.hc_magnitude
        self.fn = nn.Parameter(torch.empty(2 * n, n * H, dtype=torch.float32))
        if config.hy4_muon_exclude_ihc:
            self.fn.use_muon = False
        self.base = nn.Parameter(torch.empty(2 * n, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        with torch.no_grad():  # HF _init_weights 的初值（载入权重时会被覆盖）
            nn.init.normal_(self.fn, mean=0.0, std=config.init_method_std)
            self.scale.fill_(0.01)
            self.base.zero_()
            self.base[:n] = -torch.log(torch.tensor(float(max(n - 1, 1))))
        for p in (self.fn, self.base, self.scale):
            mark_keep_in_fp32(p)

    def forward(self, x: torch.Tensor):
        s, b, _ = x.shape
        n = self.n
        with torch.autocast(device_type=x.device.type, enabled=False):
            flat = x.float()
            mixes = F.linear(flat, self.fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
            pre_b, post_b = self.base.float().split(n, dim=-1)
            pre_s, post_s = self.scale.float().unbind(0)
            pre_l, post_l = mixes.split(n, dim=-1)
            pre = torch.sigmoid(pre_l * pre_s + pre_b) + self.hc_eps
            post = self.magnitude * torch.sigmoid(post_l * post_s + post_b) + self.hc_eps
            out = torch.sum(pre.unsqueeze(-1) * x.view(s, b, n, self.H), dim=2)
        return post, out.to(x.dtype)


def hc_residual(post: torch.Tensor, h: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """HF 解码层的残差更新：(post ⊗ h + residual) 在 FP32 里算，转回流的精度。residual [s, b, n, H]。"""
    return (post.float().unsqueeze(-1) * h.float().unsqueeze(-2) + residual.float()).to(residual.dtype)


class Hy4FinalNorm(MegatronModule):
    """放在 TransformerBlock 的 final_layernorm 位置：HF `norm(hc_head(·))`。hc_head 把 n 路流收成一路（FP32）。"""

    def __init__(self, config, hidden_size: int, eps: float = 1e-5, norm_cls=None):
        super().__init__(config)
        n = config.hc_mult
        self.n, self.H = n, hidden_size
        self.norm_eps = config.layernorm_epsilon
        self.hc_eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(n, n * hidden_size, dtype=torch.float32))
        if config.hy4_muon_exclude_ihc:
            self.hc_fn.use_muon = False
        self.hc_base = nn.Parameter(torch.empty(n, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        with torch.no_grad():
            nn.init.normal_(self.hc_fn, mean=0.0, std=config.init_method_std)
            self.hc_scale.fill_(0.01)
            self.hc_base.fill_(-float(torch.log(torch.tensor(float(max(n - 1, 1))))))
        for p in (self.hc_fn, self.hc_base, self.hc_scale):
            mark_keep_in_fp32(p)
        self.norm = (norm_cls or _default_norm_cls(config))(config=config, hidden_size=hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, b, _ = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            flat = x.float()
            mixes = F.linear(flat, self.hc_fn.float()) * torch.rsqrt(
                flat.square().mean(-1, keepdim=True) + self.norm_eps
            )
            pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.hc_eps
            out = (pre.unsqueeze(-1) * x.view(s, b, self.n, self.H)).sum(dim=2).to(x.dtype)
        return self.norm(out)


def _default_norm_cls(config):
    if config.transformer_impl == "transformer_engine":
        from megatron.core.extensions.transformer_engine import TENorm

        return TENorm
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    return WrappedTorchNorm


# ------------------------------------------------------------------ 注意力


def causal_sparse_mask(topk: torch.Tensor, sq: int, sk: int) -> torch.Tensor:
    """top-k 下标 [b, sq, k]（-1 表示无效）→ 允许的位置 [b, sq, sk]（选中且不越过因果）。"""
    b = topk.shape[0]
    idx = torch.where(topk >= 0, topk, torch.full_like(topk, sk)).long()
    allowed = torch.zeros(b, sq, sk + 1, dtype=torch.bool, device=topk.device)
    allowed.scatter_(-1, idx, True)
    allowed = allowed[..., :sk]
    causal = torch.ones(sq, sk, dtype=torch.bool, device=topk.device).tril(diagonal=sk - sq)
    return allowed & causal


def sparse_sink_attention(
    query, key, value, topk, sinks, scale: float, scores_fp32: bool = False, return_max_logit: bool = False
):
    """HF `eager_attention_forward`（带 sink）在 DSA 稀疏掩码下的同一算法。

    query/key [s, b, h, d]，value [s, b, h, dv]，topk [b, s, k]，sinks [h]（FP32）。返回 [s, b, h·dv]。
    被掩掉的位置置 -inf；HF 是加上 dtype 的最小值，softmax 之后两者都是精确的 0。
    """
    sq, b, h, _ = query.shape
    sk = key.shape[0]
    q = query.permute(1, 2, 0, 3)
    k = key.permute(1, 2, 0, 3)
    v = value.permute(1, 2, 0, 3)
    if scores_fp32:
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    else:
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    allowed = causal_sparse_mask(topk, sq, sk).unsqueeze(1)
    scores = scores.masked_fill(~allowed, float("-inf"))
    max_logit = scores.detach().float().amax(dim=(0, 2, 3)) if return_max_logit else None  # 每头，不含 sink
    sink = sinks.float().view(1, h, 1, 1).expand(b, h, sq, 1)
    comb = torch.cat([scores.float(), sink], dim=-1)
    comb = comb - comb.max(dim=-1, keepdim=True).values
    probs = torch.softmax(comb, dim=-1)[..., :-1].to(v.dtype)
    out = torch.matmul(probs, v)  # [b, h, s, dv]
    out = out.permute(2, 0, 1, 3).reshape(sq, b, h * v.shape[-1])
    return (out, max_logit) if return_max_logit else out


class Hy4Indexer(DSAIndexer):
    """Megatron-Core 的 DSA 索引器，RoPE 改放在每头的后 qk_rope_head_dim 维（HF 的布局）。"""

    def __init__(self, config, submodules, pg_collection=None):
        super().__init__(config, submodules, pg_collection=pg_collection)
        for p in self.k_norm.parameters():
            p.data = p.data.float()
            mark_keep_in_fp32(p)  # HF 的 k_norm 留 FP32；dsa_indexer_k_norm_fp32 把输入转 FP32 再归一化
        if not getattr(config, "hy4_train_indexer", False):
            for p in self.parameters():
                p.requires_grad_(False)

    def _apply_rope(self, x, rotary_pos_emb, mscale, cu_seqlens=None):
        assert cu_seqlens is None
        rope = self.qk_pos_emb_head_dim
        x_nope, x_pe = torch.split(x, [self.index_head_dim - rope, rope], dim=-1)
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
            mla_rotary_interleaved=self.config.dsa_indexer_rope_interleaved,
        )
        return torch.cat([x_nope, x_pe], dim=-1)


_TOPK_STORE_ATTR = "_hy4_topk_store"
# shared 层复用的 top-k 按 (microbatch 号, 层号) 存：流水线 1F1B 下第 m 个 microbatch 的反向（整层重算时
# shared 层要重新取 top-k）发生在第 m+1… 个 microbatch 的前向之后，只按层号存会取到别的 microbatch 的。
# 训练入口在前向开始时、以及该 microbatch 的反向到达本段输出时调 set_topk_micro（见 train_hy4mg.py）。
_TOPK_MICRO = [0]


def set_topk_micro(m: int) -> None:
    _TOPK_MICRO[0] = m


def clear_topk_store(config) -> None:
    """每步开始清空 top-k 存放处（microbatch 号每步从 0 起）。"""
    st = getattr(config, _TOPK_STORE_ATTR, None)
    if st is not None:
        st.clear()


class Hy4DSAttention(MegatronModule):
    """MLA 的 core_attention：DSA 选 top-k（full 层用自己的索引器，shared 层复用上一个 full 层）+ 带 sink 的稀疏注意力。

    `force_full`：MTP 层自带 full 索引器、不复用主干的 top-k。`capture=True` 时把这一次的 top-k 留在 `last_topk`
    （等价性检查用）。
    """

    requires_dsa_inputs = True  # MLASelfAttention 据此把 x 与 qr 传进来

    def __init__(
        self,
        config,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout=None,
        softmax_scale=None,
        k_channels=None,
        v_channels=None,
        cp_comm_type=None,
        pg_collection=None,
        force_full: bool = False,
    ):
        super().__init__(config=config)
        _check_parallel(config)
        assert attn_mask_type == AttnMaskType.causal
        self.layer_number = layer_number
        self.topk = config.dsa_indexer_topk
        types = config.hy4_indexer_types
        if force_full:
            self.is_full, self.source_layer = True, None
        else:
            assert len(types) == config.num_layers, (len(types), config.num_layers)
            self.is_full = types[layer_number - 1] == "full"
            prev = [i + 1 for i in range(layer_number) if types[i] == "full"]
            if not prev:
                raise ValueError(f"第 {layer_number} 层是 shared，但它前面没有 full 层")
            self.source_layer = None if self.is_full else prev[-1]
        self.force_full = force_full
        self.indexer = (
            build_module(submodules.indexer, config=config, pg_collection=pg_collection) if self.is_full else None
        )
        self.softmax_scale = softmax_scale
        self.sinks = nn.Parameter(torch.zeros(config.num_attention_heads, dtype=torch.float32))
        mark_keep_in_fp32(self.sinks)
        self.capture = False
        self.last_topk = None
        self.current_max_attn_logits = None  # QK-Clip 用（见 forward 末尾）
        self.inject_topk = None  # 诊断：给定就用它代替自己索引器的 top-k（等价性检查里灌 HF 的选择，隔离掉平局翻转）
        self.pg_collection = pg_collection
        self.last_indexer_loss = None  # 本层这次前向的索引器 KD loss（已乘逐层系数），记录用

    def _store(self):
        st = getattr(self.config, _TOPK_STORE_ATTR, None)
        if st is None:
            st = {}
            setattr(self.config, _TOPK_STORE_ATTR, st)
        return st

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        x,
        qr,
        position_ids=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        up_v_weight=None,
    ):
        assert packed_seq_params is None and up_v_weight is None and value is not None
        sq, sk = query.shape[0], key.shape[0]
        coeff = float(self.config.dsa_indexer_loss_coeff or 0.0)
        want_loss = self.is_full and coeff > 0 and self.training and torch.is_grad_enabled()
        causal = torch.full((sq, sk), float("-inf"), device=query.device).triu(diagonal=1 + sk - sq)
        q = k = w = None
        if self.is_full and (want_loss or self.inject_topk is None):
            # 索引器的输入先 detach（同 MaxText 的 stop_gradient、Megatron-Core 的 DSAttention）：KD loss 只更新索引器
            with torch.set_grad_enabled(want_loss):
                q, k, w = self.indexer.forward_before_topk(x.detach(), qr.detach(), None)
        if self.is_full and self.inject_topk is not None:
            topk = self.inject_topk.to(device=query.device, dtype=torch.long)
            if not self.force_full:
                self._store()[(_TOPK_MICRO[0], self.layer_number)] = topk
        elif self.is_full:
            with torch.no_grad():
                _, topk = fused_qk_topk_naive(
                    q.detach(),
                    k.detach(),
                    w.detach(),
                    self.topk,
                    mask=causal,
                    use_relu=self.config.dsa_indexer_scoring_relu,
                )
            if not self.force_full:
                self._store()[(_TOPK_MICRO[0], self.layer_number)] = topk
        else:
            topk = self._store()[(_TOPK_MICRO[0], self.source_layer)]
        indexer_loss = None
        if want_loss:
            # Megatron-Core 原样的 KD：教师 = Σ_头 softmax(缩放后 q·k) 再按行 L1 归一（不含 sink，同 MaxText），
            # 学生 = 索引分数的 log_softmax；sparse 时两边都只在 top-k ∩ 因果上；q / k 主干一侧 detach。
            # 逐层系数 = MaxText 的 indexer_loss_scaling_factor ÷ 产出 KD loss 的层数（full 层 + MTP），
            # 因为 MaxText train.py 对各层取平均，而这里每层经 DSAIndexerLossAutoScaler 各自加进目标（求和）。
            index_scores = _compute_index_scores(q, w, k, use_relu=self.config.dsa_indexer_scoring_relu)
            indexer_loss = compute_dsa_indexer_loss(
                index_scores,
                topk,
                query.detach(),
                key.detach(),
                self.softmax_scale,
                coeff,
                bool(self.config.dsa_indexer_use_sparse_loss),
                self.pg_collection,
                mask=causal,
                calculate_per_token_loss=bool(self.config.calculate_per_token_loss),
            )
            self.last_indexer_loss = indexer_loss.detach()
            acc = getattr(self, "indexer_loss_accum", None)  # 训练入口按步读取并清零（各 microbatch 相加）
            self.indexer_loss_accum = self.last_indexer_loss if acc is None else acc + self.last_indexer_loss
        if self.capture:
            self.last_topk = topk
            if self.is_full:  # 索引器 KD 检查用：同一份输入在检查里按 MaxText 的公式重算
                self.last_kd_inputs = (x.detach(), qr.detach(), query.detach(), key.detach())
        want_max = bool(self.config.qk_clip or self.config.log_max_attention_logit)
        out = sparse_sink_attention(
            query,
            key,
            value,
            topk,
            self.sinks,
            self.softmax_scale,
            self.config.hy4_attn_scores_fp32,
            return_max_logit=want_max,
        )
        if not want_max:
            return DSAIndexerLossAutoScaler.apply(out, indexer_loss) if indexer_loss is not None else out
        # QK-Clip 的统计量：同 Megatron 的 TE 注意力，记每头缩放后 q·k 的最大值（只在选中且因果的位置上，不含 sink），
        # 跨 microbatch 取最大，由 megatron.core.optimizer.qk_clip.clip_qk 在优化器步后归约、裁剪并清零
        out, m = out
        self.current_max_attn_logits = (
            m if self.current_max_attn_logits is None else torch.maximum(self.current_max_attn_logits, m)
        )
        return DSAIndexerLossAutoScaler.apply(out, indexer_loss) if indexer_loss is not None else out


class Hy4MLASelfAttention(MLASelfAttention):
    """MLA + 逐元素输出门：core_attention 的输出 [s, b, h·dv] 乘 sigmoid(linear_gate(x))，再进 linear_proj。"""

    def __init__(self, config, submodules, layer_number, gate_linear=None, **kwargs):
        super().__init__(config, submodules, layer_number, **kwargs)
        _check_parallel(config)
        self.linear_gate = build_module(
            gate_linear,
            config.hidden_size,
            config.num_attention_heads * config.v_head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="gate",
            tp_group=self.pg_collection.tp,
        )
        assert config.hy4_muon_gate in ("whole", "per_head"), config.hy4_muon_gate
        if config.hy4_muon_gate == "per_head":
            w = self.linear_gate.weight
            w.is_qkv = True
            w.qkv_split_shapes = [config.v_head_dim] * config.num_attention_heads
        self._gate_input = None

    def forward(self, hidden_states, attention_mask, *args, **kwargs):
        self._gate_input = hidden_states
        try:
            return super().forward(hidden_states, attention_mask, *args, **kwargs)
        finally:
            self._gate_input = None

    def _run_core_attention(self, *args, **kwargs):
        out = super()._run_core_attention(*args, **kwargs)
        gate, _ = self.linear_gate(self._gate_input)
        return out * torch.sigmoid(gate)


# ------------------------------------------------------------------ 解码层


class Hy4TransformerLayer(MegatronModule, BaseTransformerLayer):
    """HF `HYV4DecoderLayer`：attn_hc → input_layernorm → 门控 MLA + DSA → 残差；ffn_hc → pre_mlp_layernorm → MLP/MoE → 残差。

    层间传的是 n 路流 [s, b, n·H]（流的精度 = 计算精度）；第一层收到一路 [s, b, H]（embedding）时先复制成 n 路（HF 同）。
    """

    def __init__(
        self,
        config,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout=None,
        pg_collection=None,
        vp_stage=None,
        is_mtp_layer: bool = False,
        **kwargs,
    ):
        super().__init__(config=config)
        _check_parallel(config)
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.mlp import MLP
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.utils import get_pg_rank

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.layer_number = layer_number + get_transformer_layer_offset(config, vp_stage, get_pg_rank(pg_collection.pp))
        self.is_mtp_layer = is_mtp_layer
        H, eps = config.hidden_size, config.layernorm_epsilon
        self.input_layernorm = submodules.input_layernorm(config=config, hidden_size=H, eps=eps)
        self.self_attention = build_module(
            submodules.self_attention, config=config, layer_number=self.layer_number, pg_collection=pg_collection
        )
        self.pre_mlp_layernorm = submodules.pre_mlp_layernorm(config=config, hidden_size=H, eps=eps)
        mlp = submodules.mlp
        if isinstance(mlp, ModuleSpec) and mlp.module is MLP:  # 同 TransformerLayer 的写法
            mlp = functools.partial(MLP.as_mlp_submodule, submodules=mlp.submodules, **mlp.params)
        self.mlp = mlp(
            config=config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            name=None,
            hash_moe_layer_threshold=None,
        )
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(self.layer_number)
        self.is_moe_layer = isinstance(self.mlp, MoELayer)
        # 只有路由专家截断（HF HYV4Experts._apply_gate）；稠密层与共享专家不截
        if self.is_moe_layer:
            eb = getattr(self.mlp.router, "expert_bias", None)
            if eb is not None:
                mark_keep_in_fp32(eb)  # checkpoint 里是 FP32；Float16Module 不要把它舍成 BF16
            if getattr(self.mlp, "shared_experts", None) is not None:
                self.mlp.shared_experts.config = _no_clamp(self.mlp.shared_experts.config)
        else:
            self.mlp.config = _no_clamp(self.mlp.config)
        self.attn_hc = Hy4HyperConnection(config)
        self.ffn_hc = Hy4HyperConnection(config)

    def forward(self, hidden_states, attention_mask=None, packed_seq_params=None, position_ids=None, **kwargs):
        assert packed_seq_params is None
        n, H = self.config.hc_mult, self.config.hidden_size
        s, b, d = hidden_states.shape
        if d == H:
            hidden_states = hidden_states.unsqueeze(2).expand(s, b, n, H).reshape(s, b, n * H)
        residual = hidden_states.view(s, b, n, H)
        post, x = self.attn_hc(hidden_states)
        x = self.input_layernorm(x)
        attn_out, attn_bias = self.self_attention(x, attention_mask=attention_mask, position_ids=position_ids)
        assert attn_bias is None
        h = hc_residual(post, attn_out, residual)
        post, x = self.ffn_hc(h.view(s, b, n * H))
        x = self.pre_mlp_layernorm(x)
        mlp_out, mlp_bias = self.mlp(x)
        assert mlp_bias is None
        h = hc_residual(post, mlp_out, h)
        return h.view(s, b, n * H), None


# ------------------------------------------------------------------ spec


def _backend(use_te: bool):
    if use_te:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TELinear,
            TENorm,
            TERowParallelLinear,
        )

        return dict(col=TEColumnParallelLinear, row=TERowParallelLinear, dup=TELinear, norm=TENorm)
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    return dict(col=ColumnParallelLinear, row=RowParallelLinear, dup=DuplicatedLinear, norm=WrappedTorchNorm)


class DuplicatedLinear(nn.Module):
    """只用 Megatron-Core（没有 TE）时的不切分线性层，接口同 TELinear(parallel_mode="duplicated")。"""

    def __init__(self, input_size, output_size, *, config, init_method, bias=False, skip_bias_add=False, **_):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
        if config.perform_initialization:
            init_method(self.weight)
        assert not bias

    def forward(self, x):
        return F.linear(x, self.weight), None

    def backward_dw(self):
        pass


def attention_spec(config, use_te: bool, force_full: bool = False) -> ModuleSpec:
    be = _backend(use_te)
    indexer = ModuleSpec(
        module=Hy4Indexer,
        submodules=DSAIndexerSubmodules(
            linear_wq_b=be["dup"], linear_wk=be["dup"], k_norm=be["norm"], linear_weights_proj=be["dup"]
        ),
    )
    core = ModuleSpec(
        module=Hy4DSAttention,
        submodules=DSAttentionSubmodules(indexer=indexer),
        params={"force_full": force_full},
    )
    # q/kv 的 latent norm 不和上投影融合：索引器要归一化之后的 q（同 Megatron-Core 的 DSA spec）
    return ModuleSpec(
        module=Hy4MLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal, "gate_linear": be["col"]},
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=be["col"],
            linear_q_down_proj=be["dup"] if use_te else be["col"],
            linear_q_up_proj=be["col"],
            linear_kv_down_proj=be["dup"] if use_te else be["col"],
            linear_kv_up_proj=be["col"],
            core_attention=core,
            linear_proj=be["row"],
            q_layernorm=be["norm"],
            kv_layernorm=be["norm"],
        ),
    )


def mlp_specs(config, use_te: bool):
    """(稠密 MLP, MoE) 的 spec。稠密层的 pre_mlp_layernorm 不和 fc1 融合（权重名两种层一致）。"""
    from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend

    if use_te:
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        backend = TESpecProvider()
    else:
        from megatron.core.models.backends import LocalSpecProvider

        backend = LocalSpecProvider()
    from megatron.core.transformer.mlp import MLP, MLPSubmodules

    be = _backend(use_te)
    dense = ModuleSpec(module=MLP, submodules=MLPSubmodules(linear_fc1=be["col"], linear_fc2=be["row"]))
    moe = get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_te_activation_func=False,
    )
    del get_mlp_module_spec_for_backend
    return dense, moe


def stage_layers(config, pp_rank: int | None = None) -> range:
    """本流水线段负责的全局层号（0 起），照 Megatron-Core 的切法（含首末段不等分）。"""
    from megatron.core.transformer.transformer_block import get_num_layers_to_build

    if config.pipeline_model_parallel_size == 1:
        return range(config.num_layers)
    off = get_transformer_layer_offset(config, None, pp_rank)
    return range(off, off + get_num_layers_to_build(config, None, pp_rank))


def check_pp_split(config) -> list:
    """每段的层号范围；shared 层复用的 full 层必须在同一段里（top-k 不跨段传），否则报错。"""
    types = config.hy4_indexer_types
    out = []
    for r in range(config.pipeline_model_parallel_size):
        rg = stage_layers(config, r)
        for i in rg:
            if types[i] == "shared":
                src = max(j for j in range(i) if types[j] == "full")
                if src < rg.start:
                    raise ValueError(
                        f"流水线第 {r} 段（层 {rg.start}–{rg.stop - 1}）的第 {i} 层是 shared，复用的第 {src} 层在前一段；"
                        "切分点要落在 full 层上（num_layers_in_first_pipeline_stage / num_layers_in_last_pipeline_stage）"
                    )
        out.append([rg.start, rg.stop - 1])
    return out


def layer_specs(config, use_te: bool) -> TransformerBlockSubmodules:
    """本流水线段的解码层 spec（PP = 1 时是全部层）。"""
    be = _backend(use_te)
    attn = attention_spec(config, use_te)
    dense, moe = mlp_specs(config, use_te)
    specs = []
    for i in stage_layers(config):
        specs.append(
            ModuleSpec(
                module=Hy4TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=be["norm"],
                    self_attention=attn,
                    pre_mlp_layernorm=be["norm"],
                    mlp=moe if config.moe_layer_freq[i] else dense,
                ),
            )
        )
    final = ModuleSpec(module=Hy4FinalNorm, params={"norm_cls": be["norm"]})
    return TransformerBlockSubmodules(layer_specs=specs, layer_norm=final)


def mtp_block_spec(config, use_te: bool):
    """MTP：一层普通预归一化残差的解码层（不带 iHC），注意力自带 full 索引器，FFN 是 MoE。"""
    from megatron.core.transformer.multi_token_prediction import (
        MultiTokenPredictionBlockSubmodules,
        get_mtp_layer_spec_for_backend,
    )

    be = _backend(use_te)
    _, moe = mlp_specs(config, use_te)
    inner = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=be["norm"],
            self_attention=attention_spec(config, use_te, force_full=True),
            self_attn_bda=_bda(),
            pre_mlp_layernorm=be["norm"],
            mlp=moe,
            mlp_bda=_bda(),
        ),
    )
    if use_te:
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        backend = TESpecProvider()
    else:
        from megatron.core.models.backends import LocalSpecProvider

        backend = LocalSpecProvider()
    layer = get_mtp_layer_spec_for_backend(mtp_model_layer_spec=inner, backend=backend, rms_norm=True)
    return MultiTokenPredictionBlockSubmodules(layer_specs=[layer] * config.mtp_num_layers)


def _bda():
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

    return get_bias_dropout_add


# ------------------------------------------------------------------ 输出头


class _ChunkedHeadFn(torch.autograd.Function):
    """输出头 dgrad 分段累加：与 HF golden 一侧 `hy4_model._ChunkedHeadFn` 同一写法。"""

    @staticmethod
    def forward(ctx, x, w, chunk):
        ctx.save_for_backward(x, w)
        ctx.chunk = chunk
        return F.linear(x, w)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        V = w.shape[0]
        gx = torch.zeros(gy.shape[:-1] + (w.shape[1],), dtype=torch.float32, device=gy.device)
        for s in range(0, V, ctx.chunk):
            gx += (gy[..., s : s + ctx.chunk] @ w[s : s + ctx.chunk]).float()
        gw = gy.reshape(-1, V).t() @ x.reshape(-1, x.shape[-1])
        return gx.to(x.dtype), gw.to(w.dtype), None


def install_fp32_head(model) -> None:
    """GPTModel 的输出头：权重留 FP32（Float16Module 不转），输入转 FP32 再乘；可选 dgrad 分段。"""
    cfg = model.config
    head = model.output_layer
    if not cfg.hy4_lm_head_fp32:
        return
    mark_keep_in_fp32(head.weight)
    with torch.no_grad():
        head.weight.data = head.weight.data.float()
    chunk = cfg.hy4_head_chunk

    def forward(input_, weight=None, runtime_gather_output=None, **_):
        w = head.weight if weight is None else weight
        x = input_.float()
        y = _ChunkedHeadFn.apply(x, w, chunk) if chunk else F.linear(x, w)
        return y, None

    head.forward = forward


def apply_mlp_semantics(model) -> None:
    """整个模型（含 MTP 层）：路由偏置留 FP32；只有路由专家截断，共享专家不截。幂等。"""
    from megatron.core.transformer.moe.moe_layer import MoELayer

    for mod in model.modules():
        if isinstance(mod, MoELayer):
            eb = getattr(mod.router, "expert_bias", None)
            if eb is not None:
                mark_keep_in_fp32(eb)
            if getattr(mod, "shared_experts", None) is not None:
                mod.shared_experts.config = _no_clamp(mod.shared_experts.config)


def build_model(config, use_te: bool, vocab_size: int, max_seq: int, mtp: bool = False):
    """只用 Megatron-Core 建 GPTModel（训练入口与等价性检查用）。PP > 1 时只建本段：首段有嵌入，末段有
    hc_head、输出头与 MTP（MTP 段另有一份嵌入，由 Megatron-Core 跨段同步梯度）。"""
    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt import GPTModel

    _check_parallel(config)
    pp = config.pipeline_model_parallel_size > 1
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_specs(config, use_te),
        vocab_size=vocab_size,
        max_sequence_length=max_seq,
        pre_process=ps.is_pipeline_first_stage() if pp else True,
        post_process=ps.is_pipeline_last_stage() if pp else True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=int(config.rotary_base),
        mtp_block_spec=mtp_block_spec(config, use_te) if mtp else None,
    )
    if model.post_process:
        install_fp32_head(model)
    apply_mlp_semantics(model)
    return model


__all__ = [
    "Hy4Config",
    "Hy4Fields",
    "Hy4FinalNorm",
    "Hy4HyperConnection",
    "Hy4DSAttention",
    "Hy4Indexer",
    "Hy4MLASelfAttention",
    "Hy4TransformerLayer",
    "IdentityOp",
    "build_model",
    "causal_sparse_mask",
    "check_pp_split",
    "clear_topk_store",
    "set_topk_micro",
    "stage_layers",
    "hy4_config_kwargs",
    "install_fp32_head",
    "layer_specs",
    "mtp_block_spec",
    "sparse_sink_attention",
]
