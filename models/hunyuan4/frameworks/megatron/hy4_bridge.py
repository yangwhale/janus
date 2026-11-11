"""混元 4 的 Megatron-Bridge 桥（v0.7.0）：HF `HYV4ForCausalLM` ⇄ Megatron-Core GPTModel（hy4_mcore.py 的 spec）。

`import hy4_bridge` 之后，`AutoBridge.from_hf_pretrained(<混元 4 HF 目录>)` 就能认出 `model_type: hy_v4`：
`to_megatron_provider()` 给出 `Hy4ModelProvider`，`load_hf_weights(model)` 按 hy4_weights.py 的同一张对照表载入。

- 只支持 TP = 1（hy4_mcore.py 的限制）；非专家参数一律按 ReplicatedMapping，路由专家用 Bridge 自带的
  FusedGatedExpertMapping / FusedExpertMapping（HF 的专家是 [E, 2I, H] / [E, H, I] 三维张量）。
- Megatron-Bridge 自己的训练循环没有接上 hy4_mcore 的 FP32 输出头与 iHC 的语义检查；训练用 train_hy4mg.py。
  这个桥用于把 checkpoint 转成 Megatron 格式、或在 Bridge 的流程里建模型。
- 没有在单测里跑（要 Transformer Engine）；在 GPU Pod 里由 bridge_check.py 核：Bridge 载入的每个参数与
  hy4_weights.load_hf_into_mcore 载入的逐位相同。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hy4_mcore as H  # noqa: E402
import torch  # noqa: E402
from hy4_weights import ATTN, GLOBAL, INDEXER, LAYER, MOE_FIXED, MTP_EXTRA  # noqa: E402
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry  # noqa: E402
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge  # noqa: E402
from megatron.bridge.models.conversion.param_mapping import (  # noqa: E402
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.mla_provider import MLAModelProvider  # noqa: E402
from megatron.core.models.gpt.gpt_model import GPTModel  # noqa: E402


@dataclass
class Hy4ModelProvider(H.Hy4Fields, MLAModelProvider):
    """混元 4 的 provider：MLA 字段 + Hy4Fields；provide() 用 hy4_mcore 的 spec 建 GPTModel 并装 FP32 输出头。"""

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        assert self.pipeline_model_parallel_size == 1 and self.virtual_pipeline_model_parallel_size is None
        use_te = self.transformer_impl == "transformer_engine"
        model = GPTModel(
            config=self,
            transformer_layer_spec=H.layer_specs(self, use_te),
            vocab_size=self.vocab_size,
            max_sequence_length=self.seq_length,
            pre_process=True,
            post_process=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            rotary_base=int(self.rotary_base),
            pg_collection=self._pg_collection,
            mtp_block_spec=H.mtp_block_spec(self, use_te) if self.mtp_num_layers else None,
        )
        H.install_fp32_head(model)
        H.apply_mlp_semantics(model)
        return model


@MegatronModelBridge.register_bridge(
    source="HYV4ForCausalLM", target=GPTModel, provider=Hy4ModelProvider, model_type="hy_v4"
)
class HYV4Bridge(MegatronModelBridge):
    """混元 4（Hy4-preview）：门控 MLA + DSA（逐层 full / shared）+ 可学习 sink + iHC + MoE + FP32 输出头。"""

    MODEL_CONFIG_CLASS = None  # 走 provider 路径（同 GLM-5 桥）

    def provider_bridge(self, hf_pretrained) -> Hy4ModelProvider:
        hf = hf_pretrained.config.to_dict()
        kw = H.hy4_config_kwargs(hf)
        try:
            import transformer_engine  # noqa: F401

            impl = dict(transformer_impl="transformer_engine", moe_grouped_gemm=True)
        except ImportError:
            impl = dict(transformer_impl="local", moe_grouped_gemm=False)
        return Hy4ModelProvider(
            **kw,
            **impl,
            vocab_size=hf["vocab_size"],
            seq_length=4096,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            bf16=True,
            params_dtype=torch.bfloat16,
            make_vocab_size_divisible_by=1,
            mtp_num_layers=None,  # 默认不建 MTP（同 GLM-5 桥）；要训练 MTP 时设 1
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        cfg = self.hf_config
        n = cfg.num_hidden_layers
        types, mlp_types = list(cfg.indexer_types)[:n], list(cfg.mlp_layer_types)[:n]
        maps = [ReplicatedMapping(megatron_param=m, hf_param=h) for m, h in GLOBAL]
        for i in range(n):
            pm, ph = f"decoder.layers.{i}.", f"model.layers.{i}."
            pairs = LAYER + [("self_attention." + m, h) for m, h in ATTN]
            if types[i] == "full":
                pairs += [("self_attention." + m, h) for m, h in INDEXER]
            if mlp_types[i] == "sparse":
                pairs += [("mlp." + m, h) for m, h in MOE_FIXED]
                maps.append(
                    GatedMLPMapping(
                        megatron_param=pm + "mlp.shared_experts.linear_fc1.weight",
                        gate=ph + "mlp.shared_experts.gate_proj.weight",
                        up=ph + "mlp.shared_experts.up_proj.weight",
                    )
                )
                maps += _expert_maps(pm, ph)
            else:
                pairs += [("mlp.linear_fc2.weight", "mlp.down_proj.weight")]
                maps.append(
                    GatedMLPMapping(
                        megatron_param=pm + "mlp.linear_fc1.weight",
                        gate=ph + "mlp.gate_proj.weight",
                        up=ph + "mlp.up_proj.weight",
                    )
                )
            maps += [ReplicatedMapping(megatron_param=pm + m, hf_param=ph + h) for m, h in pairs]
        if getattr(cfg, "num_nextn_predict_layers", 0):
            pm, ph = "mtp.layers.0.", "model.mtp_layers.0."
            inner = pm + "mtp_model_layer."
            maps += [ReplicatedMapping(megatron_param=pm + m, hf_param=ph + h) for m, h in MTP_EXTRA]
            pairs = [
                ("input_layernorm.weight", "input_layernorm.weight"),
                ("pre_mlp_layernorm.weight", "post_attention_layernorm.weight"),
            ]
            pairs += [("self_attention." + m, h) for m, h in ATTN + INDEXER]
            pairs += [("mlp." + m, h) for m, h in MOE_FIXED]
            maps += [ReplicatedMapping(megatron_param=inner + m, hf_param=ph + h) for m, h in pairs]
            maps.append(
                GatedMLPMapping(
                    megatron_param=inner + "mlp.shared_experts.linear_fc1.weight",
                    gate=ph + "mlp.shared_experts.gate_proj.weight",
                    up=ph + "mlp.shared_experts.up_proj.weight",
                )
            )
            maps += _expert_maps(inner, ph)
        return MegatronMappingRegistry(*maps)


def _expert_maps(pm: str, ph: str):
    return [
        FusedGatedExpertMapping(
            megatron_param=pm + "mlp.experts.linear_fc1.weight*", hf_param=ph + "mlp.experts.gate_up_proj"
        ),
        FusedExpertMapping(megatron_param=pm + "mlp.experts.linear_fc2.weight*", hf_param=ph + "mlp.experts.down_proj"),
        FusedGatedExpertMapping(
            megatron_param=pm + "mlp.experts.local_experts.*.linear_fc1.weight",
            hf_param=ph + "mlp.experts.gate_up_proj",
        ),
        FusedExpertMapping(
            megatron_param=pm + "mlp.experts.local_experts.*.linear_fc2.weight", hf_param=ph + "mlp.experts.down_proj"
        ),
    ]


__all__ = ["HYV4Bridge", "Hy4ModelProvider"]
