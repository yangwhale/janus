# Muon 分区表：hy4-tiny（scan_layers=True）

由 `muon_partition.py` 写出，覆盖：无。Muon 142 个张量 / 4,851,712 个参数；Adam 112 个张量 / 530,609 个参数。

| 参数 | 形状 | 优化器 | 归约轴 → 输出轴 | NS 矩阵 | 个数（批维） |
|---|---|---|---|---|---|
| `decoder/decoder_norm/scale` | (256,) | Adam | — | — | — |
| `decoder/dense_layers/mhc_attention/post_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/dense_layers/mhc_attention/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_attention/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_attention/pre_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/dense_layers/mhc_attention/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_attention/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_mlp/post_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/dense_layers/mhc_mlp/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_mlp/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_mlp/pre_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/dense_layers/mhc_mlp/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_mlp/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/dense_layers/mlp/wi_0/kernel` | (256, 1, 512) | Muon | (0,) → (-1,) | 256 × 512 | 1（(1,)） |
| `decoder/dense_layers/mlp/wi_1/kernel` | (256, 1, 512) | Muon | (0,) → (-1,) | 256 × 512 | 1（(1,)） |
| `decoder/dense_layers/mlp/wo/kernel` | (512, 1, 256) | Muon | (0,) → (-1,) | 512 × 256 | 1（(1,)） |
| `decoder/dense_layers/post_self_attention_layer_norm/scale` | (256, 1) | Adam | — | — | — |
| `decoder/dense_layers/pre_self_attention_layer_norm/scale` | (256, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/gate_proj/kernel` | (256, 1, 4, 32) | Muon | (0,) → (-1,) | 256 × 32 | 4（(1, 2)） |
| `decoder/dense_layers/self_attention/indexer/k_norm/bias` | (32, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/indexer/k_norm/scale` | (32, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/indexer/weights_proj/kernel` | (256, 1, 16) | Muon | (0,) → (-1,) | 256 × 16 | 1（(1,)） |
| `decoder/dense_layers/self_attention/indexer/wk/kernel` | (256, 1, 32) | Muon | (0,) → (-1,) | 256 × 32 | 1（(1,)） |
| `decoder/dense_layers/self_attention/indexer/wq_b/kernel` | (64, 1, 16, 32) | Muon | (0,) → (-2, -1) | 64 × 512 | 1（(1,)） |
| `decoder/dense_layers/self_attention/kv_norm/scale` | (32, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/out/kernel` | (4, 1, 32, 256) | Muon | (0, -2) → (-1,) | 128 × 256 | 1（(1,)） |
| `decoder/dense_layers/self_attention/q_norm/scale` | (64, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/sinks` | (4, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/wkv_a/kernel` | (256, 1, 48) | Muon | (0,) → (-1,) | 256 × 48 | 1（(1,)） |
| `decoder/dense_layers/self_attention/wkv_b/kernel` | (32, 1, 4, 64) | Muon | (0,) → (-2, -1) | 32 × 256 | 1（(1,)） |
| `decoder/dense_layers/self_attention/wq_a/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/dense_layers/self_attention/wq_b/kernel` | (64, 1, 4, 48) | Muon | (0,) → (-2, -1) | 64 × 192 | 1（(1,)） |
| `decoder/hc_head/hc_base` | (4,) | Adam | — | — | — |
| `decoder/hc_head/hc_fn` | (1024, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（()） |
| `decoder/hc_head/hc_scale` | (1,) | Adam | — | — | — |
| `decoder/logits_dense/kernel` | (256, 1024) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (256, 1, 8) | Muon | (0,) → (-1,) | 256 × 8 | 1（(1,)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (8, 1, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0, 1)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (8, 1, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0, 1)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (8, 1, 64, 256) | Muon | (-2,) → (-1,) | 64 × 256 | 8（(0, 1)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (64, 1, 256) | Muon | (0,) → (-1,) | 64 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_0/mhc_attention/post_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_0/mhc_attention/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_attention/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_attention/pre_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_0/mhc_attention/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_attention/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_mlp/post_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_0/mhc_mlp/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_mlp/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_mlp/pre_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_0/mhc_mlp/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_mlp/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/post_self_attention_layer_norm/scale` | (256, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/pre_self_attention_layer_norm/scale` | (256, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/gate_proj/kernel` | (256, 1, 4, 32) | Muon | (0,) → (-1,) | 256 × 32 | 4（(1, 2)） |
| `decoder/moe_layers/layers_0/self_attention/indexer/k_norm/bias` | (32, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/indexer/k_norm/scale` | (32, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/indexer/weights_proj/kernel` | (256, 1, 16) | Muon | (0,) → (-1,) | 256 × 16 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/indexer/wk/kernel` | (256, 1, 32) | Muon | (0,) → (-1,) | 256 × 32 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/indexer/wq_b/kernel` | (64, 1, 16, 32) | Muon | (0,) → (-2, -1) | 64 × 512 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/kv_norm/scale` | (32, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/out/kernel` | (4, 1, 32, 256) | Muon | (0, -2) → (-1,) | 128 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/q_norm/scale` | (64, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/sinks` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/wkv_a/kernel` | (256, 1, 48) | Muon | (0,) → (-1,) | 256 × 48 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/wkv_b/kernel` | (32, 1, 4, 64) | Muon | (0,) → (-2, -1) | 32 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/wq_a/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/wq_b/kernel` | (64, 1, 4, 48) | Muon | (0,) → (-2, -1) | 64 × 192 | 1（(1,)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (256, 1, 8) | Muon | (0,) → (-1,) | 256 × 8 | 1（(1,)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (8, 1, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0, 1)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (8, 1, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0, 1)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (8, 1, 64, 256) | Muon | (-2,) → (-1,) | 64 × 256 | 8（(0, 1)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (64, 1, 256) | Muon | (0,) → (-1,) | 64 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_1/mhc_attention/post_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_1/mhc_attention/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_attention/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_attention/pre_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_1/mhc_attention/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_attention/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_mlp/post_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_1/mhc_mlp/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_mlp/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_mlp/pre_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_1/mhc_mlp/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_mlp/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/post_self_attention_layer_norm/scale` | (256, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/pre_self_attention_layer_norm/scale` | (256, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/self_attention/gate_proj/kernel` | (256, 1, 4, 32) | Muon | (0,) → (-1,) | 256 × 32 | 4（(1, 2)） |
| `decoder/moe_layers/layers_1/self_attention/kv_norm/scale` | (32, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/self_attention/out/kernel` | (4, 1, 32, 256) | Muon | (0, -2) → (-1,) | 128 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_1/self_attention/q_norm/scale` | (64, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/self_attention/sinks` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/self_attention/wkv_a/kernel` | (256, 1, 48) | Muon | (0,) → (-1,) | 256 × 48 | 1（(1,)） |
| `decoder/moe_layers/layers_1/self_attention/wkv_b/kernel` | (32, 1, 4, 64) | Muon | (0,) → (-2, -1) | 32 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_1/self_attention/wq_a/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_1/self_attention/wq_b/kernel` | (64, 1, 4, 48) | Muon | (0,) → (-2, -1) | 64 × 192 | 1（(1,)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (256, 1, 8) | Muon | (0,) → (-1,) | 256 × 8 | 1（(1,)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (8, 1, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0, 1)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (8, 1, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0, 1)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (8, 1, 64, 256) | Muon | (-2,) → (-1,) | 64 × 256 | 8（(0, 1)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (64, 1, 256) | Muon | (0,) → (-1,) | 64 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_2/mhc_attention/post_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_2/mhc_attention/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_attention/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_attention/pre_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_2/mhc_attention/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_attention/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_mlp/post_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_2/mhc_mlp/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_mlp/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_mlp/pre_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_2/mhc_mlp/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_mlp/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/post_self_attention_layer_norm/scale` | (256, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/pre_self_attention_layer_norm/scale` | (256, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/self_attention/gate_proj/kernel` | (256, 1, 4, 32) | Muon | (0,) → (-1,) | 256 × 32 | 4（(1, 2)） |
| `decoder/moe_layers/layers_2/self_attention/kv_norm/scale` | (32, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/self_attention/out/kernel` | (4, 1, 32, 256) | Muon | (0, -2) → (-1,) | 128 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_2/self_attention/q_norm/scale` | (64, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/self_attention/sinks` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/self_attention/wkv_a/kernel` | (256, 1, 48) | Muon | (0,) → (-1,) | 256 × 48 | 1（(1,)） |
| `decoder/moe_layers/layers_2/self_attention/wkv_b/kernel` | (32, 1, 4, 64) | Muon | (0,) → (-2, -1) | 32 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_2/self_attention/wq_a/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_2/self_attention/wq_b/kernel` | (64, 1, 4, 48) | Muon | (0,) → (-2, -1) | 64 × 192 | 1（(1,)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (256, 1, 8) | Muon | (0,) → (-1,) | 256 × 8 | 1（(1,)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (8, 1, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0, 1)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (8, 1, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0, 1)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (8, 1, 64, 256) | Muon | (-2,) → (-1,) | 64 × 256 | 8（(0, 1)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (64, 1, 256) | Muon | (0,) → (-1,) | 64 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_3/mhc_attention/post_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_3/mhc_attention/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_attention/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_attention/pre_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_3/mhc_attention/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_attention/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_mlp/post_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_3/mhc_mlp/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_mlp/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_mlp/pre_alpha` | (1024, 1, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（(1,)） |
| `decoder/moe_layers/layers_3/mhc_mlp/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_mlp/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/post_self_attention_layer_norm/scale` | (256, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/pre_self_attention_layer_norm/scale` | (256, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/self_attention/gate_proj/kernel` | (256, 1, 4, 32) | Muon | (0,) → (-1,) | 256 × 32 | 4（(1, 2)） |
| `decoder/moe_layers/layers_3/self_attention/kv_norm/scale` | (32, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/self_attention/out/kernel` | (4, 1, 32, 256) | Muon | (0, -2) → (-1,) | 128 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_3/self_attention/q_norm/scale` | (64, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/self_attention/sinks` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/self_attention/wkv_a/kernel` | (256, 1, 48) | Muon | (0,) → (-1,) | 256 × 48 | 1（(1,)） |
| `decoder/moe_layers/layers_3/self_attention/wkv_b/kernel` | (32, 1, 4, 64) | Muon | (0,) → (-2, -1) | 32 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_3/self_attention/wq_a/kernel` | (256, 1, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（(1,)） |
| `decoder/moe_layers/layers_3/self_attention/wq_b/kernel` | (64, 1, 4, 48) | Muon | (0,) → (-2, -1) | 64 × 192 | 1（(1,)） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (256, 8) | Muon | (0,) → (-1,) | 256 × 8 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (8, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0,)） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (8, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0,)） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (8, 64, 256) | Muon | (-2,) → (-1,) | 64 × 256 | 8（(0,)） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (256, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (256, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (64, 256) | Muon | (0,) → (-1,) | 64 × 256 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/post_alpha` | (1024, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/post_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/post_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/pre_alpha` | (1024, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/pre_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/pre_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/post_alpha` | (1024, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/post_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/post_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/pre_alpha` | (1024, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/pre_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/pre_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/post_self_attention_layer_norm/scale` | (256,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/pre_self_attention_layer_norm/scale` | (256,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/gate_proj/kernel` | (256, 4, 32) | Muon | (0,) → (-1,) | 256 × 32 | 4（(1,)） |
| `decoder/moe_layers_remainder/layers_0/self_attention/indexer/k_norm/bias` | (32,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/indexer/k_norm/scale` | (32,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/indexer/weights_proj/kernel` | (256, 16) | Muon | (0,) → (-1,) | 256 × 16 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/indexer/wk/kernel` | (256, 32) | Muon | (0,) → (-1,) | 256 × 32 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/indexer/wq_b/kernel` | (64, 16, 32) | Muon | (0,) → (-2, -1) | 64 × 512 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/kv_norm/scale` | (32,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/out/kernel` | (4, 32, 256) | Muon | (0, -2) → (-1,) | 128 × 256 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/q_norm/scale` | (64,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/sinks` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/wkv_a/kernel` | (256, 48) | Muon | (0,) → (-1,) | 256 × 48 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/wkv_b/kernel` | (32, 4, 64) | Muon | (0,) → (-2, -1) | 32 × 256 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/wq_a/kernel` | (256, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/wq_b/kernel` | (64, 4, 48) | Muon | (0,) → (-2, -1) | 64 × 192 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (256, 8) | Muon | (0,) → (-1,) | 256 × 8 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (8, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0,)） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (8, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0,)） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (8, 64, 256) | Muon | (-2,) → (-1,) | 64 × 256 | 8（(0,)） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (256, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (256, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (64, 256) | Muon | (0,) → (-1,) | 64 × 256 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/post_alpha` | (1024, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/post_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/post_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/pre_alpha` | (1024, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/pre_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/pre_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/post_alpha` | (1024, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/post_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/post_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/pre_alpha` | (1024, 4) | Muon | (0,) → (-1,) | 1024 × 4 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/pre_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/pre_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/post_self_attention_layer_norm/scale` | (256,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/pre_self_attention_layer_norm/scale` | (256,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/self_attention/gate_proj/kernel` | (256, 4, 32) | Muon | (0,) → (-1,) | 256 × 32 | 4（(1,)） |
| `decoder/moe_layers_remainder/layers_1/self_attention/kv_norm/scale` | (32,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/self_attention/out/kernel` | (4, 32, 256) | Muon | (0, -2) → (-1,) | 128 × 256 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/self_attention/q_norm/scale` | (64,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/self_attention/sinks` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/self_attention/wkv_a/kernel` | (256, 48) | Muon | (0,) → (-1,) | 256 × 48 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/self_attention/wkv_b/kernel` | (32, 4, 64) | Muon | (0,) → (-2, -1) | 32 × 256 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/self_attention/wq_a/kernel` | (256, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/self_attention/wq_b/kernel` | (64, 4, 48) | Muon | (0,) → (-2, -1) | 64 × 192 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_embedding_norm/scale` | (256,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_final_norm/scale` | (256,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_hidden_state_norm/scale` | (256,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_projection/kernel` | (512, 256) | Muon | (0,) → (-1,) | 512 × 256 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (256, 8) | Muon | (0,) → (-1,) | 256 × 8 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (8, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0,)） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (8, 256, 64) | Muon | (-2,) → (-1,) | 256 × 64 | 8（(0,)） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (8, 64, 256) | Muon | (-2,) → (-1,) | 64 × 256 | 8（(0,)） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (256, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (256, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (64, 256) | Muon | (0,) → (-1,) | 64 × 256 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/post_self_attention_layer_norm/scale` | (256,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/pre_self_attention_layer_norm/scale` | (256,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/gate_proj/kernel` | (256, 4, 32) | Muon | (0,) → (-1,) | 256 × 32 | 4（(1,)） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/indexer/k_norm/bias` | (32,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/indexer/k_norm/scale` | (32,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/indexer/weights_proj/kernel` | (256, 16) | Muon | (0,) → (-1,) | 256 × 16 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/indexer/wk/kernel` | (256, 32) | Muon | (0,) → (-1,) | 256 × 32 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/indexer/wq_b/kernel` | (64, 16, 32) | Muon | (0,) → (-2, -1) | 64 × 512 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/kv_norm/scale` | (32,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/out/kernel` | (4, 32, 256) | Muon | (0, -2) → (-1,) | 128 × 256 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/q_norm/scale` | (64,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/sinks` | (4,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/wkv_a/kernel` | (256, 48) | Muon | (0,) → (-1,) | 256 × 48 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/wkv_b/kernel` | (32, 4, 64) | Muon | (0,) → (-2, -1) | 32 × 256 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/wq_a/kernel` | (256, 64) | Muon | (0,) → (-1,) | 256 × 64 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/wq_b/kernel` | (64, 4, 48) | Muon | (0,) → (-2, -1) | 64 × 192 | 1（()） |
| `token_embedder/embedding` | (1024, 256) | Adam | — | — | — |
