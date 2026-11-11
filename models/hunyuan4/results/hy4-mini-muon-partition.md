# Muon 分区表：hy4-mini（scan_layers=True）

由 `muon_partition.py` 写出，覆盖：muon_exclude_ihc=true。Muon 113 个张量 / 179,814,400 个参数；Adam 141 个张量 / 247,968,977 个参数。

| 参数 | 形状 | 优化器 | 归约轴 → 输出轴 | NS 矩阵 | 个数（批维） |
|---|---|---|---|---|---|
| `decoder/decoder_norm/scale` | (1024,) | Adam | — | — | — |
| `decoder/dense_layers/mhc_attention/post_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/dense_layers/mhc_attention/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_attention/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_attention/pre_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/dense_layers/mhc_attention/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_attention/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_mlp/post_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/dense_layers/mhc_mlp/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_mlp/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_mlp/pre_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/dense_layers/mhc_mlp/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/dense_layers/mhc_mlp/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/dense_layers/mlp/wi_0/kernel` | (1024, 1, 3072) | Muon | (0,) → (-1,) | 1024 × 3072 | 1（(1,)） |
| `decoder/dense_layers/mlp/wi_1/kernel` | (1024, 1, 3072) | Muon | (0,) → (-1,) | 1024 × 3072 | 1（(1,)） |
| `decoder/dense_layers/mlp/wo/kernel` | (3072, 1, 1024) | Muon | (0,) → (-1,) | 3072 × 1024 | 1（(1,)） |
| `decoder/dense_layers/post_self_attention_layer_norm/scale` | (1024, 1) | Adam | — | — | — |
| `decoder/dense_layers/pre_self_attention_layer_norm/scale` | (1024, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/gate_proj/kernel` | (1024, 1, 8, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 8（(1, 2)） |
| `decoder/dense_layers/self_attention/indexer/k_norm/bias` | (128, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/indexer/k_norm/scale` | (128, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/indexer/weights_proj/kernel` | (1024, 1, 32) | Muon | (0,) → (-1,) | 1024 × 32 | 1（(1,)） |
| `decoder/dense_layers/self_attention/indexer/wk/kernel` | (1024, 1, 128) | Muon | (0,) → (-1,) | 1024 × 128 | 1（(1,)） |
| `decoder/dense_layers/self_attention/indexer/wq_b/kernel` | (512, 1, 32, 128) | Muon | (0,) → (-2, -1) | 512 × 4096 | 1（(1,)） |
| `decoder/dense_layers/self_attention/kv_norm/scale` | (512, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/out/kernel` | (8, 1, 256, 1024) | Muon | (0, -2) → (-1,) | 2048 × 1024 | 1（(1,)） |
| `decoder/dense_layers/self_attention/q_norm/scale` | (512, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/sinks` | (8, 1) | Adam | — | — | — |
| `decoder/dense_layers/self_attention/wkv_a/kernel` | (1024, 1, 576) | Muon | (0,) → (-1,) | 1024 × 576 | 1（(1,)） |
| `decoder/dense_layers/self_attention/wkv_b/kernel` | (512, 1, 8, 448) | Muon | (0,) → (-2, -1) | 512 × 3584 | 1（(1,)） |
| `decoder/dense_layers/self_attention/wq_a/kernel` | (1024, 1, 512) | Muon | (0,) → (-1,) | 1024 × 512 | 1（(1,)） |
| `decoder/dense_layers/self_attention/wq_b/kernel` | (512, 1, 8, 256) | Muon | (0,) → (-2, -1) | 512 × 2048 | 1（(1,)） |
| `decoder/hc_head/hc_base` | (4,) | Adam | — | — | — |
| `decoder/hc_head/hc_fn` | (4096, 4) | Adam | — | — | — |
| `decoder/hc_head/hc_scale` | (1,) | Adam | — | — | — |
| `decoder/logits_dense/kernel` | (1024, 120832) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (1024, 1, 16) | Muon | (0,) → (-1,) | 1024 × 16 | 1（(1,)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (16, 1, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0, 1)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (16, 1, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0, 1)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (16, 1, 256, 1024) | Muon | (-2,) → (-1,) | 256 × 1024 | 16（(0, 1)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (1024, 1, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (1024, 1, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_0/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (256, 1, 1024) | Muon | (0,) → (-1,) | 256 × 1024 | 1（(1,)） |
| `decoder/moe_layers/layers_0/mhc_attention/post_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_attention/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_attention/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_attention/pre_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_attention/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_attention/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_mlp/post_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_mlp/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_mlp/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_mlp/pre_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_mlp/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/mhc_mlp/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/post_self_attention_layer_norm/scale` | (1024, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/pre_self_attention_layer_norm/scale` | (1024, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/gate_proj/kernel` | (1024, 1, 8, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 8（(1, 2)） |
| `decoder/moe_layers/layers_0/self_attention/indexer/k_norm/bias` | (128, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/indexer/k_norm/scale` | (128, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/indexer/weights_proj/kernel` | (1024, 1, 32) | Muon | (0,) → (-1,) | 1024 × 32 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/indexer/wk/kernel` | (1024, 1, 128) | Muon | (0,) → (-1,) | 1024 × 128 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/indexer/wq_b/kernel` | (512, 1, 32, 128) | Muon | (0,) → (-2, -1) | 512 × 4096 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/kv_norm/scale` | (512, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/out/kernel` | (8, 1, 256, 1024) | Muon | (0, -2) → (-1,) | 2048 × 1024 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/q_norm/scale` | (512, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/sinks` | (8, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_0/self_attention/wkv_a/kernel` | (1024, 1, 576) | Muon | (0,) → (-1,) | 1024 × 576 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/wkv_b/kernel` | (512, 1, 8, 448) | Muon | (0,) → (-2, -1) | 512 × 3584 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/wq_a/kernel` | (1024, 1, 512) | Muon | (0,) → (-1,) | 1024 × 512 | 1（(1,)） |
| `decoder/moe_layers/layers_0/self_attention/wq_b/kernel` | (512, 1, 8, 256) | Muon | (0,) → (-2, -1) | 512 × 2048 | 1（(1,)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (1024, 1, 16) | Muon | (0,) → (-1,) | 1024 × 16 | 1（(1,)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (16, 1, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0, 1)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (16, 1, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0, 1)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (16, 1, 256, 1024) | Muon | (-2,) → (-1,) | 256 × 1024 | 16（(0, 1)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (1024, 1, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (1024, 1, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_1/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (256, 1, 1024) | Muon | (0,) → (-1,) | 256 × 1024 | 1（(1,)） |
| `decoder/moe_layers/layers_1/mhc_attention/post_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_attention/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_attention/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_attention/pre_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_attention/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_attention/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_mlp/post_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_mlp/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_mlp/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_mlp/pre_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_mlp/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/mhc_mlp/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/post_self_attention_layer_norm/scale` | (1024, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/pre_self_attention_layer_norm/scale` | (1024, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/self_attention/gate_proj/kernel` | (1024, 1, 8, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 8（(1, 2)） |
| `decoder/moe_layers/layers_1/self_attention/kv_norm/scale` | (512, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/self_attention/out/kernel` | (8, 1, 256, 1024) | Muon | (0, -2) → (-1,) | 2048 × 1024 | 1（(1,)） |
| `decoder/moe_layers/layers_1/self_attention/q_norm/scale` | (512, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/self_attention/sinks` | (8, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_1/self_attention/wkv_a/kernel` | (1024, 1, 576) | Muon | (0,) → (-1,) | 1024 × 576 | 1（(1,)） |
| `decoder/moe_layers/layers_1/self_attention/wkv_b/kernel` | (512, 1, 8, 448) | Muon | (0,) → (-2, -1) | 512 × 3584 | 1（(1,)） |
| `decoder/moe_layers/layers_1/self_attention/wq_a/kernel` | (1024, 1, 512) | Muon | (0,) → (-1,) | 1024 × 512 | 1（(1,)） |
| `decoder/moe_layers/layers_1/self_attention/wq_b/kernel` | (512, 1, 8, 256) | Muon | (0,) → (-2, -1) | 512 × 2048 | 1（(1,)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (1024, 1, 16) | Muon | (0,) → (-1,) | 1024 × 16 | 1（(1,)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (16, 1, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0, 1)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (16, 1, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0, 1)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (16, 1, 256, 1024) | Muon | (-2,) → (-1,) | 256 × 1024 | 16（(0, 1)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (1024, 1, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (1024, 1, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_2/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (256, 1, 1024) | Muon | (0,) → (-1,) | 256 × 1024 | 1（(1,)） |
| `decoder/moe_layers/layers_2/mhc_attention/post_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_attention/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_attention/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_attention/pre_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_attention/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_attention/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_mlp/post_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_mlp/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_mlp/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_mlp/pre_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_mlp/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/mhc_mlp/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/post_self_attention_layer_norm/scale` | (1024, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/pre_self_attention_layer_norm/scale` | (1024, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/self_attention/gate_proj/kernel` | (1024, 1, 8, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 8（(1, 2)） |
| `decoder/moe_layers/layers_2/self_attention/kv_norm/scale` | (512, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/self_attention/out/kernel` | (8, 1, 256, 1024) | Muon | (0, -2) → (-1,) | 2048 × 1024 | 1（(1,)） |
| `decoder/moe_layers/layers_2/self_attention/q_norm/scale` | (512, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/self_attention/sinks` | (8, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_2/self_attention/wkv_a/kernel` | (1024, 1, 576) | Muon | (0,) → (-1,) | 1024 × 576 | 1（(1,)） |
| `decoder/moe_layers/layers_2/self_attention/wkv_b/kernel` | (512, 1, 8, 448) | Muon | (0,) → (-2, -1) | 512 × 3584 | 1（(1,)） |
| `decoder/moe_layers/layers_2/self_attention/wq_a/kernel` | (1024, 1, 512) | Muon | (0,) → (-1,) | 1024 × 512 | 1（(1,)） |
| `decoder/moe_layers/layers_2/self_attention/wq_b/kernel` | (512, 1, 8, 256) | Muon | (0,) → (-2, -1) | 512 × 2048 | 1（(1,)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (1024, 1, 16) | Muon | (0,) → (-1,) | 1024 × 16 | 1（(1,)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (16, 1, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0, 1)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (16, 1, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0, 1)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (16, 1, 256, 1024) | Muon | (-2,) → (-1,) | 256 × 1024 | 16（(0, 1)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (1024, 1, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (1024, 1, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（(1,)） |
| `decoder/moe_layers/layers_3/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (256, 1, 1024) | Muon | (0,) → (-1,) | 256 × 1024 | 1（(1,)） |
| `decoder/moe_layers/layers_3/mhc_attention/post_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_attention/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_attention/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_attention/pre_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_attention/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_attention/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_mlp/post_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_mlp/post_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_mlp/post_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_mlp/pre_alpha` | (4096, 1, 4) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_mlp/pre_alpha_scale` | (1, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/mhc_mlp/pre_beta` | (4, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/post_self_attention_layer_norm/scale` | (1024, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/pre_self_attention_layer_norm/scale` | (1024, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/self_attention/gate_proj/kernel` | (1024, 1, 8, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 8（(1, 2)） |
| `decoder/moe_layers/layers_3/self_attention/kv_norm/scale` | (512, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/self_attention/out/kernel` | (8, 1, 256, 1024) | Muon | (0, -2) → (-1,) | 2048 × 1024 | 1（(1,)） |
| `decoder/moe_layers/layers_3/self_attention/q_norm/scale` | (512, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/self_attention/sinks` | (8, 1) | Adam | — | — | — |
| `decoder/moe_layers/layers_3/self_attention/wkv_a/kernel` | (1024, 1, 576) | Muon | (0,) → (-1,) | 1024 × 576 | 1（(1,)） |
| `decoder/moe_layers/layers_3/self_attention/wkv_b/kernel` | (512, 1, 8, 448) | Muon | (0,) → (-2, -1) | 512 × 3584 | 1（(1,)） |
| `decoder/moe_layers/layers_3/self_attention/wq_a/kernel` | (1024, 1, 512) | Muon | (0,) → (-1,) | 1024 × 512 | 1（(1,)） |
| `decoder/moe_layers/layers_3/self_attention/wq_b/kernel` | (512, 1, 8, 256) | Muon | (0,) → (-2, -1) | 512 × 2048 | 1（(1,)） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (1024, 16) | Muon | (0,) → (-1,) | 1024 × 16 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (16, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0,)） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (16, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0,)） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (16, 256, 1024) | Muon | (-2,) → (-1,) | 256 × 1024 | 16（(0,)） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (1024, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (1024, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (256, 1024) | Muon | (0,) → (-1,) | 256 × 1024 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/post_alpha` | (4096, 4) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/post_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/post_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/pre_alpha` | (4096, 4) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/pre_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_attention/pre_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/post_alpha` | (4096, 4) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/post_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/post_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/pre_alpha` | (4096, 4) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/pre_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/mhc_mlp/pre_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/post_self_attention_layer_norm/scale` | (1024,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/pre_self_attention_layer_norm/scale` | (1024,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/gate_proj/kernel` | (1024, 8, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 8（(1,)） |
| `decoder/moe_layers_remainder/layers_0/self_attention/indexer/k_norm/bias` | (128,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/indexer/k_norm/scale` | (128,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/indexer/weights_proj/kernel` | (1024, 32) | Muon | (0,) → (-1,) | 1024 × 32 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/indexer/wk/kernel` | (1024, 128) | Muon | (0,) → (-1,) | 1024 × 128 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/indexer/wq_b/kernel` | (512, 32, 128) | Muon | (0,) → (-2, -1) | 512 × 4096 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/kv_norm/scale` | (512,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/out/kernel` | (8, 256, 1024) | Muon | (0, -2) → (-1,) | 2048 × 1024 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/q_norm/scale` | (512,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/sinks` | (8,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_0/self_attention/wkv_a/kernel` | (1024, 576) | Muon | (0,) → (-1,) | 1024 × 576 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/wkv_b/kernel` | (512, 8, 448) | Muon | (0,) → (-2, -1) | 512 × 3584 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/wq_a/kernel` | (1024, 512) | Muon | (0,) → (-1,) | 1024 × 512 | 1（()） |
| `decoder/moe_layers_remainder/layers_0/self_attention/wq_b/kernel` | (512, 8, 256) | Muon | (0,) → (-2, -1) | 512 × 2048 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (1024, 16) | Muon | (0,) → (-1,) | 1024 × 16 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (16, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0,)） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (16, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0,)） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (16, 256, 1024) | Muon | (-2,) → (-1,) | 256 × 1024 | 16（(0,)） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (1024, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (1024, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (256, 1024) | Muon | (0,) → (-1,) | 256 × 1024 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/post_alpha` | (4096, 4) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/post_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/post_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/pre_alpha` | (4096, 4) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/pre_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_attention/pre_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/post_alpha` | (4096, 4) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/post_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/post_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/pre_alpha` | (4096, 4) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/pre_alpha_scale` | (1,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/mhc_mlp/pre_beta` | (4,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/post_self_attention_layer_norm/scale` | (1024,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/pre_self_attention_layer_norm/scale` | (1024,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/self_attention/gate_proj/kernel` | (1024, 8, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 8（(1,)） |
| `decoder/moe_layers_remainder/layers_1/self_attention/kv_norm/scale` | (512,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/self_attention/out/kernel` | (8, 256, 1024) | Muon | (0, -2) → (-1,) | 2048 × 1024 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/self_attention/q_norm/scale` | (512,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/self_attention/sinks` | (8,) | Adam | — | — | — |
| `decoder/moe_layers_remainder/layers_1/self_attention/wkv_a/kernel` | (1024, 576) | Muon | (0,) → (-1,) | 1024 × 576 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/self_attention/wkv_b/kernel` | (512, 8, 448) | Muon | (0,) → (-2, -1) | 512 × 3584 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/self_attention/wq_a/kernel` | (1024, 512) | Muon | (0,) → (-1,) | 1024 × 512 | 1（()） |
| `decoder/moe_layers_remainder/layers_1/self_attention/wq_b/kernel` | (512, 8, 256) | Muon | (0,) → (-2, -1) | 512 × 2048 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_embedding_norm/scale` | (1024,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_final_norm/scale` | (1024,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_hidden_state_norm/scale` | (1024,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_projection/kernel` | (2048, 1024) | Muon | (0,) → (-1,) | 2048 × 1024 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/MoeBlock_0/gate/kernel` | (1024, 16) | Muon | (0,) → (-1,) | 1024 × 16 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/MoeBlock_0/wi_0` | (16, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0,)） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/MoeBlock_0/wi_1` | (16, 1024, 256) | Muon | (-2,) → (-1,) | 1024 × 256 | 16（(0,)） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/MoeBlock_0/wo` | (16, 256, 1024) | Muon | (-2,) → (-1,) | 256 × 1024 | 16（(0,)） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/shared_experts/wi_0/kernel` | (1024, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/shared_experts/wi_1/kernel` | (1024, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/DeepSeekMoeBlock_0/shared_experts/wo/kernel` | (256, 1024) | Muon | (0,) → (-1,) | 256 × 1024 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/post_self_attention_layer_norm/scale` | (1024,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/pre_self_attention_layer_norm/scale` | (1024,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/gate_proj/kernel` | (1024, 8, 256) | Muon | (0,) → (-1,) | 1024 × 256 | 8（(1,)） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/indexer/k_norm/bias` | (128,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/indexer/k_norm/scale` | (128,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/indexer/weights_proj/kernel` | (1024, 32) | Muon | (0,) → (-1,) | 1024 × 32 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/indexer/wk/kernel` | (1024, 128) | Muon | (0,) → (-1,) | 1024 × 128 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/indexer/wq_b/kernel` | (512, 32, 128) | Muon | (0,) → (-2, -1) | 512 × 4096 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/kv_norm/scale` | (512,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/out/kernel` | (8, 256, 1024) | Muon | (0, -2) → (-1,) | 2048 × 1024 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/q_norm/scale` | (512,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/sinks` | (8,) | Adam | — | — | — |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/wkv_a/kernel` | (1024, 576) | Muon | (0,) → (-1,) | 1024 × 576 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/wkv_b/kernel` | (512, 8, 448) | Muon | (0,) → (-2, -1) | 512 × 3584 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/wq_a/kernel` | (1024, 512) | Muon | (0,) → (-1,) | 1024 × 512 | 1（()） |
| `mtp_block/mtp_layer_1/mtp_1_transformer_layer/self_attention/wq_b/kernel` | (512, 8, 256) | Muon | (0,) → (-2, -1) | 512 × 2048 | 1（()） |
| `token_embedder/embedding` | (120832, 1024) | Adam | — | — | — |
