[janus](../README.md) › 关于公开版

# 关于公开版

这个仓库是 janus 的公开版，由开发仓库定期同步生成。公开版不带开发历史：每个提交对应开发仓库的一次同步，提交说明里写着对应的开发仓库提交号。

## 我们修改过的 MaxText

文中的 `maxtext` 指我们修改过的 MaxText，公开在 [yangwhale/maxtext](https://github.com/yangwhale/maxtext)：

| 文中的叫法 | 公开分支 | 公开提交 | 基于上游 AI-Hypercomputer/maxtext |
|---|---|---|---|
| `hunyuan` 分支（混元 3） | [`hunyuan`](https://github.com/yangwhale/maxtext/tree/hunyuan) | `fb4c449621c2f3f3a0941f176903911da4cfb69c` | `ae6e4eb2fdb8bd115c14ad9d3331d7e6dd407e43` |
| 混元 4 集成分支 | [`hunyuan4`](https://github.com/yangwhale/maxtext/tree/hunyuan4) | `319afd46a676654e33c09a553b3d4266b82716cc` | `493d5928156869b70f7198ba7bbd1a7f779233d9` |

- 每个公开分支只有一个提交，放在它所基于的上游提交之上，`git diff HEAD~1` 就是全部改动。分支根目录的 `HUNYUAN.md` 列了模型、配置与开关。
- 混元 4 的 `stage.sh` 用 `MAXTEXT_BRANCH=hunyuan4`；混元 3 的 `stage.sh` 默认的 `origin/hunyuan` 在公开仓库里同名。
- 公开分支是开发分支某一时刻的最新代码，不带开发历史。文档、配置和结果文件（`meta.json`、`results.json`、`runs.yaml` 的 `code`、`commits`）里记的 maxtext 提交号是开发分支上的提交，公开仓库里查不到，原样保留作出处。下面两张表把它们对到各自引入的开关或改动，这些改动都已含在对应的公开分支里。
- 公开分支上开关的默认值与各提交相同（默认关，关时行为不变）。用旧提交跑出的结果，在公开分支上照结果文件里的配置打开对应开关，走的是同一套代码路径；逐位是否相同没有逐一验证过。

### 混元 3：开发分支 `hunyuan` 上的提交 → 公开分支 `hunyuan`

| 开发分支提交 | 这个提交带来的开关或改动 |
|---|---|
| `64cd9a64` | 只改协作文件（公开版已删），代码与它的父提交相同 |
| `4d4ffac2` | `mlp_activation_round_per_op`；路由专家稀疏路径读 `activations_in_float32`。此时分支上已有 `routed_gate_bf16_rounding`（路由核先舍入成 BF16，梯度不舍）与 `rmsnorm_single_rounding` |
| `237a3270` | 只改 PR 模板与检查脚本（公开版已删），代码与 `4d4ffac2` 相同 |
| `0622774a` | `dense_fp32_accumulation`（DenseGeneral FP32 累加、只舍一次）。此时分支上已有 `remat_policy: save_everything` 与诊断开关 `lm_head_dh_gauge_scale` / `lm_head_dh_wbar_remove` |

### 混元 4：混元 4 集成分支上的提交 → 公开分支 `hunyuan4`

按提交时间排序。

| 开发分支提交 | 这个提交带来的开关或改动 |
|---|---|
| `2f745f37` | `rmsnorm_single_rounding` |
| `3a892622` | `routed_gate_bf16_rounding` |
| `746d105e` | `routed_gate_bf16_rounding` 改用 `reduce_precision`，梯度不再被舍入 |
| `e9447340` | `mlp_activation_round_per_op`；路由专家稀疏路径读 `activations_in_float32` |
| `727c98b5` | `remat_policy: save_everything` |
| `23f71c66` | `dense_fp32_accumulation` 与三个梯度 FP32 开关（`rmsnorm_scale_grad_fp32`、`logits_dense_wgrad_fp32`、`embedding_grad_fp32`），各精度开关接到混元 4 的 MLA / MoE / MTP 路径 |
| `352014df` | `logits_matmul_precision`（输出头单独的矩阵乘精度） |
| `0b7dead5` | 同上（合入提交） |
| `bf39b0c1` | `save_everything` 在复数 RoPE 上报错的修复 |
| `f10b4d9a` | Muon / QK-Clip 对齐 Megatron-Core：QK-Clip 按层缩放、输出门整块正交化，`muon_exclude_ihc`、`muon_fuse_glu` |
| `0ddf8a5b` | 新增 `hy4-mini`（约 0.43B）；`hy4-tiny` 改用与 `hy4-preview` 相同的检查点读法 |
| `6ba40afa` | `muon_attention_gate_whole_matrix` |
| `27d82457` | `muon_mcore_nesterov`、`muon_adam_plain`、`muon_mcore_ns_normalize`、`qk_clip_mcore_denominator`、`indexer_kd_mcore_log` 等 Megatron-Core 兼容开关；梯度累积时 QK-Clip 的最大 logit 取最大 |
| `8611ea6b` | `indexer_kd_mcore_weight`、`mtp_loss_mcore_per_rank`（eod 掩码下的损失权重） |
| `ea539057` | `indexer_fp32_scores` |
| `657cae95` | 诊断开关 `debug_sow_indexer` |
| `1c34c5be` | `debug_sow_indexer` 另记第 0 行在图里算出的 KD |
| `d723fa2d` | `debug_sow_indexer` 另记 KD 的教师目标、索引器对数概率与逐行 KL |
| `3e6d4c9a` | `qk_clip_max_logit_exclude_sink` |
| `5a03d6a1` | `indexer_weights_proj_bf16_weight` |
| `5eaa12f3` | `qk_clip_log_per_head` |
| `dcd4b940` | `muon_mcore_ns_bf16` |
| `ce24536d` | 只调整代码行宽 |
| `e8dee18f` | `transcendental_accuracy_highest` |
| `94007054` | `transcendental_accuracy_highest` 覆盖到主 loss 与 MTP loss 的交叉熵 |
| `64652880` | `muon_adam_accurate` |
| `ce66bf9f` | QK-Clip 对最大 logit 非正的头不缩放（原来会出 NaN） |
| `c902c147` | `ihc_force_bf16_rounding` |
| `b8d4ba8b` | `transcendental_accuracy_highest` 的 sigmoid / silu 写法修正 |
| `91ece12a` | `muon_adam_accurate` 的 1 - b^t 与 sqrt 写法修正 |
| `7c95fabe` | `muon_adam_accurate` 的除法改成带一次修正的商（约 1 ULP） |
| `469dbb85` | 新增 `hy4-mid`（约 5.19B） |
| `95389004` | 按位置分开的 `accurate_sigmoid_*` / `accurate_silu_*` |
| `1d9cc8c7` | 精确 sigmoid / silu 的反向改用 PyTorch 的解析式 |
| `2b005b3d` | `moe_mcore_prob_before_fc2` |
| `4a01e4a5` | 只改测试 |
| `3741e818` | `mlp_fused_fc1_dgrad` |
| `af4ca621` | 诊断开关 `debug_routing_digest` |
| `27bdc7a9` | `splash_dsinks_fp32` |
| `2306c7a5` | `log_indexer_loss_per_layer` |
| `9eac4e60` | 诊断开关 `debug_forced_routing_deepseek` |
| `e286fd63` | 诊断开关 `debug_forced_indexer_topk` |
| `2c28a4f4` | `splash_dsinks_gpu_order` |
| `1dddc65d` | `rmsnorm_scale_grad_gpu_order` |
| `676759bc` | `rmsnorm_scale_grad_gpu_order` 打开时输入梯度与原写法逐位相同 |
| `1e880fdc` | 诊断开关 `debug_capture_choices` |
| `47d31984` | `indexer_kd_generalized_kl` |
| `fd30fbfb` | `indexer_kd_generalized_kl` 小值处改用级数 |
| `c23ca978` | `transcendental_log_freebsd` |
| `b87a55aa` | 只改 `transcendental_log_freebsd` 的注释与配置说明，不改计算 |
| `2aa332cb` | `ihc_stream_sum_unrolled` |
| `d5c9b113` | 修复：数据并行 + 梯度累积时 `embedding_grad_fp32` 的词嵌入梯度丢掉前面的微批 |
| `6d2179a0` | `embedding_grad_fp32_local_bf16_comm` |
| `f70d74cb` | 同上（合入提交） |
| `56492817` | MLA 把 `float32_qk_product`、`float32_logits` 接到 AttentionOp |
| `f76a136f` | 同上（合入提交） |
| `9099c8ce` | `embedding_shard_over_data` |

## 文中提到、但没有公开的东西

- **janus 的开发历史。** 结果文件和文档里记的 janus 提交号指开发仓库的提交，公开版里查不到，保留作出处。
- **问题跟踪记录。** 文中「一次实际排查」「项目里的排查」等说法，指开发过程中的问题跟踪记录，没有公开。相关的技术结论都已写进正文。

## 换成你自己的环境

- 桶名一律写成 `gs://<POC 桶>/…` 占位，换成你自己的存储桶。
- 集群的 kubectl 上下文、命名空间、节点池、队列等提交设置不在仓库里，放在仓库外的站点配置（`JANUS_SITE_PROFILE`），照 `models/*/frameworks/maxtext/site-profile.example.sh` 写一份，见[准备 TPU 环境](guide/03-tpu-setup.md)。
- GPU 一侧的 Pod / JobSet 模板里，节点池等写成 `<…>` 的位置换成你自己的值。

## 许可

Apache License 2.0，见 [`LICENSE`](../LICENSE) 与 [`NOTICE`](../NOTICE)。
