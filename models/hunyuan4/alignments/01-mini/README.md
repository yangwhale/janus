[janus](../../../../README.md) › [混元 4](../../README.md) ｜ 01 · mini（hy4-mini，T1）

# 混元 4 · 01 · mini：hy4-mini（427,783,489 参数），TPU v5p 8 个设备 ↔ B200 × 8

hy4-mini 保留混元 4 的全部结构与每头维度，只缩宽度与个数（7 层 + 1 层 MTP，随机初始化），在上面做生产式训练（Muon + QK-Clip + 索引器 KD + MTP）的完整 L0–L4 对齐。根源一览见 [`ROOT-CAUSES.md`](../../ROOT-CAUSES.md)。GPU（Megatron-Core）是 golden，TPU（MaxText）是被测，噪声尺子只用 GPU 重跑。

- **给人读的**：[总览](../../dashboard/dashboard.md)、[对齐报告](ALIGNMENT-REPORT.md)；依据在[对齐细节](DETAILS.md)（知识库，偏差台账、开关表、签字清单）。
- 逐项比什么：[检查清单](CHECKS.md)；历代比较：[历代比较](../../dashboard/history.md)。

## 这个文件夹

| 文件 / 文件夹 | 放什么 |
|---|---|
| [`ALIGNMENT-REPORT.md`](ALIGNMENT-REPORT.md) | 给人读的结论：现状、生产建议、挡签字、下一步 |
| [`DETAILS.md`](DETAILS.md) | 对齐细节（知识库）：偏差台账、开关表、签字清单、证据索引 |
| [`CHECKS.md`](CHECKS.md) | 检查清单（由 `models/hunyuan4/dashboard/build.py` 按 `dashboard/checks.yaml` 生成） |
| [`config-alignment.md`](config-alignment.md) | 两边每一项配置怎么对上 |
| [`goldens.md`](goldens.md) | 每一份运行的来历与代号 |
| [`setup.md`](setup.md)、[`runs.yaml`](runs.yaml) | 设置与运行登记 |
| `catalogs/`、`results/` | 检查项目录；结果文件 |
| [`gpu-noise/`](gpu-noise/README.md) | GPU 一侧自比 |

本尺寸 TPU 配置是 [`hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml)。

## 现状

现行配置（2026-11-11 起 base2 关）跑 1000 步，对 GPU 基线逐步比，噪声取同一代 GPU 16 份两两 120 对：

| 比较 | 步数 | 被测 / 基线 | 结果 |
|---|---|---|---|
| [`t1zo-l4`](results/t1zo-l4.report.md) | 1000 | TPU `tz-off-rep`（现行对齐配置，含 M50，base2 关）/ GPU `g3-a` | 过 7、不过 0、待定 2（主 loss 全程与后段，r 1.55 / 1.52；与 base2 无关，是第 500 步以后攒出来的训练轨迹差，同一份权重上前向没有，后段根因未定：已排除 MTP / KD 两项 loss、sink 与 norm 两类小参数，[D30](DETAILS.md#d30)）；上一代（base2 开的 `t1zg-a`）见[历代比较](../../dashboard/history.md) |

| mini 挡签字的项 | 状态 |
|---|---|
| 第 1 步 177 个超出噪声 | 现行配置（base2 关）5 个 iHC 张量与 MTP enorm 比 GPU 远；enorm 与其中 2 个 iHC 在 base2 开时已接受（[D11](DETAILS.md#d11)），另 3 个 iHC 是关 base2 后新出现的，未定责；仍缺 2b |
| 第 1 步噪声参照 | 只有 3 对，要求 15 对 |
| golden 对 GPU 生产程序 | 没比 |
| 三条事后判法的前瞻验证 | 已在 PROTOCOL 事先登记，用真实权重截层下一批运行验证 |

QK-Clip 全局最大 logit（[D29](DETAILS.md#d29)）、逐元素 log（[D14](DETAILS.md#d14)）已定责、已接受；param_norm 漂移（[D12](DETAILS.md#d12)）已过线，不挡签字。


## 真实权重截层的 L0 与 L1（不依赖优化器）

阶段 A–D 是混元 4 真实权重的截层（A＝第 0 层，B＝第 0–1 层，C＝第 0–2 层，D＝第 0–1 层加 1 层 MTP；`model/hy4_map.py` 的 `STAGES`）。L0 转换与阶段 A 的 L1 单算子不依赖优化器，仍有效；T2 的端到端训练在 [real-weight](../02-real-weight/README.md)。

| 检查 | 阶段 | 结果 | 结果文件 |
|---|---|---|---|
| HF 目录：135 个张量的名字、形状、dtype；参数量 42,070,313,621 | 全部 | pass | [`hy4-l0-names`](results/hy4-l0-names.results.json) |
| 转换后逐张量还原回 HF 对 `tensor_sha256.json` 逐位，含 dtype 清单 | A / B / C / D | pass（31 / 60 / 84 / 87 个张量） | `hy4-{a,b,c,d}-l0-ckpt-convert`，例 [`A`](results/hy4-a-l0-ckpt-convert.report.md) |
| RoPE 列去交错与逆置换 | A / B / C / D | pass（6 / 12 / 14 / 18 个张量） | `hy4-{a,b,c,d}-l0-rope` |
| 跨侧：TPU 初始权重对 GPU golden 初始参数清单 | A、B | pass（248 / 248、472 / 472，对 AdamW 时期的 GPU golden：A 是 `ga-b`、`ga1-a`，B 是 `gb-a`；现行 T2 的 Megatron-Core golden 上这一项没做，见 [real-weight 签字清单](../02-real-weight/ALIGNMENT-REPORT.md#签字清单) 1a） | [`A`](results/hy4-a-l0-init-vs-gpu.report.md)、[`B`](results/hy4-b-l0-init-vs-gpu.report.md) |
| 数据回放：300 步 × 8 rank × 2 microbatch 的输入指纹对 GPU 运行 | A | 4800 个单元 0 处不同 | [`ga-a`](results/hy4-a-replay-check-ga-a.txt) |
| 阶段 A 第 0 层 10 个算子隔离回放，对 FP64（GPU 一侧是 HF 参照模型） | A | 全过，R 0.18–1.00 | [`hy4-a-l1-op-tpu-fix.json`](results/hy4-a-l1-op-tpu-fix.json) |

- 单算子回放把每个算子的输入换成 GPU 第 1 步抓到的同一个输入，GPU 一侧的 FP64 结果在 [`ga1-a.fp64_ops.json`](../../results/ga1-a.fp64_ops.json)。这里 GPU 是 HF 写法（RMSNorm 两次舍入），RMSNorm 的 R 0.71 偏向 TPU（推断）。
- 接线核对（诊断）：CPU 上用阶段 A 检查点跑一次前向、按模块对 GPU 抓的张量，正常转换各模块 rel_L2 在 BF16 量级（logits 9.43e-3）；不去交错的负对照在注意力处大 20 倍以上（logits 1.67e-1）（[`hy4-a-probe-forward-cpu.json`](results/hy4-a-probe-forward-cpu.json)）。

## 其他

- **显存**：编译期估算 [`hy4-mini-aot-memory.json`](results/hy4-mini-aot-memory.json)，实测峰值在 [`hy4-mini-tpu-determinism.json`](results/hy4-mini-tpu-determinism.json) 的 `peak_bytes_in_use_max_local_devices`。
