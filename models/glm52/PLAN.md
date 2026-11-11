[janus](../../README.md) › [GLM-5.2](README.md) ｜ 实践计划

# GLM-5.2 · GPU → TPU 端到端数值对齐计划

- 比较协议见 [`PROTOCOL.md`](PROTOCOL.md)，在任何 golden 跑出来之前登记、冻结。
- 本页的数字分三类：**精确**（从公开 config 与 checkpoint 头算出）、**实测**（附运行名与结果文件）、**估算**（写明推导）。没标注的属于第一类。

## 0. 一句话

纯 GLM-5.2：模型、结构、配方都按 GLM-5.2 原样，不掺任何其他模型的取值。以 Megatron-Bridge 的 GLM-5.2 配方作为 GPU 一侧的「生产配置」，只改 8 卡单机放不下或无法照搬的几项（第 3 节）。两边都是 8 个设备，数据按 rank 一对一回放，与混元 3 相同。

## 1. 截层

| 阶段 | 截哪几层 | 参数量 | 新结构 |
|---|---|---|---|
| 1 | 第 0 层（稠密，自带索引器）+ 第 6 层（MoE，自带索引器），嵌入、最后的 norm、输出头；不带 MTP | 12,181,475,072 | MLA、DSA 索引器与 top-k、稠密 FFN、256 专家 MoE 与共享专家、sigmoid 路由与无辅助损失偏置 |
| 2 | 阶段 1 加 MTP | 22,134,395,648 | 与主干联合训练的 MTP 层 |
| 之后（届时再定） | 例如阶段 1 加第 7 层（MoE，复用第 6 层 top-k） | 22,049,507,840 | 索引器跨层共享 |

- 参数量由 [`analysis/param_counts.py`](analysis/param_counts.py) 读 checkpoint 头（revision `cf457fa7`）算出，结果在 [`results/param-counts.json`](results/param-counts.json)。全模型 59,585 个张量、753,329,940,480 个参数（含 MTP 层）；模型卡里没有写总参数量（查过），所以没有外部锚点。
- 为什么取第 6 层而不取第 3 层：config 前 3 层是稠密层（`first_k_dense_replace: 3`），第 3 层是第一个 MoE 层，但它的 `indexer_types` 是 `shared`，复用上一层的 top-k，checkpoint 里第 3、7 层都没有索引器张量（第 0、1、2、6 层各 5 个）。截成 0 + 3 时第 3 层会去复用第 0 层的 top-k，语义变了。第 6 层是第一个自己算 top-k 的 MoE 层。
- 截层后两边的配置同步截：HF 的 `indexer_types`、`mlp_layer_types` 只留选中的层；Megatron 与 MaxText 的层类型、是否带索引器逐层对上（L0 核）。
- 阶段 1 每一级都对齐之后才加 MTP。

## 2. GPU 参照程序

- 程序：Megatron-Bridge 的 GLM-5 桥（`GlmMoeDsaForCausalLM`）+ Megatron-Core；载入截层后的 HF checkpoint。HF transformers `glm_moe_dsa` 只当语义参照，用在 G1，不当训练程序。
- 里程碑：

| # | 做什么 | 判据 |
|---|---|---|
| G0 | 截层 checkpoint；Bridge 载入再导出 | 截出的张量 sha256 对上原件；往返逐位相同 |
| G1 | Megatron 对 HF `glm_moe_dsa`，FP32，逐层等价 | 层输出、logits、梯度相对差 ≤ 1e-5；loss 逐位相同；top-k 不同的行都是平局；负对照必须判不一致 |
| G2 | B200 冒烟：Muon + QK-Clip + 整层重算，10 步 | 能跑；记步时与显存峰值；降低阈值时 QK-Clip 确实触发；每种结构都跑到 |
| G3 | 确定性：同设置 2 份 | 是否逐位相同，决定协议里噪声怎么取 |
| G4 | 正式 golden：8 份 × 300 步 + 第 1 步全量抓数 3 份 | 照 [`PROTOCOL.md`](PROTOCOL.md) |

## 3. 「生产配置」与必须不同的地方

| 项 | GLM-5.2 配方 | 本实践 | 理由 |
|---|---|---|---|
| 结构 | GLM-5.2 原样 | 同 | — |
| 优化器 | 分布式融合 Adam | Megatron-Bridge 的 `distributed_muon_with_cosine_annealing`，默认值：momentum 0.95、nesterov、Newton-Schulz 5 步、spectral 缩放、NS 精度 "medium"；标量参数（嵌入、norm、偏置）走 Adam | 项目负责人定（2026-11-07）。具体取值从 Bridge 源码读出，写进 golden 的 `meta.json` 与 [`config-alignment.md`](config-alignment.md) |
| QK-Clip | 无 | 开：Megatron-Core 默认 alpha 0.5、阈值 100；TPU 一侧用 MaxText 的 QK-Clip，语义逐项对上 | 项目负责人定：开 Muon 就开 QK-Clip，两边同开 |
| 学习率 | warmup + cosine | 同形状，按本实践的步数缩放；具体数字在 PROTOCOL 里登记 | 项目负责人定 |
| 并行 | PP / EP 跨多机 | 8 卡单机：专家并行 8 + 数据并行 8，TP / PP / CP = 1 | 8 卡单机放不下多机切分；保留专家并行；PP 只改变求和顺序，截层后层数也不够切 |
| 重算 | full / uniform | 同 | 生产标准 |
| 序列 | 4096 | 同 | — |
| 批 | 配方的全局批 | 全局 16 = 8 卡 × microbatch 1 × 梯度累积 2 | 与混元 3 相同：设备 d 的第 m 个 microbatch 原样回放给 TPU |
| 数据 | 配方数据集 | fineweb-edu `sample/10BT` 第一个分片，GLM 分词器 | 与混元 3 同源同格式；每批存下给 TPU 回放 |
| TF32 | — | 真关，启动时 FP32 GEMM 对 FP64 实测 | 混元 3 的教训：设置读回值不可信 |

要先查清的数值点（结论写进 [`config-alignment.md`](config-alignment.md)）：

- Muon 的 Newton-Schulz 在什么精度里做（"medium" 的含义），TPU 一侧怎么对上。
- MLA 的投影在 Muon 里是整矩阵正交化还是拆开做（Megatron-Core 对 MLA 没有拆分）；两边按同一种写法。
- QK-Clip：每个头的统计量怎么取、q / k 的哪几段各乘缩放因子的几次方、哪些层参与；分布式优化器下参数是一维分片，按头取统计量要在完整参数上做。

## 4. TPU 一侧（现在只做编译期显存分析）

- MaxText 的 DeepSeek 解码器（MLA + DSA）上写一份 GLM-5.2 的模型配置；v5p 2x2x2（8 个设备）、FSDP 8、整层重算、Muon + QK-Clip。
- 编译三档：阶段 1；阶段 1 + MTP；阶段 1 + 第 7 层。尽量在 CPU 上按拓扑离线编译，不占 TPU。结果写进 `alignments/01-v5p-acceptance/results/`。编译期数字只作初筛，最终以 TPU 上的实测峰值为准。
- TPU 训练等 GPU 参照跑通（G4 完成）之后再开始。

## 5. 级别与份数

照混元 3（[`docs/reference/levels.md`](../../docs/reference/levels.md)）：L0 地基逐位 → L1 / L2 单算子与逐层隔离回放对 FP64 的 R 比 → L3 第 1 步整步 → L4 多步。份数与判法见 [`PROTOCOL.md`](PROTOCOL.md)。

## 6. 变更记录

| 日期 | 改了什么 | 原因 |
|---|---|---|
| 2026-11-07 | 初版 | — |
