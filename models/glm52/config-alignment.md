[janus](../../README.md) › [GLM-5.2](README.md) ｜ 配置对照表

# GLM-5.2 · 配置对照表

两边每一项配置怎么对上、证据在哪。格式照[混元 3](../hunyuan3/config-alignment.md)：按运行时实际生效的值填，每个值都要有出处；「待查」逐条列出，查清后改成「相同 / 等价 / 有差异」并附证据。GPU 一栏取自 golden 的 `meta.json`，TPU 一栏取自被测的配置与 `meta.json`。

**现状**：GPU、TPU 两栏都还没有运行，下表先列出必须核对的项与 GLM-5.2 的公开取值（Hugging Face `zai-org/GLM-5.2` revision `cf457fa7` 的 config.json）。

| 类别 | 项 | 公开 config | GPU（Megatron-Bridge） | TPU（MaxText） | 状态 | 证据 |
|---|---|---|---|---|---|---|
| 结构 | hidden / 层类型 | 6144；前 3 层稠密，其余 MoE；截层取第 0、6 层 | 待查 | 待查 | 待查 | L0 逐层核 |
| 结构 | MLA | q_lora 2048、kv_lora 512、nope 192、rope 64、v 256、64 头 | 待查 | 待查 | 待查 | L0 |
| 结构 | RoPE | theta 8,000,000，`rope_type: default`，`rope_interleave: true` | 待查 | 待查 | 待查 | L2 比 q / k |
| 结构 | DSA 索引器 | 32 头 × 128，top-k 2048，`indexer_rope_interleave: true`；第 0、1、2、6、10… 层自带（21 层），其余复用 | 待查 | 待查 | 待查 | L1 top-k 翻转统计 |
| 结构 | 稠密 FFN | 12288，SwiGLU | 待查 | 待查 | 待查 | L1 |
| 结构 | MoE | 256 专家 × 2048，top-8 + 1 共享专家，sigmoid，`routed_scaling_factor` 2.5，top-k 归一，路由 FP32（`moe_router_dtype: float32`） | 待查 | 待查 | 待查 | L1 路由回放 |
| 结构 | 归一化 | RMSNorm，eps 1e-5 | 待查 | 待查 | 待查 | L1 |
| 结构 | 词表 / 输出头 | 154880，不共享 | 待查 | 待查 | 待查 | L0 |
| 优化器 | 类型 | — | Muon（`distributed_muon_with_cosine_annealing`），标量参数走 Adam | Muon | 待查 | 第 1 步优化器状态（L3） |
| 优化器 | Muon 超参 | — | momentum 0.95、nesterov、NS 5 步、spectral 缩放、NS 精度 "medium"（从 Bridge 源码读出后填实际生效值） | 待查 | 待查 | L3 |
| 优化器 | Muon 处理 MLA 投影的方式 | — | 待查（Megatron-Core 对 MLA 不拆分） | 待查 | 待查 | L3 |
| 优化器 | **QK-Clip** | — | 开，alpha 0.5、阈值 100（Megatron-Core 默认） | 开，MaxText QK-Clip，同 alpha / 阈值 | 待查（必须核对的一行） | 每个头的统计量、缩放的分配方式与参与的层逐项对照；降低阈值的触发测试两边各做一次；L4 比被裁的头 |
| 优化器 | 学习率 | — | 配方的 warmup + cosine，按步数缩放（数字见协议第 7 节） | 同 | 待查 | L4 学习率逐位 |
| 优化器 | 权重衰减、梯度裁剪、Adam β / ε | — | 待查（实际生效值） | 待查 | 待查 | L0 配置对照 |
| 路由 | 无辅助损失偏置更新 | `topk_method: noaux_tc` | 待查（更新率与公式） | 待查 | 待查 | L4 路由偏置曲线 |
| 数据 | 全局批 / microbatch / 累积 | — | 16 = 8 卡 × 1 × 2 | `per_device_batch_size: 1`、`gradient_accumulation_steps: 2` | 待查 | L0 输入指纹逐位 |
| 并行 | 切分 | — | 专家并行 8 + 数据并行 8 | FSDP 8 | 有差异（不可避免） | L4 对噪声 |
| 精度 | 计算 / 主权重 | — | BF16 / FP32 | BF16 / FP32 | 待查 | L1 |
| 精度 | TF32 | — | 真关，启动时实测 | — | 待查 | `meta.json` 的 `tf32` |
| 重算 | 方式 | — | full / uniform | 整层重算 | 待查 | L2 |
