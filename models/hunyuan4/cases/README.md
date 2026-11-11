[janus](../../../README.md) › [混元 4](../README.md) ｜ 排查案例

# 混元 4 · 排查案例

各尺寸共用，一个问题一个文件，每篇四段：现象 / 机制 / 证据 / 结论（含修法与定责依据）。数字出自各尺寸的 `results/`、[`CHANGES-LEDGER.md`](../CHANGES-LEDGER.md) 与 [`PROTOCOL.md`](../PROTOCOL.md) 变更记录；结论以[根源一览](../ROOT-CAUSES.md)与 [mini 对齐细节](../alignments/01-mini/DETAILS.md)为准。案例里不带链接的「对齐报告」「Dn」指 mini 对齐细节（D19–D22 在 [real-weight 的报告](../alignments/02-real-weight/ALIGNMENT-REPORT.md)）。

r、R、MARE 与定责依据的读法见[总览细节「读法」](../dashboard/detail.md#读法)，术语见[知识库的术语对照表](../knowledge.md#术语对照表)。

边界类型（与根源一览的 A / B / C 类别无关）：**边界**（16 位与 32 位之间的舍入位置）／**语义**（两边写法或配置的含义不同）／**协议**（协议里的设置）／**病态**（标量本身数值病态）／**量法**（测量口径）。

| # | 案例 | 边界类型 | 台账 / 开关 | 定责依据 | 状态 |
|---|---|---|---|---|---|
| 01 | [第 1 步逐张量梯度普遍是 GPU 散布的 2–3 倍](01-step1-grad-rounding-switches.md) | 边界 | M8–M10、M12 | 读代码（对照 TE 的舍入位置） | 已修 |
| 02 | [第 0 层索引器 KD 偏低约 1.2e-3](02-indexer-kd-f32-lse.md) | 病态 | M23、M24、G16 | 只改 TPU 后对 FP64 的偏差收拢 | 已修；1000 步记录值偏高另由 M50 修（[根源一览第 25 行](../ROOT-CAUSES.md#r25)） |
| 04 | [TPU 的「FP32 输出头」只做一遍 BF16](04-lm-head-single-bf16-pass.md) | 边界 | M13 | 读代码；单开关 A/B | 已修（T1 开）；代价 +24%，生产看情况 |
| 05 | [`full` 重算下反向重选路由](05-full-remat-routing-reselect.md) | 语义 | M26 | 读代码 | 已修 |
| 06 | [GPU eager 注意力 q·k 先舍 BF16](06-gpu-attn-scores-bf16.md) | 边界 | G17 | 读代码＋修后两边的差收拢 | 已修 |
| 07 | [QK-Clip 的三处问题](07-qk-clip-bugs.md) | 语义 | M1、M19、M25、M35 | 读代码＋单测 | 已修 |
| 08 | [Muon 与标量 Adam 对上原版 Megatron-Core](08-muon-mcore-switches.md) | 语义 | M15–M18、M30、M34、G18 | 读代码＋单测；回放对 FP64 的 R | 已对齐（D17） |
| 09 | [路由偏置均值 r 19](09-router-bias-zero-mean.md) | 协议 | G14 | 读代码＋改后两边的差收拢 | 已解决 |
| 11 | [第 1 步 iHC 梯度：四路流加权和的相加顺序](11-ihc-stream-sum-highest-einsum.md) | 边界 | M28、G29 | 只改 TPU 后两边的差收拢 | 已修；现行配置 6 个 iHC 张量已定责、已接受（D11） |
| 12 | [param_norm：FP32 全局求和的测量误差与真实漂移](12-param-norm-fp32-sum-and-drift.md) | 量法＋边界 | J13、M28 | 量法（两边比精确值） | 量法已改；余下一半的漂移待查（D12） |
| 13 | [权重侧精度开关在第 1 步测不到](13-weight-side-switch-step1-blind-spot.md) | 量法 | M13、M27、M8 | 只改一边的 300 步 A/B＋机理核对 | 机理已查明 |
| 14 | [DDP 和 FSDP 对不上，一路查到词嵌入反向的一次 scatter-add](14-ddp-fsdp-embedding.md) | 边界 | M12、J48；MaxText 上的一处修改 | 对 FP64 的 R；训练步 HLO 审计；单算子逐位对仿真 | 已修（三种修法）；生产尺寸上的验收待核 |
| 14 · 二 | [案例 14 第二阶段：只把词嵌入表沿数据并行轴切开](14-ddp-fsdp-embedding.md#第二阶段只把词嵌入表沿数据并行轴切开) | 边界 | maxtext `9099c8ce` | 对 FP64 的 R；训练步 HLO 的集合通信清单 | 已验证（300 / 1000 步和 GPU 重跑分不开） |
