[janus](../../../README.md) › [混元 4](../README.md) ｜ 排查案例

# 混元 4 · 排查案例

各尺寸共用，一个问题一个文件，每篇四段：现象 / 机制 / 证据 / 结论（含修法与定责依据）。数字出自各尺寸的 `results/`、[`CHANGES-LEDGER.md`](../CHANGES-LEDGER.md) 与 [`PROTOCOL.md`](../PROTOCOL.md) 变更记录；结论以[根源一览](../ROOT-CAUSES.md)与 [mini 对齐细节](../alignments/01-mini/DETAILS.md)为准。案例里不带链接的「对齐报告」「Dn」指 mini 对齐细节（D19–D22 在 [real-weight 的报告](../alignments/02-real-weight/ALIGNMENT-REPORT.md)）。

r、R、MARE 与定责依据的读法见[总览细节「读法」](../dashboard/detail.md#读法)，术语见[知识库的术语对照表](../knowledge.md#术语对照表)。

边界类型（与根源一览的 A / B / C 类别无关）：**边界**（16 位与 32 位之间的舍入位置）／**语义**（两边写法或配置的含义不同）／**协议**（协议里的设置）／**病态**（标量本身数值病态）／**量法**（测量口径）／**编译器 bug**（编译出的程序算错，和精度无关）。

**推荐先读：[案例 14](14-ddp-fsdp-embedding.md)、[案例 17](17-late-drift-elided-rounding.md)。** 两篇都从一条看不出原因的曲线查到编译产物里的一处舍入，排查方法写得最完整。案例 17 的「先编译、再扫描」（[第九幕](17-late-drift-elided-rounding.md#第九幕全局清单先编译再扫描)）在整步程序里一次扫出所有「声明了 BF16 舍入、编译器却省掉」的位置，换模型也能直接用。

| # | 案例 | 边界类型 | 台账 / 开关 | 定责依据 | 状态 |
|---|---|---|---|---|---|
| 01 | [第 1 步逐张量梯度普遍是 GPU 散布的 2–3 倍](01-step1-grad-rounding-switches.md) | 边界 | M8–M10、M12 | 读代码（对照 TE 的舍入位置） | 已修 |
| 02 | [第 0 层索引器 KD 偏低约 1.2e-3](02-indexer-kd-f32-lse.md) | 病态 | M32、M33（修法）；M23、M24、G16（排除与诊断） | 只改 TPU 后对 FP64 的偏差收拢 | 已修；1000 步记录值偏高另由 M50 修（[根源一览第 25 行](../ROOT-CAUSES.md#r25)） |
| 04 | [TPU 的「FP32 输出头」只做一遍 BF16](04-lm-head-single-bf16-pass.md) | 边界 | M13 | 读代码；单开关 A/B | 已修（T1 开）；代价 +24%，生产看情况 |
| 05 | [`full` 重算下反向重选路由](05-full-remat-routing-reselect.md) | 语义 | M26 | 读代码 | 已修 |
| 06 | [GPU eager 注意力 q·k 先舍 BF16](06-gpu-attn-scores-bf16.md) | 边界 | G17 | 读代码＋修后两边的差收拢 | 已修 |
| 07 | [QK-Clip 的三处问题](07-qk-clip-bugs.md) | 语义 | M1、M19、M25、M35 | 读代码＋单测；M19 另有 TPU 低阈值实测 | 已修 |
| 08 | [Muon 与标量 Adam 对上原版 Megatron-Core](08-muon-mcore-switches.md) | 语义 | M15–M18、M30、M34、G18 | 读代码＋单测；回放对 FP64 的 R | 已对齐（D17） |
| 09 | [路由偏置均值 r 19](09-router-bias-zero-mean.md) | 协议 | G14 | 读代码＋改后两边的差收拢 | 协议问题已解决；现行 `tc-a` 1000 步路由偏置均值 r 1.31 待定、未定责，是另一件 |
| 11 | [第 1 步 iHC 梯度：四路流加权和的相加顺序](11-ihc-stream-sum-highest-einsum.md) | 边界 | M28、G29 | 只改 TPU 后两边的差收拢 | 已修；现行 C 口径（`s1f-c`，GPU 参照 6 份）第 1 步还有 1 个 iHC 张量未定责：第 3 层 FFN 前 `hc_fn`（1.14 N，R 1.43）；C 口径之前 `tz-off-rep` 未定责的第 3 层注意力前 `hc_scale` 在 C 口径下 0.66 N；base2 开那一代的 6 个已定责、已接受（D11） |
| 12 | [param_norm：FP32 全局求和的测量误差与真实漂移](12-param-norm-fp32-sum-and-drift.md) | 量法＋边界 | J13、M28 | 量法（两边比精确值） | 量法已改；t1v 上余下一半的漂移待查（D12）；现行 `tc-a` 前 300 步 r 0.18、1000 步带符号均值 r 0.04，都过 |
| 13 | [权重侧精度开关在第 1 步测不到](13-weight-side-switch-step1-blind-spot.md) | 量法 | M13、M27、M8 | 只改一边的 300 步 A/B＋机理核对 | 机理已查明 |
| **14 ★** | [DDP 和 FSDP 对不上，一路查到词嵌入反向的一次 scatter-add](14-ddp-fsdp-embedding.md) | 边界 | M12、M54、M55、M56、J48 | 对 FP64 的 R；训练步 HLO 审计；单算子逐位对仿真 | 已修（三种修法）；生产尺寸上的验收待核 |
| 14 · 二 | [案例 14 第二阶段：只把词嵌入表沿数据并行轴切开](14-ddp-fsdp-embedding.md#第二阶段只把词嵌入表沿数据并行轴切开) | 边界 | M54；maxtext `9099c8ce` | 对 FP64 的 R；训练步 HLO 的集合通信清单 | 已验证（300 / 1000 步和 GPU 重跑分不开） |
| 15 | [换 libtpu 版本后，RMSNorm 多舍一次被放大成主 loss 偏高](15-libtpu-upgrade-final-norm-double-rounding.md) | 边界 | M9 | 只改 TPU 的单开关 A/B；同状态前向；FP64 模拟 | 机理已查明；修法不变（开 `rmsnorm_single_rounding`） |
| 16 | [jit 静默算错：不满 8 行的行块](16-jit-miscompile-sub8-row-block.md) | 编译器 bug（不是精度） | 无（libtpu 实测 0.0.46、0.0.48、0.0.51、0.0.52 有的 bug，0.0.47、0.0.49 没测；workaround 见案例） | 逐算子执行对 jit；逐 pass 程序 CPU 回放；换 libtpu 二分 | 机理已查明；正式版未修（nightly 11-06 起这个复现已对，修的是融合决策，代码生成路径没确认），混元 3 / 混元 4 存档程序里没有特征 |
| **17 ★** | **[后段为什么慢慢偏开：iHC 流梯度、被省掉的 BF16 舍入，以及一次扫全图找同类](17-late-drift-elided-rounding.md)** | 边界 | M36、M52、M58、M59、M62；G37 | 两头验证（TPU 补舍入、GPU 照 TPU 不舍）；整步 HLO / LLO 扫描 | 主因已定位；C 口径下的残余在查（全局清单给出三类候选，GPU 验证在跑） |
