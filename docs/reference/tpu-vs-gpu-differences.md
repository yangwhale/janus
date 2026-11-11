[janus](../../README.md) › [文档](../README.md) ｜ TPU 与 GPU 训练数值哪里不一样：逐条清单

# TPU 与 GPU 训练数值哪里不一样：逐条清单

**现状**（hy4-mini，判定见[总览](../../models/hunyuan4/dashboard/dashboard.md)）：

| 配置 | 主 loss | L4 判定 | 步时 |
|---|---|---|---|
| 默认（对齐开关全关） | 300 步最大单步：N（GPU 重跑波动上限）4.53e-5，TPU 2.68e-4，r 5.91 | 不过 | 基准 |
| 按本清单逐条处理后（现行） | 1000 步 MARE：N 8.24e-6，TPU 8.05e-6，r 0.98 | 9 项过 7、待定 2（U5、U18，已接受） | 全开比全关 +44.6%（不含 M42b、M50；mid +46.8%，量法不同），大头 B1 |

**范围**：GPU B200 × 8（混元 4 Megatron-Core 0.20 + TE，混元 3 Megatron-Bridge v0.7.0）；TPU MaxText（v5p 2x2x2，混元 3 另有 v7x-4），开关在 `maxtext` 混元 4 分支、默认关；BF16 混合精度、FP32 主权重、无 FP8。数字以 hy4-mini 为主。

**读法**（完整见[混元 4 总览细节「读法」](../../models/hunyuan4/dashboard/detail.md#读法)）：

| 术语 | 含义 | 判定线 |
|---|---|---|
| r | TPU 对 GPU 的差 ÷ N（噪声上限：GPU 8 份两两 28 对的最大值），不定责；取带符号均值、MARE（逐步相对差绝对值的平均）、最大单步里最差的 | ≤ 1 过，> 2 不过 |
| R | TPU 对 CPU FP64 的误差 ÷ GPU 的同一量，能定责 | ≤ 1.2 过 |
| 生产建议 | 开（修 bug，或更准且代价 < 1%）／不开（只为照 GPU）／看情况（取舍或未查清）／不涉及 | — |
| 代价 | hy4-mini 300 步单开关 A/B：全开步时 ÷ 只关该开关的步时 − 1 | — |
| 编号 | M / G / J：[台账](../../models/hunyuan4/CHANGES-LEDGER.md)；[rn]：[根源一览](../../models/hunyuan4/ROOT-CAUSES.md)第 n 行；混元 3 #n：[混元 3 报告](../../models/hunyuan3/ALIGNMENT-REPORT.md)第 5 节；类 n：[精度边界](precision-boundaries.md) | — |

## 1. 总表

| # | 差异一句话 | 边界类别 | 修法 / 开关 | 生产建议 | 代价 |
|---|---|---|---|---|---|
| A1 | 跨分片部分和在 BF16 里相加 | 类 2 | `dense_fp32_accumulation`（M12） | 开（v7x 看情况） | +0.8% |
| A2 | RMSNorm 舍一次还是两次 | 类 4 | `rmsnorm_single_rounding`（M9） | 开 | −0.2% |
| A3 | SwiGLU 激活舍一次还是两次 | 类 5 | `mlp_activation_round_per_op`（M10） | 不开 | −0.1% |
| A4 | 路由权重在 fc2 前还是 wo 后乘 | 舍入位置 | `moe_mcore_prob_before_fc2`（M40） | 看情况（对 FP64 未测） | +0.2% |
| A5 | SwiGLU 输入梯度舍一次还是两次 | 舍入次数 | `mlp_fused_fc1_dgrad`（M41） | 看情况（对 FP64 未测） | −1.8% |
| A6 | 编译器省掉 GPU 必做的 BF16 往返（推断） | 编译器 | `ihc_force_bf16_rounding`（M36） | 不开 | +0.1% |
| A7 | FP32 参数的梯度先以 BF16 落地 | 类 10 | 三个梯度 FP32 开关 | embed、norm 开；输出层看情况 | 部分未量 |
| A8 | GPU 参照把 q·k 先存成 BF16 | 类 1 | GPU `--attn-scores-fp32`（G17） | 不涉及 | — |
| A9 | 注意力缩放常数在 BF16 里舍过 | 舍入位置 | 不处理 | 不涉及 | — |
| A10 | splash 反向 sink 梯度在 BF16 里累加 | 类 3 | M42；对齐 M42b | 开 M42 | M42b +3.9% |
| A11 | splash base2 指数把注意力 logits 放大 1.0018 倍、q 多舍一次 | 舍入位置（常数舍成 BF16） | `sa_use_base2_exp: false`（M43） | 关 base2 | 无（对齐配置用 base2 慢 0.1%） |
| B1 | FP32 矩阵乘默认单遍 BF16（输出头） | 类 7 | `logits_matmul_precision: highest`（M13） | 看情况 | +23.7% |
| B2 | 索引器打分单遍 BF16 | 类 7 | `indexer_fp32_scores`（M23） | 看情况 | +6.3% |
| B3 | 路由 logits 精度决定选哪个专家 | 类 6 | `float32_gate_logits` 等 | 开 | 波动内 |
| B4 | 乘 FP32 主权重还是 BF16 权重 | 类 8 | M27、M8 | 不开 | ≤ +0.1% |
| B5 | iHC 加权和的相加顺序 | 舍入位置 | `ihc_stream_sum_unrolled`（M28） | 看情况 | +0.2% |
| B6 | 格点上的权重 HIGHEST 与 default 逐位同 | 机理 | — | 不涉及 | — |
| B7 | GPU 上 TF32 悄悄打开 | 类 9 | GPU 实测 `tf32_probe` | 不涉及 | — |
| B8 | GPU 张量核累加截断，长求和有偏 | 类 3 | GPU dgrad 分段累加 | 不涉及 | GPU +61% |
| C1 | FP32 exp / log 是近似实现 | 类 13 | `transcendental_accuracy_highest`（M32） | 看情况 | +3.0% |
| C2 | `lax.logistic` 带 HIGHEST 不起作用 | 类 13 | 同开关的 M37 | 同 C1 | 同 C1 |
| C3 | 精确 sigmoid / silu 的反向式子 | 类 13 | M39（同开关） | 同 C1 | 同 C1 |
| C4 | 病态标量把一格误差放大 | 类 12 | 找被放大的源头；KD 记录改记广义 KL（M50） | M50 开 | 无 |
| C5 | FP32 除法、sqrt、1 − β^t 不精确 | 类 11 | `muon_adam_accurate`（M34） | 开 | +0.1% |
| C6 | 开关覆盖不到的 | 类 13 | 无 | 不涉及 | — |
| D1 | Nesterov 做不做偏差修正 | 语义 | `muon_mcore_nesterov`（M15） | 看情况 | 未测 |
| D2 | 标量参数 Adam 还是 NAdam | 语义 | `muon_adam_plain`（M16） | 看情况 | 未测 |
| D3 | Newton–Schulz 前怎么归一化 | 语义 | `muon_mcore_ns_normalize`（M17） | 看情况 | 未测 |
| D4 | Newton–Schulz 在 BF16 还是 FP32 | 类 11 | `muon_mcore_ns_bf16`（M30） | 不开 | +1.1% |
| D5 | Muon 分区与正交化单位 | 语义 | `muon_exclude_ihc` 等 | 按配方 | — |
| D6 | QK-Clip 缩放的分母 | 语义 | `qk_clip_mcore_denominator`（M18） | 看情况 | 未测 |
| D7 | QK-Clip 扫描层共用缩放 | bug | M1 | 已默认 | — |
| D8 | 梯度累积把最大 logit 求和 | bug | M19 | 已默认 | — |
| D9 | QK-Clip 最大 logit 含不含 sink | 语义 | `qk_clip_max_logit_exclude_sink`（M25） | 开 | +0.9% |
| D10 | 最大 logit 非正时开负数根号 | bug | M35 | 已默认 | — |
| E1 | `full` 重算在反向重选路由 | 语义 | `remat_policy: custom` 等（M26） | 开 | 无 |
| E2 | 路由偏置零均值更新 | 设置 | 两边一致（G14） | 两边都开 | 未测 |
| E3 | eod 掩码下 aux loss 的权重 | 语义 | M20–M22 | 看情况 | 未测 |
| E4 | 近似平局两边各落一边 | 离散选择 | 各自对 FP64 自选比 | 不涉及 | — |
| E5 | 并行、累积分组、分布式优化器 | 设置 | 两边一致 | 按配方 | — |
| E6 | 数据加载器约定不同 | 设置 | TPU 回放 GPU 的 batch | 不涉及 | — |
| F1 | XLA 默认允许 excess precision | 编译器 | 局部用 `reduce_precision` | 保持默认 | — |
| F2 | algsimp 把 HIGHEST einsum 改成逐元素 | 编译器 | 看优化后 HLO | 不涉及 | — |
| F3 | 融合本身改变数值 | 编译器 | — | 保持默认 | — |
| F4 | 关 pass 造不出无优化参照 | 编译器 | 参照用 CPU FP64 | 不涉及 | — |
| F5 | 单算子逐位不变不能证明开关无效 | 编译器 | 整网里验 | 不涉及 | — |
| G1 | 真值是 CPU FP64 整网重算 | 判法 | `step1_ref.py` | 不涉及 | — |
| G2 | 噪声尺子只用 GPU 重跑 | 判法 | GPU 8 份 28 对 | 不涉及 | — |
| G3 | r 只量差，不定责 | 判法 | 三种定责依据 | 不涉及 | — |
| G4 | GPU 噪声量不出舍入位置的差 | 判法 | 对 FP64 算 R | 不涉及 | — |
| G5 | 权重侧开关第 1 步测不到 | 量法 | 训练后权重重放 | 不涉及 | — |
| G6 | 判格点看位，不做往返 | 量法 | 看低 16 位 | 不涉及 | — |
| G7 | param_norm 两边都比精确值 | 量法 | J13 | 不涉及 | — |
| G8 | 误差小于一次 BF16 舍入时 R 不可信 | 判法 | 分辨率下限 | 不涉及 | — |
| G9 | 单算子回放要换掉全部输入 | 工具 | 注入覆盖关键字参数 | 不涉及 | — |
| G10 | grad_norm 相对差以均值作分母 | 量法 | (t − g) / 均值 | 不涉及 | — |
| G11 | 大文件核分块与整文件哈希 | 量法 | sha256 | 不涉及 | — |
| G12 | 第 1 步用自选路由判 | 判法 | 灌选择只作诊断 | 不涉及 | — |
| G13 | 路由一致率按对真值判 | 判法 | 翻转率的 R | 不涉及 | — |
| G14 | TPU 对 8 份 GPU 各判一遍 | 量法 | 只作诊断 | 不涉及 | — |
| G15 | top-k 翻转多不等于输出差 | 判法 | 翻转只报 | 不涉及 | — |
| G16 | 往返过不等于训练时载入 | 工具 | 第 1 步前逐参数自检 | 不涉及 | — |
| G17 | 单次运行的 r 在 1 附近不分好坏 | 判法 | 只看过 / 待定 / 不过与定责 | 不涉及 | — |

## 2. A 存储与舍入位置

同一个值在两边以什么精度落地、舍几次、在哪舍。

### A1 跨分片的部分和在 BF16 里相加

- **原理**：FSDP 切收缩维，XLA 的 collective matmul 在片间以 BF16 加部分和；TE GEMM 全程 FP32 累加、只舍一次。
- **量到**：混元 3 单算子 qkv 前向对 FP64：GPU 1.89e-3，TPU 2.33e-3，R 1.24；开后约 1.00。
- **开关与生产**：`dense_fp32_accumulation: true`（M12）。v5p 开（+0.8%）；v7x 看情况（混元 3 +1.9%）。（详见 [r2]、混元 3 #6）

### A2 RMSNorm 舍一次还是两次

- **原理**：MaxText 归一化、乘 scale 各舍一次 BF16；TE 在 FP32 里算完舍一次。默认写法中间那次在 excess precision 下被后端省掉，开关把单次舍入写死。
- **量到**：hy4-mini 单算子 RMSNorm 输出对 FP64：两次舍入 2.33e-3，单次 1.67e-3。
- **开关与生产**：`rmsnorm_single_rounding: true`（M9）。开，−0.2%。（详见 [r2]、混元 3 #7）

### A3 SwiGLU 激活舍一次还是两次

- **原理**：GPU 的 silu、乘 up 各舍一次 BF16；TPU 在 FP32 里算完舍一次，更准。
- **量到**：hy4-mini 第 1 步稠密 MLP 对 FP64：GPU 3.73e-3，TPU 关 2.87e-3、开 3.31e-3，R 0.77 / 0.89（关掉更近 FP64）。
- **开关与生产**：`mlp_activation_round_per_op: true`（M10）。不开（只为照 GPU），−0.1%。（详见 [r2]、混元 3 #8）

### A4 路由权重在 fc2 之前乘，还是在 wo 之后乘

- **原理**：GPU 在 fc2 之前乘路由权重再舍一次，各专家输出逐个加进 BF16；TPU 在 wo 之后的 FP32 加权和里乘。
- **量到**：1000 步主 loss 带符号均值（M41 开）：N 3.78e-6，TPU 7.20e-6 → 2.45e-6，r 1.91 → 0.65。
- **开关与生产**：`moe_mcore_prob_before_fc2: true`（M40）。看情况：对 FP64 未测。（详见 [r19]、[D25]）

### A5 SwiGLU 两个上投影的输入梯度舍一次还是两次（单算子已对上；整网单开变差，待查，见 U3）

- **原理**：GPU 的 gate | up 是一个融合 GEMM，dX 舍一次；TPU 两个矩阵乘各舍一次再相加。
- **量到**：CPU 单测 dX 对 GPU 写法：GPU 0，TPU 3.1e-3 → 0；但不开 A4 单开时整网变差（U3）。
- **开关与生产**：`mlp_fused_fc1_dgrad: true`（M41）。看情况：对 FP64 未测；单开时整网离 GPU 更远（U3）。（详见 [r20]、[D25]）

### A6 编译器省掉了 GPU 一定会做的 BF16 往返（机制是推断，见 U12）

- **原理**：GPU（eager）每次把 iHC 分支输入与更新后的流存成 BF16；TPU 写了 astype，但 excess precision 下融合内部这次往返可以不做（推断，HLO 未核）。
- **量到**：1000 步主 loss 带符号均值：N 3.78e-6，TPU 7.99e-6 → 2.87e-6，r 2.12 → 0.76。
- **开关与生产**：`ihc_force_bf16_rounding: true`（M36）。不开（只为照 GPU），+0.1%。**规则：要和 GPU 的舍入点一致，用 `reduce_precision`，不用 astype**（F5）。（详见 [r16]）

### A7 FP32 参数的梯度先以 BF16 落地

- **原理**：MaxText 的 norm scale、输出层、embedding 表以 BF16 参与计算，梯度先以 BF16 落地再累加；GPU 直接累加进 FP32。
- **量到**：hy4-mini embed 梯度对 FP64：GPU 中位数 4.17e-3，TPU 4.02e-3 → 3.83e-3，R 0.96 → 0.92。
- **开关与生产**：`embedding_grad_fp32` 开（无代价）；`rmsnorm_scale_grad_fp32` 开（代价待测；只在真实权重 B 上超线，R 中位数 1.60 → 0.98）；`logits_dense_wgrad_fp32` 看情况（混元 3 三个一起 +0.79%）。对齐配置都不开。（详见 [r22]、[r23]、[D22]、混元 3 #20）

### A8 GPU 参照自己把 q·k 先存成 BF16

- **原理**：janus 的 GPU eager 稀疏注意力把 FP32 的 q·k 先存成 BF16 再 softmax；生产融合内核与 TPU 都留 FP32。
- **量到**：第 1 步只改这一处，lm_loss 相对变化 8.1e-7（GPU 8 份第 1 步极差 2.1e-6）。
- **开关与生产**：GPU `--attn-scores-fp32`（G17，默认开）。不涉及 TPU。**参照一侧自己写的代码也要按精度边界查。**（详见 [r4]）

### A9 注意力缩放常数在 BF16 里舍过

- 混元 3 GPU 用精确的 1/√128，MaxText 等于乘舍过的 0.08837890625。300 步看不出，不处理。详见混元 3 #14。

### A10 splash 反向的 sink 梯度在 BF16 里累加（生产用 M42，对齐用 M42b；余量见 U1）

- **原理**：tokamax splash 反向把 p_sink 舍成 BF16 再在 BF16 里求和；GPU（autograd 穿过 FP32 softmax）p 与求和都在 FP32，只有 dp 舍一次。
- **量到**：hy4-mini 单层探针 sink 梯度对 FP64，最差的第 6 层：GPU 写法 1.51e-3，TPU 4.35e-3 → 3.03e-3（M42），R 2.88 → 2.00；M42b 后 R 1.000–1.001。
- **开关与生产**：生产开 M42 `splash_dsinks_fp32`（代价待测）；对齐开 M42b `splash_dsinks_gpu_order`（照 GPU 写法，+3.9%）。只开 M42 时整网离 GPU 更远（1000 步主 loss 带符号均值：N 3.78e-6，TPU 2.45e-6 → 4.99e-6，r 0.65 → 1.32）。（详见 [r21]、[D24]）

### A11 splash 的 base2 指数把注意力 logits 放大 1.0018 倍，q 多舍一次 BF16（只记录）

- **原理**：base2 模式下内核写 `q *= LOG2E`，q 是 BF16，JAX 把 Python 常数也转成 BF16：乘的是 1.4453125，比 log2(e) 大 1.81e-3；之后 exp2、换回自然底的 logsumexp 与 sink 都按精确值。结果是 logits（不含 sink）整体乘 1.0018，再加 q 多舍一次 BF16（前向、反向各一处）。系统性的那一半来自常数，和 logits 的大小成正比：mini 第 1 步全局最大 logit 0.129，看不出；真实权重截层 B 第 1 步逐头最大 logit 最大 14.5、中位数约 7，第 1 步主 loss 就偏低约 1e-4。
- **真实宽度**：06 截层 D3 第 1 步主 loss 对 FP64，TPU 只关这个开关 −9.87e-5 → −1.24e-6（R 1.85 → 0.02），CPU FP64 前向只加这一处放大事先预测移动 −9.69e-5；逐张量梯度对 FP64 的中位数 2.03e-3 → 1.72e-3，注意力一组 2.44e-3 → 1.78e-3。截层 B 只关这个开关，TPU −8.06e-5 → +7.3e-6。
- **量到**：hy4-mini 前向 logsumexp 对 FP64 最大绝对差：base2 8.5e-5–1.5e-4，自然指数 3.6e-6–4.3e-6；截层 B 第 0 层 `q_a_layernorm` 梯度 R 2.89 → 0.88。
- **开关与生产**：`sa_use_base2_exp`（默认 true，M43 只记录）。生产关（更准，无代价）。对齐配置仍开，关不关待定：mini 上关掉后整网离 GPU 更远（U20），真实宽度第 1 步关掉后对 FP64 与对 GPU 都更近（主 loss 对 GPU 的差 −1.52e-4 → −5.45e-5），D3 250 步回放关掉后对 GPU 9 项过 2 → 4、不过 5 → 3，主 loss 后段两臂相同、索引器 KD 变坏；06 建议只在真实宽度截层关（06 README「对齐配置要不要关 base2」）。（详见 [r24]、[D22]）

## 3. B 矩阵乘精度

TPU 矩阵单元原生 BF16 × BF16、FP32 累加：FP32 操作数默认先截成 BF16 乘一遍，`highest` 拆成几段分几遍乘。GPU 一侧要防 TF32 与张量核累加截断。

### B1 TPU 的 FP32 矩阵乘默认只算一遍 BF16（输出头）

- **原理**：默认精度下 FP32 权重先截成 BF16 再乘；GPU 输出头是真 FP32。
- **量到**：300 步主 loss 最大单步：N 4.53e-5，TPU 2.73e-4 → 2.75e-5，r 6.03 → 0.61。
- **开关与生产**：`logits_matmul_precision: highest`（M13）。看情况：开着输出头是真 FP32（对 FP64 更准），+23.7%（mid +24.4%），按精度与吞吐取舍。（详见 [r5]、[案例 13]）

### B2 索引器打分只算一遍 BF16

- **原理**：索引器打分在默认精度下单遍 BF16；原版是 `.float()`。
- **量到**：第 0 层教师 q 对 FP64：GPU 3.55e-3，TPU 4.16e-3 → 3.44e-3，R 1.17 → 0.97。
- **开关与生产**：`indexer_fp32_scores: true`（M23）。看情况，+6.3%。（详见 [r3]）

### B3 路由 logits 的精度决定选哪个专家

- **原理**：logits 精度影响平局处的 top-k；混元 3 两边路由输入相同时翻转为 0。
- **量到**：hy4-mini L1 路由 logits 对 FP64：GPU 1.13e-6，TPU 1.20e-7，R 0.106。
- **开关与生产**：TPU `float32_gate_logits` + `router_matmul_precision: highest`（已设），GPU `moe_router_dtype: fp32`。开：对 FP64 更准（默认 2.34e-3、HIGHEST 9.86e-8），步时波动内；混元 4 未做关掉对照（[r26]）。

### B4 乘 FP32 主权重还是 BF16 模型权重

- **原理**：GPU 索引器 `weights_proj`、路由 GEMM 乘 BF16 模型权重；MaxText 乘 FP32 主权重，`highest` 下低位参与计算。
- **量到**：训练后权重重放，`weights_proj` 输出对 FP64（BF16 权重）：关 9.6e-4–1.24e-3 → 开 8e-8。
- **开关与生产**：`indexer_weights_proj_bf16_weight`（M27）、`routed_gate_bf16_rounding`（M8）。不开（多舍一次）。M8 +0.1%，M27 +0.1%。（详见 [r5]、[案例 13]）

### B5 iHC 四路流加权和的相加顺序（机制待查，见 U11）

- **原理**：HIGHEST einsum 被改写成 FP32 逐元素乘加（F2），只差加法顺序：TPU 配对相加，GPU `torch.sum` 依次相加（G29）。
- **量到**：300 步 param_norm MARE：N 1.90e-7，TPU 3.04e-7 → 1.68e-7，r 1.60 → 0.88。
- **开关与生产**：`ihc_stream_sum_unrolled: true`（M28）。看情况：相加顺序同 GPU，对 FP64 更准未证实（U11），+0.2%。**规则：判断 HIGHEST 是不是多遍 BF16，看优化后 HLO 里还是不是 convolution。**（详见 [r6]）

### B6 权重在 BF16 格点上时，HIGHEST 与 default 逐位相同

- **原理**：从 BF16 检查点起训时主权重全在格点上，低位全 0，先舍再乘是恒等变换。
- **量到**：第 1 步只关 B1，输出头 logits 对 FP64 R 0.32 → 0.32，逐位相同；第 1 步更新后除嵌入外 99% 以上离开格点，盲区只有第 1 步前向。
- **开关与生产**：不涉及；怎么量见 G5。**详见**：[案例 13]。

### B7 GPU 上 TF32 悄悄打开

- **原理**：混元 3 容器设了 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，FP32 GEMM 进 TF32；只有 `NVIDIA_TF32_OVERRIDE=0` 连 TE 一起关。
- **量到**：混元 3 开着时 FP32 GEMM 对 FP64 2.94e-4。
- **开关与生产**：GPU 启动时 `tf32_probe` 实测，写进 `meta.json`。不涉及 TPU。（详见 混元 3 #3、类 9）

### B8 GPU 张量核的累加截断：长求和有系统偏差

- **原理**：B200 张量核每 16 个乘积对齐到最大指数，低位丢弃、写回朝零舍，同号长求和同向累积。
- **量到**：混元 3 输出头 dgrad 沿 ŵ̄（第 1 步）：GPU −6.51e-4，TPU −2.4e-5；GPU 分段累加后 L4 最差 r 0.85。
- **开关与生产**：GPU 分段累加（≤ 1024 一段，G19）。不涉及 TPU；混元 3 GPU +61%。（详见 混元 3 #1、#2，类 3）

## 4. C 超越函数与除法开方

TPU 默认的 FP32 超越函数、除法、开方是快速近似。GPU：CUDA 文档 exp 2、log 1 ulp；sigmoid / silu B200 实测最大 3.4 ulp。

| 函数（v5p，FP32） | 默认 ulp_max | 开关写法 | 开关 |
|---|---|---|---|
| exp / log | 105 / 62 | 1 / 3 | M32 |
| sigmoid / silu | 83 / 86 | 3 / 4 | M37 |
| 除法 / sqrt / 1 − β^t | 2 / 3 / 108 | 1 / 1 / 2 | M34 |
| rsqrt | 2 | — | 无 |

出处：[`tpu-elementwise-ulp.json`](../../models/hunyuan4/alignments/01-mini/results/tpu-elementwise-ulp.json)。

### C1 TPU 的 FP32 exp / log 是近似实现（第 1 步偏差已消失；多步 KD 仍有余量，见 U4、U15）

- **原理**：XLA 默认用快速近似的 FP32 exp / log；索引器 KD 是病态标量（C4），一格误差被放大成可见偏差。
- **量到**：300 步索引器 KD 带符号均值：N 1.38e-4，TPU −5.06e-4 → −1.89e-6，r 3.67 → 0.01。
- **开关与生产**：`transcendental_accuracy_highest: true`（M32、M33），同一开关管 C1–C3。看情况：开着近正确舍入，+3.0%，按精度与吞吐取舍。（详见 [r13]）

### C2 `lax.logistic` 带 HIGHEST 不起作用

- **原理**：v5p 上 `lax.logistic(accuracy=HIGHEST)` 与默认逐位相同；M37 改用 HIGHEST 的 exp 搭。
- **量到**：sigmoid / silu ulp_max 83 / 86 → 3 / 4（GPU 3.4）。
- **开关与生产**：同 C1 的开关。**开了开关要实测效果**。（详见 [r14]）

### C3 精确 sigmoid / silu 的反向式子要与 GPU 相同

- **原理**：GPU 反向用 PyTorch 解析式、FP32 里舍一次；C2 写法自动微分出的式子与舍入点不同。
- **量到**：1000 步路由偏置均值带符号均值：N 0.501，TPU 1.27 → 0.444，r 2.54 → 0.89。
- **开关与生产**：M39随 C1 的开关。**前向更准不等于整网更近，反向也要对上。**（详见 [r18]）

### C4 病态标量：一格误差被放大成大差

- **原理**：大项相减剩下的小量相对误差被放大：KD 的 KL 每行约 1.1e-4 nat、lse 约 7.1，lse 差 1 格就是 KL 的 4.3e-3。
- **怎么分辨**：输入与梯度两边一样近、只有这个标量对不上，就是病态；再查同一程序在 CPU 上偏不偏。
- **第二个例子**：KD 记录写法本身在 FP32 下病态，教师分布行和与学生 log q 的 1e-8 级偏差被放大（CPU 上同样偏高）；改记广义 KL（M50，只改记录值、梯度不变）后 1000 步 KD 由待定变过（U15）。
- **生产**：不涉及；M50 建议开（只改记录值）。差大不代表有 bug：条件数大时一格误差就被放大（这里是 C1）。**详见**：类 12；[根源一览第 25 行](../../models/hunyuan4/ROOT-CAUSES.md#r25)。

### C5 TPU 的 FP32 除法、sqrt、1 − β^t 不精确

- **原理**：Adam 的除法、sqrt、1 − β^t 在 TPU 上是 2 / 3 / 108 ulp；GPU 正确舍入。
- **量到**：优化器 8 步回放，第 5 步 `hc_head_base` 主权重增量对 FP64：GPU 1.49e-6，TPU 5.70e-6 → 1.49e-6，R 3.82 → 1.00。
- **开关与生产**：`muon_adam_accurate: true`（M34）。开，+0.1%。整网几项随之变差未解释（U10）。（详见 [r17]）

### C6 开关覆盖不到的

- log 用 HIGHEST 仍是 3 ulp，真实输入上 1 ulp（B200 实测 1；M51 精确写法 ≤ 1，默认关；已定责、已接受，[D14]）；rsqrt 2 ulp 无开关；注意力内核的 exp 不在开关内。
- **生产**：不涉及。**详见**：[r13]、台账 M32。

## 5. D 优化器语义

两种写法都合法。D1–D3、D6、E3 的「看情况」都指：已有在 GPU 上调好的超参时开。

### D1 Muon 的 Nesterov 做不做偏差修正

- **原理**：原版的 Nesterov 方向 β·mu_t + (1 − β)·g_t 不做偏差修正；MaxText 原用 optax 写法。
- **量到**：同一组梯度上逐步比 Megatron-Core，M15–M17 全开每步更新相对差 1.3e-6，全关 0.13。
- **开关与生产**：`muon_mcore_nesterov: true`（M15）。看情况。**详见**：[r7]。

### D2 标量参数用 Adam 还是 NAdam

- optax 的 Muon 组合里标量参数的 Adam 随 Nesterov 变成 NAdam，原版用普通 Adam。`muon_adam_plain: true`（M16）后对 torch Adam 逐步相对差 0.56 → ≤ 1e-5。看情况，同 D1。

### D3 Newton–Schulz 前怎么归一化

- MaxText x / (‖x‖ + eps)（eps 取 `adam_eps`），原版 x / max(‖x‖, 1e-7)。`muon_mcore_ns_normalize: true`（M17），单独开影响约 1e-6。看情况，同 D1。

### D4 Newton–Schulz 在 BF16 还是 FP32 里迭代

- **原理**：GPU 在 BF16 上迭代；TPU 原在 FP32，**更准**（关时 97% 的 Muon 张量步 TPU 更近 FP64）。
- **量到**：8 步回放，第 5 步第 0 层 `weights_proj` 增量对 FP64：GPU 1.23e-2，TPU 1.91e-2，R 1.55；开后 Muon 增量 TPU 对 GPU 最多差 12.8% → 1.6%。
- **开关与生产**：`muon_mcore_ns_bf16: true`（M30、M31）。不开（只为照 GPU），+1.1%。**详见**：[r10]。

### D5 Muon 的分区与正交化单位

- 走 Muon 还是 Adam、拼起来还是分开正交化，两边默认不同：`muon_exclude_ihc`（M3 / G5）、`muon_fuse_glu`（M4）、输出门按头正交化（M2 / G6）；更新缩放两边都是 √max(1, fan_out / fan_in)（G18）。分区逐参数对上。按训练配方，两边一致。

### D6 QK-Clip 缩放的分母

- MaxText min(1, τ / (max_logit + 1e-6))，原版不加 1e-6。`qk_clip_mcore_denominator: true`（M18），τ = 100 时相对影响 ≤ 1e-8。看情况，同 D1。

### D7 QK-Clip 在扫描层下各层共用一个缩放

- **原理**：层扫描时 max_logits 也对扫描维取最大，各层共用缩放，触发时裁错。
- **量到**：修后 11 层每头 √scale 与期望一致（rtol 1e-5），修前 4 条单测全不过。
- **开关与生产**：MaxText 修 bug（M1，无开关），已默认生效。**详见**：[r8]。

### D8 梯度累积把最大 logit 也求和了

- aux 按 microbatch 求和时最大 logit 被加成约累积步数倍；M19 改成取最大，已默认生效。**详见**：[r8]。

### D9 QK-Clip 的最大 logit 含不含 sink

- **原理**：splash 带 sink 时 running max 从 sink 起算；sink 不是 q·k，缩 q / k 降不了它，原版不含 sink。
- **量到**：T1 里 QK-Clip 从不触发（τ 100）；第 1 步单开关主 loss 逐位相同；但只关它的 300 步主 loss 带符号均值 0.48 → 0.93 倍（推断：编译结果变了）。
- **开关与生产**：`qk_clip_max_logit_exclude_sink: true`（M25），**必须与 D10 一起用**。开，+0.9%。**详见**：[r12]。

### D10 最大 logit 不是正数时开负数的根号

- **原理**：η = min(1, τ / s_max)，s_max 为负时乘 η^0.5 是负数开根号（D9 打开后拿到的正是它）。只在真实权重截层 B 上出现：第 1 步后 `wq_b` / `wkv_b` 变 NaN。
- **开关与生产**：M35（无开关）把这种头的缩放取 1，之后不再 NaN；已默认生效。原版 Megatron-Core 有同样隐患，未修。（详见 [r15]、[D20]）

## 6. E 训练语义

不是精度，是「算的是不是同一件事」。

### E1 `full` 重算在反向重选路由

- **原理**：`remat_policy: full` 下 MaxText 不读 `moe_routing_maps`，反向重算时重新选路由。
- **量到**：第 1 步 786,432 个槽位里 922 个（0.117%）与前向不同 → 0。
- **开关与生产**：`remat_policy: custom` + `moe_routing_maps: device`（M26）。开，无代价。（详见 [r5]）

### E2 路由偏置的零均值更新：原版 Megatron-Core 没有

- **原理**：TPU 每步把路由偏置更新减去均值，不减时共同分量漂移；减均值不改 top-k。
- **量到**：路由偏置均值带符号均值：N 0.799，差 15.3 → 0.135，r 19.1 → 0.17。
- **开关与生产**：对齐时两边都不减（G14）。生产**两边都开**（漂移吃掉偏置分辨率）；原版 Megatron-Core 不减均值。（详见 [r1]）

### E3 eod 掩码下 aux loss 的权重

- KD 按行数还是有效 token 加权、MTP loss 怎么重标、log 截断写法两边不同；M20–M22 后单测对原版一致（修前权重差估算 ≤ 2.4e-3）。看情况，同 D1。**详见**：[r9]。

### E4 路由与 DSA 的近似平局在两边各落一边

- **原理**：两种舍入把平局各推一边；翻转不累积。
- **量到**：混元 3 第 1 步每层 0.44%–0.56% 的路由槽位翻转，1000 步在噪声内；hy4-mini 第一行对 FP64 的 DSA 翻转 GPU 7,913 行、TPU 7,901 行。
- **生产**：不涉及。正式判定按各自的选择对 FP64 自选比（G12、G13）。**详见**：混元 3 #5、#17。

### E5 并行方式、累积分组、分布式优化器

- 混元 3 只换累积分组，embedding 梯度差到噪声上限的 31 倍。对齐时累积分组一致、golden 关分布式优化器；并行方式可以不同。按配方。**详见**：混元 3 #11–#13。

### E6 两边数据加载器的约定不同

- pad 处理与 microbatch 顺序两边不同。对齐时 TPU 原样回放 GPU 存下的每步 batch，输入逐位相同。不涉及。**详见**：混元 3 #16、台账 G9。

## 7. F 编译器行为

XLA 的改写会让舍入点和代码字面不一样。

### F1 XLA 默认允许 excess precision

- **原理**：融合内部的中间值可以不按代码写的 BF16 落地、留在 FP32（`embed_tokens` 实测：允许时等于 FP32 累加最后舍一次）。
- **量到**：hy4-mini 第 1 步只关它：205 个梯度 R 中位数 0.967 → 0.986；`embed_tokens` 梯度对 FP64：GPU 4.2e-3，TPU 4.0e-3 → 0.109（坏 27 倍）。
- **开关与生产**：**保持默认，不要全局关**；要照 GPU 舍入点的地方局部用 `reduce_precision`（A6），依赖多保留精度的地方用开关写死（A7）。（详见 [r22]、混元 3 #10）

### F2 algsimp 把 HIGHEST einsum 改写成 FP32 逐元素

- iHC 的 HIGHEST einsum（收缩维 4）：algsimp 把 54 个 dot 改写成逐元素乘加，不上矩阵单元，HIGHEST 的精度要以优化后 HLO 为准（[案例 11]）。

### F3 融合本身改变数值

- 只关 `tpu_fusion`：205 个梯度 R 中位数 0.967 → 0.995；步时 0.506 → 1.002 s。保持默认；第 1 步 iHC 的 R 随编译设置摆动（超线的在 11–16 个之间）。

### F4 关优化 pass 造不出「无优化参照」

- 全关编不过；关 `algsimp,tpu_fusion` 能跑但结果错：loss 只差 3e-5（相对），grad_norm 4.05（FP64 6.359）。参照用 CPU FP64（G1）；**loss 对得上不代表梯度对。**

### F5 单算子回放逐位不变，不能证明开关无效

- A6 的开关单算子回放逐位不变，整网里起作用（只关它的 300 步主 loss 带符号均值 N 3.01e-6，TPU 1.43e-6 → 3.03e-6，r 0.48 → 1.01）。推断单独编译时 XLA 没省那次往返（U12）。编译相关的开关要在整网编译里验。**详见**：[r16]。

## 8. G 测量方法

### G1 真值是 CPU 上的 FP64 整网重算，与被测不同源

- **怎么量**：[`step1_ref.py`](../../models/hunyuan4/frameworks/megatron/step1_ref.py) 在 CPU 上用 GPU golden 同一套代码全程 FP64 重算第 1 步；FP32 版离它 1.34e-4（GPU 误差的 1/51）；离散选择由 FP64 自己选。
- **为什么**：真值要比两边准一个量级以上，且不和任何一边共用写法。

### G2 噪声尺子只用 GPU 重跑

- N 取 GPU 同设置 8 份两两 28 对的最大值（第 1 步取三份两两最大 rel_L2）。TPU 同设置重跑逐位相同，不进尺子。判定只对一份 GPU golden，其余重跑只定 N。

### G3 r 只量两边差多大，不定责

- r 对两边对称，说不出哪边错。定责只认三种依据：① 对 CPU FP64 的 R；② 读代码，对照写明的标准；③ 只改一边后两边的差收拢。

### G4 GPU 重跑的噪声量不出舍入位置的差

- GPU 重跑之间舍入位置相同：hy4-mini 第 1 步 GPU 三份第 0、1 层索引器梯度逐位相同（N = 0）。所以第 1 步超出 N 的张量对 CPU FP64 算 R（G1）。

### G5 权重侧开关第 1 步测不到

- 只作用在权重一侧的开关（B1、B4）第 1 步是空操作（B6）：B1 第 1 步几乎不变，300 步关掉后主 loss 最大单步：N 4.53e-5，TPU 2.75e-5 → 2.73e-4，r 0.61 → 6.03。量法：训练后权重重放，或多步 A/B（[案例 13]）。

### G6 判断一个值在不在 BF16 格点上，看位

- 用 `x != f32(bf16(x))` 判，TPU 上那对 convert 在融合里被省掉，按 FP32 存的 iHC 也只判出 6.1e-5；看低 16 位是否全 0，第 0 步就是 1（[案例 13]）。

### G7 param_norm 两边都比精确值（真实漂移见 U16）

- GPU `param_norm_fp64`，TPU `param_norm_exact`（每 128 个元素一段 FP32 平方和、段间 FP64，J13）。FP32 全局求和对精确值逐步误差 −4.4e-7 至 +2.9e-7，与 N（1.90e-7）同量级（[案例 12]）。

### G8 误差小于一次 BF16 舍入时，R 量不出对错

- 第 1 层索引器 `k_norm.weight` 梯度（N = 0）对 FP64：GPU 6.69e-4，TPU 两代 1.07e-3 / 4.83e-4，R 1.60 / 0.72。两边都 < u = 2^-8 时 R 只是两种舍入之比，N = 0 的这类张量按分辨率下限过。

### G9 单算子回放要换掉全部输入

- GPU 抓到的同一份输入喂 TPU 单算子，对 FP64 算 R。注入覆盖位置参数与关键字参数，换不了就报错；否则用关键字传输入的算子会吃 TPU 自己上游的输入，误差里带着上游。

### G10 grad_norm 的相对差以两边均值作分母

- 用 (t − g) / ((t + g) / 2)。混元 3 grad_norm 逐步涨落约 ±3%，以一边作分母有约 +1e-3 的偏置（混元 3 #19）。

### G11 取回的大文件核分块与整文件哈希

- 分块取回时每块核字节数，拼好后核整文件 sha256，不一致整体失败。截断的块是整段 0、总大小照样对得上，会造出假的大 R：单个张量突然离 FP64 很远，先排除传输损坏。

### G12 第 1 步正式判定用 TPU 自选的路由；灌 GPU 的选择只作诊断

- 生产时 TPU 不回放 GPU 的选择，L3 用自选的路由与 top-k 对 FP64 比；灌 GPU 选择只用来拆差：hy4-mini 第 1 步约 2% token 选得不同，灌后 205 个梯度对 GPU rel_L2 中位数 4.25e-3 → 2.23e-3。

### G13 路由一致率：TPU 选得离真值是否比 GPU 远

- 按层统计自选的路由与 DSA top-k 对 FP64 的翻转率，R = TPU ÷ GPU 各份最大值，阈值同 L3；对 GPU 的翻转率照报不判（GPU 重跑前几层逐位相同，N 为 0）。hy4-mini 第 1、8、30、100 步各层 R 0.89–1.00，过。

### G14 TPU 对 8 份 GPU 各判一遍

- 诊断，不进判定：对 8 份 GPU 主 loss 带符号均值 r 中位数 0.51，GPU 留一 0.36；关 base2 后 2.06。

### G15 DSA top-k 翻转多，不等于输出差

- 第 1 步 GPU 对 FP64：2048 个稀疏 query 里 1936 个至少翻了一个（槽位 0.147%），灌 GPU 的 top-k 给 FP64 注意力，输出误差几乎不变（翻转在第 2048 名附近、权重很小）。要报，不能直接判不过。

### G16 转换往返通过不代表训练时载入了

- 训练入口没设载入路径时会随机建权重，往返检查照样全过；要在第 1 步前逐参数对检查点自检（GPU 台账 G7）。

### G17 单次运行的 r 在 1 附近不分好坏

- 只改一个按理不改数值的开关（M51，CPU 逐位相同），TPU 融合一变，第 1 步 10 个梯度张量就不逐位，1000 步主 loss 带符号均值：N 3.78e-6，TPU 1.36e-6 → 5.44e-6，r 0.36 → 1.44；诊断程序与训练步分开编译，第 1 步梯度相对差中位数 2.9e-4。

## 9. 还没解释清的

未定的以[根源一览](../../models/hunyuan4/ROOT-CAUSES.md#还没查明根因的)为准；编号 U 与上面各条独立，已解决的不列。

| # | 项 | 读数 | 状态 |
|---|---|---|---|
| U1 | sink 梯度余量 | mid R 1.23–1.47，真实权重 D1 第 0 层 R 2.66（mini 已收拢） | 带 M42b 的第 1 步没跑（[D24]、[D22]） |
| U2 | iHC 小参数梯度 | mid 第 1 步 3 个（mini 的 6 个已定责、已接受） | 带 M42b 的第 1 步没跑（[D11]） |
| U3 | A4 / A5 耦合 | 单开 M41 时 1000 步主 loss 带符号均值 N 3.78e-6，TPU 7.20e-6，r 1.91 | 前向逐位不变（[D25]） |
| U4 | 真实权重 D1 索引器 KD | 带符号均值 N 8.67e-4，TPU +2.01e-3，r 2.32 | 那一代没有 M50（[D21]） |
| U5 | grad_norm 待定 | 1000 步带符号均值 N 3.36e-4，TPU −4.14e-4，r 1.23 | 已接受：同一状态 6 次里 5 次 TPU 更近 FP64（[D2]） |
| U6 | 1000 步主 loss 余量 | 开 M42b 前后 MARE：N 8.24e-6，TPU 8.78e-6 → 8.05e-6，r 1.07 → 0.98 | 因果没证实 |
| U7 | 真实权重 D1 第 1 步主 loss | 对 FP64：GPU +9.24e-4，TPU −2.23e-4 | 差在 GPU 一侧（[D19]） |
| U8 | mid 第 1 步主 loss | 对 FP64：GPU +1.68e-4，TPU +2.8e-6 | 推断差在 GPU |
| U10 | C5 打开后整网几项变差 | — | 没拆开（[r17]） |
| U11 | B5 对第 1 步 iHC 梯度的改善 | 11 → 8 个 | 后一代没复现（[r6]） |
| U12 | A6 的往返被省掉 | — | 推断（[r16]） |
| U14 | 没量步时的开关 | A7 两个开关、M42、B3、D1–D3、D6、E2、E3 | — |
| U15 | 索引器 KD 记录值偏高 | M50 前后 1000 步带符号均值 N 1.83e-4，TPU +1.93e-4 → −1.27e-4，r 1.06 → 0.69 | 已修（C4、[r25]、[D1]）；GPU 记录值也偏高 |
| U16 | param_norm 真实漂移 | — | 现行配置没重做逐张量定位（[D12]） |
| U17 | 真实权重第 1 步逐张量不过 | B 16 个、D1 11 个 R > 1.2 | norm scale 见 A7，q latent norm 见 A11；其余待查（[D22]） |
| U18 | 路由偏置标准差待定 | 1000 步带符号均值 N 1.60e-3，TPU −1.82e-3，r 1.14 | 已接受：同一状态 35 处偏置更新方向里 TPU 与 FP64 不同 2 处、GPU 每份 1 处（[D23]） |
| U20 | 关 base2 让整网离 GPU 更远（mini） | 主 loss 带符号均值对 8 份 GPU 的 r 中位数 0.51 → 2.06；真实宽度 D3 方向相反：第 1 步关后对 GPU 的差 −1.52e-4 → −5.45e-5，250 步主 loss（1–250 步）r 11.2 → 7.32、9 项过 2 → 4 | [r24] |
| U21 | mid 300 步 L4 四项略超 | 主 loss 带符号均值 N 4.66e-6，TPU 6.41e-6，r 1.38 | 都在 2N 以内（[D27]） |

## 10. 怎么用这份清单

| 场景 | 做法 |
|---|---|
| 对齐 | TPU 照 [T1 配置](../../models/hunyuan4/frameworks/maxtext/hy4-mini-t1.yml)（含 B3、E1；A4、A5 一起开；A10 用 M42b；不含 A7 与关 base2），回放 GPU 的 batch；GPU 实测 TF32（B7）、dgrad 分段（B8）、自写内核查精度边界（A8）、关分布式优化器（E5）、存每步 batch（E6）、偏置更新一致（E2） |
| 生产 | 看第 1 节「生产建议」：对真值的误差加代价 |
| 新开关 | 前向、反向、配套开关整套对上再上整网，先在单算子上对 GPU 写法逐元素比（C2、A10 只修一半时整网离 GPU 更远） |
| 验证顺序 | L0 逐位 → L1（G9）→ L3（G1、G4、G8、G12、G13）→ 权重侧重放（G5）→ L4（G2、G17）；编译相关开关在整网里验（F5） |

## 证据索引

- [混元 4 根源一览](../../models/hunyuan4/ROOT-CAUSES.md)、[排查案例](../../models/hunyuan4/cases/README.md)、[比较协议](../../models/hunyuan4/PROTOCOL.md)
- 各尺寸报告：[mini](../../models/hunyuan4/alignments/01-mini/ALIGNMENT-REPORT.md)、[real-weight](../../models/hunyuan4/alignments/02-real-weight/ALIGNMENT-REPORT.md)、[mid](../../models/hunyuan4/alignments/03-mid/ALIGNMENT-REPORT.md)
- [混元 3 对齐报告](../../models/hunyuan3/ALIGNMENT-REPORT.md)（台账第 5 节、开关表第 6 节）
- [精度边界](precision-boundaries.md)、[判法](judging.md)

[r1]: ../../models/hunyuan4/ROOT-CAUSES.md#r1
[r25]: ../../models/hunyuan4/ROOT-CAUSES.md#r25
[r26]: ../../models/hunyuan4/ROOT-CAUSES.md#r26
[r2]: ../../models/hunyuan4/ROOT-CAUSES.md#r2
[r3]: ../../models/hunyuan4/ROOT-CAUSES.md#r3
[r4]: ../../models/hunyuan4/ROOT-CAUSES.md#r4
[r5]: ../../models/hunyuan4/ROOT-CAUSES.md#r5
[r6]: ../../models/hunyuan4/ROOT-CAUSES.md#r6
[r7]: ../../models/hunyuan4/ROOT-CAUSES.md#r7
[r8]: ../../models/hunyuan4/ROOT-CAUSES.md#r8
[r9]: ../../models/hunyuan4/ROOT-CAUSES.md#r9
[r10]: ../../models/hunyuan4/ROOT-CAUSES.md#r10
[r12]: ../../models/hunyuan4/ROOT-CAUSES.md#r12
[r13]: ../../models/hunyuan4/ROOT-CAUSES.md#r13
[r14]: ../../models/hunyuan4/ROOT-CAUSES.md#r14
[r15]: ../../models/hunyuan4/ROOT-CAUSES.md#r15
[r16]: ../../models/hunyuan4/ROOT-CAUSES.md#r16
[r17]: ../../models/hunyuan4/ROOT-CAUSES.md#r17
[r18]: ../../models/hunyuan4/ROOT-CAUSES.md#r18
[r19]: ../../models/hunyuan4/ROOT-CAUSES.md#r19
[r20]: ../../models/hunyuan4/ROOT-CAUSES.md#r20
[r21]: ../../models/hunyuan4/ROOT-CAUSES.md#r21
[r22]: ../../models/hunyuan4/ROOT-CAUSES.md#r22
[r23]: ../../models/hunyuan4/ROOT-CAUSES.md#r23
[r24]: ../../models/hunyuan4/ROOT-CAUSES.md#r24
[D1]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d1
[D2]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d2
[D11]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d11
[D12]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d12
[D13]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d13
[D15]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d15
[D23]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d23
[D24]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d24
[D25]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d25
[D14]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d14
[D27]: ../../models/hunyuan4/alignments/03-mid/ALIGNMENT-REPORT.md#d27
[D19]: ../../models/hunyuan4/alignments/02-real-weight/ALIGNMENT-REPORT.md#d19
[D20]: ../../models/hunyuan4/alignments/02-real-weight/ALIGNMENT-REPORT.md#d20
[D21]: ../../models/hunyuan4/alignments/02-real-weight/ALIGNMENT-REPORT.md#d21
[D22]: ../../models/hunyuan4/alignments/02-real-weight/ALIGNMENT-REPORT.md#d22
[案例 11]: ../../models/hunyuan4/cases/11-ihc-stream-sum-highest-einsum.md
[案例 12]: ../../models/hunyuan4/cases/12-param-norm-fp32-sum-and-drift.md
[案例 13]: ../../models/hunyuan4/cases/13-weight-side-switch-step1-blind-spot.md
