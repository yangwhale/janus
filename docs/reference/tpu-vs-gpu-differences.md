[janus](../../README.md) › [文档](../README.md) ｜ TPU 与 GPU 训练数值哪里不一样：逐条清单

# TPU 与 GPU 训练数值哪里不一样：逐条清单

测过哪些取值、哪些没测（尺寸、片数、切分、步数、版本、检查点、数据管线等），见[混元 4 mini 报告「适用范围」](../../models/hunyuan4/alignments/01-mini/ALIGNMENT-REPORT.md#适用范围测过什么没测什么)与[混元 3 报告开头的「适用范围」](../../models/hunyuan3/ALIGNMENT-REPORT.md)。

**现状**（hy4-mini，以现行比较 `tc-a` 为准，判定见[总览](../../models/hunyuan4/dashboard/dashboard.md)）：

| 配置 | 主 loss | L4 判定 | 步时 |
|---|---|---|---|
| 默认（对齐开关全关） | 300 步最大单步：N（GPU 重跑波动上限，8 份 28 对）4.53e-5，TPU 2.68e-4，r 5.91 | 不过 | 基准 |
| 按本清单逐条处理后（现行 `tc-a`，C 口径：M36、M52、M58 都开，base2 关） | 1000 步带符号均值：N（16 份 120 对）4.65e-6，TPU +3.53e-6，r 0.76；三个量里最差的是 MARE，r 0.98；第 101 步起 0.97 | 9 项过 8、待定 1（路由偏置均值 r 1.31，未定责；只改执行方式的另 4 份 0.39–0.80，`tc-e` 0.36，它是 full 重算、反向重选路由，不算同配置；[`t1zc-repeats-6.json`](../../models/hunyuan4/alignments/01-mini/results/t1zc-repeats-6.json)）；grad_norm 0.79、路由偏置标准差 0.85 都过。前 300 步另判一套，也是过 8、待定 1：grad_norm r 1.01（U5；[`t1zc300-l4`](../../models/hunyuan4/alignments/01-mini/results/t1zc300-l4.report.md)）。C 口径 6 份（`tc-a`…`tc-f`）都判，都没有不过；主 loss `tc-b` r 1.14、grad_norm `tc-f` 最大单步 r 1.04 待定（MARE r 0.95，过）（[`t1zc-repeats-6.json`](../../models/hunyuan4/alignments/01-mini/results/t1zc-repeats-6.json)） | 全开比全关 +44.6%（不含 M42b、M50、M52、M58；mid +46.8%，量法不同），大头 B1。C 口径整体 1000 步步时中位数 0.4963 s，只开 M36 的对照 0.4960 s（对照 `ti-o-a`，两份 MaxText 提交不同：`8d3d7561` / `751fdc6d`；单次运行，不是 A/B；[`mini-step-time.json`](../../models/hunyuan4/alignments/01-mini/results/mini-step-time.json)） |
| C 口径之前（`tz-off-rep`，只开 M36，base2 关） | 1000 步带符号均值 TPU +7.21e-6，r 1.55；第 101 步起 1.52 | 9 项过 7、待定 2（主 loss 全程与第 101 步起；主因后来定到 M58，见 U22） | — |

读数出自 [`t1zc-l4`](../../models/hunyuan4/alignments/01-mini/results/t1zc-l4.report.md)，C 口径之前那一行出自 [`t1zo-l4`](../../models/hunyuan4/alignments/01-mini/results/t1zo-l4.report.md)。更早一代（base2 开的 `t1zg-a`，8 份尺子：主 loss MARE r 0.98，grad_norm、路由偏置标准差待定）见[历代比较](../../models/hunyuan4/dashboard/history.md)。下文各条的「量到」多数是 base2 开那几代、8 份尺子下的单开关 A/B：1000 步主 loss 带符号均值写 N 3.78e-6 的都是 8 份 28 对的上限，16 份 120 对时是 4.65e-6。

**范围**：GPU B200 × 8（混元 4 Megatron-Core 0.20 + TE，混元 3 Megatron-Bridge v0.7.0）；TPU MaxText（v5p 2x2x2，混元 3 另有 v7x-4；混元 4 的 v7x 只有 TPU 自比，外加一次 1000 步直接对 GPU 16 份，9 项都在噪声内，用的是 C 口径之前的配置（同 TPU 自比 golden，没有 M52、M58）、base2 开、libtpu 0.0.46，现行 C 口径没在 v7x 上跑过，见 [04 README](../../models/hunyuan4/alignments/04-tpu-self/README.md)「v7x 直接对 GPU 比」；libtpu 0.0.46 上 v7x 默认配置第 1 步 `hc_head` 梯度偏一个常数（0.0.52 上换成纯数据并行不累积那个程序偏），靠 TPU 自比的 iHC 豁免才不挡判档，见 [A16](#a16-hc_head输出头反向-kernel-合并末层-rmsnorm-反向行和时省掉一次-bf16-舍入)），开关在 `maxtext` 混元 4 分支、默认关；BF16 混合精度、FP32 主权重、无 FP8。数字以 hy4-mini 为主。例外：06 真实宽度三层（A11、U25 引的回放）是 GPU B200 × 16（2 台，DP 16 + EP 16，多机代码包）对 TPU v5p 2x2x4（FSDP 16），见 [06 setup](../../models/hunyuan4/alignments/06-rw3/setup.md)。

**适用条件**：现行 mini 的读数对应 MaxText `8d3d7561`、jax 0.11.1、libtpu 0.0.46（`tc-a` 的 `meta.json`；C 口径之前的 `tz-off-rep` 是 MaxText `b87a55aa`，jax、libtpu 相同）；GPU 数据并行 8 + 专家并行 4、整层重算，TPU FSDP 8；序列 4096、全局批 16、1000 步、hy4-mini 随机初始化。换 libtpu 版本、换 TPU 一侧的并行切法都可能改结论（[F7](#f7-libtpu-版本改的不只是求和顺序)、[A12](#a12-纯数据并行时词嵌入梯度在-bf16-里-scatter不排序)、[A14](#a14-上下文并行时-sink-梯度在-bf16-里算float32_logits-没接到-mla)），结论只对实测过的组合成立。

**读法**（完整见[混元 4 总览细节「读法」](../../models/hunyuan4/dashboard/detail.md#读法)）：

| 术语 | 含义 | 判定线 |
|---|---|---|
| r | TPU 对 GPU 的差 ÷ N（噪声上限：GPU 同设置重跑两两配对的最大值；mini 前 300 步 8 份 28 对，1000 步 2026-11-11 起 16 份 120 对，之前 8 份 28 对，见[尺子 1.5](../guide/ruler.md#15-噪声尺子-n)），不定责；取带符号均值、MARE（逐步相对差绝对值的平均）、最大单步里最差的 | ≤ 1 过，1–2 待定，> 2 不过 |
| R | TPU 对 CPU FP64 的误差 ÷ GPU 的同一量，能定责 | ≤ 1.2 过 |
| 生产建议 | 开（修 bug，或更准且代价 < 1%）／不开（只为照 GPU）／看情况（取舍或未查清）／不涉及 | — |
| 代价 | hy4-mini 300 步单开关 A/B：全开步时 ÷ 只关该开关的步时 − 1 | — |
| 编号 | M / G / J：[台账](../../models/hunyuan4/CHANGES-LEDGER.md)；[rn]：[根源一览](../../models/hunyuan4/ROOT-CAUSES.md)第 n 行；混元 3 #n：[混元 3 报告](../../models/hunyuan3/ALIGNMENT-REPORT.md)第 5 节；类 n：[精度边界](precision-boundaries.md) | — |

## 1. 总表

| # | 差异一句话 | 边界类别 | 修法 / 开关 | 生产建议 | 代价 |
|---|---|---|---|---|---|
| A1 | 跨分片部分和在 BF16 里相加 | 类 2 | `dense_fp32_accumulation`（M12） | 开（v7x 看情况） | +0.8% |
| A2 | RMSNorm 舍一次还是两次 | 类 4 | `rmsnorm_single_rounding`（M9） | 开（libtpu 0.0.52 上不开不能忍） | −0.2% |
| A3 | SwiGLU 激活舍一次还是两次 | 类 5 | `mlp_activation_round_per_op`（M10） | 不开 | −0.1% |
| A4 | 路由权重在 fc2 前还是 wo 后乘 | 舍入位置 | `moe_mcore_prob_before_fc2`（M40） | 看情况（对 FP64 未测） | +0.2% |
| A5 | SwiGLU 输入梯度舍一次还是两次 | 舍入次数 | `mlp_fused_fc1_dgrad`（M41） | 看情况（对 FP64 未测） | −1.8% |
| A6 | 编译器省掉 GPU 必做的 BF16 往返（HLO 里 convert 在，后端省） | 编译器（类 15） | `ihc_force_bf16_rounding`（M36） | 不开 | +0.1% |
| A7 | FP32 参数的梯度先以 BF16 落地 | 类 10 | 三个梯度 FP32 开关 | embed、norm 开；输出层看情况 | 部分未量 |
| A8 | GPU 参照把 q·k 先存成 BF16 | 类 1 | GPU `--attn-scores-fp32`（G17） | 不涉及 | — |
| A9 | 注意力缩放常数在 BF16 里舍过 | 舍入位置 | 不处理 | 不涉及 | — |
| A10 | splash 反向 sink 梯度在 BF16 里累加 | 类 3 | M42；对齐 M42b | 开 M42 | M42b +3.9% |
| A11 | splash base2 指数把注意力 logits 放大 1.0018 倍、q 多舍一次 | 舍入位置（常数舍成 BF16） | `sa_use_base2_exp: false`（M43） | 关 base2 | 无（用 base2 反而慢 0.1%）；对齐配置 2026-11-11 起也关 |
| A12 | 纯数据并行时词嵌入梯度在 BF16 里 scatter、不排序 | 编译器（后端排不排序；类 15） | `embedding_shard_over_data`（M54，与 ZeRO-1 互斥）或 `embedding_grad_fp32`（梯度累积下要带 M55，否则静默丢梯度） | 用纯数据并行就开 | M54 步时不增加 |
| A13 | hc_head 收拢之后没舍 BF16 | 编译器（类 15） | `hc_head_force_bf16_rounding`（M52；对齐配置开） | 不开 | 未测 |
| A17 | iHC 流梯度三份各舍 BF16、按 autograd 顺序相加（推断 TPU 在 FP32 里加完才舍，没看 HLO / LLO） | 编译器（后端省）；梯度在 BF16 里多次相加（类 16） | `ihc_stream_grad_gpu_order`（M58；对齐配置开） | 不开 | 未单独测 |
| A14 | 上下文并行时 sink 梯度在 BF16 里算 | 类 1 | `float32_logits`（+ `float32_qk_product`） | 用上下文并行就开 | +12.4%（上下文并行 2） |
| A15 | GPU 参照的索引器 RoPE 在 BF16 里算 | 舍入位置 | 不修（GPU 一侧） | 不涉及 | — |
| A16 | hc_head：输出头反向 kernel 合并末层 RMSNorm 反向行和时省掉一次 BF16 舍入 | 编译器（后端省；类 15） | 待拍板（候选：对余切 dy 加 `optimization_barrier` 或 `reduce_precision`）；`hc_head_dgrad_fusion.py` 离线自查 | 待拍板 | 未测 |
| B1 | FP32 矩阵乘默认单遍 BF16（输出头） | 类 7 | `logits_matmul_precision: highest`（M13） | 看情况 | +23.7% |
| B2 | 索引器打分单遍 BF16 | 类 7 | `indexer_fp32_scores`（M23） | 看情况 | +6.3% |
| B3 | 路由 logits 精度决定选哪个专家 | 类 6 | `float32_gate_logits` 等 | 开 | 波动内 |
| B4 | 乘 FP32 主权重还是 BF16 权重 | 类 8 | M27、M8 | 不开；M27 要不要改成开待拍板（组合里关 M27 时前 300 步 KD 不过，开回后 r 4.91 → 1.44） | ≤ +0.1% |
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
| E5 | 并行、累积分组、分布式优化器 | 设置 | 累积分组一致；TPU 换切法见 A12、A14 | 按配方 | — |
| E6 | 数据加载器约定不同 | 设置 | TPU 回放 GPU 的 batch | 不涉及 | — |
| F1 | XLA 默认允许 excess precision | 编译器 | 局部用 `reduce_precision` | 保持默认 | — |
| F2 | algsimp 把 HIGHEST einsum 改成逐元素 | 编译器 | 看优化后 HLO | 不涉及 | — |
| F3 | 融合本身改变数值 | 编译器 | — | 保持默认 | — |
| F4 | 关 pass 造不出无优化参照 | 编译器 | 参照用 CPU FP64 | 不涉及 | — |
| F5 | 单算子逐位不变不能证明开关无效 | 编译器 | 整网里验 | 不涉及 | — |
| F6 | 关小矩阵乘改写，RMSNorm 乘法走单遍 BF16 | 编译器 | 保持默认开 | 保持默认 | 关了更慢 |
| F7 | libtpu 版本改的不只是求和顺序 | 后端 | 换版本先建 golden、重测 | 钉版本 | — |
| F8 | jit 静默算错：DUS 和读同一缓冲的算子被合进一颗多输出 fusion（v5p，实测 libtpu 0.0.46、0.0.48、0.0.51、0.0.52 都错） | 编译器 bug | `hlo_mof_dus_scan.py` 自查；`optimization_barrier` 隔开读写，或关 multi-output fusion | 扫到特征就修那一处 | 屏障：只挡一处；全局关：−2.4% ～ +0.5%（MaxText 两个模型） |
| G1 | 真值是 CPU FP64 整网重算 | 判法 | `step1_ref.py` | 不涉及 | — |
| G2 | 噪声尺子只用 GPU 重跑 | 判法 | GPU 重跑（1000 步 16 份 120 对） | 不涉及 | — |
| G3 | r 只量差，不定责 | 判法 | 三种定责依据 | 不涉及 | — |
| G4 | GPU 噪声量不出舍入位置的差 | 判法 | 对 FP64 算 R | 不涉及 | — |
| G5 | 权重侧开关第 1 步测不到 | 量法 | 训练后权重重放 | 不涉及 | — |
| G6 | 判格点看位，不做往返 | 量法 | 看低 16 位 | 不涉及 | — |
| G7 | param_norm 两边都比精确值 | 量法 | J13 | 不涉及 | — |
| G8 | 误差小于一次 BF16 舍入时 R 不可信 | 判法 | 分辨率下限 | 不涉及 | — |
| G9 | 单算子回放要换掉全部输入 | 工具 | 注入覆盖关键字参数 | 不涉及 | — |
| G10 | grad_norm 相对差的分母 | 量法 | 混元 4 以参照作分母；混元 3 用均值 | 不涉及 | — |
| G11 | 大文件核分块与整文件哈希 | 量法 | sha256 | 不涉及 | — |
| G12 | 第 1 步用自选路由判 | 判法 | 灌选择只作诊断 | 不涉及 | — |
| G13 | 路由一致率按对真值判 | 判法 | 翻转率的 R | 不涉及 | — |
| G14 | TPU 对每份 GPU 各判一遍 | 量法 | 只作诊断 | 不涉及 | — |
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
- **版本**：TPU 自比里不开它的影响随 libtpu 变：0.0.46 上曲线 r 1.06，能忍；0.0.52 上主 loss 系统性偏高，r 5.50（1000 步 3.68），不能忍。机理见 [impacts/04](../../models/hunyuan4/alignments/04-tpu-self/impacts/04-rmsnorm-double-rounding.md)。
- **开关与生产**：`rmsnorm_single_rounding: true`（M9）。对齐、生产都开，−0.2%。（详见 [r2]、混元 3 #7）

### A3 SwiGLU 激活舍一次还是两次

- **原理**：GPU 的 silu、乘 up 各舍一次 BF16；TPU 在 FP32 里算完舍一次，更准。
- **量到**：hy4-mini 第 1 步稠密 MLP 对 FP64：GPU 3.73e-3，TPU 关 2.87e-3、开 3.31e-3，R 0.77 / 0.89（关掉更近 FP64）。
- **开关与生产**：`mlp_activation_round_per_op: true`（M10）。不开（只为照 GPU），−0.1%。（详见 [r2]、混元 3 #8）

### A4 路由权重在 fc2 之前乘，还是在 wo 之后乘

- **原理**：GPU 在 fc2 之前乘路由权重再舍一次，各专家输出逐个加进 BF16；TPU 在 wo 之后的 FP32 加权和里乘。
- **量到**：1000 步主 loss 带符号均值（M41 开）：N 3.78e-6，TPU 7.20e-6 → 2.45e-6，r 1.91 → 0.65。这只是开、关各一份 1000 步（8 份尺子）：同一配置只改执行方式的 4 份重复，主 loss 带符号均值的 r 就能在 0.36–2.08 之间（base2 开；关 0.62–1.91。按同一把 8 份尺子推算：[`mini-base2-loo.json`](../../models/hunyuan4/alignments/01-mini/gpu-noise/results/mini-base2-loo.json) 的 `loss_signed_mean` ÷ N 3.78e-6），base2 开关 r 挪 1.55 也已查明是轨迹偶然（见 A11），这里挪 1.26 分不清是开关还是轨迹。
- **开关与生产**：`moe_mcore_prob_before_fc2: true`（M40）。看情况：对 FP64 未测。（详见 [r19]、[D25]）

### A5 SwiGLU 两个上投影的输入梯度舍一次还是两次（单算子已对上；整网单开的效果没分清，见 U3）

- **原理**：GPU 的 gate | up 是一个融合 GEMM，dX 舍一次；TPU 两个矩阵乘各舍一次再相加。
- **量到**：CPU 单测 dX 对 GPU 写法：GPU 0，TPU 3.1e-3 → 0；不开 A4 单开时整网那一份离 GPU 更远（U3）。
- **开关与生产**：`mlp_fused_fc1_dgrad: true`（M41）。看情况：对 FP64 未测；单开时整网离 GPU 更远（U3）。「更远」只基于开、关各一份 1000 步：同一配置只改执行方式的 4 份重复，主 loss 带符号均值的 r 就能在 0.36–2.08 之间（base2 开；关 0.62–1.91。按同一把 8 份尺子推算：[`mini-base2-loo.json`](../../models/hunyuan4/alignments/01-mini/gpu-noise/results/mini-base2-loo.json) 的 `loss_signed_mean` ÷ N 3.78e-6），一份运行分不清是开关还是轨迹。（详见 [r20]、[D25]）

### A6 编译器省掉了 GPU 一定会做的 BF16 往返（后端省；长跑归因见 U12）

- **原理**：GPU（eager）每次把 iHC 分支输入与更新后的流存成 BF16；TPU 写了 astype，excess precision 下这次往返在后端被省掉一部分：融合内部的来回 convert 不产生舍入指令，跨融合边界的值按声明的 BF16 写回内存（实验确认）。训练步优化后 HLO 里 convert(bf16)→convert(f32) 都在（14 个前向融合逐一核过），所以 HLO 看不出；v5p 最小复现的 LLO 里，RMSNorm 均方那颗 kernel 用的是没舍的 FP32（`vpack.i.bf16` 0 条），归一化那颗读的是舍过的 BF16，同一个值两处不一致；加 `reduce_precision` 后两处都真舍（[TPU 自比影响逐件「iHC 不强制舍成 BF16」](../../models/hunyuan4/alignments/04-tpu-self/impacts/05-ihc-bf16-rounding.md)）。
- **量到**：1000 步主 loss 带符号均值：N 3.78e-6，TPU 7.99e-6 → 2.87e-6，r 2.12 → 0.76。这条曲线也只是开、关各一份（8 份尺子），挪 1.36 比 base2 那次轨迹偶然的 1.55 还小，单看曲线分不清是开关还是轨迹（见 A4）；开关改了舍入点：最小复现的 LLO 已核（上面）；整网训练步的后端没核，整网里起作用的证据是 TPU 自比只关它时第 1 步 `k_norm` 梯度变远（F5）。
- **开关与生产**：`ihc_force_bf16_rounding: true`（M36）。不开（只为照 GPU），+0.1%。**规则：要和 GPU 的舍入点一致，用 `reduce_precision`，不用 astype**（F5）。hc_head 收拢后是同一类，见 A13。（详见 [r16]）

### A7 FP32 参数的梯度先以 BF16 落地

- **原理**：MaxText 的 norm scale、输出层、embedding 表以 BF16 参与计算，梯度先以 BF16 落地再累加；GPU 直接累加进 FP32。
- **量到**：hy4-mini embed 梯度对 FP64：GPU 中位数 4.17e-3，TPU 4.02e-3 → 3.83e-3，R 0.96 → 0.92。
- **开关与生产**：`embedding_grad_fp32` 开（无代价；数据并行＋梯度累积下要 MaxText `d5c9b113`（[台账](../../models/hunyuan4/CHANGES-LEDGER.md) M55）之后的代码：之前的代码里三者同时成立时，XLA 把 FP32 词嵌入梯度的 all-reduce 挪出累积循环并清零累加器，静默丢掉前面微批的梯度（第 1 步词嵌入梯度范数只有正确值的 0.51 倍，300 步 r 611）；移植这个开关要连 M55 一起移植）；`rmsnorm_scale_grad_fp32` 开（代价待测；只在真实权重 B 上超线，R 中位数 1.60 → 0.98）；`logits_dense_wgrad_fp32` 看情况（混元 3 三个一起 v5p +0.79%、v7x +1.6%）。对齐配置都不开。另有 M48 `rmsnorm_scale_grad_gpu_order`（dγ 照 GPU 的顺序求和），CPU 上逐位对上，TPU 整网结果作废、待查，不进任何配置（[台账](../../models/hunyuan4/CHANGES-LEDGER.md) M48）。（详见 [r22]、[r23]、[D22]、混元 3 #20）

### A8 GPU 参照自己把 q·k 先存成 BF16

- **原理**：janus 的 GPU eager 稀疏注意力把 FP32 的 q·k 先存成 BF16 再 softmax；生产融合内核与 TPU 都留 FP32。
- **量到**：第 1 步只改这一处，lm_loss 相对变化 8.1e-7（GPU 8 份第 1 步极差 2.1e-6）。
- **开关与生产**：GPU `--attn-scores-fp32`（G17，默认开）。不涉及 TPU。**参照一侧自己写的代码也要按精度边界查。**（详见 [r4]）

### A9 注意力缩放常数在 BF16 里舍过

- 混元 3 GPU 用精确的 1/√128，MaxText 等于乘舍过的 0.08837890625。300 步看不出，不处理。详见混元 3 #14。

### A10 splash 反向的 sink 梯度在 BF16 里累加（生产用 M42，对齐用 M42b；余量见 U1）

- **原理**：tokamax splash 反向把 p_sink 舍成 BF16 再在 BF16 里求和；GPU（autograd 穿过 FP32 softmax）p 与求和都在 FP32，只有 dp 舍一次。
- **量到**：hy4-mini 单层探针 sink 梯度对 FP64，最差的第 6 层：GPU 写法 1.51e-3，TPU 4.35e-3 → 3.03e-3（M42），R 2.88 → 2.00；M42b 后 R 1.000–1.001。
- **开关与生产**：生产开 M42 `splash_dsinks_fp32`（代价待测）；对齐开 M42b `splash_dsinks_gpu_order`（照 GPU 写法，+3.9%）。只开 M42 时整网离 GPU 更远（1000 步主 loss 带符号均值，8 份尺子：N 3.78e-6，TPU 2.45e-6 → 4.99e-6，r 0.65 → 1.32）；这也只是开、关各一份，只改执行方式的重复 r 就能差这么多（见 A5），不能据此说 M42 让整网变差。（详见 [r21]、[D24]）

### A11 splash 的 base2 指数把注意力 logits 放大 1.0018 倍，q 多舍一次 BF16（生产与对齐都关）

- **原理**：base2 模式下内核写 `q *= LOG2E`，q 是 BF16，JAX 把 Python 常数也转成 BF16：乘的是 1.4453125，比 log2(e) 大 1.81e-3；之后 exp2、换回自然底的 logsumexp 与 sink 都按精确值。结果是 logits（不含 sink）整体乘 1.0018，再加 q 多舍一次 BF16（前向、反向各一处）。系统性的那一半来自常数，和 logits 的大小成正比：mini 第 1 步全局最大 logit 0.129，看不出；真实权重截层 B 第 1 步逐头最大 logit 最大 14.5、中位数约 7，第 1 步主 loss 就偏低约 1e-4。
- **真实宽度**：06 截层 D3 第 1 步主 loss 对 FP64，TPU 只关这个开关 −9.87e-5 → −1.24e-6（R 1.85 → 0.02），CPU FP64 前向只加这一处放大事先预测移动 −9.69e-5；逐张量梯度对 FP64 的中位数 2.03e-3 → 1.72e-3，注意力一组 2.44e-3 → 1.78e-3。截层 B 只关这个开关，TPU −8.06e-5 → +7.3e-6。
- **量到**：hy4-mini 前向 logsumexp 对 FP64 最大绝对差：base2 8.5e-5–1.5e-4，自然指数 3.6e-6–4.3e-6；截层 B 第 0 层 `q_a_layernorm` 梯度 R 2.89 → 0.88。
- **开关与生产**：`sa_use_base2_exp`（MaxText 默认 true，M43）。生产关（更准，无代价）。对齐配置 2026-11-11 起所有尺寸都关（四份对齐 yml 都是 `false`，PROTOCOL 变更记录同日）：真实宽度第 1 步关掉后对 FP64 与对 GPU 都更近（主 loss 对 GPU 的差 −1.52e-4 → −5.45e-5），D3 250 步回放关掉后对 GPU 9 项过 2 → 4、不过 5 → 3（主 loss 后段两臂相同、索引器 KD 变坏，是 GPU 一侧偏高），1000 步回放过 2 → 3、不过 3 → 0（主 loss 1–1000 步 r 1.87 → 1.20，后段 1.24 → 1.02，都是三量最差，后段关那臂由最大单步领头、带符号均值是 0.86，[`rw3-1k-replay.json`](../../models/hunyuan4/alignments/06-rw3/results/rw3-1k-replay.json)）；mini 上「关掉后整网离 GPU 更远」查明是轨迹偶然（开、关各 4 份，16 份尺子下主 loss r 开 0.90–1.69、关 0.93–1.55，两组交叠，[D30]）。TPU 自比的 golden `ts-g-a` 仍是开 base2 的一代，不改（06 README「对齐配置要不要关 base2」）。（详见 [r24]、[D22]）

### A12 纯数据并行时词嵌入梯度在 BF16 里 scatter，不排序

- **原理**：词嵌入的反向是对 BF16 表的一次 scatter-add。后端在一次 scatter 的下标数超过词表行数的 1/8 时插排序（libtpu 0.0.46 上二分所得），排好序时同一个词在片上 FP32 缓冲里加完只舍一次，不排序时加一次舍一次。FSDP 下每张卡收 8 张卡的 32768 个下标，过线；纯数据并行每张卡只有自己的 4096 个，不过线，高频词的梯度在 BF16 里累积误差。
- **量到**：TPU 自比（对 TPU golden，FSDP 8）纯数据并行曲线 r 2.62，第 1 步词嵌入梯度 R 3.62（libtpu 0.0.46）；0.0.52 上 r 2.82、R 3.60。三种修法都和 GPU 重跑分不开（在 GPU 留一法最大值以内），但 0.0.52 上三种修法前 300 步主 loss 都同向偏高（r 1.55 / 1.20 / 1.23，0.0.46 上三份都略低），扣掉 golden 自身的差后仍比 0.0.46 上同一修法高约 2.4e-6–4.8e-6，原因没查（U26）；修法① 1000 步回到 0.92（[`libtpu-052.md`](../../models/hunyuan4/alignments/04-tpu-self/libtpu-052.md)）。
- **版本**：门槛随 libtpu、列宽和芯片代际变；0.0.52 上 v5p 这两种并行排不排序不变，v7x 上纯数据并行不做梯度累积时开始排序（mini 的表 120832 × 1024，每卡 8192 个下标；[`libtpu-052.md`](../../models/hunyuan4/alignments/04-tpu-self/libtpu-052.md)）。
- **生产形状**（词表 120832、隐藏维 6144，纯数据并行每卡整张表；单算子离线编译，[`emb-scatter-sort-hy4-prod.json`](../../models/hunyuan4/alignments/04-tpu-self/results/emb-scatter-sort-hy4-prod.json)）：0.0.52 上 v5p、v7x 每卡 4096 / 8192 / 16384 个下标都不排序（门槛 v5p 65536 以内都不排、v7x 43648）；0.0.46 上门槛 15105，16384 才排序。所以 mini 上「v7x 0.0.52 不累积就排序」搬不到生产形状，纯数据并行照样要修；修法 `embedding_shard_over_data` 在生产形状上仍过线（[案例 14]「先看结论」末尾的表）。
- **开关与生产**：用纯数据并行时开 `embedding_shard_over_data`（M54，只把词嵌入表沿数据并行轴切开，词嵌入通信少 93%，步时不增加）。它和 ZeRO-1（`shard_optimizer_over_data`）互斥，配置检查直接拒绝；开着 ZeRO-1 时改用 `embedding_grad_fp32`。ZeRO-1 本身的精度没测（04 B1c 在本仓库训练入口上没跑通，[TPU 自比「纯数据并行」](../../models/hunyuan4/alignments/04-tpu-self/impacts/01-data-parallel.md)）。不能切表时用 `embedding_grad_fp32`（A7；数据并行＋梯度累积下要 MaxText `d5c9b113`（[台账](../../models/hunyuan4/CHANGES-LEDGER.md) M55）之后的代码：之前的代码里三者同时成立时，XLA 把 FP32 词嵌入梯度的 all-reduce 挪出累积循环并清零累加器，静默丢掉前面微批的梯度（第 1 步词嵌入梯度范数只有正确值的 0.51 倍，300 步 r 611）；移植这个开关要连 M55 一起移植）。这两个开关只在 `maxtext` 混元 4 集成分支上，上游没有（[MaxText 说明「这些开关在哪个分支」](../../models/hunyuan4/frameworks/maxtext/README.md#这些开关在哪个分支)）。FSDP 在 mini 上过线（每卡收齐 8 张卡共 32768 个下标），不用修；FSDP 每卡下标数 = 序列长 × 每卡微批 × FSDP 组卡数，太少时同样不排序（推断，没跑过），要按同一办法核（[04 README「落到实践」](../../models/hunyuan4/alignments/04-tpu-self/README.md#一页看完六件真有影响的和三条规律)）。（详见 [案例 14]、[TPU 自比「纯数据并行」](../../models/hunyuan4/alignments/04-tpu-self/impacts/01-data-parallel.md)）

### A13 hc_head 收拢之后没舍 BF16

- **原理**：GPU 在 hc_head 把四路流加权收成一路后 `.to(bfloat16)` 再进末层 RMSNorm，是一次真实的舍入；TPU 写的是 `astype`，推断编译器在 excess precision 下把这一来回省掉（同 A6；没看 LLO）。
- **量到**：mini 1000 步前段，8 份 TPU 第 51–90 步主 loss 都偏高、第 50–77 步 param_norm 落后 GPU；开 M52 的 4 份 1000 步第 75 步 param_norm 对 GPU 均值 −1.9e-6 到 +7.1e-6（其余 66 份 TPU 运行 −1.1e-5 到 −4.8e-5，GPU 16 份标准差 2.5e-6）；第 751–1000 步不变。
- **开关与生产**：`hc_head_force_bf16_rounding`（M52），默认关；对齐配置 2026-11-11 起开（C 口径，与 M36、M58 一起，四份对齐 yml 都开）；生产不开（只为照 GPU），代价未单独测，对 FP64 未测。C 口径 6 份（`tc-a`…`tc-f`）第 551–1000 步主 loss 对 GPU 16 份均值还剩 +6.9e-6，比只开 M58 的 +1.9e-6 大（6 对 5，p 0.024；只算只改执行方式的 5 对 4，p 0.048）。线索：M52 与 M58 相冲点在 hc_head 的余切舍入（推断）。M52 只在前向舍、余切不舍时（M62，4 份）M58 又起作用，后段 +3.7e-6（对只开 M52 p 0.029），但前段鼓包回来；M58 不覆盖 hc_head、配完整 M52 时后段 +4.3e-6（p 0.057），前段没有鼓包（[`mini-factorial-c-ab-vs-m52.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-c-ab-vs-m52.json)，见 A17、U22）。（详见 [r28]、[r30]）

### A14 上下文并行时 sink 梯度在 BF16 里算（float32_logits 没接到 MLA）

- **原理**：MaxText 对混元 4 的上下文并行只能配普通注意力；建 MLA 时没把 `float32_qk_product`、`float32_logits` 传下去，配置里打开也不生效，打分与 softmax 都在 BF16，sink 梯度整条在 BF16 里算，对求和顺序敏感。MaxText `56492817` 起接上。
- **量到**：TPU 自比上下文并行 2 对普通注意力的底，曲线 r 1.27，第 6 层 sink 梯度 R 1.27（libtpu 0.0.46）；0.0.52 上这次在噪声内（r 0.95）。这个张量只换顺序就在离 FP64 2.1e-2 到 4.2e-2 之间跳，所以判「超」是一个张量压线（R 1.27，刚过 1.2）判出来的，曲线在 GPU 重跑范围内，影响逐件判「能忍」，不能当作上下文并行算错了的证据。只开 `float32_logits` 跑 300 步，上下文并行 2 对不切 r 1.00；修法没跑 1000 步。
- **开关与生产**：用上下文并行时开 `float32_logits`（最好加 `float32_qk_product`），要 MaxText `56492817` 之后的代码（混元 4 集成分支；上游有这两个键，但在 MLA 上静默不生效，见 [MaxText 说明「这些开关在哪个分支」](../../models/hunyuan4/frameworks/maxtext/README.md#这些开关在哪个分支)）；上下文并行 2 时 +12.4%。不用上下文并行不涉及。（详见 [TPU 自比「上下文并行」](../../models/hunyuan4/alignments/04-tpu-self/impacts/06-context-parallel.md)）

### A15 GPU 参照的索引器 RoPE 在 BF16 里算

- **原理**：Megatron-Core 不融合 RoPE（`apply_rope_fusion=False`）时，`rope_utils` 把 cos、sin 转成 BF16，索引器 q、k 的 RoPE 段乘加也在 BF16。参照一侧自己的精度边界，同 A8。
- **量到**：真实宽度三层，GPU 只把这一处改 FP32 算，第 1 步索引器 KD 降 4.85e-4；250 步 KD 带符号均值降 4.2e-4 – 4.7e-4，落到 TPU 一侧；主 loss 不变。
- **开关与生产**：不修（只作诊断）。不涉及 TPU；判索引器 KD 时要知道 GPU 一侧偏高。（详见 [r27]）

### A16 hc_head：输出头反向 kernel 合并末层 RMSNorm 反向行和时省掉一次 BF16 舍入

- **原理**：输出头反向（dgrad）的卷积 kernel 算出 FP32 的余切 dy，HLO 里写着先舍成 BF16 再给末层 RMSNorm 的反向用。后端有时把末层 RMSNorm 反向的每 token 行和 S = Σ_d x·s·dy 并进这颗卷积 kernel。excess precision 开着时，这次舍入在生成机器码时被省掉：S 用的是没舍的 FP32 dy，主项 r·s·dy 读的是写进显存的 BF16 dy，同一个 dy 两处精度不一致。GPU 两处用的是同一份 BF16 dy。和 A13 不是一件事：A13 是 hc_head 前向收拢之后那次舍入，这一条在输出头的反向。
- **出现条件**：优化后 HLO 里主输出头 dgrad 卷积 kernel 里有 S，且 excess precision 开着。TPU 自比 17 次存了 HLO、有第 1 步抓数的运行里，满足这两条的 6 次都偏，其余 11 次都在 FP64 附近（有 S 但关了 excess precision 的 H4b 不偏；[`hc-head-dgrad-fusion.json`](../../models/hunyuan4/alignments/04-tpu-self/results/hc-head-dgrad-fusion.json)）。会不会切成「有 S」随 libtpu 版本和切分变：
  - v7x FSDP 8（v7x 默认配置）：0.0.46 有（H2），0.0.52 没有（V16）；
  - v7x 纯数据并行不累积：0.0.46 没有（H4d），0.0.52 有（V17）；
  - v5p 0.0.46：golden 没有；关小矩阵乘改写（C15，F6）、einsum 改广播乘（D3）有；mini 现行比较 `tc-a`（C 口径，开 M52，MaxText `8d3d7561`）没有：按 `meta.json` 在 CPU 上重编（没存 TPU 上的 HLO，离线等于上卡是推断），主头卷积 kernel 里没有 S，与 golden 同型；这颗 kernel 默认设置下 `vpack.i.bf16` 0 条，关 excess precision 后 288 条，但它里面本来就没有 S，这 288 条是 kernel 里别的 BF16 转换，不是 A16 那一次。`tc-a` 自己没抓第 1 步；C 口径第 1 步全量抓数 `s1f-c`（同 MaxText `8d3d7561`、同 C 口径开关）里 `hc_head` 三个梯度离 FP64：`hc_head_base` 2.2e-2、`hc_head_fn` 5.9e-2、`hc_head_scale` 2.3e-2，都在 GPU 噪声内，对 GPU 的 R 0.45 / 0.81 / 1.62，没有 A16 偏时的 0.56–1.27（[`t1zc-step1-train-grad-vs-fp64-n6.json`](../../models/hunyuan4/alignments/01-mini/results/t1zc-step1-train-grad-vs-fp64-n6.json)）。上卡结果与离线重编「没有 S」一致；抓数会不会改编出来的程序没核。
  - v5p 0.0.46，两个切片走 DCN（TPU 自比 B15，`ts-ms2-dp2-r2`，自比的配置，不是 C 口径）：第 1 步 `hc_head` 三个梯度离 FP64 0.51–0.57（单切片 0.02–0.07），对自比 golden 的 R 最大 21.1，幅度与 A16 偏的另外 5 次同量级；离线查过是 A16 同一型：B15 存下的训练步程序里主头卷积 kernel 有 S，单切片 16 片同一切法（B2c，`ts-f16-dp2`）没有、不偏，所以是走 DCN 改了融合切分；`hc_head_base` 四个元素对 FP64 的差同号、接近同一个常数（−4.49、−4.70、−4.79、−4.39 e-6）。LLO 里省没省那次舍入、上卡加 barrier 都没核（U29；[`hc-head-dgrad-fusion.json`](../../models/hunyuan4/alignments/04-tpu-self/results/hc-head-dgrad-fusion.json)、[TPU 自比 README](../../models/hunyuan4/alignments/04-tpu-self/README.md)）。上多切片前先用 `models/hunyuan4/analysis/hc_head_dgrad_fusion.py` 查自己的训练步程序里主头 kernel 有没有 S。

  所以「升 0.0.52 就不偏」只对 v7x FSDP 8 这一个程序成立。v7x 正是迁移目标芯片。

  对齐配置的程序：现行 `tc-a` 见上面「v5p 0.0.46」那行（主头没有 S，`s1f-c` 第 1 步 `hc_head` 梯度不偏）；C 口径之前的 `tz-off-rep` 没用 `hc_head_dgrad_fusion.py` 查过。
- **MTP 头是否同样受影响**：融合切分上是，梯度上没看到。
  - 切分：MTP 头 dgrad 卷积 kernel 里并进 MTP 末层 RMSNorm（`mtp_1_final_norm`）反向的每 token 行和 S，上面那 15 次存了 HLO 的程序**全部都有**，包括主头不偏的 golden（v5p 0.0.46 / 0.0.52）、V16、H4c、H4d；离线重编的 `tc-a` 也有。
  - LLO（MTP 那颗 kernel）：H2 `vpack.i.bf16` 0 条，关 excess precision 后 1024 条（H4b），其余指令只差 `vld` / `vst` / `vmov` 几条；V17 `vpack.i.f32.bf16` 0 条，关后 384 条（只离线编），其余只差 `vld`、`vmov` 各 1 条；V16 0 条，没编关掉的对照；v5p `tc-a` 0 条，关后 128 条，但关掉后这颗 kernel 的其余指令也大变（如 `vld` 15519 → 6840），不如 v7x 那两对干净。所以 MTP 头那次舍入在默认设置下同样被省（v7x 两对核过，v5p 是推断）。
  - 梯度：golden 自己也是这种切分，TPU 自比看不出来，只能对 FP64。第 1 步 MTP 三个 norm 权重离 FP64，关 excess precision 前后之比（r64）：`final_layernorm` 0.96 / 0.97 / 0.99，`enorm` 1.04 / 1.02 / 1.00，`hnorm` 0.97 / 0.97 / 0.98（C1 对 golden、H4b 对 H2、0.0.52 关 excess precision 对 `ts-g52-a`），没有 `hc_head` 那种 18–46 倍。TPU 自比 golden 的 MTP `enorm` 离 FP64 是 GPU 的 1.18 倍（1.86e-3 对 1.57e-3），关掉后不回落，不是这一处。`hc_head` 也不受 MTP 这一路影响：MTP loss 系数设 0 后 golden 与 D3 的 `hc_head_base` 都照旧（[MATRIX「hc_head」](../../models/hunyuan4/alignments/04-tpu-self/MATRIX.md#hc_head被-ihc-豁免盖住的三个张量)）。
  - 为什么主头放大、MTP 头不放大：推断是主头的 S 误差经末层 RMSNorm 反向落到 `hc_head` 这种以 ε 项为主、互相抵消的小梯度上，MTP 末层 norm 之后没有这样的参数；没有直接测。没查的：MTP 层其余张量逐个对 FP64、v5p golden 的 MTP kernel LLO（[`hc-head-dgrad-fusion.json`](../../models/hunyuan4/alignments/04-tpu-self/results/hc-head-dgrad-fusion.json) 的 `mtp_head`）。
- **量到**：`hc_head` 第 1 步梯度整体偏一个常数。`hc_head_base` 离 FP64：偏的 5 次 0.56–1.27，golden 6.1e-2。逐元素差：H2 −11.22、−11.19、−11.13、−11.76（e-6），V17 −4.26、−5.51、−5.06、−5.13（e-6）；FP64 是 6.87–11.18（e-6），所以 0.0.46 上 H2 四个元素都变号（C15、D3 是前三个）。LLO 里这颗 kernel 的 in-kernel 舍入打包指令：H2 0 条，关 excess precision 后 256 条（H4b）；V17 0 条，关后 768 条（只离线编，没上卡）（[`hc-head-dgrad-fusion.json`](../../models/hunyuan4/alignments/04-tpu-self/results/hc-head-dgrad-fusion.json)）。
- **被 iHC 豁免盖住**：`hc_head` 三个张量属 iHC 小参数，离 FP64 是对照的 18–46 倍（V17 8.8–25.7 倍），按 TPU 自比的 iHC 豁免（`ihc_accepted`，没有倍数上限）单列、不挡判档；曲线在 GPU 重跑范围内，读不出来。这条豁免只用在 TPU 自比。GPU ↔ TPU 的第 1 步按另一条规则判（[D11 豁免规则](../../models/hunyuan4/alignments/01-mini/DETAILS.md#d11-rule)）：超出 N、R > 1.2、又没有能复核的登记接受理由的张量计入未定责，L3 至少待定；同样的 `hc_head` 偏差出现在 GPU ↔ TPU 上会计入。H2「略超」、V17「在噪声内」都是靠豁免判出来的（[TPU 自比实验矩阵「hc_head」](../../models/hunyuan4/alignments/04-tpu-self/MATRIX.md#hc_head被-ihc-豁免盖住的三个张量)）。
- **因果验证到哪**：v5p 上只改末层 RMSNorm 的程序（D3 的拆法）里，给末层 RMSNorm 输出的余切加 `optimization_barrier`（逼 dy 先写成 BF16 再给反向各处用），`hc_head_base` 回到 golden，逐值相同（`ts-n3-d3final-barrier`）。`reduce_precision` 这条路没上卡试过。v7x 上（H2、V17）、关小矩阵乘改写都没加 barrier 上卡核，同一机制是推断（H2、V17：融合切分相同、LLO 里省掉了同一次舍入、逐元素差同号接近常数；C15 只核了 HLO 的融合切分，没编 LLO）。没查的：V17 幅度只有 0.0.46 那几次的一半，原因没查；就近舍入只能解释约 1/100 的幅度，推断写进显存那次带了截断式的同向偏差，没有直接测。
- **处理**：待拍板。候选修法是在 S 之前对 dy 加 `optimization_barrier` 或 `reduce_precision`（只有前者在 v5p 上验证过）；不能靠全局关 excess precision（F1）。要不要进对齐配置、生产要不要修，都没定。自查：`models/hunyuan4/analysis/hc_head_dgrad_fusion.py` 读优化后 HLO，看主头卷积 kernel 里有没有 S；要确认省没省，再看 LLO。混元 3 没有 iHC，但同样有输出头 dgrad 接末层 RMSNorm，会不会被这样切分、它的末层 norm 梯度受不受影响，没核。

### A17 iHC 流梯度：GPU 三份各舍 BF16 逐个相加，TPU 在 FP32 里加完才舍（mini 主 loss 后段偏高的主因）

- **原理**：GPU（Megatron-Core autograd）里 BF16 的流在一层里用到三处（加权和、门控、残差），autograd 给每处一份 BF16 梯度，按 autograd 的顺序逐个加进流的梯度，每加一次舍一次 BF16（CPU 上用 GPU 同一段代码逐位核过，`analysis/ihc_stream_grad_order_probe.py`）；分支输出也是 BF16 张量。TPU 一侧是推断：在 excess precision 下这几处按 FP32 加完才舍，分支输出可能留在 FP32（没看 HLO / LLO；依据是 TPU 在不开 M52 时补上舍入，第 551–1000 步均值降到 +1.9e-6（5 份里 2 份仍超尾部阈值）；GPU 照 TPU 的写法后出现尾部，见下）。和 A6 不是一件事：A6 是前向分支输入与更新后的流那两次往返，这一条在反向，是梯度怎么相加。
- **量到**：mini 1000 步，第 551–1000 步主 loss 对 GPU 16 份逐步均值（每格 4–6 份，置换检验）：只开 M36 的对照 4 份 +9.4e-6，只再开 M58 的 5 份 +1.9e-6（p 0.016）。反过来，GPU 换成 TPU 的舍法（GPU 诊断开关 G37，对 GPU 16 份留一法）4 份都正，+3.9e-6（p 0.002）。CPU 单测：一块 iHC 的流梯度开开关后与 torch 上 GPU 写法逐位相同。C 口径（M36、M52、M58 都开）6 份 +6.9e-6，比只开 M58 大（6 对 5，p 0.024；只算只改执行方式的 5 对 4，p 0.048）；对只开 M36 的对照只低 2.4e-6（p 0.32），6 份里 5 份第 751–1000 步尾部还在（门槛 3e-6）；现行 `tc-a` 自己第 551–1000 步 +8.15e-6，接近对照均值 +9.4e-6。以只开 M36 + M52 的 4 份为对照（+12.5e-6），再补 M58 全套 / 只补流梯度一半 / 只补分支输出一半（各 4 份）是 +6.7e-6 / +7.9e-6 / +10.8e-6，p 0.14 / 0.14 / 0.54：和完整的 hc_head 舍入（M52）一起开时，M58 测不出作用。把 M52 拆开（M62，各 4 份，默认关，只作诊断）：(a) M52 只在前向舍、余切不舍，配 M58：后段 +3.7e-6，对只开 M52 的 4 份 p 0.029、对只开 M36 的对照 p 0.029，但前段鼓包回来（第 51–100 步 +4.3e-6）；(b) M58 不覆盖 hc_head，配完整 M52：后段 +4.3e-6（对只开 M52 p 0.057，边缘），前段没有鼓包（+0.7e-6）。所以 M52 开着时 TPU 侧同样测得到下降，测不到的是 M52 在 hc_head 舍余切、且 M58 覆盖 hc_head 的组合。推断：相冲点是 hc_head 那一段的余切舍入与流梯度舍入；(b) 不是 GPU 的写法（GPU 在 hc_head 同样舍流梯度）。C 口径的后段没修掉，hc_head 反向里两次舍入的先后待查（U22）。结果文件：[`mini-factorial-c-ab-vs-m52.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-c-ab-vs-m52.json)（拆 M52 两格对只开 M52）、[`mini-factorial-c-ab-vs-base.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-c-ab-vs-base.json)（对对照）、[`mini-factorial-c-parts-vs-m52.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-c-parts-vs-m52.json)（M52 开着时补 M58 全套与两半）、[`mini-factorial-tpu-6.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-tpu-6.json)（对照、只开 M58、C 口径各格对对照）、[`mini-factorial-c-vs-m58-6.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-c-vs-m58-6.json)（C 口径对只开 M58）、[`mini-factorial-c-vs-m58-exec-only-6.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-c-vs-m58-exec-only-6.json)（只算只改执行方式）、[`mini-gpu-tpulike.json`](../../models/hunyuan4/alignments/01-mini/results/mini-gpu-tpulike.json)（GPU 照 TPU 的舍法）。
- **开关与生产**：`ihc_stream_grad_gpu_order`（M58，MaxText `06244684`），默认关，只作用于 iHC 的 XLA 路径（另一条是 iHC 的 Pallas 内核路径 `use_ihc_pallas_kernel`，MaxText 默认关；四份对齐 yml 都没开它，走 XLA 路径；开了内核、且每卡 token 数排得进内核分块时，M36、M58 都不生效；生产走哪条没核）；对齐配置 2026-11-11 起开（C 口径，四份对齐 yml 都开）。生产不开：只为照 GPU，离 FP64 远近没测。代价没单独测：C 口径整体 1000 步步时中位数 0.4963 s，只开 M36 的对照 0.4960 s（对照 `ti-o-a`，两份 MaxText 提交不同：`8d3d7561` / `751fdc6d`；单次运行，不是 A/B；[`mini-step-time.json`](../../models/hunyuan4/alignments/01-mini/results/mini-step-time.json)）。（详见 [r30]、[D30]）

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
- **开关与生产**：`indexer_weights_proj_bf16_weight`（M27）、`routed_gate_bf16_rounding`（M8）。不开（多舍一次）。M8 +0.1%，M27 +0.1%。推荐组合里关 M27 时前 300 步索引器 KD 对 GPU 不过（三份 r 4.78–5.23）；组合里只把 M27 开回来跑 300 步（一份，`x5-prod-m27`），KD 带符号均值 −6.76e-4 → −1.98e-4，r 4.91 → 1.44，从不过变成待定，其余 8 项过 7、待定 1（grad_norm）（[`t1zp300m27-l4`](../../models/hunyuan4/alignments/01-mini/results/t1zp300m27-l4.report.md)）。所以这处不过主要来自关 M27。第 1 步两者逐位相同，对 FP64 量不到 M27（起训时主权重在 BF16 格点上，见案例 13），这只说明离 GPU 远，对 FP64 没比。生产建议「M27 不开」要不要改：待拍板（[mini 对齐细节「推荐组合整体跑一次」](../../models/hunyuan4/alignments/01-mini/DETAILS.md#推荐组合整体跑一次)）。（详见 [r5]、[案例 13]）

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
- **量到**：1000 步路由偏置均值带符号均值（8 份尺子，上一代）：N 0.501，TPU 1.27 → 0.444，r 2.54 → 0.89。
- **开关与生产**：M39随 C1 的开关。**前向更准不等于整网更近，反向也要对上。**（详见 [r18]）

### C4 病态标量：一格误差被放大成大差

- **原理**：大项相减剩下的小量相对误差被放大：KD 的 KL 每行约 1.1e-4 nat、lse 约 7.1，lse 差 1 格就是 KL 的 4.3e-3。
- **怎么分辨**：输入与梯度两边一样近、只有这个标量对不上，就是病态；再查同一程序在 CPU 上偏不偏。
- **第二个例子**：KD 记录写法本身在 FP32 下病态，教师分布行和与学生 log q 的 1e-8 级偏差被放大（CPU 上同样偏高）；改记广义 KL（M50，只改记录值、梯度不变）后 1000 步 KD 由待定变过（U15）。
- **生产**：不涉及；M50 建议开（只改记录值）。差大不代表有 bug：条件数大时一格误差就被放大（这里是 C1）。**详见**：类 12；[根源一览第 25 行](../../models/hunyuan4/ROOT-CAUSES.md#r25)。

### C5 TPU 的 FP32 除法、sqrt、1 − β^t 不精确

- **原理**：Adam 的除法、sqrt、1 − β^t 在 TPU 上是 2 / 3 / 108 ulp；GPU 正确舍入。
- **量到**：优化器 8 步回放，第 5 步 `hc_head_base` 主权重增量对 FP64：GPU 1.49e-6，TPU 5.70e-6 → 1.49e-6，R 3.82 → 1.00。
- **开关与生产**：`muon_adam_accurate: true`（M34）。开，+0.1%。（详见 [r17]）

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
- **量到**：T1 里 QK-Clip 从不触发（τ 100）；第 1 步单开关主 loss 逐位相同；但只关它的 300 步主 loss 带符号均值 0.48 → 0.93 倍（推断：编译结果变了；升幅 0.45 在只改执行方式的波动线 1.07 以内，单看曲线归不到开关。波动线 = 8 份只改执行方式的运行截前 300 步、任取同组两份，「升得最多的一项」的最大值，[`exec-spread-300.json`](../../models/hunyuan4/alignments/01-mini/gpu-noise/results/exec-spread-300.json)）。
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
- **开关与生产**：对齐时两边都不减（G14）。生产**两边都开**（漂移吃掉偏置分辨率）；原版 Megatron-Core 不减均值。两边都开的实测：混元 4 mini 的生产配方臂 P1（学习率 cosine 峰值 2e-5、warmup 200，YaRN 8，路由偏置零均值两边都开）1000 步，对同配方 GPU 8 份两两 28 对的尺子 9 项全过（[`t1pr-l4-gpu8`](../../models/hunyuan4/alignments/01-mini/results/t1pr-l4-gpu8.results.json)，[知识库「生产配方臂」](../../models/hunyuan4/knowledge.md#生产配方臂与-qk-clip-梯度累积要点与出处)）。这只是诊断，不进签字：8 份尺子是看过借现行尺子的判档之后才定的（尺子事后选），份数少于登记的 16 份，而且比现行尺子宽（主 loss 带符号均值 4.8 倍）；换现行 16 份尺子，主 loss（全程 r 2.97、后段 3.02）与 grad_norm（2.45）不过（[`t1pr-ruler-compare`](../../models/hunyuan4/alignments/01-mini/results/t1pr-ruler-compare.json)）。另外，P1 上 TPU 主 loss 带符号均值 +1.38e-5（后段 +1.52e-5），和 C 口径之前的 `tz-off-rep`（+7.21e-6）同号、约 1.9 倍（U22 那件事）；TPU 只有 1 份，分不出是轨迹偶然还是生产配方下偏高更大，「不是跨侧差」要等 TPU 补份数（[知识库「生产配方臂」](../../models/hunyuan4/knowledge.md#生产配方臂与-qk-clip-梯度累积要点与出处)）。（详见 [r1]）

### E3 eod 掩码下 aux loss 的权重

- KD 按行数还是有效 token 加权、MTP loss 怎么重标、log 截断写法两边不同；M20–M22 后单测对原版一致（修前权重差估算 ≤ 2.4e-3）。看情况，同 D1。**详见**：[r9]。

### E4 路由与 DSA 的近似平局在两边各落一边

- **原理**：两种舍入把平局各推一边；翻转不累积（混元 3 的证据只到截 4 层、1000 步、常数学习率 1e-5，更长的训练和生产学习率没测；混元 4 mini 的生产配方臂 P1（学习率 cosine 峰值 2e-5、warmup 200，YaRN 8，路由偏置零均值两边都开）1000 步，对同配方 GPU 8 份两两 28 对的尺子 9 项全过（[`t1pr-l4-gpu8`](../../models/hunyuan4/alignments/01-mini/results/t1pr-l4-gpu8.results.json)，[知识库「生产配方臂」](../../models/hunyuan4/knowledge.md#生产配方臂与-qk-clip-梯度累积要点与出处)）。这只是诊断，不进签字：8 份尺子是看过借现行尺子的判档之后才定的（尺子事后选），份数少于登记的 16 份，而且比现行尺子宽（主 loss 带符号均值 4.8 倍）；换现行 16 份尺子，主 loss（全程 r 2.97、后段 3.02）与 grad_norm（2.45）不过（[`t1pr-ruler-compare`](../../models/hunyuan4/alignments/01-mini/results/t1pr-ruler-compare.json)）。另外，P1 上 TPU 主 loss 带符号均值 +1.38e-5（后段 +1.52e-5），和 C 口径之前的 `tz-off-rep`（+7.21e-6）同号、约 1.9 倍（U22 那件事）；TPU 只有 1 份，分不出是轨迹偶然还是生产配方下偏高更大，「不是跨侧差」要等 TPU 补份数（[知识库「生产配方臂」](../../models/hunyuan4/knowledge.md#生产配方臂与-qk-clip-梯度累积要点与出处)）；更长的训练没测）。
- **量到**：混元 3 第 1 步每层 0.44%–0.56% 的路由槽位翻转，1000 步在噪声内；hy4-mini 第一行对 FP64 的 DSA 翻转 GPU 7,913 行、TPU 7,901 行。
- **生产**：不涉及。正式判定按各自的选择对 FP64 自选比（G12、G13）。**详见**：混元 3 #5、#17。

### E5 并行方式、累积分组、分布式优化器

- 混元 3 只换累积分组，embedding 梯度差到噪声上限的 31 倍。对齐时累积分组一致、golden 关分布式优化器（混元 3 截 4 层、8 个设备上 TPU 改 EP 8，只验了 L4 的 loss、grad_norm、param_norm 与学习率）。
- 并行方式可以不同，但有条件：混元 4 TPU 自比里 FSDP、专家并行、张量并行、重算档位只换了求和顺序，和 GPU 重跑分不开（libtpu 0.0.46；专家并行 4、8 另在 0.0.52 上测过，结论相同，见 [TPU 自比 libtpu 0.0.52](../../models/hunyuan4/alignments/04-tpu-self/libtpu-052.md)）；纯数据并行要开 `embedding_shard_over_data` 或 `embedding_grad_fp32`（A12；后者在梯度累积下要带 M55，否则静默丢梯度，见 A7），上下文并行要开 `float32_logits`（A14）。生产切法和验收用的切法不同时，先查 [TPU 自比](../../models/hunyuan4/alignments/04-tpu-self/README.md)，没测过的切法要重测。按配方。**详见**：混元 3 #11–#13。

### E6 两边数据加载器的约定不同

- pad 处理与 microbatch 顺序两边不同。对齐时 TPU 原样回放 GPU 存下的每步 batch，输入逐位相同。不涉及。**详见**：混元 3 #16、台账 G9。

## 7. F 编译器行为

XLA 的改写会让舍入点和代码字面不一样。

### F1 XLA 默认允许 excess precision

- **原理**：融合内部的中间值可以不按代码写的 BF16 落地、留在 FP32（`embed_tokens` 实测：允许时等于 FP32 累加最后舍一次）。
- **量到**：hy4-mini 第 1 步只关它：205 个梯度 R 中位数 0.967 → 0.986；`embed_tokens` 梯度对 FP64：GPU 4.2e-3，TPU 4.0e-3 → 0.109（坏 27 倍）。
- **开关与生产**：**保持默认，不要全局关**；要照 GPU 舍入点的地方局部用 `reduce_precision`（A6），依赖多保留精度的地方用开关写死（A7）。（详见 [r22]、混元 3 #10）
- **多步也更差**：hy4-mini 1000 步只关它，第 751–1000 步主 loss 比 GPU 16 份均值高 4.1e-5（`mlh-noexc`）；真实宽度三层 250 步主 loss r 7.32 → 29.6（`b2r-noexc`）。
- **残差**：GPU eager 一定做、TPU 编译器可能省掉的舍入，没有通用开关，只能逐处补 `reduce_precision`。混元 4 对齐配置（C 口径）补了三处：iHC 分支输入与更新后的流（M36，A6）、hc_head 收拢（M52，A13）、iHC 流梯度与分支输出（M58，A17）。别的模型迁移时要自己逐处找。

### F2 algsimp 把 HIGHEST einsum 改写成 FP32 逐元素

- iHC 的 HIGHEST einsum（收缩维 4）：algsimp 把 54 个 dot 改写成逐元素乘加，不上矩阵单元，HIGHEST 的精度要以优化后 HLO 为准（[案例 11]）。

### F3 融合本身改变数值

- 只关 `tpu_fusion`：205 个梯度 R 中位数 0.967 → 0.995；步时 0.506 → 1.002 s。保持默认；第 1 步 iHC 的 R 随编译设置摆动（超线的在 11–16 个之间）。

### F4 关优化 pass 造不出「无优化参照」

- 全关编不过；关 `algsimp,tpu_fusion` 能跑但结果错：loss 只差 3e-5（相对），grad_norm 4.05（FP64 6.359）。参照用 CPU FP64（G1）；**loss 对得上不代表梯度对。**

### F5 单算子回放逐位不变，不能证明开关无效

- A6 的开关单算子回放逐位不变，整网里起作用：TPU 自比只关它，第 1 步第 1 层索引器 `k_norm` 的权重梯度离 FP64 1.08e-3（golden 4.83e-4，GPU 6.69e-4），只把分支输入那次舍入补回去就回到 golden（[TPU 自比影响逐件「iHC 不强制舍成 BF16」](../../models/hunyuan4/alignments/04-tpu-self/impacts/05-ihc-bf16-rounding.md)）。省不省这次往返取决于后端怎么切融合（最小复现的 LLO 里均方那颗省、归一化那颗不省，A6；整网编译里是否同样省、各层是否一致没核），推断单独编译时没省。曲线读数不作依据：只关它的 300 步主 loss 带符号均值 N 3.01e-6，TPU 1.43e-6 → 3.03e-6，r 0.48 → 1.01，单份，升幅 0.53 在只改执行方式的波动线 1.07 以内；1000 步 r 2.12 → 0.76 是 8 份尺子（上一代）上的单份。编译相关的开关要在整网编译里验。**详见**：[r16]。

### F6 关小矩阵乘改写：RMSNorm 的乘法改走单遍 BF16

- 小矩阵乘改写默认开。关掉后 RMSNorm 的乘法改走矩阵单元单遍 BF16：TPU 自比曲线在噪声内，第 1 步 MTP `enorm` 梯度 R 1.40（0.0.52 上 1.35）；同一臂（`ts-c-nosrb`，libtpu 0.0.46）`hc_head` 三个张量离 FP64 1.25–1.29（golden 2.7e-2–7.0e-2），`hc_head_base` 前三个元素变号，被 iHC 豁免单列，判档里看不到。这一处偏差推断是 A16（HLO 里主头卷积 kernel 有 S；没编 LLO、没上卡加 barrier）。另一个观察：关改写时给 RMSNorm 那几处 einsum 写明 HIGHEST，`hc_head_base` 回到 6.00e-2（`ts-n2-nosrb-hi`，[TPU 自比实验矩阵](../../models/hunyuan4/alignments/04-tpu-self/MATRIX.md)「C15」一节），它和 A16 怎么对上没查；0.0.52 上关小矩阵乘改写（V6）不再偏（5.2e-2）。关了还更慢。保持默认，调性能开关时别关（[TPU 自比「关小矩阵乘改写」](../../models/hunyuan4/alignments/04-tpu-self/impacts/03-small-matmul-rewrite.md)）。

### F7 libtpu 版本改的不只是求和顺序

- **原理**：换 libtpu 时后端代码生成一起换了，HLO 相同也不代表机器码相同，离线编译比程序指纹判不了。新版本可能改默认行为。
- **量到**：0.0.46 → 0.0.52，golden 配置（v5p，FSDP 8）300 步、1000 步都在噪声内，第 1 步没有梯度离精确值更远；这只是这一个配置。别的配置变了：「RMSNorm 多舍一次」从 r 1.06 变成 5.50（A2）；词嵌入 scatter 插排序的门槛随列宽和芯片代际变（A12）；v7x 上 `hc_head` 第 1 步梯度偏不偏（A16）随版本反转——FSDP 8 从偏（`hc_head_base` 离 FP64 1.267）变成不偏（0.014），纯数据并行不累积从不偏（0.021）变成偏（0.56），两次判档都靠 iHC 豁免、曲线看不出。本清单的读数都在 0.0.46 上（「适用条件」）。0.0.50、0.0.50.1 有 scatter 正确性问题，不用（[环境](environment.md#tpu-侧)）。
- **生产**：钉版本。换版本先建新 golden、和旧 golden 比，再把已知敏感项（TPU 自比的六件）重测，并用 `hlo_mof_dus_scan.py` 扫 F8 的特征、用 `hc_head_dgrad_fusion.py` 查 A16；第 1 步被 iHC 豁免单列的张量要看倍数和符号，不能只看判档；结论只对实测过的版本对和配置成立（[`libtpu-052.md`](../../models/hunyuan4/alignments/04-tpu-self/libtpu-052.md)）。

### F8 jit 静默算错：DUS 和读同一缓冲的算子被合进一颗多输出 fusion

- **现象**：v5p 上 jit 的结果错到 0.47（相对），逐算子执行（`jax.disable_jit()`）是对的，没有报错。不是精度问题。
- **原理**：multi-output fusion 把「用 DUS 原地改写一块缓冲」的 kernel 和「读同一块缓冲」的 kernel 合成一颗。v5p 为这颗 kernel 生成的代码把后者的输出算丢。出错的组合是：目标缓冲按 8 行一块存，读出的小块只有 2 行或 4 行。v7x 上同样的 kernel 算对。每个 pass 之后的 HLO 在 CPU 上都算对，错在代码生成。
- **版本**：0.0.46、0.0.48、0.0.51、0.0.52 都错；nightly 0.0.53.dev20261106 起不再这样合并，算对。截至 2026-11-11 没有带修复的正式版。
- **开关与生产**：
  - 用 `hlo_mof_dus_scan.py` 扫训练步的优化后程序；混元 3、混元 4 存档的 300 份是 0 处。这 300 份只有混元 3 截 4 层与混元 4 mini 存了 HLO 的程序（混元 3 正式被测不在内）；mini 现行比较 `tc-a`（C 口径，MaxText `8d3d7561`，多了 M52 的 `reduce_precision` 与 M58 的 custom_vjp，程序和 `tz-off-rep` 不同）与 C 口径另 5 份 `tc-b … tc-f`（只改执行方式的 4 份加 full 重算的 `tc-e`）、C 口径之前的现行比较 `tz-off-rep`、推荐组合 `x5-prod` 都没存 HLO，在 CPU 上按 `meta.json` 重编后扫，也都是 0 处（离线程序与上卡程序是同一份是推断，依据：存了 HLO 的 `ts-g-a` 按同一办法在 CPU 重编，规范化后与 TPU 上存下的逐行相同，[`jit-miscompile-offline-scan.json`](../../models/hunyuan4/results/jit-miscompile-offline-scan.json) 的 `calib`；`ts-g-a` 与 `tz-off-rep`、`x5-prod` 同一 MaxText 提交，`tc-*` 的 `8d3d7561` 没有存了 HLO 的运行可校准，依据弱一层）。没扫的：L3 用的 1 步抓数运行 `s1f-c`、混元 4 mid、真实宽度截层（06）、生产尺寸，混元 3 正式被测（`t-rmap-on-1000`、`t-v7x-on-1000`），所以不能完全排除（待做：扫这几类程序）；
  - 扫到特征时，对被改写缓冲的读取加 `jax.lax.optimization_barrier`（只挡这一处）；
  - 找不到位置时，`LIBTPU_INIT_ARGS=--xla_jf_enable_multi_output_fusion=false` 全局关（会改别处数值）。

  （详见 [案例 16]）

## 8. G 测量方法

### G1 真值是 CPU 上的 FP64 整网重算，与被测不同源

- **怎么量**：[`step1_ref.py`](../../models/hunyuan4/frameworks/megatron/step1_ref.py) 在 CPU 上用 GPU golden 同一套代码全程 FP64 重算第 1 步；FP32 版离它 1.34e-4（GPU 误差的 1/51）；离散选择由 FP64 自己选。
- **为什么**：真值要比两边准一个量级以上，且不和任何一边共用写法。

### G2 噪声尺子只用 GPU 重跑

- N 取 GPU 同设置重跑两两配对的最大值：mini 前 300 步 8 份 28 对，1000 步 2026-11-11 起 16 份 120 对（之前 8 份 28 对偏紧，扩到 16 份后上限放大 1.0–2.1 倍，[尺子 1.6](../guide/ruler.md#16-r-与判法)）；第 1 步取 GPU 各份两两最大 rel_L2：mini GPU ↔ TPU 的梯度 2026-11-11 起 6 份 15 对（之前 3 份）；协议第 1.1 节规定有效份数少于 6 份（15 对）判「参照不足」，这条管不管第 1 步待拍板（[PROTOCOL](../../models/hunyuan4/PROTOCOL.md) 第 1.1 节），照做时按 6 份准备；优化器一步与 TPU 自比仍是 3 份。TPU 同设置重跑逐位相同，不进尺子。判定只对一份 GPU golden，其余重跑只定 N。

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

### G10 grad_norm 相对差的分母

- 混元 4 2026-11-10 起所有曲线一律以参照作分母，grad_norm 也是：实测偏置远小于噪声，重判 262 份结果只有 `t1zz-l4` 一项判定变（[对照表](../../models/hunyuan4/results/grad-norm-denominator-rejudge.md)）。
- 混元 3 用两边均值 (t − g) / ((t + g) / 2)：grad_norm 逐步涨落约 ±3%，以一边作分母有约 +1e-3 的偏置（混元 3 #19），混元 3 的结果不重判。

### G11 取回的大文件核分块与整文件哈希

- 分块取回时每块核字节数，拼好后核整文件 sha256，不一致整体失败。截断的块是整段 0、总大小照样对得上，会造出假的大 R：单个张量突然离 FP64 很远，先排除传输损坏。

### G12 第 1 步正式判定用 TPU 自选的路由；灌 GPU 的选择只作诊断

- 生产时 TPU 不回放 GPU 的选择，L3 用自选的路由与 top-k 对 FP64 比；灌 GPU 选择只用来拆差：hy4-mini 第 1 步约 2% token 选得不同，灌后 205 个梯度对 GPU rel_L2 中位数 4.25e-3 → 2.23e-3。

### G13 路由一致率：TPU 选得离真值是否比 GPU 远

- 按层统计自选的路由与 DSA top-k 对 FP64 的翻转率，R = TPU ÷ GPU 各份最大值，≤ 1.2 过、1.2–2 待定、> 2 不过（1.2 这条线同 L3 的 R；L3 的 R 只有 ≤ 1.2 一条线，待定这一档是路由一致率自己的，PROTOCOL 第 5 节）；对 GPU 的翻转率照报不判（GPU 重跑前几层逐位相同，N 为 0）。hy4-mini 第 1、8、30、100 步各层 R 0.87–1.00，过：第 1 步是 C 口径之前的配置（base2 关，`tw-ra1z`）0.87–0.99，C 口径上没重测；第 8、30、100 步沿用 base2 开的一代 0.90–1.00，关 base2 后没有重测（[`t1-route-agree-s1.json`](../../models/hunyuan4/alignments/01-mini/results/t1-route-agree-s1.json) 等四份）。

### G14 TPU 对每份 GPU 各判一遍

- 诊断，不进判定。下面是 8 份尺子那一代的数：对 8 份 GPU 主 loss 带符号均值 r 中位数 0.51，GPU 留一 0.36；关 base2 后 2.06。后来开、关各补到 4 份、换 16 份尺子，主 loss r 开 0.90–1.69、关 0.93–1.55，GPU 16 份留一法 0.77–0.97，查明是轨迹偶然（[`mini-base2-loo.json`](../../models/hunyuan4/alignments/01-mini/gpu-noise/results/mini-base2-loo.json)、[D30]）。

### G15 DSA top-k 翻转多，不等于输出差

- 第 1 步 GPU 对 FP64：2048 个稀疏 query 里 1936 个至少翻了一个（槽位 0.147%），灌 GPU 的 top-k 给 FP64 注意力，输出误差几乎不变（翻转在第 2048 名附近、权重很小）。要报，不能直接判不过。

### G16 转换往返通过不代表训练时载入了

- 训练入口没设载入路径时会随机建权重，往返检查照样全过；要在第 1 步前逐参数对检查点自检（GPU 台账 G7）。

### G17 单次运行的 r 在 1 附近不分好坏

- 只改一个按理不改数值的开关（M51 精确 log，CPU 上逐位相同），TPU 融合一变，第 1 步 10 个梯度张量就不逐位，1000 步主 loss 带符号均值：N 3.78e-6，TPU 1.36e-6 → 5.44e-6，r 0.36 → 1.44；诊断程序与训练步分开编译，第 1 步梯度相对差中位数 2.9e-4。

## 9. 还没解释清的

未定的以[根源一览](../../models/hunyuan4/ROOT-CAUSES.md#还没查明根因的)为准；编号 U 与上面各条独立，已解决的不列。

| # | 项 | 读数 | 状态 |
|---|---|---|---|
| U1 | sink 梯度余量 | mid R 1.23–1.47，真实权重 D1 第 0 层 R 2.66（mini 已收拢） | 带 M42b 的第 1 步没跑（[D24]、[D22]） |
| U2 | iHC 小参数梯度 | mid 第 1 步 3 个；mini 上一代（base2 开）6 个已定责、已接受；现行 C 口径（`s1f-c`）对 GPU 6 份 1 个未定责（第 3 层 FFN 前 `hc_fn`），见 U24 | 带 M42b 的第 1 步没跑（[D11]） |
| U3 | A4 / A5 耦合 | 单开 M41 时 1000 步主 loss 带符号均值（8 份尺子）N 3.78e-6，TPU 7.20e-6，r 1.91 | 前向逐位不变（[D25]）；只有一份运行，只改执行方式的重复，主 loss 带符号均值的 r 就能在 0.36–2.08 之间（8 份尺子推算，见 A4），分不清是开关还是轨迹 |
| U4 | 真实权重 D1 索引器 KD | 带符号均值 N 8.67e-4，TPU +2.01e-3，r 2.32 | 那一代没有 M50（[D21]） |
| U5 | grad_norm 前 300 步待定 | 现行 `tc-a` 前 300 步 MARE：N 3.85e-4，TPU 3.88e-4，r 1.01；1000 步过（0.79）。C 口径之前的 `tz-off-rep` 前 300 步 r 1.23 | base2 开的上一代已接受（同一状态 6 次里 5 次 TPU 更近 FP64，[D2]）；那套证据是在开 base2 的状态上做的，关后没有重做 |
| U7 | 真实权重 D1 第 1 步主 loss | 对 FP64：GPU +9.24e-4，TPU −2.23e-4 | 差在 GPU 一侧（[D19]） |
| U8 | mid 第 1 步主 loss | 对 FP64：GPU +1.68e-4，TPU +2.8e-6 | 推断差在 GPU |
| U11 | B5 对第 1 步 iHC 梯度的改善 | 11 → 8 个 | 后一代没复现（[r6]） |
| U12 | A6 的往返被省掉，能不能解释长跑偏高 | 机制已核到训练步 HLO 与最小复现的 LLO（A6），整网训练步的后端没核；第 1 步只有三处一起模拟才对上第 0、1 层 KD 的变化，第 5 层模拟 −2.2e-4、TPU +1.55e-3 | 第 5 层原因没查；1000 步后段偏高的主因后来定到 A17（[r30]），A6 在长跑里起多大作用没单独分（[r16]） |
| U14 | 没量步时的开关 | A7 两个开关、M42、D1–D3、D6、E2、E3 | — |
| U15 | 索引器 KD 记录值偏高 | M50 前后 1000 步带符号均值 N 1.83e-4，TPU +1.93e-4 → −1.27e-4，r 1.06 → 0.69 | 已修（C4、[r25]、[D1]）；GPU 记录值也偏高 |
| U16 | param_norm 真实漂移 | — | 现行配置没重做逐张量定位（[D12]） |
| U17 | 真实权重第 1 步逐张量不过 | B 16 个、D1 11 个 R > 1.2 | norm scale 见 A7，q latent norm 见 A11；其余待查（[D22]） |
| U18 | 路由偏置标准差 | 现行 `tc-a` 1000 步过（0.85；C 口径之前的 `tz-off-rep` 0.77）；base2 开的上一代 8 份尺子下待定（带符号均值 N 1.60e-3，TPU −1.82e-3，r 1.14） | 上一代已接受（同一状态 35 处偏置更新方向里 TPU 与 FP64 不同 2 处、GPU 每份 1 处，[D23]）；关 base2 后没有重做 |
| U21 | mid 300 步 L4 四项略超 | 主 loss 带符号均值 N 4.66e-6，TPU 6.41e-6，r 1.38 | 都在 2N 以内（[D27]） |
| U22 | mini 主 loss 后段偏高：C 口径没修掉 | C 口径之前的 `tz-off-rep` 1000 步带符号均值 N 4.65e-6，TPU +7.21e-6，r 1.55（第 101 步起 1.52）；现行 `tc-a` 主 loss 过（MARE r 0.98，带符号均值 0.76）。第 551–1000 步对 GPU 16 份逐步均值：只开 M36 的对照 +9.4e-6，再开 M58 +1.9e-6，C 口径（M36、M52、M58）6 份 +6.9e-6（[`mini-factorial-tpu-6.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-tpu-6.json)） | 主因已定：iHC 流梯度的舍入写法（A17、[r30]，两头验证）；前段小鼓包定到 A13。是训练轨迹的差，不是前向的差；与 base2 无关。C 口径后段没修掉：C 口径 6 份后段 +6.9e-6，比只开 M58 大（p 0.024；只算只改执行方式的 5 对 4，p 0.048），对只开 M36 的对照只低 2.4e-6（p 0.32），6 份里 5 份尾部仍在，`tc-a` 自身后段 +8.15e-6。M52 开着时补 M58 全套 / 只补流梯度一半 / 只补分支输出一半（各 4 份）+6.7e-6 / +7.9e-6 / +10.8e-6，对只开 M52 的 4 份（+12.5e-6）p 0.14 / 0.14 / 0.54，M58 测不出作用（[`mini-factorial-c-parts-vs-m52.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-c-parts-vs-m52.json)）。线索：相冲点在 hc_head 的余切舍入（推断）：M52 只在前向舍、余切不舍时配 M58，后段 +3.7e-6（对只开 M52 p 0.029），前段鼓包回来；M58 不覆盖 hc_head、配完整 M52，后段 +4.3e-6（p 0.057），前段没有鼓包（M62，各 4 份，[`mini-factorial-c-ab-vs-m52.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-c-ab-vs-m52.json)）；hc_head 反向里两次舍入的先后待查（[D30]） |
| U23 | 关 base2 后低阈值 QK-Clip 第 1 步逐头最大 logit | C 口径之前的配置（`tz-off-qk01`）：第 4 层对 FP64：GPU 2.97e-3，TPU 4.61e-3，R 1.55（base2 开时 0.94）；C 口径上没重测 | 差在 BF16 舍入单位的量级，没查（[D31]） |
| U28 | mini 路由偏置均值 1000 步待定 | 现行 `tc-a` 带符号均值 TPU 1.36、N 1.04，r 1.31；C 口径只改执行方式的另 4 份 0.39–0.80（`tc-b` 0.80、`tc-c` 0.67、`tc-d` 0.39、`tc-f` 0.69）；`tc-e`（full 重算、反向重选路由，不算同配置）0.36（[`t1zc-repeats-6.json`](../../models/hunyuan4/alignments/01-mini/results/t1zc-repeats-6.json)） | 未定责；`tc-a` 之外的 5 份（`tc-b`…`tc-f`）都在 0.36–0.80，定责没定（[mini 报告「现状」](../../models/hunyuan4/alignments/01-mini/ALIGNMENT-REPORT.md#现状)） |
| U24 | mini 第 1 步第 3 层 FFN 前 `hc_fn` | 以现行 C 口径（`s1f-c`，2026-11-11）为准，GPU 第 1 步参照 6 份 15 对：205 个梯度里 163 个超出 N，超出 N 且 R > 1.2 的 2 个：第 3 层 FFN 前 `hc_fn`（1.14 N；GPU 2.04e-2，TPU 2.90e-2，R 1.43）与 MTP `enorm`（1.41 N，R 1.20）。C 口径之前（`tz-off-rep`）计入未定责的是第 3 层注意力前 `hc_scale`（1.05 N，R 5.36），C 口径下 0.66 N、在噪声内 | `hc_fn` 计入未定责（base2 开那一代登记接受的 6 个之一，接受理由没有数值门槛、复核不了；是否沿用待拍板），`enorm` 另列、按 D13 不计入（[D11 豁免规则](../../models/hunyuan4/alignments/01-mini/DETAILS.md#d11-rule)）；单列的挡不挡签字待拍板（[`t1zc-step1-train-grad-vs-fp64-n6.json`](../../models/hunyuan4/alignments/01-mini/results/t1zc-step1-train-grad-vs-fp64-n6.json)） |
| U26 | 0.0.52 上纯数据并行三种修法前 300 步主 loss 同向偏高 | r 1.55 / 1.20 / 1.23，扣掉 golden 自身的差后仍比 0.0.46 上同一修法高约 2.4e-6–4.8e-6（推断，按带符号均值近似相减；A12） | 原因没查，和 0.0.52 上 RMSNorm 那一件（A2）是否同源没核；修法① 1000 步回到 0.92（[`libtpu-052.md`](../../models/hunyuan4/alignments/04-tpu-self/libtpu-052.md)） |
| U27 | A16 的幅度 | 就近舍入只能解释 `hc_head_base` 偏差的约 1/100；V17（v7x 0.0.52）偏 −4.3e-6 到 −5.5e-6，约是 0.0.46 那几次（约 −11e-6）的一半；B15（v5p 0.0.46 两个切片走 DCN）−4.39e-6 到 −4.79e-6，约四成 | 推断写进显存那次带截断式的同向偏差，没有直接测；V17 为什么是一半、B15 为什么约四成都没查；v7x 上没加 barrier 上卡验证（A16） |
| U29 | 多切片第 1 步 `hc_head` 偏 | TPU 自比 B15（两个 v5p 切片走 DCN，300 步一份）：`hc_head` 三个梯度离 FP64 0.51–0.57（单切片 0.02–0.07），对自比 golden 的 R 8.4 / 7.5 / 21.1；曲线对 golden 与对单切片都在噪声内（r 0.91、0.99），靠 iHC 豁免 | 离线定位为 A16 同一型：B15 的训练步程序里主头卷积 kernel 有 S，单切片同一切法 B2c 没有、不偏，是走 DCN 改了融合切分（[`hc-head-dgrad-fusion.json`](../../models/hunyuan4/alignments/04-tpu-self/results/hc-head-dgrad-fusion.json)）；LLO 与上卡加 barrier 没核；更多切片、DCN 上做 FSDP、1000 步都没测（[TPU 自比 MATRIX](../../models/hunyuan4/alignments/04-tpu-self/MATRIX.md)） |
| U25 | 真实宽度三层主 loss 后段偏低 | 第 101–250 步带符号均值：N 9.22e-6，TPU base2 开 −2.60e-5、关 −2.51e-5，r 2.82 / 2.72；1000 步回放第 101–1000 步：N 2.10e-5，开 −2.60e-5、关 −1.80e-5，带符号均值 r 1.24 / 0.86；按三量最差 1.24 / 1.02（关那臂领头的是最大单步）（[`rw3-1k-replay.json`](../../models/hunyuan4/alignments/06-rw3/results/rw3-1k-replay.json)） | 不是前向的差，是轨迹的差；已排除 base2、权重侧精度、GPU 索引器 RoPE、iHC 舍入（[06 README](../../models/hunyuan4/alignments/06-rw3/README.md#后段主-loss-偏低不是前向的差是轨迹的差)） |

## 10. 怎么用这份清单

| 场景 | 做法 |
|---|---|
| 对齐 | TPU 照 [T1 配置](../../models/hunyuan4/frameworks/maxtext/hy4-mini-t1.yml)（含 B3、E1、关 base2；A4、A5 一起开；A10 用 M42b；A6、A13、A17 三处补舍入都开，即 C 口径；不含 A7），回放 GPU 的 batch；GPU 实测 TF32（B7）、dgrad 分段（B8）、自写内核查精度边界（A8）、关分布式优化器（E5）、存每步 batch（E6）、偏置更新一致（E2） |
| 生产 | 看第 1 节「生产建议」：对真值的误差加代价。这些建议是在 T1 上逐个关开关得到的；推荐组合整体在 hy4-mini 上跑过一次 1000 步（`x5-prod`，3 份；E2 零均值更新照对齐配置关（GPU golden 没有），「看情况」各行照对齐配置，所以不覆盖 E2 两边都开的生产设置）：1000 步没有不过项，主 loss 与 C 口径之前的对齐配置交叠，但索引器 KD 三份都待定（r 1.23–1.26；C 口径之前的对齐配置 `tz-off-rep` 0.77，[`t1zp-repeats.json`](../../models/hunyuan4/alignments/01-mini/results/t1zp-repeats.json)；现行 `tc-a` 0.75，[`t1zc-repeats-6.json`](../../models/hunyuan4/alignments/01-mini/results/t1zc-repeats-6.json)）；前 300 步三份的索引器 KD 都不过（r 4.78–5.23；组合里开回 M27 后 4.91 → 1.44，主要来自关 M27，见 B4），见 [mini 对齐细节「推荐组合整体跑一次」](../../models/hunyuan4/alignments/01-mini/DETAILS.md#推荐组合整体跑一次)；生产切法与验收不同时看 E5，换 libtpu 看 F7 |
| 新开关 | 前向、反向、配套开关整套对上再上整网，先在单算子上对 GPU 写法逐元素比（例：C2 只换了前向的精确写法时，1000 步路由偏置均值 r 2.54，C3 把反向也对上后 0.89） |
| 验证顺序 | L0 逐位 → L1（G9）→ L3（G1、G4、G8、G12、G13）→ 权重侧重放（G5）→ L4（G2、G17）；编译相关开关在整网里验（F5） |
| 差大得不像舍入 | 相对误差 1e-2 以上、或只在某些形状下错：先在 TPU 上用 `jax.disable_jit()` 逐算子跑同一函数。逐算子对、jit 错就是编译 bug，不查精度开关；再扫一遍 F8 的特征 |

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
[r27]: ../../models/hunyuan4/ROOT-CAUSES.md#r27
[r28]: ../../models/hunyuan4/ROOT-CAUSES.md#r28
[r30]: ../../models/hunyuan4/ROOT-CAUSES.md#r30
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
[D30]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d30
[D31]: ../../models/hunyuan4/alignments/01-mini/DETAILS.md#d31
[D19]: ../../models/hunyuan4/alignments/02-real-weight/ALIGNMENT-REPORT.md#d19
[D20]: ../../models/hunyuan4/alignments/02-real-weight/ALIGNMENT-REPORT.md#d20
[D21]: ../../models/hunyuan4/alignments/02-real-weight/ALIGNMENT-REPORT.md#d21
[D22]: ../../models/hunyuan4/alignments/02-real-weight/ALIGNMENT-REPORT.md#d22
[案例 11]: ../../models/hunyuan4/cases/11-ihc-stream-sum-highest-einsum.md
[案例 12]: ../../models/hunyuan4/cases/12-param-norm-fp32-sum-and-drift.md
[案例 13]: ../../models/hunyuan4/cases/13-weight-side-switch-step1-blind-spot.md
[案例 14]: ../../models/hunyuan4/cases/14-ddp-fsdp-embedding.md
[案例 16]: ../../models/hunyuan4/cases/16-jit-miscompile-sub8-row-block.md
