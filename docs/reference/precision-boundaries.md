[janus](../../README.md) › [文档](../README.md) ｜ 精度边界：对不齐几乎都出在 16 位与 32 位之间

# 精度边界：对不齐几乎都出在 16 位与 32 位之间

**一句话**：到目前为止，两边对不齐几乎都发生在同一类地方：一个值从 16 位进 32 位，或从 32 位落回 16 位，再加上 GPU 上悄悄打开的 TF32（尾数约 10 位，即常说的「19 位」格式）。看代码找对不齐，就是把每个算子在两边的 dtype 路径写出来，找出不一样的那一格。例外是 32 位对 32 位的实现精度：混元 4 上 TPU 默认的 FP32 超越函数 ulp_max 62–105（GPU：CUDA 文档给 exp 2、log 1 ULP，sigmoid / silu 在 B200 上实测最大 3.4 ulp；第 13 类），dtype 路径完全相同也会对不上。另一类例外是 dtype 路径看着一样、舍几次由后端决定（第 15 类）。

这一页与模型无关。例子都取自本仓库的混元 3、混元 4，每条都给出处（台账行、开关、结果文件），读者可以自己核。判法与噪声的论证见 [`judging.md`](judging.md)，参照与 FP64 的角色见 [`principles.md`](principles.md)。

## 1. 每个算子问五个问题

对一个算子，两边各写一行，逐格比：

| 格 | 问什么 | 常见的坑 |
|---|---|---|
| 输入 dtype | 进这个算子之前是 BF16 还是 FP32 | 上一个算子把 FP32 结果先存成 BF16 再传下来 |
| 乘法 dtype / 矩阵乘精度 | 乘积在什么精度里算 | TPU 上 FP32 操作数在默认精度下只算一遍 BF16；GPU 上 FP32 GEMM 可能走 TF32 |
| 累加 dtype | 部分和、长求和在什么精度里加 | 跨芯片 / 分片的部分和先舍成 BF16 再相加；很长的 FP32 求和在硬件上有偏 |
| 输出 / 存储 dtype | 结果以什么精度落地 | FP32 算完立刻存 BF16，下一个算子再升回 FP32 |
| 舍入发生在哪 | 每一次 cast 在哪一步 | 一侧舍一次、另一侧舍两次；一侧先舍权重、另一侧不舍 |

五格都一样，这个算子就只剩求和顺序的差，属于噪声；有一格不一样，就是一个要对上的边界。

## 2. 边界类型清单

每一类给一个本仓库里的例子与处理办法。「台账」指 [`models/hunyuan4/CHANGES-LEDGER.md`](../../models/hunyuan4/CHANGES-LEDGER.md) 的条目编号（M = MaxText 一侧，G = GPU 一侧，J = janus 工具），「混元 3 台账 #n」指 [`../../models/hunyuan3/ALIGNMENT-REPORT.md`](../../models/hunyuan3/ALIGNMENT-REPORT.md) 第 5 节偏差台账的第 n 行（不是 issue 号）。

| # | 边界 | 本仓库的例子 | 处理 |
|---|---|---|---|
| 1 | 矩阵乘在 FP32 里累加，结果却先存 BF16 再给下一个算子 | 混元 4 注意力 logits：GPU 的稀疏注意力原来把 q·k 算成 BF16 再进 FP32 softmax（我们自己 eager 实现的写法），TPU flash 内核里 q·k 是 FP32 累加、不舍入（台账 G17） | GPU 一侧开关 `hy4_attn_scores_fp32`（q·k 在 FP32 里算）：已开（项目负责人 2026-11-07 20:17），GPU 重跑 `g2-*`；训练入口 `--attn-scores-fp32` 默认打开（[`train_hy4mg.py`](../../models/hunyuan4/frameworks/megatron/train_hy4mg.py)）；[混元 4 案例 06](../../models/hunyuan4/cases/06-gpu-attn-scores-bf16.md) |
| 2 | 跨芯片 / 分片的部分和先舍 BF16 再相加 | 混元 3 台账 #6：收缩维被 FSDP 切开的投影编成窗口化 collective matmul，分片部分积在 BF16 里累加；单算子对 FP64 的 rel_L2，v5p 上 qkv GPU 1.89e-3、TPU 2.33e-3，R 1.24；稠密 fc1 GPU 1.99e-3、TPU 2.58e-3，R 1.29（v7x R 1.57 / 1.58；混元 3 `phase2-l1-ops-own`） | TPU 开关 `dense_fp32_accumulation`（混元 4 台账 M12）：打开后各层最差观测点 v5p 1.01–1.09；混元 4 第 1 步见[混元 4 案例 01](../../models/hunyuan4/cases/01-step1-grad-rounding-switches.md) |
| 3 | 很长的 FP32 求和本身有偏 | 混元 3 台账 #1、#2：输出头 dgrad 沿 12 万词表的长求和，B200 张量核每组累加都朝 0 截断，长求和里同向累积；cuBLAS FP32 一次累加 12 万长的收缩维同样有偏。**测量量也会中招**：混元 4 TPU 记的 param_norm 是整棵参数树（约 4.3 亿个元素）在 FP32 里一路相加，对精确值逐步误差到 ±4e-7、平均偏低约 6e-8，单步误差和 L4 param_norm 的噪声上限同量级（[混元 4 案例 12](../../models/hunyuan4/cases/12-param-norm-fp32-sum-and-drift.md)） | 分段累加（混元 3 最终 golden 的 `dgrad-chunk`；混元 4 GPU 程序 `--head-chunk`，默认每 1024 个词一段，`hy4_head_chunk`）。起作用的是分段，不是升 FP32。测量量同理：混元 4 TPU 改记 `param_norm_exact`（每 128 个元素一段 FP32、段间 FP64），两边比精确值（台账 J13） |
| 4 | 归一化舍一次还是两次 | 混元 3 台账 #7：TE RMSNorm 在 FP32 里算完只舍一次；MaxText 写法是归一化结果、scale 各舍一次 | TPU 开关 `rmsnorm_single_rounding`（混元 4 台账 M9）；[混元 4 案例 01](../../models/hunyuan4/cases/01-step1-grad-rounding-switches.md) |
| 5 | 逐元素函数在 FP32 里算再舍，还是直接在 BF16 上算 | 混元 4 输出门：TPU 的 sigmoid 在 FP32 里算、舍 BF16 后乘；GPU 的 `torch.sigmoid` 输入是 BF16，内部在 FP32 里算、结果舍成 BF16（台账 G26 读码与 B200 实测：BF16 进 BF16 出 100% 正确舍入；[`../../models/hunyuan4/alignments/01-mini/CHECKS.md`](../../models/hunyuan4/alignments/01-mini/CHECKS.md)「逐元素函数的精度」）。混元 3 台账 #8：门控激活 silu · up，GPU 舍两次、TPU 默认舍一次，只有 73.7% 元素逐位 | 门控激活用 TPU 开关 `mlp_activation_round_per_op`（打开后 100% 逐位；[混元 4 案例 01](../../models/hunyuan4/cases/01-step1-grad-rounding-switches.md)）；输出门 sigmoid 已定责、不处理（现行 C 口径第 1 步全量抓数 `s1f-c` 的 8 个 `linear_gate` 梯度对 FP64：GPU 5.86e-3–9.44e-3，TPU 5.78e-3–8.54e-3，R 0.90–1.00，TPU 不比 GPU 远，[`t1zc-step1-train-grad-vs-fp64-n6.json`](../../models/hunyuan4/alignments/01-mini/results/t1zc-step1-train-grad-vs-fp64-n6.json)；C 口径之前的 `tz-off-rep` 是 R 0.96–0.98；[根源一览](../../models/hunyuan4/ROOT-CAUSES.md#a-out-gate)） |
| 6 | 路由 logits 要 FP32 | 混元 3：路由 GEMM 默认精度下 top-k 集合与 FP64 不同（测试模型上约 1%–2% 的 token，见 [`environment.md`](environment.md)） | TPU `float32_gate_logits` + `router_matmul_precision: highest`；GPU `moe_router_dtype: fp32`（混元 4 `hy4_mcore.py`） |
| 7 | FP32 操作数在 TPU 默认精度下只算一遍 BF16；**HIGHEST 也只是多遍 BF16 近似 FP32** | 混元 4 输出头：只开 `logits_dot_in_fp32` 时 FP32 矩阵乘在默认精度下仍是单遍 BF16，GPU 输出头是真 FP32（台账 M13）；索引器打分（台账 M23）；iHC 混合系数。混元 4 iHC 四路流的加权和不属于这一类（台账 M28）：那个 4 维的 HIGHEST einsum 被 XLA 的 algsimp 改写成 FP32 逐元素乘 + FP32 reduce，根本不上矩阵单元（[混元 4 案例 11](../../models/hunyuan4/cases/11-ihc-stream-sum-highest-einsum.md)「HLO 实证」）。**判断一处 HIGHEST 是不是多遍 BF16，要看优化后 HLO 里它还是不是 convolution** | 给这一处单独设精度：`logits_matmul_precision: highest`、`indexer_fp32_scores`、`hc_matmul_precision: highest`；输出头见[混元 4 案例 04](../../models/hunyuan4/cases/04-lm-head-single-bf16-pass.md)，索引器见[混元 4 案例 02](../../models/hunyuan4/cases/02-indexer-kd-f32-lse.md)。`ihc_stream_sum_unrolled`（逐路 FP32 乘加）让第 1 步 iHC 梯度对 CPU FP64 R > 1.2 的张量 11 → 8（后一代配置里没有复现，差异清单 U11），两种写法编译后都是 FP32 逐元素乘加，只差加法顺序（TPU 上实测：einsum 是配对相加 (t₀ + t₂) + (t₁ + t₃)，逐路是依次相加；单算子对 FP64 配对的还略好）；GPU 的 `torch.sum` 在 B200 上实测是依次相加，M28 让 TPU 的顺序与它相同；它对第 1 步梯度离 FP64 远近的作用，机制待查（[混元 4 案例 11](../../models/hunyuan4/cases/11-ihc-stream-sum-highest-einsum.md)） |
| 8 | 乘的是 FP32 主权重还是 BF16 模型权重 | 混元 4 索引器 weights_proj：原版 Megatron-Core 乘 BF16 权重、FP32 输出，MaxText 原来乘 FP32 主权重（台账 M27）。混元 3 台账 #9：路由 GEMM 的权重操作数 GPU 是 BF16；但在 TPU 默认精度下 FP32 操作数本来就先舍成 BF16，开不开开关逐位相同，差别要到混元 4 设了 `router_matmul_precision: highest` 才出现 | TPU 开关 `indexer_weights_proj_bf16_weight`（M27）、`routed_gate_bf16_rounding`（M8；[混元 4 案例 01](../../models/hunyuan4/cases/01-step1-grad-rounding-switches.md)） |
| 9 | GPU 上 TF32 悄悄打开 | 混元 3 台账 #3：容器设了 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，FP32 GEMM 进 TF32 张量核；开着时 FP32 乘法里输出头 dgrad 沿 ŵ̄ −8.1e-4 | 不信读回的开关值，启动时对 FP64 实测：混元 4 GPU 训练入口的 `tf32_probe`（[`models/hunyuan4/frameworks/megatron/train_hy4mg.py`](../../models/hunyuan4/frameworks/megatron/train_hy4mg.py)）用 4096 × 4096 的 FP32 GEMM 对 FP64 算相对误差，写进每个运行的 `meta.json` |
| 10 | 梯度以 BF16 落地 | 混元 3 台账 #20：TPU 把一维 norm 权重、输出层、embedding 这几类 FP32 参数先转 BF16 再参与计算，梯度先以 BF16 落地；整网按能量对 FP64 的 rel_L2：GPU 7.66e-4、TPU 9.22e-4，R 1.20 | TPU 开关 `rmsnorm_scale_grad_fp32`、`logits_dense_wgrad_fp32`、`embedding_grad_fp32`（前向不变；混元 4 台账 M12）。`embedding_grad_fp32` 在数据并行＋梯度累积下要带 MaxText `d5c9b113`（混元 4 台账 M55）之后的代码，否则 XLA 静默丢掉前面微批的词嵌入梯度（第 1 步词嵌入梯度范数只有正确值的 0.51 倍），见[差异清单 A7](tpu-vs-gpu-differences.md#a7-fp32-参数的梯度先以-bf16-落地) |
| 11 | 优化器内部的精度 | 混元 4 Muon 的 Newton–Schulz：GPU（emerging_optimizers，`muon_fp32_matmul_prec: medium`）在 BF16 张量上迭代，TPU 原来留在 FP32 量级；连续 8 步回放里两边的 Muon 增量最多差 12.8%，GPU 离 FP64 1.2%–13%（[`t1-optim-replay-tpu.json`](../../models/hunyuan4/alignments/01-mini/results/t1-optim-replay-tpu.json)）。同一回放里 Adam 管的小张量：TPU 的 FP32 除法、sqrt、偏差修正 1 − β^t 分别是 2 / 3 / 108 ULP（GPU 的除法与 sqrt 正确舍入），第 5 步 `hc_head_base` 的主权重增量对 FP64：GPU 1.49e-6、TPU 5.70e-6，R 3.82 | TPU 开关 `muon_mcore_ns_bf16`（混元 4 台账 M30、M31），差降到 1.6%；Adam 用 `muon_adam_accurate`（M34：带修正的商、一步牛顿的 sqrt、−expm1(t·log β)），单算子 1 / 1 / 2 ULP；Muon 的语义差别（Nesterov 写法、NS 前归一化、标量参数的 Adam）用开关对上（M15–M17；[混元 4 案例 08](../../models/hunyuan4/cases/08-muon-mcore-switches.md)） |
| 12 | FP32 里的病态标量：**放大器，不是 bug** | 混元 4 索引器 KD：KL 每行约 1.1e-4 nat，log-sum-exp 约 7.1；f32 里 lse 差 1 格就是 KL 的 4.3e-3，TPU 的 log-sum-exp 平均低 0.95 格（来自 TPU 近似实现的 FP32 exp / log，混元 4 对齐报告 D14），第 0 层同一行记下的 KD 偏低 1.16e-3（[`t1-layer0-kd-attribution.json`](../../models/hunyuan4/alignments/01-mini/results/t1-layer0-kd-attribution.json)） | 见下面「怎么分辨病态与 bug」。病态只是放大器，被放大的偏差来自第 13 类；T1 已打开高精度开关（台账 M33），第 1 步偏差消失，300 步 L4（`t1z-l4`）的 KD 系统偏差也消失（带符号均值 0.01 N）；[混元 4 案例 02](../../models/hunyuan4/cases/02-indexer-kd-f32-lse.md) |
| 13 | FP32 超越函数的实现精度（32 位对 32 位） | 混元 4：TPU v5p 上 XLA 默认的 FP32 exp / log / sigmoid / silu 是近似实现，ulp_max 105 / 62 / 83 / 86；GPU 一侧 CUDA 文档给 exp 2、log 1，sigmoid / silu 是 PyTorch eager 的 FP32 opmath，B200 实测最大 3.4 ulp（台账 G26）（[`tpu-elementwise-ulp.json`](../../models/hunyuan4/alignments/01-mini/results/tpu-elementwise-ulp.json)）；第 12 类 KD 的 lse 偏低 0.95 格来自这里，hy4-mini 逐算子回放里交叉熵（对 FP64 的 rel_L2：GPU 3.33e-8、TPU 9.81e-8，R 2.95）也来自这里，换成高精度写法后 TPU 3.82e-8，R 1.15 | MaxText 开关 `transcendental_accuracy_highest`（台账 M32，`lax` 的 `accuracy=HIGHEST`），T1 打开（M33）：第 1 步 KD 的 lse 偏差 −0.953 → +0.025 格（[`t1-step1-transcendental-accuracy.json`](../../models/hunyuan4/alignments/01-mini/results/t1-step1-transcendental-accuracy.json)）。**`lax.logistic` 带 HIGHEST 与默认逐位相同**，sigmoid / silu 要改用 HIGHEST 的 exp 搭（M37）才降到 3 / 4 ULP（`tw-elem4`）；开了也要实测：混元 4 只打开 M37 时 TPU loss 反而偏高变多，原因在反向（自动微分的式子与 PyTorch 不同），M39 改用 PyTorch 的解析式后两边的差收拢（混元 4 根源一览第 18 行）。log 用 HIGHEST 仍是 3 ULP。注意力内核里的 exp 不在其内 |
| 14 | Python 常数跟着 BF16 张量被转成 BF16 | 混元 4 splash 的 base2 模式：内核写 `q *= LOG2E`，q 是 BF16，JAX 把常数也转成 BF16 的 1.4453125（比 log2 e 大 1.81e-3），而之后换回自然底与 sink 都按精确值；注意力 logits（不含 sink）整体乘 1.0018。真实宽度截层第 1 步主 loss 因此偏低约 1e-4（[混元 4 根源一览第 24 行](../../models/hunyuan4/ROOT-CAUSES.md#r24)） | 读内核代码，看常数和谁相乘、那个张量是什么 dtype；CPU 上 FP64 前向只加这一处放大，看能不能预测出 TPU 的移动。这一处用 `sa_use_base2_exp: false`（M43） |
| 15 | 同一份 HLO，后端决定舍几次 | 混元 4 纯数据并行：词嵌入 scatter 排不排序由后端按下标数与表行数的门槛决定，HLO 里看不出，不排序时高频词的梯度在 BF16 里加一次舍一次（[TPU 自比「纯数据并行」](../../models/hunyuan4/alignments/04-tpu-self/impacts/01-data-parallel.md)）；iHC：融合内部的 convert 不产生舍入、跨融合边界的按 BF16 写回，同一个值在两颗 kernel 里被读成两种精度（[TPU 自比「iHC 不强制舍成 BF16」](../../models/hunyuan4/alignments/04-tpu-self/impacts/05-ihc-bf16-rounding.md)）；hc_head 收拢成一路后 GPU `.to(bfloat16)`、TPU 推断被省掉（台账 M52，[差异清单 A13](tpu-vs-gpu-differences.md#a13-hc_head-收拢之后没舍-bf16)）；输出头反向：末层 RMSNorm 反向的每 token 行和被并进输出头 dgrad 卷积 kernel 时，它用没舍的 FP32 余切、主项用写进显存的 BF16 余切，`hc_head` 第 1 步梯度偏一个常数，会不会这样切随 libtpu 版本和切分变（[差异清单 A16](tpu-vs-gpu-differences.md#a16-hc_head输出头反向-kernel-合并末层-rmsnorm-反向行和时省掉一次-bf16-舍入)）。门槛和融合切法都随 libtpu 版本变（[`libtpu-052.md`](../../models/hunyuan4/alignments/04-tpu-self/libtpu-052.md)） | HLO 看不出，要看后端：后端指令里数舍入指令（如 `vpack`）、看 scatter 前有没有排序；要固定的舍入点用 `reduce_precision`；换 libtpu 版本重测 |
| 16 | 一个 BF16 张量在前向被几处用到，反向时各处的梯度在 BF16 里按 autograd 顺序逐个相加，每加一次舍一次 | 混元 4 iHC：BF16 的流在一层里用到三处（加权和、门控、残差），GPU（Megatron-Core autograd）给每处一份 BF16 梯度、逐个加进流的梯度，每次舍 BF16；TPU 在 excess precision 下按 FP32 加完才舍（推断，没看 HLO / LLO，依据是下面两头的因果实验）。mini 1000 步主 loss 后段偏高的主因就在这里：TPU 照 GPU 舍法（只开 M58 那一格）后第 551–1000 步对 GPU 均值 +9.4e-6 → +1.9e-6（p 0.016），GPU 反过来照 TPU 舍法自己 +3.9e-6（p 0.002，台账 G37）。限定：C 口径（M36、M52、M58 都开）6 份 +6.9e-6，对只开 M36 的对照只低 2.4e-6（p 0.32），6 份里 5 份尾部仍在，比只开 M58 大（p 0.024，[`mini-factorial-tpu-6.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-tpu-6.json) 与 [`mini-factorial-c-vs-m58-6.json`](../../models/hunyuan4/alignments/01-mini/results/mini-factorial-c-vs-m58-6.json)），现行 `tc-a` 自身后段 +8.15e-6（[根源一览第 30 行](../../models/hunyuan4/ROOT-CAUSES.md#r30)、[差异清单 A17](tpu-vs-gpu-differences.md#a17-ihc-流梯度gpu-三份各舍-bf16-逐个相加tpu-在-fp32-里加完才舍mini-主-loss-后段偏高的主因)） | 先在 CPU 上用参照一侧的同一段代码逐位核出它的相加分组与舍入点；TPU 用 custom_vjp 把余切逐份舍 BF16、按同样分组相加（开关 `ihc_stream_grad_gpu_order`，台账 M58，只作用于 XLA 路径）。有残差流、或一个张量被多处复用的模型都要查这一类 |

### 一张图看一个边界

把一个算子沿计算顺序拆成几格，每格写这一步落在 32 位还是 16 位，两边各一行。两行走法不同的那一格（🟥）就是要对上的边界。例：混元 4 注意力 logits（台账 G17）。

| 一侧 | q、k 输入 | q·k 乘积 | 累加 | logits 落地 | softmax |
|---|---|---|---|---|---|
| TPU（tokamax splash） | 16 | 32 | 32 | 32 | 32 |
| GPU 改前（我们自己的 eager 实现，`g1-*`） | 16 | 32 | 32 | 🟥 **16** | 32 |
| GPU 改后（`--attn-scores-fp32`，`g2-*` 起） | 16 | 32 | 32 | 32 | 32 |

### 对照：bug 和病态，按写法判，不按差的大小判

| | 混元 4 注意力 logits（G17） | 混元 4 索引器 KD（D1） |
|---|---|---|
| 写法 | 错：矩阵乘在 FP32 里累加完，又先存成 BF16 才进 softmax，生产的融合内核和 TPU 都不这样 | 对：两边都在 FP32 里算 log_softmax 与 KL |
| 差有多大 | 很小：第 1 步只改这一处，grad_norm 的变化（相对 −3.0e-5）在 GPU 自己重跑的散布（7.3e-5）之内，lm_loss 只变 +8.1e-7 | 不小：修之前 L4 的 KD loss 超出噪声 3 倍以上 |
| 怎么判 | bug，改（GPU 修正） | 病态放大了一个实现精度差：KL ≪ log-sum-exp，f32 里 lse 差 1 格就是 KL 的 4.3e-3；被放大的那 0.95 格来自 TPU 近似的 exp / log（第 13 类，D14）。T1 已改用高精度版本（M33），第 1 步与 300 步 L4 的偏差都消失 |
| 教训 | 差小不代表没问题，看代码路径的精度边界 | 差大不代表有 bug，先看这个标量的条件数 |

### 量法的盲区：权重侧开关在第 1 步测不到

从 BF16 检查点起训时，FP32 主权重一开始的值全部落在 BF16 格点上（混元 4 hy4-mini 的输出头、weights_proj、路由权重都是这样；输出头的输入是 BF16 的 norm 输出，同样在格点上）。这时只作用在**权重一侧**舍入或精度的开关是空操作：HIGHEST 把 FP32 操作数拆成高、低两部分 BF16 分几遍乘，低位部分全是 0；「先舍 BF16 再乘」对格点上的值是恒等变换。所以第 1 步前向逐位相同、反向的差远小于 GPU 自己的噪声。第 1 步更新之后主权重就离开格点（[`t1-bf16-offgrid-frac.json`](../../models/hunyuan4/alignments/01-mini/results/t1-bf16-offgrid-frac.json)），开关才开始起作用。混元 4 的 M13（输出头矩阵乘精度）第 1 步逐位不变，300 步 L4 上关掉它两边的差却到 r 6.03。**第 0 / 1 步逐位相同不能当作开关没用的证据**；权重侧开关要用训练后的权重重放，或看多步 A/B。从 FP32 随机初始化起训时不成立。[混元 4 案例 13](../../models/hunyuan4/cases/13-weight-side-switch-step1-blind-spot.md)

### 编译器省掉的舍入：excess precision

XLA 默认允许 excess precision：融合内部的中间值可以不按代码写的 BF16 落地。这让 TPU 的舍入位置和代码字面不一样，但**不一定离精确值更远**。混元 4 第 1 步关掉它（`--xla_allow_excess_precision=false`）做过对照：205 个梯度对 CPU FP64 的 R 中位数 0.967 → 0.986，R > 1.2 的 12 → 17，`embed_tokens` 的梯度坏 27 倍，更差（[`t1-step1-compiler-pass-variants.json`](../../models/hunyuan4/alignments/01-mini/results/t1-step1-compiler-pass-variants.json)；差异清单 F1）。混元 3 关掉它时 loss 偏移变小一半、param_norm 不变（混元 3 台账 #10）。所以它是一个要实测的变量，不能默认「关掉就和 GPU 一样」。

反过来的例子：混元 4 iHC 的分支输入与更新后的流，GPU（eager）每次都存成 BF16，TPU 代码里也写了 astype，但打开逐路 FP32 乘加（M28）以后，TPU 1000 步主 loss 一直同号偏高（5 次运行对 GPU 8 份均值 +3.7e-6 至 +8.6e-6，GPU 28 对的带符号均值最大 3.8e-6）。当时推断原因是这次 FP32 → BF16 → FP32 往返在 excess precision 下被后端省掉了一部分（第 1 步三处一起模拟才对上第 0、1 层 KD 的变化，第 5 层对不上，差异清单 U12）。M36 只管分支输入与更新后的流这两处；后来实测，1000 步主 loss 后段偏高的主因是流梯度的相加写法（第 16 类，台账 M58），前段鼓包来自 hc_head 收拢后没舍（台账 M52）；三处都补上（C 口径）后第 551–1000 步还剩约 +6.9e-6（6 份），机制待查（[根源一览第 28、30 行](../../models/hunyuan4/ROOT-CAUSES.md#r30)）。优化后 HLO 里 convert 都在，看不出；v5p 最小复现的后端指令里核到了：RMSNorm 均方那颗 kernel 用的是没舍的 FP32，归一化那颗读的是舍过的 BF16（[TPU 自比「iHC 不强制舍成 BF16」](../../models/hunyuan4/alignments/04-tpu-self/impacts/05-ihc-bf16-rounding.md)；差异清单 A6）。只对这两处改用 `reduce_precision` 强制舍入（开关 `ihc_force_bf16_rounding`，台账 M36）后，1000 步主 loss 带符号均值（8 份尺子）N 3.78e-6、TPU 7.99e-6 → 2.87e-6，r 2.12 → 0.76（[`t1zr-l4.report.md`](../../models/hunyuan4/alignments/01-mini/results/t1zr-l4.report.md)）；这条曲线只是开、关各一份，只改执行方式的 4 份重复，主 loss 带符号均值的 r 就能在 0.36–2.08 之间（同一把 8 份尺子推算：[`mini-base2-loo.json`](../../models/hunyuan4/alignments/01-mini/gpu-noise/results/mini-base2-loo.json) 的 `loss_signed_mean` ÷ 3.78e-6），单看曲线分不清开关还是轨迹。舍入点确实被改回来的证据是最小复现的后端指令，加上整网里只关它时第 1 步索引器 `k_norm` 梯度变远（差异清单 F5）；整网训练步的后端没核。要和 GPU 的舍入点一致时，用 `reduce_precision` 而不是 astype。

度量工具同样会中招：混元 4 量主权重离开 BF16 网格的比例时，用 `x != f32(bf16(x))` 判，TPU 上结果几乎全是 0（连按 FP32 存的参数也是）；改成看位（FP32 能被 BF16 精确表示当且仅当低 16 位全是 0）才对。优化后的 HLO 里那对 convert 仍在，省掉发生在融合的代码生成里，HLO 上看不出来（[混元 4 案例 13](../../models/hunyuan4/cases/13-weight-side-switch-step1-blind-spot.md) 第 8 节）。**判断一个值在不在某个格点上，看位，不要做往返比较。**

### 怎么分辨病态与 bug

- 这个标量的**输入**对参照两边一样近（R ≤ 1.2）：混元 4 KD 的教师 q / k、索引器分数对 CPU FP64 的 rel_L2，GPU 3.33e-3–5.27e-3、TPU 3.15e-3–5.22e-3，R 0.94–0.99（[`t1-l1-kd-qkclip-vs-fp64.json`](../../models/hunyuan4/alignments/01-mini/results/t1-l1-kd-qkclip-vs-fp64.json)）。
- 由它出发的**梯度**两边一样近：混元 4 的 20 个索引器梯度对 CPU FP64 R 0.81–1.05（t1s 一代；R 最大的张量 GPU 6.81e-3、TPU 7.14e-3；[`t1s-step1-grad-vs-fp64.json`](../../models/hunyuan4/alignments/01-mini/results/t1s-step1-grad-vs-fp64.json)）。
- 只有这个标量本身对不上，而它是**大项相减剩下的小量**：用大项的 ulp 量它的差。差在一两格以内，就是病态，不是两边算法不同。还要先排除逐元素函数本身的实现差：同一个程序在 CPU 上偏不偏（混元 4 的 KD：CPU 不偏，TPU 偏，偏差来自 TPU 近似的 exp / log）。
- **长求和的条件数**也是同一类问题：混元 4 iHC post 门控的梯度是对 4096 个 token 的求和，κ = Σ|项| ÷ |Σ项| 约 25–1400，每项 1e-4 量级的系统差加总后成 1e-2 量级的相对差（[`t1-ihc-post-grad-conditioning.json`](../../models/hunyuan4/alignments/01-mini/results/t1-ihc-post-grad-conditioning.json)）。
- **误差小于一个 BF16 ulp 时，R 是两次舍入实现之比**：混元 4 第 1 层索引器 `k_norm.weight` 的梯度对 CPU FP64，t1z 上 GPU 6.69e-4、TPU 1.07e-3，R 1.60；只换一代 TPU 写法 R 0.72（两边都远小于 BF16 的单位舍入 u = 2^-8 ≈ 3.9e-3；[`t1z-step1-grad-vs-fp64.json`](../../models/hunyuan4/alignments/01-mini/results/t1z-step1-grad-vs-fp64.json)、[`t1-step1-compiler-pass-variants.json`](../../models/hunyuan4/alignments/01-mini/results/t1-step1-compiler-pass-variants.json)）。读 R 时要看误差的绝对大小；混元 4 的 L3 对 GPU 噪声为 0、两边都小于 u 的张量按**分辨率下限**直接判过（PROTOCOL 第 5 节；差异清单 G8）。

### 不是 dtype 的一类：语义 bug

| 例子 | 机制 | 出处 |
|---|---|---|
| 梯度累积把「取最大」的统计量也求和了 | MaxText 对各 microbatch 的 aux 整棵树求和，QK-Clip 的最大 logit 被加成约累积步数倍，裁得过多 | 台账 M19；[混元 4 案例 07](../../models/hunyuan4/cases/07-qk-clip-bugs.md) |
| eod 掩码下 aux loss 的权重 | 索引器 KL 按查询行数还是有效 token 数加权、MTP loss 按本 rank 还是整个 microbatch 重标，两边写法不同 | 台账 M21、M22（G10、G11） |

## 3. 十分钟看代码

1. 在两边代码里 grep 这些词，每一处都是一个可能的边界：

   ```
   astype  .to(  .float()  .bfloat16()  .half()  preferred_element_type  precision=  matmul_precision
   allow_tf32  NVIDIA_TF32_OVERRIDE  autocast  reduce_precision  dtype=  _keep_in_fp32  fp32
   ```

2. 沿前向（和反向）路径，每个算子两边各写一行 dtype 路径，然后逐格 diff：

   | 算子 | 一侧 | 输入 | 乘法 / 精度 | 累加 | 输出 / 存储 | 舍入在哪 |
   |---|---|---|---|---|---|---|
   | 例：输出门 | TPU | BF16 | FP32（sigmoid） | — | BF16 | sigmoid 之后舍一次，乘完再舍 |
   | 例：输出门 | GPU | BF16 | FP32（sigmoid 内部 opmath） | — | BF16 | sigmoid 在 FP32 里算、舍 BF16 后乘，再舍 |

3. 有差的格子，按第 2 节归类，找对应的开关；没有开关的，在台账里记一行再决定谁改。

## 4. 怎么验证

- **参照**：CPU 上的 FP64 重算，与被测不同源（混元 4 第 1 步整网用 [`step1_ref.py`](../../models/hunyuan4/frameworks/megatron/step1_ref.py)，单算子、单层同样对 FP64）。参照要比两边都准一个量级以上，且不能与任何一边共用写法，否则误差同向、R 偏向那一边；被测一侧产出的数不作参照（差异清单 [G1](tpu-vs-gpu-differences.md#g1-真值是-cpu-上的-fp64-整网重算与被测不同源)）。
- **R 比**：R = rel_L2(TPU, 参照) ÷ rel_L2(GPU, 参照)，≤ 1.2 过，即 TPU 不比 GPU 离精确算术远（[`judging.md`](judging.md)）。
- **第 1 步逐张量梯度**：每个张量对 GPU 第 1 步参照的噪声算 r，超过 N 的对参照定责（R ≤ 1.2 算已定责；有超 2N 而未定责的判不过）；N = 0 且两边对参照都小于 u 的按分辨率下限过；混元 4 的做法见 [`../../models/hunyuan4/alignments/01-mini/CHECKS.md`](../../models/hunyuan4/alignments/01-mini/CHECKS.md) L3。
- **单算子隔离回放**：把 GPU 抓到的同一个输入喂给 TPU 的单个算子，对 FP64 算 R。注入要覆盖全部输入（包括关键字参数），换不了就报错；否则包在别的模块里、用关键字传输入的算子（如混元 4 iHC 包着的分支）吃的是本侧上游的输入，误差里带着上游（janus `8cacde9`；混元 4 阶段 A 第 0 层稠密 MLP 对 FP64：GPU 2.33e-3、TPU 2.22e-3，R 0.95，[`hy4-a-l1-op-tpu-fix.json`](../../models/hunyuan4/alignments/01-mini/results/hy4-a-l1-op-tpu-fix.json)；差异清单 [G9](tpu-vs-gpu-differences.md#g9-单算子回放要换掉全部输入)）。
- **为什么 GPU 重跑的噪声量不出舍入位置的差**：GPU 同设置重跑之间只差原子加的顺序，每一份的舍入位置完全相同；混元 4 第 1 步里第 0、1 层的索引器梯度三份甚至逐位相同。舍入位置不同造成的差在这把尺子上要么被判成「超出噪声」、要么因为噪声为 0 根本无法判，所以要另用 FP64 参照定责。
