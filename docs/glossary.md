[janus](../README.md) › [用户指南](guide/README.md) ｜ 术语表；所属[附录](guide/appendix.md)

# 术语表

janus 文档里反复出现的词，按主题分组。每条先用一两句定义，再指向详细出处。其他文档用到这些词时不再重新定义，只链到这里。混元 4 文档里另有一批词（定责依据、分辨率下限、留一法 /「和 GPU 重跑分不开」、臂、轨迹偶然、离线筛等），定义在[混元 4 知识库的术语对照表](../models/hunyuan4/knowledge.md#术语对照表)。括号里的英文名是代码、目录文件、结果文件里的写法。

本页依次是参照与运行、判定、loss 怎么比、分级、golden 与尺子与验收、迁移验收与日常回归、仓库与工具七组。

## 参照与运行

<a id="golden"></a>
### golden（golden reference）

源平台（通常是 GPU）跑出的参照数据，包括逐步 loss 等标量和第 1 步的逐层激活与梯度。同一配置跑几份，`meta.json` 的 `side` 依次写 `golden-1`、`golden-2`……

- 详见：[`principles.md`](reference/principles.md)

<a id="test"></a>
### 被测（test）

拿来和参照比的那次运行。迁移验收里是 TPU，第一阶段 GPU 自比时是另一次 GPU 运行，日常回归里是今天的 TPU 运行。目录文件里写 `runs.test`。

- 详见：[`reference/catalog-schema.md`](reference/catalog-schema.md)

<a id="noise-reference"></a>
### 噪声参照（noise reference）

回答「本来就会有多大的差」：两次只改执行方式、不改数学的运行之间的差，按运行对给出（`runs.noise`）。门槛从它算出，不手写。

- 详见：[`judging.md`](reference/judging.md#噪声参照尺子从哪来)「噪声参照」

<a id="count-units"></a>
### 份 / 对 / 条 / 张量（计数单位）

四种计数不能混用：

- 份：一次运行，如 `rcc-a`（最终 golden 的第 1 份）。
- 对：两份运行配成的一次比较。n 份两两配对得 n(n−1)/2 对：3 份 3 对，5 份 10 对，6 份 15 对，8 份 28 对，10 份 45 对。
- 条：检查项目录里的一项检查。
- 张量：一条检查里比的单个数组。

写成「跑了 10 份重跑，两两配对得到 45 对比较」，不写「45 组测试」。各对共用运行，不是独立样本。

- 详见：[`judging.md`](reference/judging.md#参照要几对)「参照要几对」

<a id="tensor-counts"></a>
### 1781 / 1193 / 1480 / 50（混元 3 的张量个数）

同一份 4 层权重，按不同的切法数出来的张量个数不同，文中几个数的来历：

| 个数 | 怎么数 | 出现在哪 |
|---|---|---|
| 1781 | HF 命名，一个矩阵一个张量。3 个 MoE 层 × 192 个专家 × 3 个矩阵（gate、up、down）= 1728，加其余 53 个：第 0 层 11 个（4 个 norm、q / k / v / o 四个投影、gate / up / down）、每个 MoE 层 13 个（4 个 norm、4 个投影、路由权重、路由偏置、共享专家 3 个），再加词嵌入、最后的 RMSNorm、输出层 | 权重转换 L0（`L0.ckpt.convert.*`） |
| 1193 | Megatron 布局：q / k / v 融成一个 `linear_qkv`，gate / up 融成一个 `linear_fc1`，输入 RMSNorm 并进投影的 `layer_norm_weight`。每个专家 2 个矩阵，3 × 192 × 2 = 1152，加其余 41 个 | GPU 一侧的参数 |
| 1480 | 按 GPU 8 个 rank 分的初始参数清单：1152 个专家张量各在一个 rank 上，41 个非专家张量每个 rank 一份（41 × 8 = 328） | L0「载入后的参数」（`L0.weights.init`） |
| 50 | 第 1 步抓数只留每层 0、1 号两个专家的参数：第 0 层 8 个、每个 MoE 层 13 个，加词嵌入、最后的 RMSNorm、输出层 | L0「FP32 主权重」、L2 / L3 的梯度与状态 |

param_norm、grad_norm 是全部可训练参数（不含 3 个路由偏置，它们是 buffer）的全局 L2 范数，约 124 亿个元素合成一个数。

<a id="setting"></a>
### 设置（setting）

会改变数值轨迹的开关组合，如确定性开关、分布式优化器开关。噪声参照只能取同一种设置的运行对。不同设置之间本身就有持续偏移，混进来会把尺子撑松。

- 详见：[`judging.md`](reference/judging.md)

<a id="perturbation-arm"></a>
### 扰动臂（perturbation arm）

TPU 同一程序重跑逐位相同，噪声为 0，当不了尺子，所以改用只改执行方式的 TPU 运行，例如换 remat 档位、改片上内存上限、改调度开关。`side` 写 `tpu-perturb-<n>`。每个臂要先证明自己只改了执行。

混元 3 用了三组，用途不同：L4 说明 TPU 不是挑出来的好样本（v5p 5 份、v7x 4 份，对齐报告第 3 节）；（历史）第 1 步整步的噪声并集（每个平台 7 份第 1 步状态读，两两 21 对），2026-11-08 撤回，现行第 1 步只用 GPU 3 对，见[两侧执行噪声的并集](#union-noise)；日常回归的 TPU 执行噪声（v5p 基线加 7 个臂共 8 份，两两 28 对，第 7 章 7.6）。

- 详见：[`reference/regression.md`](reference/regression.md) 第五节

<a id="lkg"></a>
### LKG（last-known-good）

日常回归的参照：同一平台、同一档位、同一并行变体上，上一次全部通过的 TPU 运行。按「模型 × 平台 × 档位 × 并行变体」各登记一份，在 `models/<模型>/regression/lkg.yaml`。

- 详见：[`reference/regression.md`](reference/regression.md) 第七节

### 对照臂

当天在同一平台上重跑 LKG 那个提交。它和 LKG 不逐位，说明变的是环境（镜像、硬件、集群），不是代码。

- 详见：[`reference/regression.md`](reference/regression.md) 第二节

### 运行 / 运行名 / 运行目录（run）

一次训练加抓数。运行名全局唯一。运行目录按统一格式存 `meta.json`、`metrics.jsonl`、张量与索引，两侧格式相同。

- 详见：[`reference/dump-format.md`](reference/dump-format.md)

### side

运行在对比里的角色，写在 `meta.json`，取值为 `golden-<n>`、`tpu`、`tpu-perturb-<n>`、`gpu-perturb-<n>`。写成别的值，训练脚本启动即报错，`janus run` 读到时报 `error`。

- 详见：[`reference/dump-format.md`](reference/dump-format.md)

### POC

项目的云资源（存储桶、集群）。仓库里不写真实桶名，一律用 `gs://<POC 桶>/…` 占位。

- 详见：[`layout.md`](dev/layout.md#每层放什么不放什么)「每层放什么、不放什么」

## 判定

<a id="threshold"></a>
### 门槛（threshold）

判过不过的刻度。它和 golden 是两回事：golden 是被比的对象，门槛是尺子上的刻度。目录里每条检查写一种门槛：`exact`、`calibrated`、`pending` 或数值。

- 详见：[`reference/checks/README.md`](reference/checks/README.md)

<a id="calibrated"></a>
### calibrated

门槛由噪声参照现场算出。噪声上限 N 是各参照对在同一指标上的最大值，r = 被测值 / N。r ≤ 1 过，r > 2 不过，其间待定（结果文件里分别是 `pass`、`fail`、`inconclusive`）。参照不足 3 对时最多判待定。

- 详见：[`judging.md`](reference/judging.md#判定1-倍和-2-倍)「判定：1 倍和 2 倍」

<a id="exact"></a>
### exact

逐位相同才过。用在本该逐位的地方：L0 输入、确定性重跑、抓数不改数值、日常回归的回归门。

- 详见：[`reference/checks/bitwise_equal.md`](reference/checks/bitwise_equal.md)

### pending（待标定）

噪声参照还没标定时的门槛。被测值照算，最多判待定；非有限值和超过单步硬上限照样判不过。

- 详见：[`reference/checks/README.md`](reference/checks/README.md)

### 数值门槛 + source

少数检查用固定数值门槛，必须写 `source`，注明出自哪次运行的实测，或写「推断：<推导>」。任何门槛都不能比噪声严。

- 详见：[`judging.md`](reference/judging.md#数值门槛)「数值门槛」

<a id="status"></a>
### pass / fail / inconclusive / error

每条检查的四种状态。`inconclusive` 表示证据不足，落在 1–2 倍之间或参照不够，不能直接签字。`error` 表示没法判，比如数据缺失、目录不合规。

- 详见：[`reference/result-schema.md`](reference/result-schema.md)

### 硬检查（hard）

不写 `hard` 时默认为硬检查：`fail` 或 `error` 会让 `janus run` 的退出码非 0。例外是规格写明「默认只诊断」的 `ulp_distribution`。写 `hard: false` 则只报不拦。

- 详见：[`reference/catalog-schema.md`](reference/catalog-schema.md)

### 逐位相同（bitwise equal）

两份数据按原始字节完全一样。BF16 就按 BF16 的 16 位比。

- 详见：[`principles.md`](reference/principles.md)

<a id="ulp"></a>
### ULP

一个浮点数到相邻浮点数的距离，即「最后一位的一格」。ULP 分布只用于诊断，不当门槛。

- 详见：[`judging.md`](reference/judging.md)

<a id="fp64-attribution"></a>
### FP64 定责

某个算子两边差超线时，用同一份真实输入在 FP64 里算一遍，看谁离得远。两边一样远是平台固有的差，TPU 更远是 TPU 的问题。FP64 只用来定责，不当标准答案。

- 详见：[`principles.md`](reference/principles.md)

<a id="r-ratio"></a>
### R-ratio

R 比是单算子的三方比：R = rel_L2(被测, FP64) / rel_L2(参照, FP64)。R ≈ 1 表示两边离真值一样远；R 约 1.41 是多舍入了一次同量级的特征。门槛 1.2 是推断的起点：干净实现实测 R = 1.00，多舍一次实测 R = 1.41（≈ √2），1.2 取在两者之间。

R ≈ 1 的前提是两边走同一条精度路径（BF16 乘、FP32 累加、出口舍一次），误差同量级。两种情况不能这么读：分母接近 0（GPU 一侧恰好几乎精确），R 会被放大；两边精度路径不同，例如路由打分 GPU 3.8e-6、TPU 4.8e-8，R 约 0.013，只说明 TPU 更近精确。门槛只拦 R > 1.2 一侧。

混元 4 的路由一致率借用这个比值（每层 TPU 的翻转率 ÷ GPU 各份的最大值），多一档：≤ 1.2 过、1.2–2 待定、> 2 不过。

分母有三种用法：单算子与混元 3 用参照（GPU）离 FP64 的值；混元 4 GPU ↔ TPU 第 1 步用 GPU 三份离 FP64 的中位数；混元 4 TPU 自比用 max(参照离 FP64, GPU 三份中位数)。混元 4 另有分辨率下限：GPU 噪声为 0、且两边离 FP64 都小于一次 BF16 舍入 u = 2^-8 的张量按下限判过，不用 R（[尺子 2.3](guide/ruler.md#23-两种用法)）。

- 详见：[`design/0003-l1-op-replay.md`](dev/design/0003-l1-op-replay.md)

### 降精度复现

在 CPU 上模拟一边的舍入方式（如 GPU 张量核的累加），逐位命中那一边的输出，以此证明差异完全由已知舍入决定。

- 详见：[`design/0003-l1-op-replay.md`](dev/design/0003-l1-op-replay.md)

<a id="sign-off"></a>
### 签字

宣布「TPU 和 GPU 对齐了」。需要同时满足三个条件：地基逐位且两边可复现；差在噪声内且没有方向性偏移；超线处都已定责。

- 详见：[`principles.md`](reference/principles.md#什么时候能签字)「什么时候能签字」

<a id="signoff-checklist"></a>
### 签字清单（1a–3b）

把签字的三个条件拆成 8 行可核对的条目：1a–1c 是输入与可复现，2a–2c 是在噪声内，3a–3b 是超线已定责。文中「签字项 2b」指第 1 步的梯度与优化器状态那一行。

- 详见：[第 5 章 5.6](guide/05-reading-results.md#56-签字清单现在离签字还差什么)

<a id="ledger"></a>
### 偏差台账（#1–#20）

对齐报告第 5 节的表：我们测到的每一处 GPU↔TPU 差别，各写 GPU 一侧与 TPU 一侧的机制、量级、处理方式与证据。文中「台账 #5」「台账注 20」指这张表的第 5 行、第 20 行的注。

- 详见：[对齐报告](../models/hunyuan3/ALIGNMENT-REPORT.md)第 5 节

<a id="near-tie"></a>
### 路由近似平局

某个 token 的第 8 名与第 9 名专家「分数 + 偏置」只差舍入量级（混元 3 第 1 步隔离回放里，翻转 token 的分差中位数 2.7e-5 – 6.3e-5，全部 token 是 3.1e-4 – 8.0e-4）。路由器自己的精度不是原因：喂两边相同的输入，一个都不翻；差在路由器的输入，即上游注意力的舍入。两种硬件的舍入不同，这类 token 会各自选到不同的专家。第 1 步每层 0.44%–0.56% 的路由槽位如此，是第 1 步梯度差的大头（台账 #5）。

- 详见：[对齐报告](../models/hunyuan3/ALIGNMENT-REPORT.md)第 4 节注 3

## loss 怎么比

逐步相对差 dₜ = (被测ₜ − goldenₜ) / |goldenₜ|，带符号，窗口 [start, end) 与分母写死在目录里。

| 术语 | 意思 |
|---|---|
| d₀（首步差） | 第一步的 dₜ。只检验数据、权重、前向语义；GPU 自比时恒为 0，单独给门槛 `first_step_tol`，不进噪声判定 |
| <a id="signed-mean"></a>带符号均值（signed mean） | mean(dₜ)。随机差正负抵消后剩下固定方向的偏移，是判定量 |
| <a id="mare"></a>MARE | mean(\|dₜ\|)，整体波动幅度，同报 |
| 最大单步（max step） | max\|dₜ\| 及步号。平均会盖住尖峰，可设单步硬上限 |
| 标准误（`signed_mean_se`） | 带符号均值的标准误；\|均值 / 标准误\| 远大于 2 提示有持续偏移。只提示，不判定 |

详见 [`judging.md`](reference/judging.md#loss-怎么比四个数)「loss 怎么比：四个数」。

## 分级（L0–L4）

| 级 | 问的问题 | 怎么判 |
|---|---|---|
| L0 地基 | 两边拿到的数据、权重、配置是不是同一份 | 逐位相同 |
| L1 单算子 | 同一份真实输入，单个算子两边算得一样吗；不一样时谁离 FP64 远 | 回放逐位；对 FP64 定责 |
| L2 单层（逐层、逐子模块） | 同一份权重、同一份输入，每层每个子模块的前向、反向一样吗 | 前向、反向对 FP64 定责（R 比 ≤ 1.2）；参数梯度 calibrated |
| L3 第一步整网 | 整网第一步的 loss、logits、全部梯度、更新后参数一样吗 | logits、逐 token loss 对 FP64 定责；梯度与状态对 GPU 第 1 步重跑的噪声 calibrated（混元 3 当时用[两侧执行噪声的并集](#union-noise)），超出时对 FP64 定责；首步 loss 单独给门槛 |
| L4 多步 | 多步训练的 loss、grad_norm、param_norm、路由偏置是不是同一条轨迹（学习率归 L0 设置核对，检查 id 仍是 `L4.lr`） | calibrated（三个判定量各对 N 判，取最差） |

**准出**：每一级的过关标准。低一级不过，不进高一级。详见 [`levels.md`](reference/levels.md) 与 [`levels.md`](reference/levels.md)。

## golden、尺子与验收（混元 3 实际用到的）

指南里判过不过、读结论时最常碰到的词。混元 3 的 golden 怎么一步步变成现在这样，见[第 2 章 2.2](guide/02-gpu-golden.md#22-为什么最终-golden-是生产程序只修一处)。

<a id="final-golden"></a>
### 最终 golden

L4 第一层验收所对的那组 golden。混元 3 的最终 golden 是 [`rcc`](#rcc)：GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加。修掉的是 GPU 一侧唯一测得出的系统偏差，其余都是生产程序本身。噪声带取它自己同设置重跑的两两配对，混元 3 是 `rcc-a..j` 10 份、45 对。

- 详见：[第 2 章 2.2](guide/02-gpu-golden.md#22-为什么最终-golden-是生产程序只修一处)

<a id="baseline"></a>
### 基线（baseline）

事先指定的、正式比较唯一对照的那一份 golden。其余几份 golden 只用来造噪声带，不取平均，也不取最差。混元 3 的基线是 `rcc-a`（最终 golden 的第 1 份），噪声带是 `rcc-a..j` 10 份两两配对得到的 45 对。

- 详见：[第 2 章 2.4](guide/02-gpu-golden.md#24-噪声带与基线)

<a id="noise-ceiling"></a>
<a id="r"></a>
### 噪声带 / 噪声上限 N

噪声参照里每一对运行算一遍判定量，所有对里的最大值就是噪声上限 N。被测值除以 N 得 r：r ≤ 1 过，r > 2 不过，其间待定。

- 详见：[第 4 章 4.2](guide/04-level-by-level.md#42-怎么判)

<a id="union-noise"></a>
### 两侧执行噪声的并集（第 1 步整步的尺子）

第 1 步 GPU 重跑路由一个都不翻，GPU 噪声对路由翻转的宽度是 0；TPU 只改不影响数学的执行参数（片上内存上限、splash 分块）就会翻同类的近似平局路由。被测对 golden 这一对同时含两侧的执行选择，所以尺子取两侧之和：GPU 重跑两两的对，并上 TPU 只改执行参数的运行两两的对，取所有对的最大值当 N。规则：总数不少于 15 对，两侧各至少 3 份。混元 3 是 GPU 3 对 + TPU 7 份两两 21 对 = 24 对。只用于第 1 步整步的梯度与状态；逐层隔离回放每层输入固定、两侧都不翻，改按 FP64 R 比判。这是混元 3 当时的判法；现行判法噪声只用 GPU 重跑，超出的逐张量对 CPU FP64 定责（混元 4 即如此，混元 3 已按它重判）。

- 详见：[`judging.md`](reference/judging.md#第-1-步张量两侧执行噪声的并集)

<a id="g1k-rc-rcx-rce"></a><a id="golden-codenames"></a>
### golden 代号怎么读

混元 3 的 GPU golden 名字由「配置代号 + 份号」组成。前半是配置代号（`g1k`、`rc`、`rcc`……，下面逐条解释），后缀 `-a`、`-b`…… 是同一配置的第几份重跑；`rcc-a..j` 指 `rcc-a` 到 `rcc-j` 共 10 份。GPU 程序的反向规约顺序不确定，同配置重跑每份都略有不同，所以每组跑多份，用来量噪声。TPU 被测和其他运行的名字见下面「运行名前缀」。

| 代号 | 白话名 | 一句话 | 用在哪 |
|---|---|---|---|
| `g1k` | 第一阶段 GPU golden（默认设置） | BF16、不重算 | L0–L3 第 1 步的参照；加 7 份同设置运行共 10 份，是 GPU 默认设置的噪声 |
| `rc` | GPU 生产 golden | 生产近似配置：整层重算 + 路由回放 | 第二层：已解释差异 |
| `rcx` | 生产近似 + FP32 一次累加的 dgrad（历史） | 输出头 dgrad 换 cuBLAS FP32 一次乘完，本身仍偏 | 只作归因证据 |
| `rce` | 精确轨迹 | BF16 权重、全程 FP32 计算 | 只作参考，曾短暂当第一层基线 |
| `rcc` | 最终 golden | 生产近似 + 输出头 dgrad 分段累加 | 第一层基线（`rcc-a`）与噪声带 |
| `rcb` | 最终 golden 的便宜等价版 | 同 `rcc`，分段内用 BF16 张量核 | 与 `rcc` 不可分，不是基线 |
| `rcec` | 真精确轨迹 | 精确轨迹 + 分段累加 | 只作参考 |

- 详见：[第 2 章 2.3](guide/02-gpu-golden.md#23-要造哪几组-golden怎么跑)、[`../models/hunyuan3/goldens.md`](../models/hunyuan3/goldens.md)

<a id="rc"></a>
### `rc`（GPU 生产 golden）

GPU 生产训练程序原样跑 1000 步：B200 × 8 上的 Megatron-Bridge，BF16 混合精度，整层重算加路由回放。共 5 份 `rc-a..e`。它的输出头 dgrad 有系统偏差（见「输出头 dgrad」），所以 L4 不拿它判过不过，只作[两层验收](#two-tier)的第二层：TPU 对它的差写成[已解释差异](#explained-difference)。

- 详见：[第 2 章 2.3](guide/02-gpu-golden.md#23-要造哪几组-golden怎么跑)

<a id="rcc"></a>
### `rcc`（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加）

在 [`rc`](#rc) 上只改一处：输出头 dgrad 那次沿 12 万词表的长求和改成分段 FP32 累加，修掉 B200 上这次求和的系统偏差，其余设置和生产程序一律相同。共 10 份 `rcc-a..j`。L4 第一层对它判，必须过。

它只是对参照的一处修正。对齐的其余环节，包括 TPU 一侧的精度开关、逐样本数据回放和测量口径，见[对齐报告](../models/hunyuan3/ALIGNMENT-REPORT.md)第 5、6、8 节。

- 详见：[第 2 章 2.2](guide/02-gpu-golden.md#22-为什么最终-golden-是生产程序只修一处)

<a id="rcc-a"></a>
### `rcc-a`（最终 golden 的第 1 份，验收的比较基线）

`rcc` 的第一份，也是事先指定的[基线](#baseline)：TPU 只和这一份逐步比。`rcc-a..j` 10 份两两配对得到的 45 对只用来量噪声上限 N，不取平均，也不挑最差。任意两份 golden 之间的差不超过 N，所以换一份当基线，r 最多挪动 1。实测十份每份轮流当基线：单条检查的最差 r 最多挪 0.47（param_norm，v7x EP = 8），四条检查合起来取最差后最多挪 0.33（v7x FSDP 8），v7x 当前配置有三份落进待定（最高 1.11），没有不过（[`case01-rebaseline-rcc.json`](../models/hunyuan3/alignments/01-v5p-acceptance/results/case01-rebaseline-rcc.json)、混元 3 对齐报告第 3 节）。

- 详见：[第 2 章 2.4](guide/02-gpu-golden.md#24-噪声带与基线)

<a id="g1k"></a>
### `g1k`（第一阶段 GPU golden）

第一版 GPU golden：BF16，不重算，1000 步，全量抓数（第 1 步的逐层激活与梯度）。共 3 份 `g1k-a/b/c`。L0–L3 的第 1 步比较和日常回归 T1 的「不变差」都对它比。`g1k-det` 是确定性设置的一份，只用来归因。

- 详见：[第 2 章 2.3](guide/02-gpu-golden.md#23-要造哪几组-golden怎么跑)

<a id="rce"></a>
### `rce`（精确轨迹，BF16 权重、全程 FP32 计算）

把 `rc` 改成全程 FP32 计算，是[精确轨迹](#exact-trajectory)的第一版。权重仍按 BF16 存，TF32 真关。它的输出头 dgrad 仍是 cuBLAS FP32 一次累加，还带偏差，所以只作参考。2026-11-05 晚到 11-06 早曾用 `rce-a` 当第一层基线，已取消。

- 详见：[`goldens.md`「从 g1k 到 rcc」](../models/hunyuan3/goldens.md#从-g1k-到-rcc每一步加了什么带来了什么)

<a id="rcec"></a>
### `rcec`（真精确轨迹，精确轨迹加输出头 dgrad 分段累加）

在 [`rce`](#rce) 上再把输出头 dgrad 改成分段累加，得到真精确轨迹。GPU 与 TPU 的真精确在这里重合。只作参考交叉核对。

- 详见：同上

<a id="rcb"></a>
### `rcb`（最终 golden 的便宜等价版，BF16 张量核分段）

`rcc` 的设置只把输出头 dgrad 换成 BF16 张量核分段：每 1024 一段，段内仍是 BF16 张量核乘法，段输出留 FP32，段间 FP32 累加。它验证「去掉偏差的是分段，不是升 FP32」：1000 步 2 份对 `rcc-a`、用 `rcc` 的 45 对噪声判，全过，最差 r 0.91，与同机重跑 `rcc` 的对照一样；步时比生产多 6%（`rcc` 多 61%）。正式基线仍是 `rcc-a`。

- 详见：[案例 01](../models/hunyuan3/alignments/01-v5p-acceptance/cases/01-loss-offset-param-norm.md) 注 42、[`../models/hunyuan3/goldens.md`](../models/hunyuan3/goldens.md) 注 5

<a id="rcx"></a>
### `rcx`（生产近似 + 输出头 dgrad 换 FP32 一次累加，历史）

在 `rc` 上只把输出头 dgrad 换成 cuBLAS FP32 一次乘完。排查时用来归因，量出去掉 BF16 张量核截断能挽回多少。后来查出 cuBLAS FP32 一次累加本身也偏，当时算的比例作废。2026-11-05 晚到 11-06 早曾用 `rcx-a..e` 当噪声带，已取消。现在只是历史证据。

- 详见：同上

<a id="run-prefixes"></a>
### 运行名前缀（`t-`、`r-`、`g-`、`reg-`……）

- 运行名的前缀说明它是哪类运行：
  - `t-` 是 TPU 被测，后半写这次改了什么：`t-loss30` 30 步冒烟，`t-rmap-on-1000` v5p 当前对齐配置 1000 步，`t-v7x-on-1000` 同配置在 v7x；
  - `r-` 是只换执行方式的 GPU 运行（如 `r-remat`），例外是 `r-l2*`，它是 TPU 的逐层隔离回放；
  - `g-` 是案例 01 的 GPU 诊断臂；
  - `reg-` 是日常回归。
- 指南命令里的 `t-prod-1000`、`t-smoke30` 是给你自己跑时用的示例名，对应混元 3 已提交的 `t-rmap-on-1000`、`t-loss30`。对照表在第 4 章 4.1。
- 详见：
  - [混元 3 README「运行名前缀」](../models/hunyuan3/README.md#运行名前缀)
  - [第 4 章 4.1](guide/04-level-by-level.md#41-跑之前运行根检查项目录每级对哪份-golden)

<a id="exact-trajectory"></a>
### 精确轨迹

同一个 BF16 模型在没有计算舍入误差时本该走的轨迹。权重仍按 BF16 存，每步前向用 BF16 舍过的权重，但前向和反向全用 FP32，TF32 真关。它用作裁判：看被测和 golden 各离它多远，找出偏差在哪一边。

混元 3 的第一版 [`rce`](#rce)（BF16 权重、全程 FP32 计算）查出输出头 dgrad 仍有 cuBLAS FP32 长累加的偏差。加上分段累加的 [`rcec`](#rcec)（真精确轨迹）才是真精确，GPU 与 TPU 的真精确在这里重合。精确轨迹只作参考交叉核对，不是最终 golden，因为离它剩下的差是两边生产程序共有的。

- 详见：[第 2 章 2.2](guide/02-gpu-golden.md#22-为什么最终-golden-是生产程序只修一处)

<a id="two-tier"></a>
### 两层验收

这里的「第一层 / 第二层」指验收的两层，不是模型的层。它和[两档](#strict-reference)是两件事：两层是跟谁比，两档是线的松紧。L4 分两层：第一层对修掉偏差的最终 golden，按 calibrated 判，必须过；第二层对 GPU 生产 golden，不判过不过，差写成已解释差异。只有 golden 自己有系统偏差时才需要两层验收。

- 详见：[第 4 章「两层验收」](guide/04-level-by-level.md#两层验收l4-对哪份-golden)

<a id="explained-difference"></a>
### 已解释差异

对 GPU 生产 golden 的差里，已查清来自 GPU 一侧哪个偏差的那部分。写清量级、原因和证据，不当成被测的错。

混元 3 里 TPU 对 [`rc`](#rc) 的差：

- param_norm +2.09e-2；
- loss −1.37e-4（v5p）；
- loss −1.07e-4（v7x，[`v7x-tpu-vs-rc.json`](../models/hunyuan3/alignments/02-v7x-acceptance/results/v7x-tpu-vs-rc.json)）。

v7x 的 loss 由 `t-v7x-on-1000` 的逐步序列对 `rc` 五份均值直接算出（序列不在仓库，结果文件记了它 `meta.json` 的 sha256）；和由 v7x 对 `rcc-a`（最终 golden 的第 1 份，验收的比较基线）的 +6.32e-5 加上 `rcc-a` 对 `rc` 均值的 −1.71e-4 推出的值只差约 2e-8。这些差全部来自输出头 dgrad 的长求和，`rcc`（最终 golden）修掉这一处后就和 TPU 重合。

- 详见：[`judging.md`](reference/judging.md#golden-本身有系统偏差时修掉偏差再按一般规矩比)

<a id="strict-reference"></a>
### 严格档 / 参考档

报表每格同时报两档。严格档是 janus 自己的门槛（噪声、逐位、数值），验收只看它。参考档是同类 GPU→TPU 迁移项目的实践门槛，只并列报告；其中 L4 `param_norm` 一条外部没有，由本项目设定。

- 详见：[第 4 章「两档门槛」](guide/04-level-by-level.md#两档门槛严格档与参考档)

<a id="remat"></a>
### 重算 / 路由回放（remat、recompute / routing replay）

重算是在反向时重新算前向的中间量，用来省显存。路由回放是在重算时沿用第一次前向选出的专家，不重新路由。TPU 侧叫 rmap，把路由结果存在 HBM 里。

- 详见：[第 3 章 3.4](guide/03-tpu-setup.md#34-提交-tpu-作业)

<a id="head-dgrad"></a>
### 输出头 dgrad（dh）

输出层对隐状态的梯度 dh = dlogits · W，要沿 120832 个词的词表做一次长求和。混元 3 的 GPU 生产 golden 在这一步有系统偏差：BF16 张量核会[截断](#truncation)，换成 cuBLAS FP32 一次累加也偏。最终 golden 改成了分段 FP32 累加。去掉偏差的是分段（每段 ≤ 1024，段结果留 FP32），不是升 FP32：BF16 张量核按同样分段也一样准，单次乘法 7.2 ms，升 FP32 分段 71.7 ms（[案例 01](../models/hunyuan3/alignments/01-v5p-acceptance/cases/01-loss-offset-param-norm.md) 注 41）；1000 步训练上 BF16 张量核分段（[`rcb`](#rcb)）与 `rcc` 不可分。

- 详见：[第 2 章 2.2](guide/02-gpu-golden.md#22-为什么最终-golden-是生产程序只修一处)

<a id="tensor-core-mxu"></a>
### 张量核 / MXU

两边做矩阵乘的硬件单元：GPU 的张量核（tensor core）和 TPU 的 MXU。MXU 是 BF16 乘、FP32 累加。两者累加方式不同，是长求和结果有差的来源之一。

- 详见：[`principles.md`](reference/principles.md#差从哪来按顺序排除)

<a id="truncation"></a>
### 张量核累加截断（truncation）

B200 张量核的 BF16 GEMM 累加方式（实测拟合的模型，非官方规格）：每 16 个乘积连同当前累加值为一组，全部对齐到组内最大的指数，只保留约 25 位的窗口，移出窗口的低位直接丢弃，不舍入。普通浮点加法保留保护位再就近舍入，误差有正有负；截断丢掉的总是绝对值的一部分，所以每一组都朝 0。组数越多丢得越多；累加值越大，窗口越往高位移，后面的小乘积丢得越多。按这个模型在 CPU 上重算路由 GEMM，和 B200 实测逐位相同。

- 详见：[对齐报告](../models/hunyuan3/ALIGNMENT-REPORT.md)第 2 节「根子在 GPU 一侧」、台账 #1 与注 1

<a id="w-bar"></a>
### ŵ̄ 方向（logits 整体水平的方向）

输出层各行权重的均值方向，û 是它的单位向量。隐状态沿 ŵ̄ 移动，所有词的 logits 一起升降，softmax 概率不变，但 logits 的整体水平（lse）变了。冷门行每步被「往下压」多少取决于这个整体水平，所以 dh 沿 ŵ̄ 的分量哪怕只偏万分之几，也会每步同向推动冷门行。「沿 ŵ̄ 偏 x」指 Σₜ (dhₜ − dhₜ^FP64)·û ÷ Σₜ |dhₜ^FP64·û|。

- 详见：[对齐报告](../models/hunyuan3/ALIGNMENT-REPORT.md)第 2 节「定位」

<a id="eps-regime"></a>
### Adam ε 区

Adam 的更新量 u = m̂ / (√v̂ + ε)。√v̂ ≫ ε 时 u ≈ m̂ / √v̂，梯度的整体缩放被约掉，相对误差大部分抵消；√v̂ ≪ ε 时 u ≈ m̂ / ε，更新与梯度成正比，梯度的系统偏差原样进入更新，且每步同向。混元 3 第 300 步没出现过的词那些行，92% 的元素 √v̂ < ε（中位数 2.6e-9，ε = 1e-8）。对齐的做法是两边用同一个 ε，并让这些小梯度本身算准，不靠调 ε 掩盖差别。

- 详见：[`../models/hunyuan3/goldens.md`](../models/hunyuan3/goldens.md) 注 9

<a id="param-norm"></a>
### `param_norm` / `grad_norm`

每步记下的全部参数或全部梯度的 L2 范数，L4 和 loss 一起逐步比。混元 3 里最先漂起来、暴露问题的是 `param_norm`。

- 详见：[`reference/metrics.md`「逐步序列」](reference/metrics.md#逐步序列lossgrad_normparam_norm)

<a id="phase"></a>
### 第一阶段 / 第二阶段

对应结果文件名的前缀 `phase1-` / `phase2-`。第一阶段是 GPU 自比（实践 00）；第二阶段是 TPU 对 GPU，实践 01 在 v5p 上，实践 02 在 v7x 上。

- 详见：[用户指南「名字速查」](guide/README.md#名字速查)

<a id="rel-l2"></a>
### rel_L2

相对 L2 误差 ‖a − b‖₂ / ‖b‖₂，张量比较最常用的指标。记范数比 k = ‖a‖ / ‖b‖，则 rel_L2² = (k − 1)² + 2k(1 − cos)：前一项是长度差，后一项是方向差。等长时 rel_L2 = 2 sin(θ/2)，cos = 0.999 对应 rel_L2 ≈ 4.5%，所以只看 cos 会低估差。判定逐张量算，一条检查取最差的张量（[第 4 章 4.2](guide/04-level-by-level.md#一个数是怎么聚合出来的)）。

- 详见：[`reference/metrics.md`「张量」](reference/metrics.md#张量第-1-步的激活梯度状态)

<a id="parallel"></a>
### DP / EP / FSDP

三种并行方式：数据并行、专家并行、全切分数据并行。混元 3 的 GPU 用 DP8 + EP8，TPU 用 FSDP8。切法不同，归约顺序就不同。TPU 上换切法不一定只换顺序：纯数据并行下词嵌入梯度在 BF16 里 scatter、不排序，丢精度（[差异清单 A12](reference/tpu-vs-gpu-differences.md#a12-纯数据并行时词嵌入梯度在-bf16-里-scatter不排序)）。

- 详见：[第 3 章 3.2](guide/03-tpu-setup.md#32-配置逐项对照)

## 迁移验收与日常回归

两件事，参照、尺子、频率都不同：

| | 迁移验收 | 日常回归 |
|---|---|---|
| 回答 | TPU 和 GPU 训的是同一个模型 | 今天的 TPU 和上一次验证过的 TPU 训的是同一个模型 |
| 参照 | GPU golden | LKG |
| 尺子 | GPU 自己重跑的噪声 | 逐位为主；不逐位时用 TPU 扰动臂的噪声 |
| 频率 | 一次（换平台、换模型、开新配方时重做） | 目标是每次合并、每 2–4 小时、每天；现在全部手动触发 |
| 放哪 | `models/<模型>/alignments/<实践>/` | `models/<模型>/regression/` |

日常回归里的几个词：

| 术语 | 意思 |
|---|---|
| 回归门 | 被测对 LKG 比：数值变没变；变了是不是在 PR 声明的范围和幅度内 |
| <a id="tolerance-gate"></a>容差门 | PR 声明「会改数值 · 执行级」或升级依赖时，被测对 LKG 不再要求逐位，改用 TPU 自己的执行噪声判 |
| 对齐门 | 数值变了以后，被测对 GPU golden 比：差有没有变大（「不变差」） |
| 档位（tier，T0–T4） | 回归的几档配置：<br>T0 冒烟（每次合并、几十步）<br>T1 主力（每 2–4 小时）<br>T2 每日（抓第 1 步张量，未实施）<br>T3 生产形状（待定）<br>T4 低精度配方（未启用）<br>另有金丝雀、对照臂、标定。节奏都是目标，现在全部手动触发（第 7 章 7.3） |
| PR 声明 | 训练代码 PR 正文的「数值影响」一栏三选一：数学不变 / 会改数值（执行级或数学级）/ 升级依赖。决定回归门每一行怎么判 |

详见 [`reference/regression.md`](reference/regression.md)。

## 仓库与工具

| 术语 | 意思 | 详见 |
|---|---|---|
| <a id="alignment"></a>对齐实践（alignment） | 一组从源平台到目标平台的对比，对应一个文件夹：`models/<模型>/alignments/NN-<做什么>/`（如 `01-v5p-acceptance`） | [`layout.md`](dev/layout.md) |
| <a id="catalog"></a>检查项目录（catalog） | 一份 YAML：一个模型、一个阶段要过的全部检查。`janus run` 读它。文档里单说「目录」「目录文件」都指它；文件系统里的目录写成「文件夹」或具体的名字（运行目录、结果目录） | [`reference/catalog-schema.md`](reference/catalog-schema.md) |
| 检查类型（check type） | 一种比较方法，如 `tensor_tolerance`、`multi_golden`；一类一份规格 | [`reference/checks/README.md`](reference/checks/README.md) |
| 检查 id | 目录里一条检查的名字，如 `L2.layer1.fwd`；结果、报告、大表都按它引用 | [`conventions.md`](dev/conventions.md) |
| 结果文件 | `<目录名>.results.json`（汇总）与 `.report.md`（从它渲染），放实践的 `results/`。数字只从这里引用，不手写 | [`reference/result-schema.md`](reference/result-schema.md) |
| 规范名（canonical name） | 两侧张量统一的名字（如 `layers.2.attn.out`），比较按它配对 | [`reference/dump-format.md`](reference/dump-format.md) |
| 抓数（dump） | 训练时按统一格式存下中间量，并证明开不开抓数结果逐位相同 | [`dump.md`](reference/dump.md) |
| 逐层注入 | 把参照这一层的输入直接喂给被测这一层，误差隔离在本层 | [`levels.md`](reference/levels.md) L2 |
| 串跑找台阶 | 整网一起跑，找第一个从逐位相同变成不同的点 | [`levels.md`](reference/levels.md) L2 |
| 故障演练（fault drill） | 故意埋错（改权重、错位数据、缩放梯度……），确认该报警的检查都报了。工具先自证 | 第一阶段 [`faults/phase1-faults.md`](../models/hunyuan3/alignments/00-gpu-noise-and-drills/faults/phase1-faults.md) |
| 排查案例（case） | 一次问题从发现到结论的完整记录，放实践的 `cases/` | [`guide/06-debugging.md`](guide/06-debugging.md) |
| FP32 主权重（main params） | 混合精度训练里优化器更新的 FP32 那份权重。小于 BF16 半格的错只在它上面看得见，所以 L0 单独比 | [`levels.md`](reference/levels.md) L0 |
| 假 FP32（fake FP32） | 标着 float32、低 16 位却全是 0 的张量：某处先舍成 BF16 再升回 | [`reference/checks/fake_fp32.md`](reference/checks/fake_fp32.md) |

---

[回用户指南](guide/README.md) ｜ [附录](guide/appendix.md)
