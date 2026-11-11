[janus](../../README.md) › [混元 3](README.md) ｜ 检查清单

# 混元 3 · 检查清单

这一页回答：迁移验收到底比了哪些量，每一项的对象是哪些模块和张量，用什么指标、什么门槛判，在 TPU v5p-8 和 v7x-4 上各是什么结果。

- 页面由 `models/hunyuan3/dashboard/build.py` 按 [`dashboard/checks.yaml`](dashboard/checks.yaml) 从各对齐实践的结果文件生成，数字和判定都不手写。
- 结论和偏差的机制在[对齐报告](ALIGNMENT-REPORT.md)；每一级为什么这样比、不过时先查什么，在[用户指南第 4 章](../../docs/guide/04-level-by-level.md)。
- 被测是两个平台各自的当前对齐配置（MaxText `hy3-4layers.yml` 版本 `a238883b`，开关见对齐报告第 6 节）；用了换代前配置的两行在行内注明（对 `rc-a` 的 L4、v7x 同程序重跑）。参照是 GPU golden（定义见[`goldens.md`](goldens.md)）。

## 规模

- **检查项**：表里的一行，回答一个问题。
- **检查条数**：结果文件里的检查，一个检查 id 算一条（v5p / v7x）。
- **张量级比较**：这些检查逐个比了多少个张量（一个参数、一个算子输出或一个层内观测点算一个）。同一个张量被几条检查各比一次时分别计数，所以是比较次数，不是去重后的张量个数。
- **其他比较单元**：数据指纹、dtype 条目、逐步曲线的步数等。

| 级 | 目的 | 检查项 | 检查条数 | 张量级比较 | 其他比较单元（TPU v5p-8） | TPU v5p-8 判定 | TPU v7x-4 判定 |
|---|---|---|---|---|---|---|---|
| [L0](#l0--地基数据权重配置) | 两边从同一份输入、同一份权重、同一套配置出发 | 11 | 29 / 28 | 4,844 / 4,844 | 82,600 个数据单元、6,704 项 dtype、15 条规则、6 个字段、3,000 步、64,000 个名字 | ✅ 10；只报 1 | ✅ 10；只报 1 |
| [L1](#l1--单算子) | 每个算子单独看，TPU 不比 GPU 离 FP64 真值远 | 19 | 21 / 14 | 261 / 241 | — | ✅ 19 | ✅ 11 ❌ 1 |
| [L2](#l2--逐层) | 每层输入取 GPU 的，层内观测点不比 GPU 离 FP64 远 | 7 | 15 / 15 | 95 / 95 | — | ✅ 6 ❌ 1 | ✅ 6 ❌ 1 |
| [L3](#l3--整网第-1-步) | 第 1 步整网的输出、梯度、更新后状态 | 17 | 34 / 22 | 1,543 / 1,043 | 623 个名字 | ✅ 6 🟡 1 ❌ 3；只报 7 | ✅ 5 🟡 1 ❌ 4；只报 5 |
| [L4](#l4--多步训练) | 1000 步的 loss、grad_norm、param_norm 等逐步曲线在 GPU 重跑的噪声内 | 17 | 30 / 34 | — | 13 个字段、21,810 步 | ✅ 8；只报 8 | ✅ 9；只报 8 |
| 合计 |  | 71 | 129 / 113 | 6,743 / 6,223 |  |  |  |

## 怎么读

### 一个数是怎么聚合出来的

表里每个数属于下面三种之一，每一行「对象」格最后一句写明是哪一种：

| 种类 | 怎么算 | 例 |
|---|---|---|
| 逐步标量 | 每步一个数，算 TPU 对参照的逐步相对差，再在窗口内算带符号均值、MARE（平均绝对相对差）、最大单步三个判定量，各自对噪声上限 N 判，最差的决定结果 | L4 的 loss（全局批的平均 loss）、grad_norm 与 param_norm（全部可训练参数的全局 L2 范数，约 124 亿个元素合成一个数，不含 3 个路由偏置） |
| 逐张量取最差 | 每个张量拉直成一个向量，单独算一个数（如 rel_L2 = ‖a − b‖ / ‖b‖，或 R 比）；一条检查选中一批张量时取最差的那个，不取平均 | L1 单算子、L2 逐层、L3 的梯度与状态 |
| 按能量合成（只用于定责） | 把一组张量当成一个大向量：√(Σ‖a − b‖²) ÷ √(Σ‖b‖²)；R 比的合成写法是 √(Σ‖TPU − FP64‖²) ÷ √(Σ‖GPU − FP64‖²)。大张量权重大，小张量的抖动被压住 | 第 1 步梯度「TPU 对 GPU 整网只差 1.3e-3」、「整网按能量 R 比 1.20」 |

张量个数为什么有 1781、1193、1480、50 几种，见[术语表](../../docs/glossary.md#tensor-counts)。

### 三种严格门槛

| 门槛 | 用在哪 | 怎么判 |
|---|---|---|
| 逐位 | L0 数据、权重、配置；续训 | 两边逐位相同才过 |
| R 比 ≤ 1.2 | L1 单算子；L2 逐层前向、反向；L3 logits 与逐 token loss | R = TPU 离 FP64 真值的 rel_L2 ÷ GPU 离 FP64 真值的 rel_L2。TPU 不比 GPU 离精确算术远就过 |
| 噪声上限 N | L2、L3 的第 1 步梯度与状态；L4 的逐步曲线 | N 是 GPU 自己重跑两两之间的最大差（第 1 步只用 GPU 第 1 步重跑 3 对；2026-11-08 起，之前并上 TPU 只改执行参数的 21 对，已撤回）。实测 ≤ N 过，> 2N 不过，之间待定 |

定义与依据见 [`docs/reference/judging.md`](../../docs/reference/judging.md)。

### 表里各栏

- **证据来源**：「生产程序」是不加钩子的生产构建（L4 与训练步之间的状态读），是生产图的直接证据；「隔离回放」「单算子回放」「整网前向回放」是单独编译的程序，图和生产不同，结果只用来定责、确认修复方向，不当生产程序逐位的证据；「离线 FP64 计算」在 CPU 上算真值。回放怎样贴近生产、生产 HLO 里怎样核对精度开关，见[用户指南第 4 章 4.8](../../docs/guide/04-level-by-level.md#48-tpu-上第-1-步的证据为什么不用钩子)。
- **判法**：「｜」前是严格档，决定验收；「参考」后是同类 GPU→TPU 迁移项目用过的经验门槛，只并列报告，「无」表示那边没有对应的线。
- **平台格子**：实测值，括号里是离严格门槛多远（×N 或 实测 ÷ 门槛），然后是判定。「｜参」后是参考档判定。点格子打开那份结果报告。
  - 一个格子选中多条检查（例如 4 层各一条）时写最差的那条。
  - 本列的指标或 N 和判法栏不同时，括号里另写本列自己的。
- **符号**：✅ 过　🟡 待定　❌ 不过　— 不适用。参考档另有 ☑️（记录了的都过、有门槛项没记录）与 ❔（所需指标没记录）。
- 每张表下面写非 ✅ 格子的原因；机制与量级在对齐报告的偏差台账（第 5 节）。
- 检查名后标「（只报）」的行不在验收路径上：对照（如对未修的 GPU 生产 golden）、可选配置（如 EP = 8、三个梯度 FP32 开关）、只记事实的检查（如跨平台激活逐位）、定责用的 FP64 比较。规模表的判定计数不含这些行，单列为「只报」。
- 格子写的是 rel_L2，而最差的指标离噪声上限更远时，括号里另写「最差 …×N」。
- 「比较单元」是这一行全部检查的合计；格子只写最差那一条检查自己的数，所以两者可以不同（例如 L0「载入后的参数」合计 2,960 个参数张量，是两次各 1,480）。
- 同一批张量分组不同，N 也不同：N 取每一对噪声运行在这组张量里最差的那个，所以整网一组（L3）的 N 比每层一组（L2）大。
- 张量名用两侧共用的规范名（[`model/name_map.yaml`](model/name_map.yaml)）。`layers.{0..3}` 指 4 层；第 0 层是稠密层，第 1–3 层是 MoE 层（192 个路由专家、top-8、1 个共享专家）。

## L0 · 地基：数据、权重、配置

目的：证明两边从同一份输入、同一份权重、同一套配置出发。这一级不过，上面各级的差都说不清来源。

L0 共 11 项检查，检查条数 29 / 28（v5p / v7x），张量级比较 4,844 / 4,844 次。除「TPU 原生数据加载器」一项外，全部要求逐位相同。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| 配置逐项对照<br>`L0.config.diff.dropout` | 两边训练配置里机读得到、且必须相同的字段 | 6 个字段：dropout = 0、学习率、权重衰减、路由偏置更新速率、检查点清单 sha256、回放数据清单 sha256。其余 80 行配置的人工逐项对照在 [`config-alignment.md`](config-alignment.md)<br>比较单元：6 个字段<br>聚合：逐字段比，数不同的个数 | 逐位｜参考：无 | 生产程序（30 步，不加钩子） | [0 / 6 不同 ✅](alignments/01-v5p-acceptance/results/phase2-t-loss30.report.md) | [0 / 6 不同 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-t-loss30.report.md) |
| 每步输入指纹<br>`L0.data.fingerprint` | TPU 回放的每一步、每个 rank、每个 microbatch 的输入，对 GPU golden 存下的输入 | 前 30 步 × 8 个 rank × 2 个 microbatch 的 tokens、labels、loss_mask、position_ids，加每步首个 loss<br>比较单元：1,950 个数据单元<br>聚合：逐单元比 sha256，数不同的个数 | 逐位｜参考：无 | 生产程序（30 步，不加钩子） | [0 / 1,950 不同 ✅](alignments/01-v5p-acceptance/results/phase2-t-loss30.report.md) | [0 / 1,950 不同 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-t-loss30.report.md) |
| 载入后的参数<br>`L0.weights.init`<br>`L0.weights.init.prod`<br>`L0.weights.loaded` | 训练开始前，TPU 载入的参数对 GPU 载入的参数 | 初始参数清单（BF16 参数 + FP32 expert_bias，按 GPU 的 8 个 rank 分）逐项 sha256，正式被测与 1000 步生产配置各比一次；第 1 步训练前从状态读出的参数张量（专家参数每层只读 0、1 号两个专家）<br>比较单元：2,960 个参数张量、50 个张量<br>聚合：逐张量比 sha256，数不同的个数；逐张量按字节比，数不同的个数 | 逐位｜参考：无 | 生产程序（第 1 步训练前读状态） | [0 / 1,480 不同 ✅](alignments/01-v5p-acceptance/results/phase2-l0-state.report.md) | [0 / 1,480 不同 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-l0-state.report.md) |
| FP32 主权重<br>`L0.main_params.loaded` | 训练开始前的 FP32 主权重。BF16 参数看不出主权重上不到半格的差，所以单独比 | 第 1 步训练前的 `main_param.*`（专家参数每层 0、1 号）<br>比较单元：50 个张量<br>聚合：逐张量按字节比，数不同的个数 | 逐位｜参考：无 | 生产程序（第 1 步训练前读状态） | [0 / 50 不同 ✅](alignments/01-v5p-acceptance/results/phase2-l0-state.report.md) | [0 / 50 不同 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-l0-state.report.md) |
| 路由偏置 expert_bias<br>`L0.buffers.loaded` | 训练开始前的 FP32 路由偏置。1 个 bit 的差就能改变 top-8 选择 | 第 1–3 层各一个 `expert_bias`（192 维，FP32）<br>比较单元：3 个张量<br>聚合：逐张量按字节比，数不同的个数 | 逐位｜参考：无 | 生产程序（第 1 步训练前读状态） | [0 / 3 不同 ✅](alignments/01-v5p-acceptance/results/phase2-l0-state.report.md) | [0 / 3 不同 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-l0-state.report.md) |
| dtype 清单<br>`L0.dtype.manifest`<br>`L0.dtype.state` | 每个参数与训练状态的 dtype 是否符合规则 | 参数清单（expert_bias 为 FP32，其余 BF16）；训练状态（主权重、两个动量、路由偏置必须 FP32，只在 v5p 的回归运行上读）<br>比较单元：TPU v5p-8 3,142 项 dtype；TPU v7x-4 2,960 项 dtype<br>聚合：逐项核 dtype，数不合规的个数 | 逐位｜参考：无 | 生产程序（第 1 步训练前读状态） | [0 / 3,142 不同 ✅](alignments/01-v5p-acceptance/results/phase2-l0-state.report.md) | [0 / 2,960 不同 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-l0-state.report.md) |
| 模型结构<br>`L0.model.structure` | TPU 实例化出来的模型和 HF 配置是否一致 | 15 条规则：1 层稠密 + 3 层 MoE、192 个专家、专家中间维 1536、1 个共享专家、64 个 q 头 / 8 个 kv 头 × 128、稠密 MLP 13312、词表 120832、输出层不与词表共享、主权重全部 FP32、没有 MTP<br>比较单元：15 条规则<br>聚合：逐条规则核，数不满足的条数 | 按规则｜参考：无 | 生产程序（第 1 步训练前读状态） | [全部规则满足 ✅](alignments/01-v5p-acceptance/results/phase2-l0-state.report.md) | [全部规则满足 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-l0-state.report.md) |
| 权重转换<br>`L0.ckpt.convert.bitwise`<br>`L0.ckpt.convert.dtype`<br>`L0.ckpt.source.dtype` | HF 权重转成 MaxText 检查点后，逐张量转回 HF 对原件。v5p、v7x 用同一份检查点 | 全部 1781 个 HF 张量逐位，加转换前后两份 dtype 清单<br>比较单元：1,781 个参数张量、3,562 项 dtype<br>聚合：逐张量比 sha256，数不同的个数；逐项核 dtype，数不合规的个数 | 逐位｜参考：无 | 离线核对（CPU，检查点文件） | [0 / 1,781 不同 ✅](alignments/01-v5p-acceptance/results/phase2-ckpt-convert.report.md) | [0 / 1,781 不同 ✅](alignments/01-v5p-acceptance/results/phase2-ckpt-convert.report.md) |
| 续训的数据位置<br>`L0.resume.data` | 从存档恢复后，数据是否从正确的位置接着读。TPU 侧数据是按步号回放 golden 的 batch，这一项只验证回放的步号索引，不覆盖原生加载器（Grain 迭代器）状态的存取 | 恢复后各步、各 rank 的输入指纹<br>比较单元：650 个数据单元<br>聚合：逐单元比 sha256，数不同的个数 | 逐位｜参考：无 | 生产程序（存档与续训） | [0 / 650 不同 ✅](alignments/01-v5p-acceptance/results/phase2-resume.report.md) | [0 / 650 不同 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-resume.report.md) |
| TPU 原生数据加载器（只报）<br>`L0.native.gpu_integrity`<br>`L0.native.index`<br>`L0.native.stream`<br>`L0.native.device_map.native`<br>… | 不回放 GPU 的 batch、改用 MaxText 自己的数据管线时，读出的样本是否与 Megatron 相同 | 14 条：三个索引表、样本流、每步非 padding token 数、字段约定、读取进程数、多 host、多 epoch、文档边界合成用例、pad id、3 个负对照、覆盖率；分配到设备与 microbatch 的布局。加载器与 TPU 型号无关，只在 CPU 上跑了一次，v7x 列引用同一份结果<br>比较单元：80,000 个数据单元、2,000 步、64,000 个名字<br>聚合：逐条规则核，数不满足的条数；逐单元比 sha256，数不同的个数；逐步、逐张量比两次运行，数不可复现的项；逐用例看埋的错有没有被抓到；比名字清单，数缺的个数 | 逐位｜参考：无 | 离线核对（CPU，数据管线） | [14,000 / 16,000 不同 ❌](alignments/01-v5p-acceptance/results/phase2-native-loader.report.md) | [14,000 / 16,000 不同 ❌](alignments/01-v5p-acceptance/results/phase2-native-loader.report.md) |
| 设置核对：学习率逐步<br>`L4.lr` | 逐步学习率两边逐位相同。学习率是设置的常数（1e-5，无 warmup、无衰减），不是观测量，所以归设置核对，不算 L4 的比较项（报告口径，判定不变，见对齐报告「判法变更记录」） | `lr`，第 1–1000 步（检查 id 仍是 `L4.lr`，取自 L4 的验收结果文件）<br>比较单元：1,000 步<br>聚合：逐步比，任一步超出容差就不过 | rtol ≤ 0、abs_ulp ≤ 0｜参考：无 | 生产程序 | [逐步在容差内 ✅](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-on.report.md) | [逐步在容差内 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc.report.md) |

非 ✅ 格子的原因：

- **TPU 原生数据加载器**（TPU v5p-8 ❌、TPU v7x-4 ❌）：样本本身逐位相同（`L0.native.stream`），差在两处约定：MaxText 梯度累积把样本分到 microbatch 的顺序不同（`L0.native.device_map.native`，台账 #16）；数据里出现 pad id 时 Megatron 把它换成 0 并不计 loss，MaxText 原样保留（`L0.native.pad_id`）。验收不走这条路：TPU 原样回放 GPU 每一步每个 rank 的 batch，「每步输入指纹」一行逐位相同。

## L1 · 单算子

目的：把每个算子单独拿出来，看 TPU 的实现是否不比 GPU 离 FP64 真值远。整网的差是各算子的差叠起来的，这一级定位到具体算子。

L1 共 19 项检查，检查条数 21 / 14（v5p / v7x），覆盖模型的 11 个算子（下表 TPU 两列有检查的行），张量级比较 261 / 241 次。

算子的输入有两种取法：

- **各用自己的输入**（主判，v5p 与 v7x）：每一侧的真值都从这一侧自己抓到的算子输入在 FP64 里算，比 TPU 离自己的真值多远、GPU 离自己的真值多远。上游的差不进来，只看算子本身。
- **用 golden 的输入**（v5p 补充）：把 GPU 抓到的算子输入原样喂给 TPU 的算子，可以直接和 GPU 的输出比，也能核对 TPU 的舍入方式。

### 模型的每个算子由哪条检查覆盖

按前向顺序列出模型的算子。「—」是这一列没有单独的检查；最后一列写更高一级在哪里覆盖它。GPU 一列是 GPU 侧的算子回放（参照自己的可信度，见文末「GPU 侧的检查」）。

| # | 算子 | 做什么 | TPU 各用自己的输入（v5p / v7x） | TPU 用 golden 的输入（v5p） | GPU 侧回放 | 更高一级在哪覆盖 |
|---|---|---|---|---|---|---|
| 1 | 词嵌入查表 | token id → 4096 维向量 | — | — | — | L0 权重逐位；L3 整网链路从 golden 的 `embed.out` 起算 |
| 2 | 输入 RMSNorm + q / k / v 投影 | 层输入归一化后乘 qkv 权重（GPU 把归一化融进投影，所以两者合成一个算子比） | [`L1.ops.op.qkv`](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) ✅ | — | [`L1.gemm.qkv.replay`](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) ✅ | L2 观测点 `attn.qkv.out` |
| 3 | q / k 每头 RMSNorm | q、k 各自按头做 RMSNorm（RoPE 之前） | [`L1.ops.op.qk_norm`](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) ✅ | — | [`L1.norm.replay`](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) ✅ | L2 观测点 `attn.q_norm.out`、`attn.k_norm.out` |
| 4 | 注意力核心 | RoPE、1/√d 缩放、因果 softmax、乘 V | [`L1.ops.kernel.attn_core`](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) ✅ | [`L1.ops.attn.core`](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) ✅ | [`L1.kernel.attn.fwd.replay、L1.kernel.attn.bwd.replay`](alignments/00-gpu-noise-and-drills/results/phase1-kernels.report.md) ✅ | L2 观测点 `attn.core.out` |
| 5 | 注意力输出投影 | core.out 乘输出投影权重 | [`L1.ops.op.attn_proj`](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) ✅ | — | [`L1.gemm.attn_proj.replay`](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) ✅ | L2 观测点 `attn.proj.out`、`attn.out` |
| 6 | MLP 前 RMSNorm（MoE 层） | 注意力残差和 bf16(层输入 + attn.out) 归一化 | [`L1.ops.op.mlp_norm`](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) ✅ | [`L1.ops.rmsnorm`](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) ✅ | [`L1.norm.replay`](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) ✅ | L2 观测点 `mlp_norm.out` |
| 7 | 路由 GEMM | mlp_norm.out 乘路由权重，得到 192 个专家的 FP32 logits | [`L1.ops.op.router_gemm`](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) ✅ | [`L1.ops.router.gemm`](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) ✅<br>[`L1.ops.router.precision`](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) ✅ | [`L1.gemm.router.replay`](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) ✅ | L2 观测点 `mlp.router.logits` |
| 8 | 路由打分与 top-8 | sigmoid、加 expert_bias、选 8 个专家 | [`L1.router.topk.fp64.layer1、L1.router.topk.fp64.layer2、L1.router.topk.fp64.layer3`](alignments/01-v5p-acceptance/results/phase2-l2-replay-rratio-on.report.md) ✅ | [`L1.ops.router.topk`](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) ✅ | [`L1.router.topk.fp64`](alignments/00-gpu-noise-and-drills/results/phase1-full-noise28.report.md) ✅<br>[`L1.ties.synthetic、L1.ties.cpu_torch、L1.ties.real`](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) ✅ | L2 各层路由选中集合 |
| 9 | MoE 块（整块） | 按路由分发 token、192 个专家的 gate / up / down GEMM 与门控激活、按路由权重合并、加共享专家 | [`L1.ops.kernel.moe`](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) ✅ | [`L1.ops.moe`](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) ✅ | [`L1.kernel.experts.fwd.replay、L1.kernel.experts.bwd.replay、L1.kernel.experts.vs_loop、L1.kernel.experts.wgrad`](alignments/00-gpu-noise-and-drills/results/phase1-kernels.report.md) ✅<br>[`L2.moe.movement`](alignments/00-gpu-noise-and-drills/results/phase1-moe-stages.report.md) ✅ | L2 观测点 `mlp.shared.out`、`mlp.out` |
| 10 | MoE 块内：分发与合并 | token 按选中的专家重排、发到专家、按权重加回原位置 | 在 MoE 块里整体比，没有单独的检查 | — | [`L2.moe.movement`](alignments/00-gpu-noise-and-drills/results/phase1-moe-stages.report.md) ✅ | — |
| 11 | MoE 块内：路由专家 GEMM | 每个专家的 gate、up（fc1）与 down（fc2） | 在 MoE 块里整体比，没有单独的检查 | — | [`L1.kernel.experts.fwd.replay`](alignments/00-gpu-noise-and-drills/results/phase1-kernels.report.md) ✅<br>[`L1.kernel.experts.bwd.replay`](alignments/00-gpu-noise-and-drills/results/phase1-kernels.report.md) ✅<br>[`L1.kernel.experts.wgrad`](alignments/00-gpu-noise-and-drills/results/phase1-kernels.report.md) ✅ | — |
| 12 | MoE 块内：共享专家 | 每个 token 都过的一个专家 | 在 MoE 块里整体比，没有单独的检查 | — | — | L2 观测点 `mlp.shared.out` |
| 13 | 稠密 MLP 前 RMSNorm + gate、up 投影（第 0 层） | 注意力残差和归一化后乘 fc1 权重（GPU 融合成一个算子） | [`L1.ops.op.dense_fc1`](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) ✅ | — | — | L2 观测点 `mlp.fc1.out` |
| 14 | 门控激活 + down 投影（第 0 层） | silu(gate) · up，再乘 fc2 权重 | [`L1.ops.op.dense_act_fc2`](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) ✅ | [`L1.ops.act`](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) ✅ | [`L1.elem.silu.fp32`](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) ✅ | L2 观测点 `mlp.fc2.out` |
| 15 | 残差加 | 层输入 + attn.out、h + mlp.out（BF16） | — | — | — | L2 观测点 `mlp_norm.out`（第 0 层）、`out` |
| 16 | 最后的 RMSNorm、输出层、交叉熵 | 归一化、乘输出层得到 120832 维 logits、算逐 token loss | — | — | — | L3 logits、逐 token loss（整网链路对 FP64） |
| 17 | AdamW 一步更新 | 由梯度、两个动量、偏差修正、权重衰减、裁剪系数算出更新量 | [`L1.optim.adamw.replay`](alignments/01-v5p-acceptance/results/phase2-st1-on-state-gpuonly.report.md) ✅<br>[`L1.optim.adamw.8steps`](alignments/01-v5p-acceptance/results/phase2-st8-opt8.report.md) ✅ | — | [`L1.optim.adamw.replay`](alignments/00-gpu-noise-and-drills/results/phase1-full-noise28.report.md) ✅<br>[`L1.optim.adamw.8steps`](alignments/00-gpu-noise-and-drills/results/phase1-opt8.report.md) ✅ | L3 更新后主权重、优化器状态 |

### 各用自己的输入（主判）

真值由 [`analysis/fp64_forward.py`](analysis/fp64_forward.py) 在 FP64 里算，判法见 [`analysis/fwd_rratio.py`](analysis/fwd_rratio.py)。每层一个用例。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| 输入 RMSNorm + q / k / v 投影<br>`L1.ops.op.qkv` | 层输入 → 归一化 + qkv 投影（输入两边相同，都是 golden 的层输入） | `layers.{0..3}.attn.qkv.out`<br>比较单元：4 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（单独编译）＋离线 FP64 计算 | [1（0.833×） ✅｜参✅](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) | [1（0.833×） ✅｜参✅](alignments/02-v7x-acceptance/results/phase2-v7x-l1-ops-own-on.report.md) |
| q / k 每头 RMSNorm<br>`L1.ops.op.qk_norm` | 本侧 qkv.out 里的 q、k → 每头 RMSNorm | `layers.{0..3}.attn.q_norm.out`、`layers.{0..3}.attn.k_norm.out`<br>比较单元：8 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（单独编译）＋离线 FP64 计算 | [1.073（0.894×） ✅｜参❔](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) | [1.073（0.894×） ✅｜参❔](alignments/02-v7x-acceptance/results/phase2-v7x-l1-ops-own-on.report.md) |
| 注意力核心<br>`L1.ops.kernel.attn_core` | 本侧 q_norm、k_norm、v → RoPE、1/√d、因果注意力 | `layers.{0..3}.attn.core.out`<br>比较单元：4 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（单独编译）＋离线 FP64 计算 | [1.009（0.841×） ✅｜参❔](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) | [1.009（0.841×） ✅｜参❔](alignments/02-v7x-acceptance/results/phase2-v7x-l1-ops-own-on.report.md) |
| 注意力输出投影<br>`L1.ops.op.attn_proj` | 本侧 core.out → 输出投影 | `layers.{0..3}.attn.proj.out`<br>比较单元：4 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（单独编译）＋离线 FP64 计算 | [1.001（0.834×） ✅｜参❔](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) | [1.001（0.834×） ✅｜参❔](alignments/02-v7x-acceptance/results/phase2-v7x-l1-ops-own-on.report.md) |
| MLP 前 RMSNorm<br>`L1.ops.op.mlp_norm` | bf16(层输入 + 本侧 attn.out) → RMSNorm | MoE 层 `layers.{1..3}.mlp_norm.out`<br>比较单元：3 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（单独编译）＋离线 FP64 计算 | [1.158（0.9648×） ✅｜参❔](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) | [1.157（0.9644×） ✅｜参❔](alignments/02-v7x-acceptance/results/phase2-v7x-l1-ops-own-on.report.md) |
| 路由 GEMM<br>`L1.ops.op.router_gemm` | 本侧 mlp_norm.out → 路由 logits | `layers.{1..3}.mlp.router.logits`<br>比较单元：3 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（单独编译）＋离线 FP64 计算 | [0.0152（0.0127×） ✅｜参❔](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) | [0.0148（0.0123×） ✅｜参❔](alignments/02-v7x-acceptance/results/phase2-v7x-l1-ops-own-on.report.md) |
| MoE 块<br>`L1.ops.kernel.moe` | 本侧 mlp_norm.out → 路由、分组专家、合并、共享专家（真值按 FP64 的分数重新选专家） | `layers.{1..3}.mlp.out`<br>比较单元：3 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（单独编译）＋离线 FP64 计算 | [0.9412（0.784×） ✅｜参❔](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) | [1.269（1.057×） ❌｜参❔](alignments/02-v7x-acceptance/results/phase2-v7x-l1-ops-own-on.report.md) |
| 稠密 MLP 前 RMSNorm + gate、up 投影<br>`L1.ops.op.dense_fc1` | 第 0 层本侧残差和 → RMSNorm + fc1 | `layers.0.mlp.fc1.out`<br>比较单元：1 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（单独编译）＋离线 FP64 计算 | [1.117（0.9308×） ✅｜参❔](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) | [1.118（0.9313×） ✅｜参❔](alignments/02-v7x-acceptance/results/phase2-v7x-l1-ops-own-on.report.md) |
| 门控激活 + down 投影<br>`L1.ops.op.dense_act_fc2` | 第 0 层本侧 fc1.out → silu(gate) · up → fc2 | `layers.0.mlp.fc2.out`<br>比较单元：1 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（单独编译）＋离线 FP64 计算 | [1.113（0.9271×） ✅｜参❔](alignments/01-v5p-acceptance/results/phase2-l1-ops-own-on.report.md) | [1.112（0.9266×） ✅｜参❔](alignments/02-v7x-acceptance/results/phase2-v7x-l1-ops-own-on.report.md) |
| 路由选中集合对 FP64<br>`L1.router.topk.fp64.layer1`<br>`L1.router.topk.fp64.layer2`<br>`L1.router.topk.fp64.layer3` | 本侧路由输入 mlp_norm.out → FP64 logits → 加偏置取 top-8，与本侧实际选中的比 | 第 1–3 层，每层 4096 个 token 的 top-8 集合<br>聚合：逐 token 比选中的 top-8 集合，算集合相同的比例，取最差的层 | 集合相同比例 ≥ 0.97｜参考：无 | 隔离回放（单独编译）＋离线 FP64 计算 | [0.9998 ✅](alignments/01-v5p-acceptance/results/phase2-l2-replay-rratio-on.report.md) | [1 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-l2-replay-rratio-on.report.md) |

非 ✅ 格子的原因：

- **MoE 块**（TPU v7x-4 ❌）：v7x 第 2 层 4096 个 token 里有 1 个在近似平局处选了和 FP64 不同的专家，去掉这个 token R 是 0.95；v5p 同一层有 2 个这样的 token，R 0.94（[`v7x-l1-moe-routeflip.json`](alignments/02-v7x-acceptance/results/v7x-l1-moe-routeflip.json)，台账 #5）。路由本身另由「路由选中集合对 FP64」与 L2「各层路由选中集合」判，两平台都过。

### 优化器

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| AdamW 单步 FP64 重放<br>`L1.optim.adamw.replay` | 用 TPU 第 1 步的梯度与动量在 FP64 里重算一步 AdamW 更新量，对 TPU 实际的更新量 | 全部参数的更新量（`update.*`）与二阶动量（专家参数每层 0、1 号）<br>比较单元：144 个张量<br>聚合：逐张量算 rel_L2，取最差的张量，对固定门槛判 | rel_L2 ≤ 0.005｜参考：rel_L2 ≤ 0.02、cos ≥ 0.9999 | 生产程序（第 1 步状态读）＋离线 FP64 重放 | [0.00213（0.425×） ✅｜参✅](alignments/01-v5p-acceptance/results/phase2-st1-on-state-gpuonly.report.md) | [0.00213（0.425×） ✅｜参✅](alignments/02-v7x-acceptance/results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| AdamW 连续 8 步重放<br>`L1.optim.adamw.8steps` | 第 k 步用第 k−1 步的动量，连续重放 8 步 | 第 1–8 步的更新量<br>比较单元：66 个张量<br>聚合：逐张量算 rel_L2，取最差的张量，对固定门槛判 | rel_L2 ≤ 0.005｜参考：无 | 生产程序（状态读）＋离线 FP64 重放 | [0.0048（0.9593×） ✅](alignments/01-v5p-acceptance/results/phase2-st8-opt8.report.md) | [0.00479（0.9575×） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-st8-opt8.report.md) |

### 用 golden 的输入（v5p 补充）

TPU 算子的输入换成 GPU 抓到的同一份，输出可以直接和 GPU 比。判法见 [`analysis/l1_replay_results.py`](analysis/l1_replay_results.py)。v7x 没有跑这一组。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 |
|---|---|---|---|---|---|
| 路由 GEMM<br>`L1.ops.router.gemm` | golden 的 mlp_norm.out → TPU 路由 GEMM | `layers.{1..3}.mlp.router.logits`<br>比较单元：3 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：无 | 单算子回放（单独编译）＋离线 FP64 计算 | [0.0152（0.0127×） ✅](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) |
| 路由 top-8<br>`L1.ops.router.topk` | TPU 的 logits 过 sigmoid、加 golden 的 expert_bias 取 top-8，与 GPU 选中的集合逐行比 | `layers.{1..3}.mlp.router.topk_idx`<br>比较单元：3 个用例<br>聚合：逐用例比，数不一致的用例 | 逐位｜参考：无 | 单算子回放（单独编译）＋离线 FP64 计算 | [逐位全中 ✅](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) |
| 路由 GEMM 的精度档<br>`L1.ops.router.precision` | 用不能被 BF16 表示的探针权重，核对 TPU 默认精度等于「两个操作数先舍 BF16、单遍乘、FP32 累加」 | 第 1–3 层各一个探针<br>比较单元：3 个用例<br>聚合：逐用例比，数不一致的用例 | 逐位｜参考：无 | 单算子回放（单独编译）＋离线 FP64 计算 | [逐位全中 ✅](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) |
| MLP 前 RMSNorm<br>`L1.ops.rmsnorm` | golden 的 bf16(层输入 + attn.out) → TPU RMSNorm | `layers.{1..3}.mlp_norm.out`<br>比较单元：3 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：无 | 单算子回放（单独编译）＋离线 FP64 计算 | [1（0.833×） ✅](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) |
| 门控激活<br>`L1.ops.act` | golden 的第 0 层 fc1.out → TPU silu(gate) · up，对主机上按 GPU 写法 bf16(bf16(silu(gate)) · up) 算的结果逐位比 | `layers.0.mlp.act.out`<br>比较单元：1 个用例<br>聚合：逐用例比，数不一致的用例 | 逐位｜参考：无 | 单算子回放（单独编译）＋离线 FP64 计算 | [逐位全中 ✅](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) |
| 注意力核心<br>`L1.ops.attn.core` | golden 的 q_norm、k_norm、v → TPU RoPE、缩放、splash 注意力 | `layers.{0..3}.attn.core.out`<br>比较单元：4 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：无 | 单算子回放（单独编译）＋离线 FP64 计算 | [1.002（0.835×） ✅](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) |
| MoE 块<br>`L1.ops.moe` | golden 的 mlp_norm.out → TPU 路由、分组 GEMM、合并、共享专家 | `layers.{1..3}.mlp.out`<br>比较单元：3 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：无 | 单算子回放（单独编译）＋离线 FP64 计算 | [0.9333（0.778×） ✅](alignments/01-v5p-acceptance/results/phase2-l1-ops-on.report.md) |

## L2 · 逐层

目的：一层一层比，误差不跨层累积。TPU 用逐层隔离回放：每层的输入都取 GPU golden 的 `layers.k.in`（第 1 步），所以每层只带着自己的误差。例外是「各层参数梯度」一行：它来自整网第 1 步的状态读，只是按层分组，带着上游各层的差和路由翻转。

参考档的逐层线（rel_L2、超差元素比例、均值差 < 1%）只用在第 0 层：它来自同类 GPU→TPU 迁移项目对注意力层输出的逐层前向检查，层内没有路由。第 1–3 层是 MoE 层，路由之后的观测点受近似平局翻转影响，参考档写「不适用」，按严格档 R 比判。

L2 共 7 项检查，检查条数 15 / 15（v5p / v7x），张量级比较 95 / 95 次。

每层的前向观测点（每层 11 个）：

| 层 | 观测点 |
|---|---|
| 第 0 层（稠密） | `attn.qkv.out`、`attn.q_norm.out`、`attn.k_norm.out`、`attn.core.out`、`attn.proj.out`、`attn.out`、`mlp_norm.out`、`mlp.fc1.out`、`mlp.fc2.out`、`mlp.out`、`out` |
| 第 1–3 层（MoE） | `attn.qkv.out`、`attn.q_norm.out`、`attn.k_norm.out`、`attn.core.out`、`attn.proj.out`、`attn.out`、`mlp_norm.out`、`mlp.router.logits`、`mlp.shared.out`、`mlp.out`、`out` |

第 1 步的参数梯度按层分组，每层比这些张量（专家参数每层只比 0、1 号两个专家，原因见文末「已知的覆盖缺口」）：

| 层 | 张量 |
|---|---|
| 第 0 层（8 个） | 输入 RMSNorm 权重、qkv、q_norm、k_norm、输出投影、MLP 前 RMSNorm 权重、fc1、fc2 |
| 第 1–3 层（各 13 个） | 输入 RMSNorm 权重、qkv、q_norm、k_norm、输出投影、MLP 前 RMSNorm 权重、路由权重、0 号与 1 号专家的 fc1 和 fc2、共享专家 fc1 和 fc2 |

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| 第 0 层前向<br>`L2.layer0.fwd` | 隔离回放的第 0 层，每个观测点对 FP64 真值，比 R | 第 0 层 11 个观测点<br>比较单元：11 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 < 0.01、超差 < 1%、均值差 < 0.01 | 隔离回放（单独编译）＋离线 FP64 计算 | [1.059（0.883×） ✅｜参✅](alignments/01-v5p-acceptance/results/phase2-l2-replay-rratio-on.report.md) | [1.059（0.883×） ✅｜参✅](alignments/02-v7x-acceptance/results/phase2-v7x-l2-replay-rratio-on.report.md) |
| 第 1 层前向<br>`L2.layer1.fwd` | 同上，第 1 层 | 第 1 层 11 个观测点<br>比较单元：11 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：不适用（受路由影响） | 隔离回放（单独编译）＋离线 FP64 计算 | [1.01（0.841×） ✅](alignments/01-v5p-acceptance/results/phase2-l2-replay-rratio-on.report.md) | [1.01（0.841×） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-l2-replay-rratio-on.report.md) |
| 第 2 层前向<br>`L2.layer2.fwd` | 同上，第 2 层 | 第 2 层 11 个观测点<br>比较单元：11 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：不适用（受路由影响） | 隔离回放（单独编译）＋离线 FP64 计算 | [1.011（0.843×） ✅](alignments/01-v5p-acceptance/results/phase2-l2-replay-rratio-on.report.md) | [1.012（0.843×） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-l2-replay-rratio-on.report.md) |
| 第 3 层前向<br>`L2.layer3.fwd` | 同上，第 3 层 | 第 3 层 11 个观测点<br>比较单元：11 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：不适用（受路由影响） | 隔离回放（单独编译）＋离线 FP64 计算 | [1.095（0.9122×） ✅](alignments/01-v5p-acceptance/results/phase2-l2-replay-rratio-on.report.md) | [1.015（0.846×） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-l2-replay-rratio-on.report.md) |
| 各层反向<br>`L2.layer0.bwd`<br>`L2.layer1.bwd`<br>`L2.layer2.bwd`<br>`L2.layer3.bwd` | 隔离回放的反向：每层的输入与输出端梯度都取 golden 的，比层输入的梯度对 FP64 真值的 R | `layers.{0..3}.in.grad`，每层 1 个<br>比较单元：4 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：不适用（受路由影响） | 隔离回放（单独编译）＋离线 FP64 计算 | [1.005（0.838×） ✅｜参✅](alignments/01-v5p-acceptance/results/phase2-l2-replay-bwd-fp64-on.report.md) | [0.982（0.818×） ✅｜参✅](alignments/02-v7x-acceptance/results/phase2-v7x-l2-replay-bwd-fp64-on.report.md) |
| 各层路由选中集合<br>`L2.layer1.router`<br>`L2.layer2.router`<br>`L2.layer3.router` | 隔离回放的 top-8 对 FP64 真值的集合一致比例。真值从层输入一路在 FP64 里算到路由，GPU 自己对它也只有 0.95 左右，所以门槛定为翻转率（1 − 一致比例）不超过 GPU 的 1.2 倍；L1 那条只看路由这一个算子，接近 1 | 第 1–3 层，每层 4096 个 token<br>聚合：逐 token 比选中的 top-8 集合，算集合相同的比例，取最差的层 | 集合相同比例 ≥ 0.942｜参考：无 | 隔离回放（单独编译）＋离线 FP64 计算 | [0.9553 ✅](alignments/01-v5p-acceptance/results/phase2-l2-replay-rratio-on.report.md) | [0.9543 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-l2-replay-rratio-on.report.md) |
| 各层参数梯度<br>`L2.layer0.grads`<br>`L2.layer1.grads`<br>`L2.layer2.grads`<br>`L2.layer3.grads` | 第 1 步每层的参数梯度，对 GPU golden；噪声只用 GPU 第 1 步重跑 3 对（2026-11-08 起；之前并上 TPU 只改执行参数 21 对，结果留作历史） | 第 0 层 8 个、第 1–3 层各 13 个张量（见上表）<br>比较单元：47 个张量<br>聚合：逐张量算 rel_L2、cos、范数比（每个张量拉直）；每层各有自己的 N（第 0–3 层 rel_L2 的 N 依次 5.34e-4、5.27e-4、1.36e-4、3.85e-6；GPU 重跑后几层几乎逐位相同，N 很小），每层按「实测 ÷ 本层 N」取最差，4 层再取最大比值，判法栏的 N 是起决定作用那一层的；这些梯度来自整网第 1 步，按层分组，不是逐层隔离回放 | ≤ N 过，> 2N 不过（rel_L2 的 N = 3.85e-06）｜参考：无 | 生产程序（第 1 步状态读） | [0.105（rel_L2，2.7e4×N，最差 1−cos 7.3e8×N） ❌](alignments/01-v5p-acceptance/results/phase2-st1-on-state-gpuonly.report.md) | [0.1（rel_L2，2.6e4×N，最差 1−cos 6.7e8×N） ❌](alignments/02-v7x-acceptance/results/phase2-v7x-st1-on-state-gpuonly.report.md) |

非 ✅ 格子的原因：

- **各层参数梯度**（TPU v5p-8 ❌、TPU v7x-4 ❌）：第 1 步每层 0.44% – 0.56% 的路由槽位在近似平局处两边各落一边，分到 token 少的专家梯度相对差到 0.10（台账 #5）；第 0 层是稠密层、没有路由，它的超出来自一维 norm 权重的梯度先以 BF16 落地（台账 #20，可选开关能消掉）。按梯度能量加权，TPU 对 GPU 整网只差 1.3e-3，1000 步统计在噪声内。对 FP64 的定责见 L3「对 FP64 定责」一表。

## L3 · 整网第 1 步

目的：整网跑完第 1 步，比输出、全部参数梯度、优化器状态、更新后的权重。L2 隔离了层，这一级看层与层串起来、加上优化器以后还对不对。

L3 共 17 项检查，检查条数 34 / 22（v5p / v7x），张量级比较 1,543 / 1,043 次。

比较的张量组（参数类专家只比每层 0、1 号）：

| 张量组 | 内容 | 个数 |
|---|---|---|
| 激活 | 第 1 步 rank 0 第一个 microbatch 的各层观测点 | 41 |
| 输出 | logits（120832 维）、逐 token loss | 2 |
| 参数梯度 `grad.*` | 4 层的参数（第 0 层 8 个、第 1–3 层各 13 个）+ 词嵌入、最后的 RMSNorm、输出层 | 50 |
| 更新后主权重 `main_param_after.*` | 同上 50 个参数更新后的 FP32 主权重 | 50 |
| 优化器状态 `exp_avg*` | 同上 50 个参数的一阶、二阶动量 | 100 |
| 更新后路由偏置 `buffer_after.*` | 第 1–3 层的 expert_bias | 3 |

### 对 GPU 比

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| 逐 token loss<br>`L3.step1.loss.per_token` | 整网链路：从 golden 的 embed.out 起，FP64 一路算到逐 token loss，比 R | rank 0 第一个 microbatch 的 4096 个 token<br>比较单元：1 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：无 | 整网前向回放（单独编译）＋离线 FP64 计算 | [0.9811（0.818×） ✅](alignments/01-v5p-acceptance/results/phase2-ff-chain.report.md) | [1.025（0.854×） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-ff-chain.report.md) |
| logits<br>`L3.step1.logits` | 同一条链路的 logits，比 R；另记三方 KL（TPU‖FP64、GPU‖FP64、TPU‖GPU） | `logits`，4096 × 120832<br>比较单元：1 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：无 | 整网前向回放（单独编译）＋离线 FP64 计算 | [0.957（0.798×） ✅](alignments/01-v5p-acceptance/results/phase2-ff-chain.report.md) | [0.9995（0.833×） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-ff-chain.report.md) |
| 全部参数梯度<br>`L3.step1.grads.all` | 第 1 步全部参数梯度对 GPU golden；噪声只用 GPU 第 1 步重跑 3 对（2026-11-08 起） | `grad.*` 50 个张量<br>比较单元：50 个张量<br>聚合：逐张量算 rel_L2、cos、范数比，取 50 个里最差的张量；对齐报告里「整网只差 1.3e-3」是同一批梯度按能量合成的数，只用于定责 | ≤ N 过，> 2N 不过（rel_L2 的 N = 5.34e-04）｜参考：无 | 生产程序（第 1 步状态读） | [0.105（rel_L2，196×N，最差 1−cos 3.9e4×N） ❌](alignments/01-v5p-acceptance/results/phase2-st1-on-state-gpuonly.report.md) | [0.1（rel_L2，188×N，最差 1−cos 3.6e4×N） ❌](alignments/02-v7x-acceptance/results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| 梯度分布<br>`L3.step1.grads.dist` | 同一批梯度的标准差比、零元素比例差、逐元素相对误差 p99 | `grad.*` 50 个张量<br>比较单元：50 个张量<br>聚合：逐张量算（每个张量拉直成一个向量），每个指标取这批里最差的张量，各自对 N 判 | ≤ N 过，> 2N 不过（零元素比例差 的 N = 1.40e-05）｜参考：无 | 生产程序（第 1 步状态读） | [7.26e-04（零元素比例差，51.9×N） ❌](alignments/01-v5p-acceptance/results/phase2-st1-on-state-gpuonly.report.md) | [0.00781（零元素比例差，559×N） ❌](alignments/02-v7x-acceptance/results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| 优化器状态<br>`L3.step1.optim_state` | 第 1 步更新后的一阶、二阶动量。第 1 步的一阶动量就是 0.1 × 梯度，所以这一格和「全部参数梯度」几乎相同 | `exp_avg.*`、`exp_avg_sq.*` 100 个张量<br>比较单元：100 个张量<br>聚合：逐张量算（每个张量拉直成一个向量），每个指标取这批里最差的张量，各自对 N 判 | ≤ N 过，> 2N 不过（rel_L2 的 N = 0.00101）｜参考：无 | 生产程序（第 1 步状态读） | [0.105（rel_L2，104×N，最差 1−cos 1.2e4×N） ❌](alignments/01-v5p-acceptance/results/phase2-st1-on-state-gpuonly.report.md) | [0.1（rel_L2，99.6×N，最差 1−cos 1.1e4×N） ❌](alignments/02-v7x-acceptance/results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| 更新后主权重<br>`L3.step1.updated` | 第 1 步更新后的 FP32 主权重 | `main_param_after.*` 50 个张量<br>比较单元：50 个张量<br>聚合：逐张量算（每个张量拉直成一个向量），每个指标取这批里最差的张量，各自对 N 判 | ≤ N 过，> 2N 不过（rel_L2 的 N = 4.13e-05）｜参考：rel_L2 ≤ 0.02、cos ≥ 0.9999 | 生产程序（第 1 步状态读） | [5.28e-05（rel_L2，1.28×N，最差 1−cos 1.63×N） 🟡｜参✅](alignments/01-v5p-acceptance/results/phase2-st1-on-state-gpuonly.report.md) | [5.24e-05（rel_L2，1.27×N，最差 1−cos 1.61×N） 🟡｜参✅](alignments/02-v7x-acceptance/results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| 更新后路由偏置<br>`L3.step1.bias_after` | 第 1 步按负载更新后的 expert_bias | `buffer_after.*` 3 个张量（每层 192 个专家）<br>比较单元：3 个张量<br>聚合：逐张量算（每个张量拉直成一个向量），每个指标取这批里最差的张量，各自对 N 判 | ≤ N 过，> 2N 不过（rel_L2 的 N = 0）｜参考：无 | 生产程序（第 1 步状态读） | [0（rel_L2，0×N） ✅](alignments/01-v5p-acceptance/results/phase2-st1-on-state-gpuonly.report.md) | [0.0127（rel_L2，∞×N） ❌](alignments/02-v7x-acceptance/results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| 主权重与动量是真 FP32<br>`L3.step1.fake_fp32.main_param`<br>`L3.step1.fake_fp32.optim_state` | FP32 张量的低 16 位是否几乎全 0（是的话说明实际只存了 BF16 精度） | 主权重 50 个、动量 100 个<br>比较单元：150 个张量<br>聚合：逐张量看低 16 位全 0 的比例，取最高的张量 | low16_zero_frac ≤ 0.99｜参考：无 | 生产程序（第 1 步状态读） | [0.00276（0.00278×） ✅](alignments/01-v5p-acceptance/results/phase2-st1-on-state-gpuonly.report.md) | [0.00781（0.00789×） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| 激活逐位（只报）<br>`L3.step1.act.bitwise` | 隔离回放的各层激活对 golden 逐位比。跨平台不要求逐位，只记事实 | 41 个激活张量<br>比较单元：41 个张量<br>聚合：逐张量按字节比，数不同的个数 | 逐位｜参考：无 | 隔离回放（单独编译） | [41 / 41 不同 ❌](alignments/01-v5p-acceptance/results/phase2-l2-replay-bitwise.report.md) | [41 / 41 不同 ❌](alignments/02-v7x-acceptance/results/phase2-v7x-l2-replay-bitwise.report.md) |
| 比全了没有<br>`L3.step1.coverage` | TPU 抓到的张量名单对 golden 的名单（两边本来就没有对应量的三类除外） | 第 1 步全部张量名<br>比较单元：367 个名字<br>聚合：比名字清单，数缺的个数 | 逐位｜参考：无 | 观测构建（加钩子，只比名字清单） | [全部比到 ✅](alignments/01-v5p-acceptance/results/phase2-t-a-coverage.report.md) | [全部比到 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-coverage.report.md) |

非 ✅ 格子的原因：

- **全部参数梯度**（TPU v5p-8 ❌、TPU v7x-4 ❌）：由路由近似平局主导（台账 #5），同 L2「各层参数梯度」。最差的张量是 MoE 层 0 号专家的 fc1 / fc2 梯度。
- **梯度分布**（TPU v5p-8 ❌、TPU v7x-4 ❌）：同「全部参数梯度」：被路由翻转的专家梯度把逐元素相对误差 p99 推到 1.68，GPU 3 对的上限只有 0.043（约 39 倍；当时并集的上限是 1.6，1.05 倍）。
- **优化器状态**（TPU v5p-8 ❌、TPU v7x-4 ❌）：动量由梯度直接决定，原因同「全部参数梯度」。
- **更新后主权重**（TPU v5p-8 🟡、TPU v7x-4 🟡）：v5p 是梯度差的下游，量级很小（rel_L2 5e-5；最差的 1−cos 是 N 的 1.11 倍）；对 FP64 按能量算 R 0.92，TPU 的更新量更接近精确值。参考档（rel_L2 ≤ 0.02、cos ≥ 0.9999）两平台都过。
- **更新后路由偏置**（TPU v7x-4 ❌）：v7x 第 1 层 65 号专家的偏置更新方向和 GPU 相反（192 个里 1 个，台账 #17）。GPU 第 1 步重跑 3 对里没有一对翻号（撤回前的并集 24 对里也没有），所以 N = 0，任何翻号都判不过。按 FP64 这个专家选中 2730 个 token，比平均 2730.67 少 0.67，偏置应当加：v7x 和精确算术一致，GPU 三份与 v5p 都减。
- **激活逐位**（TPU v5p-8 ❌、TPU v7x-4 ❌）：两种硬件的舍入顺序不同，按定义不会逐位；这条 `hard: false`，不参与验收，数值上的判定看 L2 前向的 R。

### 对 FP64 定责

第 1 步梯度超出噪声时，用 FP64 整步参照判是谁离精确算术更远：同一份权重、同一个全局 batch 在 FP64 里求梯度，比 R = TPU 离 FP64 ÷ GPU 离 FP64。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| 各层参数梯度对 FP64（只报）<br>`L2.layer0.grads`<br>`L2.layer1.grads`<br>`L2.layer2.grads`<br>`L2.layer3.grads` | 每层参数梯度的 R（逐张量取最差） | 第 0 层 8 个、第 1–3 层各 13 个张量<br>比较单元：47 个用例<br>聚合：逐张量算 R 比，取最差的张量；另报按能量合成的 R 比（√(Σ‖TPU − FP64‖²) ÷ √(Σ‖GPU − FP64‖²)），只用于定责 | R 比 ≤ 1.2｜参考：无 | 生产程序（第 1 步状态读）＋离线 FP64 计算 | [2.2（1.83×） ❌](alignments/01-v5p-acceptance/results/phase2-st1-fp64-on.report.md) | [2.7（2.25×） ❌](alignments/02-v7x-acceptance/results/phase2-v7x-st1-fp64-on.report.md) |
| 全部参数梯度对 FP64（只报）<br>`L3.step1.grads.all` | 50 个梯度张量的 R（逐张量取最差） | `grad.*` 50 个张量<br>比较单元：50 个用例<br>聚合：逐张量算 R 比，取 50 个里最差的张量（v5p 2.20、v7x 2.70）；按能量合成的整网 R 比是 1.20 / 1.21，只用于定责 | R 比 ≤ 1.2｜参考：无 | 生产程序（第 1 步状态读）＋离线 FP64 计算 | [2.2（1.83×） ❌](alignments/01-v5p-acceptance/results/phase2-st1-fp64-on.report.md) | [2.7（2.25×） ❌](alignments/02-v7x-acceptance/results/phase2-v7x-st1-fp64-on.report.md) |
| 更新后主权重与动量对 FP64（只报）<br>`L3.step1.updated`<br>`L3.step1.optim_state` | 用这一侧自己的主权重与 FP64 梯度在 FP64 里重放一步 AdamW 得到真值，比 R | 主权重 50 个、动量 100 个<br>比较单元：150 个用例<br>聚合：逐张量算 R 比，取最差的张量；按能量合成的整体 R 比：主权重 0.92 / 0.94，动量 1.20 / 1.21（v5p / v7x） | r_ratio_own ≤ 1.2｜参考：部分检查有，见格子 | 生产程序（第 1 步状态读）＋离线 FP64 计算 | [7.95（6.62×） ❌｜参❔](alignments/01-v5p-acceptance/results/phase2-st1-fp64-on.report.md) | [51.3（42.8×） ❌｜参❔](alignments/02-v7x-acceptance/results/phase2-v7x-st1-fp64-on.report.md) |
| 更新后路由偏置对 FP64（只报）<br>`L3.step1.bias_after` | 按 FP64 选中数算出的偏置更新方向，对两侧实际的方向。R 比是「更新方向和真值相反的专家数」之比的平方根，格子写 3 层里最大的那个，两边都是 0 个时记为 1：v5p 第 1 层 1 对 1（GPU 与 v5p 是同一个专家方向反），其余两层 0 对 0；v7x 第 1 层 GPU 有 1 个专家方向反、TPU 没有（这一层 R = 0），其余两层 0 对 0 | 第 1–3 层 192 个专家<br>比较单元：3 个用例<br>聚合：逐层数方向和 FP64 相反的专家个数，R 比 = 两侧个数之比的平方根，取 3 层最大 | R 比 ≤ 1.2｜参考：无 | 生产程序（第 1 步状态读）＋离线 FP64 计算 | [1（0.833×） ✅](alignments/01-v5p-acceptance/results/phase2-st1-fp64-on.report.md) | [1（0.833×） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-st1-fp64-on.report.md) |

非 ✅ 格子的原因：

- **各层参数梯度对 FP64**（TPU v5p-8 ❌、TPU v7x-4 ❌）：超线的是两类张量：一维 norm 权重、输出层与 embedding 的梯度 TPU 先以 BF16 落地（R 最高 2.20，v7x 2.70，台账 #20）；经过路由的专家梯度随近似平局翻转变（台账 #5）。投影、共享专家、稠密 MLP 的梯度 R 0.45 – 0.96，TPU 更接近精确值。
- **全部参数梯度对 FP64**（TPU v5p-8 ❌、TPU v7x-4 ❌）：同上。按梯度能量加权整网 R 1.20（v7x 1.21）；打开台账 #20 的三个可选开关后 0.67（v7x 0.72），见下一表。
- **更新后主权重与动量对 FP64**（TPU v5p-8 ❌、TPU v7x-4 ❌）：超线的是几个 128 维的 q / k norm：Adam 第 1 步的更新量近似 lr · sign(g)，小张量上少数元素的符号就决定比值。按能量算整体，更新后主权重 0.92（v7x 0.94）；两个动量 1.20（v7x 1.21），和梯度同源（台账 #20），打开三个可选开关后降到 0.67。

### 可选开关：梯度以 FP32 落地

台账 #20 的三个开关（`rmsnorm_scale_grad_fp32`、`logits_dense_wgrad_fp32`、`embedding_grad_fp32`，默认关，不在当前对齐配置里）打开后重判。是否打开由使用方决定，代价见对齐报告第 6 节。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| 全部参数梯度对 FP64（开关打开）（只报）<br>`L3.step1.grads.all` | 同「全部参数梯度对 FP64」 | `grad.*` 50 个张量<br>比较单元：50 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：无 | 生产程序（第 1 步状态读）＋离线 FP64 计算 | [1.785（1.49×） ❌](alignments/01-v5p-acceptance/results/phase2-st1-fp64-gradfp32.report.md) | — |
| 第 1 步整步（开关打开）（只报）<br>`L2.layer0.grads`<br>`L2.layer1.grads`<br>`L2.layer2.grads`<br>`L2.layer3.grads`<br>… | 同「对 GPU 比」一表的梯度、状态、更新后权重，噪声同样只用 GPU 3 对 | 梯度 50、主权重 50、动量 100、路由偏置 3<br>比较单元：450 个张量<br>聚合：逐张量算（每个张量拉直成一个向量），每个指标取这批里最差的张量，各自对 N 判；逐张量看低 16 位全 0 的比例，取最高的张量 | ≤ N 过，> 2N 不过（rel_L2 的 N = 3.85e-06）｜参考：部分检查有，见格子 | 生产程序（第 1 步状态读） | [0.105（rel_L2，2.7e4×N，最差 1−cos 7.3e8×N） ❌｜参✅](alignments/01-v5p-acceptance/results/phase2-st1-gradfp32-gpuonly.report.md) | — |

非 ✅ 格子的原因：

- **全部参数梯度对 FP64（开关打开）**（TPU v5p-8 ❌）：剩下的超线张量是第 2 层 MLP 前 RMSNorm（1.79）与第 2 层 1 号专家的 fc2（1.32），都经过路由，随近似平局翻转变（台账 #5）。
- **第 1 步整步（开关打开）**（TPU v5p-8 ❌）：只用 GPU 3 对作尺子（2026-11-08 起）：第 0 层参数梯度 10.6 → 5.41 倍 N（仍不过），更新后主权重 1.63 → 1.30 倍 N（仍待定）；其余格子由路由近似平局主导（台账 #5）。并集尺子下（历史）是 2.12 → 1.34、1.11 → 1.001。

### 续训

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| 存档恢复后的状态<br>`L3.resume.state`<br>`L3.resume.step11`<br>`L3.resume.coverage` | 存档再恢复后的训练状态，对不中断的运行逐位比 | 恢复时的状态张量；恢复后第 11 步全量张量<br>比较单元：91 个参数张量、256 个张量、256 个名字<br>聚合：逐张量比 sha256，数不同的个数；逐张量按字节比，数不同的个数；比名字清单，数缺的个数 | 逐位｜参考：无 | 生产程序（存档与续训） | [0 / 91 不同 ✅](alignments/01-v5p-acceptance/results/phase2-resume.report.md) | [0 / 91 不同 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-resume.report.md) |

## L4 · 多步训练

目的：训练 1000 步，看逐步曲线是否落在 GPU 自己重跑的散布里。前三级的差如果会累积，在这里显出来。

L4 共 17 项检查，检查条数 30 / 34（v5p / v7x）。

- 参照：TPU 只和最终 golden 的基线 `rcc-a`（GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加，见[对齐报告](ALIGNMENT-REPORT.md)第 2 节）逐步比。
- 噪声上限 N：`rcc-a..j` 10 份重跑两两配对的 45 对比较里的最大差。
- 每条序列算三个判定量：带符号均值、MARE（平均绝对相对差）、最大单步；三个都对 N 判，取最差。首步差只报。

### 正式验收（对最终 golden `rcc-a`）

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| loss<br>`L4.loss` | 第 1–1000 步逐步 loss 的相对差 | `lm_loss`，第 1–1000 步<br>比较单元：1,000 步<br>聚合：每步一个标量（全局批的平均 loss），算逐步相对差，再在第 1–1000 步内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 1.38e-04）｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | 生产程序 | [1.10e-04（MARE，0.797×N） ✅｜参✅](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-on.report.md) | [1.13e-04（MARE，0.817×N） ✅｜参✅](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc.report.md) |
| loss 后段<br>`L4.loss.tail` | 第 301–1000 步（前 300 步曲线下降快、相对差大，单独看后段的系统偏移） | `lm_loss`，第 301–1000 步<br>比较单元：700 步<br>聚合：同上，窗口是第 301–1000 步 | ≤ N 过，> 2N 不过（最大单步 的 N = 5.91e-04）｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | 生产程序 | [4.71e-04（最大单步，0.797×N） ✅｜参✅](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-on.report.md) | [1.24e-04（MARE，0.831×N，N=1.50e-04） ✅｜参✅](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc.report.md) |
| grad_norm<br>`L4.grad_norm` | 逐步全局梯度范数；分母用两边均值，交换两边只差一个符号（对齐报告第 8 节） | `grad_norm`，第 1–1000 步<br>比较单元：1,000 步<br>聚合：每步一个标量：全部可训练参数梯度的全局 L2 范数（裁剪前，约 124 亿个元素合成一个数）；逐步相对差的分母用两边均值；再算带符号均值、MARE、最大单步 | ≤ N 过，> 2N 不过（MARE 的 N = 0.033）｜参考：无 | 生产程序 | [0.028（MARE，0.846×N） ✅](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-on.report.md) | [0.0281（MARE，0.851×N） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc.report.md) |
| grad_norm（以 golden 为分母）（只报）<br>`L4.grad_norm.golden_den` | 同上，分母用 golden（原口径），并排保留 | `grad_norm`，第 1–1000 步<br>比较单元：1,000 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 0.0335）｜参考：无 | 生产程序 | [0.028（MARE，0.835×N） ✅](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-on.report.md) | [0.0283（MARE，0.844×N） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc.report.md) |
| param_norm<br>`L4.param_norm` | 逐步全局参数范数；TPU 读按组合成的 `param_norm_from_groups`（整网一次 FP32 归约测得太粗，对齐报告第 8 节） | `param_norm`，第 1–1000 步<br>比较单元：1,000 步<br>聚合：每步一个标量：全部可训练参数 FP32 主权重的全局 L2 范数（约 124 亿个元素，不含路由偏置）；TPU 由按组记录的平方和合成；再算带符号均值、MARE、最大单步 | ≤ N 过，> 2N 不过（最大单步 的 N = 4.86e-07）｜参考：带符号均值 ≤ 5e-6、最大单步 ≤ 1.25e-5 | 生产程序 | [2.44e-07（最大单步，0.502×N） ✅｜参✅](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-on.report.md) | [2.58e-07（最大单步，0.531×N） ✅｜参✅](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc.report.md) |
| param_norm（TPU 原始值）（只报）<br>`L4.param_norm.raw` | 同上，TPU 读日志里的原始 param_norm | `param_norm`，第 1–1000 步<br>比较单元：1,000 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（最大单步 的 N = 4.86e-07）｜参考：无 | 生产程序 | [3.55e-07（最大单步，0.731×N） ✅](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-on.report.md) | [2.79e-07（最大单步，0.574×N） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc.report.md) |
| 路由偏置曲线<br>`L4.router.bias_std` | 逐步路由偏置的统计量（均值、标准差等） | 第 1–1000 步，第 1–3 层<br>聚合：每步每层一个统计量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步 | ≤ N 过，> 2N 不过（最大单步 的 N = 0.016）｜参考：无 | 生产程序 | [0.013（最大单步，0.81×N） ✅](alignments/01-v5p-acceptance/results/phase2-acc-health-rcc.report.md) | [0.013（最大单步，0.809×N） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-acc-health-rcc.report.md) |
| 健康信号<br>`L4.health` | loss、grad_norm 等有没有 NaN / Inf、有没有突跳（单步超过中位数 3 倍） | 第 1–1000 步<br>比较单元：3 个字段<br>聚合：逐步看 NaN / Inf 与突跳 | spike_ratio ≤ 3｜参考：无 | 生产程序 | [健康 ✅](alignments/01-v5p-acceptance/results/phase2-acc-health-rcc.report.md) | [健康 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-acc-health-rcc.report.md) |

### 换并行方式

TPU 改成和 GPU 一样的专家并行 EP = 8（当前对齐配置是非专家参数 FSDP 8、专家不切），同样对 `rcc-a` 比。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| EP = 8 的 L4（只报）<br>`L4.loss`<br>`L4.loss.tail`<br>`L4.grad_norm`<br>`L4.param_norm`<br>… | loss、loss 后段、grad_norm、param_norm、学习率 | 第 1–1000 步<br>比较单元：4,700 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判；逐步比，任一步超出容差就不过 | ≤ N 过，> 2N 不过（最大单步 的 N = 5.91e-04）｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | 生产程序 | [4.61e-04（最大单步，0.779×N） ✅｜参✅](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-ep8.report.md) | [4.78e-04（最大单步，0.809×N） ✅｜参✅](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc-ep8.report.md) |

### 可选开关：梯度以 FP32 落地

台账 #20 的三个开关打开后的 1000 步，不是当前对齐配置。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| 开关打开后的 L4（只报）<br>`L4.loss`<br>`L4.loss.tail`<br>`L4.grad_norm`<br>`L4.param_norm`<br>… | loss、loss 后段、grad_norm、param_norm、学习率 | 第 1–1000 步<br>比较单元：4,700 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判；逐步比，任一步超出容差就不过 | ≤ N 过，> 2N 不过（MARE 的 N = 1.50e-04）｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | 生产程序 | [1.36e-04（MARE，0.9067×N） ✅｜参✅](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-gradfp32.report.md) | [0.482（最大单步，1.049×N，N=0.459） 🟡｜参✅](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| 开关打开后的只报变体（只报）<br>`L4.grad_norm.golden_den`<br>`L4.param_norm.raw` | grad_norm 以 golden 为分母、param_norm 读 TPU 原始值（同正式验收一表里的两条只报行） | 第 1–1000 步<br>比较单元：2,000 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 0.0335）｜参考：无 | 生产程序 | [0.0261（MARE，0.778×N） ✅](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-gradfp32.report.md) | [0.635（最大单步，1.14×N，N=0.559） 🟡](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| 开关打开后的路由偏置与健康信号（只报）<br>`L4.router.bias_std`<br>`L4.health` | 同「路由偏置曲线」「健康信号」 | 第 1–1000 步<br>比较单元：3 个字段<br>聚合：每步每层一个统计量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步；逐步看 NaN / Inf 与突跳 | ≤ N 过，> 2N 不过（带符号均值 的 N = 0.00193）｜参考：无 | 生产程序 | [0.00164（带符号均值，0.852×N） ✅](alignments/01-v5p-acceptance/results/phase2-acc-health-rcc-gradfp32.report.md) | [0.0013（带符号均值，0.675×N） ✅](alignments/02-v7x-acceptance/results/phase2-v7x-acc-health-rcc-gradfp32.report.md) |

非 ✅ 格子的原因：

- **开关打开后的 L4**（TPU v7x-4 🟡）：v7x 的 grad_norm 最大单步差 0.482，噪声上限 0.459（1.05 倍）；带符号均值 0.10 倍、MARE 0.97 倍。不开这三个开关时两平台对 `rcc-a` 已全过。
- **开关打开后的只报变体**（TPU v7x-4 🟡）：v7x 的 grad_norm 以 golden 为分母时最大单步 0.635，噪声上限 0.559（1.14 倍）；这一口径交换两边不对称（对齐报告第 8 节），只报。

### 对未修的 GPU 生产 golden `rc`（已解释差异）

同样的比较，参照换成没有修输出头 dgrad 的 GPU 生产 golden `rc`（单基线 `rc-a`，噪声 `rc` 5 份两两 10 对）。这里的超出是 GPU 一侧的系统偏差，不是 TPU 的问题。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| 对 `rc-a` 的 L4（只报）<br>`L4.loss`<br>`L4.loss.tail`<br>`L4.grad_norm`<br>`L4.param_norm`<br>… | loss、loss 后段、grad_norm、param_norm、学习率。被测是换代前配置（v5p `t-rmap-1000`、v7x `t-v7x-rmap-1000-a`，打开 `dense_fp32_accumulation` 之前） | 第 1–1000 步<br>比较单元：4,700 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判；逐步比，任一步超出容差就不过 | ≤ N 过，> 2N 不过（带符号均值 的 N = 1.23e-07）｜参考：带符号均值 ≤ 5e-6、最大单步 ≤ 1.25e-5 | 生产程序 | [2.93e-06（带符号均值，23.9×N） ❌｜参✅](alignments/01-v5p-acceptance/results/phase2-acceptance-rc.report.md) | [2.93e-06（带符号均值，23.9×N） ❌｜参✅](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rc.report.md) |

非 ✅ 格子的原因：

- **对 `rc-a` 的 L4**（TPU v5p-8 ❌、TPU v7x-4 ❌）：B200 上输出头 dgrad 沿 12 万词表的长求和有朝 0 的系统偏差（台账 #1、#2），让 `rc` 的 loss 稳定偏高、param_norm 少漂 2.07e-2。最终 golden `rcc` 只修这一处，TPU 就落在它的重跑散布里（上一表）。排查全过程见案例 01。

### 程序与工具

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | TPU v5p-8 | TPU v7x-4 |
|---|---|---|---|---|---|---|
| 梯度 NaN 先于 loss 报警（演练）<br>`L4.health.grad_norm` | 第 5 步故意让梯度变成 NaN，健康检查必须在 grad_norm 上先报，不能等到 loss | 8 步的演练运行<br>比较单元：1 个字段<br>聚合：逐步看 NaN / Inf 与突跳 | 期望报警｜参考：— | 生产程序（埋错演练） | [按期望报警 ✅](alignments/01-v5p-acceptance/results/phase2-t-nan8.report.md) | [按期望报警 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-t-nan8.report.md) |
| 加抓数不改数值（只报）<br>`DET.hooks.noop` | 同一个 TPU 程序加不加抓数输出，逐步标量是否逐位相同 | 6 个逐步标量<br>比较单元：6 个字段<br>聚合：逐步、逐张量比两次运行，数不可复现的项 | 逐位｜参考：无 | 观测构建对生产程序 | [不可复现：['lm_loss@1', 'grad_norm@1', 'param_norm@2']，张量不同 0 ❌](alignments/01-v5p-acceptance/results/phase2-obs-gate.report.md) | [不可复现：['lm_loss@1', 'grad_norm@1', 'param_norm@12']，张量不同 0 ❌](alignments/02-v7x-acceptance/results/phase2-v7x-obs-gate.report.md) |
| 续训重叠步<br>`L4.resume.overlap` | 存档恢复后与不中断的运行重叠的步，逐步标量逐位相同 | 10 个重叠步<br>比较单元：10 步<br>聚合：逐步比重叠的步，数不同的步 | 逐位｜参考：无 | 生产程序（存档与续训） | [重叠步全部逐位相同 ✅](alignments/01-v5p-acceptance/results/phase2-resume.report.md) | [重叠步全部逐位相同 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-resume.report.md) |
| TPU 自己重跑逐位（v7x）<br>`L4.repro.lm_loss`<br>`L4.repro.grad_norm`<br>`L4.repro.param_norm`<br>`L0.repro.data.fingerprint` | 同一份代码包、配置、数据在 v7x 上跑两次（各占一个切片），1000 步的逐步 loss、grad_norm、param_norm 与输入指纹逐位相同。用的是换代前配置（`t-v7x-rmap-1000-a/-b`，打开 `dense_fp32_accumulation` 之前）；v5p 的同程序双跑在日常回归里（`reg-t0-fa-a`、`reg-t1-fa-a`），不在本页 | 第 1–1000 步<br>比较单元：65,000 个数据单元、3,000 步<br>聚合：逐步比，任一步超出容差就不过；逐单元比 sha256，数不同的个数 | rtol ≤ 0、abs_ulp ≤ 0｜参考：无 | 生产程序 | — | [逐步在容差内 ✅](alignments/02-v7x-acceptance/results/phase2-v7x-rmap-repro.report.md) |

非 ✅ 格子的原因：

- **加抓数不改数值**（TPU v5p-8 ❌、TPU v7x-4 ❌）：多出的抓数输出改变了 XLA 的融合，第 1 步 loss 就不逐位。所以 TPU 一侧的正式证据都不经过抓数钩子：L0–L3 用训练步之间的状态读、逐层隔离回放与单算子回放，L4 用不加钩子的生产构建（用户指南第 4 章 4.8 节）。

## GPU 侧的检查：参照本身可不可信、工具能不能抓到错

这些检查只在 GPU 上做（对齐实践 [00](alignments/00-gpu-noise-and-drills/README.md) 与 [03](alignments/03-gpu-remat-golden/README.md)）。它们回答两件事：GPU golden 能不能当参照（重跑的噪声多大、哪些执行方式会改数值），以及 janus 的检查能不能抓到真错。故障演练（9 种埋错 × 24 项检查）的矩阵在[总览](dashboard/dashboard.md)末尾。

GPU 侧共 19 项检查，检查条数 67，张量级比较 922 次。

### 只换执行方式，数值变不变

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | GPU B200 × 8 |
|---|---|---|---|---|---|
| 数据并行副本一致<br>`L2.dp.replica` | 8 个数据并行副本上的同一个对象是否逐位相同 | 第 1 步的参数、梯度等对象<br>比较单元：41 个对象<br>聚合：逐对象比各副本，数不一致的个数 | 逐位｜参考：无 | GPU 训练运行 | [41 个对象副本全部一致 ✅](alignments/00-gpu-noise-and-drills/results/phase1-full-noise28.report.md) |
| 截断模型前缀<br>`L2.truncate.embed`<br>`L2.truncate.layer0`<br>`L2.truncate.layer1`<br>`L2.truncate.router1` | 截 2 层的模型与截 4 层的模型，前 2 层的第 1 步输出逐位相同 | embed、第 0–1 层输出、第 1 层路由<br>比较单元：24 个张量<br>聚合：逐张量按字节比，数不同的个数 | 逐位｜参考：无 | GPU 训练运行 | [0 / 24 不同 ✅](alignments/00-gpu-noise-and-drills/results/phase1-truncate.report.md) |
| 专家并行 EP8 ↔ EP4<br>`L2.parallel.fwd.bitwise`<br>`L2.parallel.grads.tol` | 只换专家并行度，前向、反向与更新后参数 | 第 1 步激活、梯度、更新后稠密参数<br>比较单元：75 个张量<br>聚合：逐张量按字节比，数不同的个数；逐张量算（每个张量拉直成一个向量），每个指标取这批里最差的张量，各自对 N 判 | ≤ N 过，> 2N 不过（1−cos 的 N = 1.30e-07）｜参考：无 | GPU 训练运行 | [5.34e-07（1−cos，4.1×N） ❌](alignments/00-gpu-noise-and-drills/results/phase1-parallel.report.md) |
| 张量 / 上下文并行：loss<br>`L3.loss.step1` | 打开 TP 2 或 CP 2，第 1 步 loss 与窗口内统计对不切分的运行 | 逐步 loss（TP 2、CP 2 各一份）<br>比较单元：6 步<br>聚合：每步一个标量，算逐步相对差，首步差、带符号均值、最大单步 对固定门槛判 | 首步差 ≤ 1.00e-04｜参考：无 | GPU 训练运行 | [4.99e-06（0.0499×） ✅](alignments/00-gpu-noise-and-drills/results/phase1-tp-cp.tp2.report.md) |
| 张量 / 上下文并行：grad_norm<br>`L4.grad_norm` | 同上，逐步 grad_norm | 逐步 grad_norm（TP 2、CP 2 各一份）<br>比较单元：6 步<br>聚合：每步一个标量，算逐步相对差，首步差、带符号均值、最大单步 对固定门槛判 | 首步差 ≤ 0.001｜参考：无 | GPU 训练运行 | [2.37e-04（0.237×） ✅](alignments/00-gpu-noise-and-drills/results/phase1-tp-cp.tp2.report.md) |
| 梯度累积切分<br>`L3.ga.loss`<br>`L3.ga.grads`<br>`L3.ga.updated` | 同样的全局批，只改梯度累积的切分 | 第 1 步 loss、梯度、更新后参数<br>比较单元：100 个张量、3 步<br>聚合：每步一个标量，算逐步相对差，首步差、带符号均值、最大单步 对固定门槛判；逐张量算（每个张量拉直成一个向量），每个指标取这批里最差的张量，各自对 N 判 | ≤ N 过，> 2N 不过（rel_L2 的 N = 6.64e-04）｜参考：无 | GPU 训练运行 | [0.0206（rel_L2，31.1×N，最差 1−cos 1.45e+03×N） ❌](alignments/00-gpu-noise-and-drills/results/phase1-ga.report.md) |
| 激活重算<br>`DET.remat.vs_first`<br>`DET.replay.vs_first` | 开整层重算时，重算的前向与首次前向是否逐位相同；开路由回放后再比 | 第 1 步各层激活<br>比较单元：108 个张量<br>聚合：逐张量按字节比，数不同的个数 | 逐位｜参考：无 | GPU 训练运行 | [9 / 51 不同 ❌](alignments/00-gpu-noise-and-drills/results/phase1-remat.report.md) |
| 确定性设置<br>`L4.loss.exact`<br>`L4.lr`<br>`L3.step1.bitwise`<br>`L3.step1.coverage`<br>… | 确定性设置下同一程序重跑是否逐位 | 1000 步逐步标量、第 1 步张量<br>比较单元：48 个张量、2,000 步、24 个名字<br>聚合：逐步 / 逐张量按位比，数不同的个数；逐步比，任一步超出容差就不过；逐张量按字节比，数不同的个数；比名字清单，数缺的个数 | 逐位｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | GPU 训练运行 | [见报告 ✅｜参✅](alignments/00-gpu-noise-and-drills/results/phase1-determinism.report.md) |

非 ✅ 格子的原因：

- **专家并行 EP8 ↔ EP4**（GPU B200 × 8 ❌）：前向逐位；反向的规约顺序随专家并行度变，梯度差到 GPU 噪声的 4.1 倍（1−cos）。所以全部 golden 与噪声参照固定 EP 8，不混用。
- **梯度累积切分**（GPU B200 × 8 ❌）：embedding 梯度差到噪声上限的 31 倍：累加分组决定舍入位置。所以两边的梯度累积分组设成一样（每 rank 2 个 microbatch，台账 #12）。
- **激活重算**（GPU B200 × 8 ❌）：不开路由回放时 9 / 51 个张量不同（重算里路由会重选）；开路由回放后 0 / 57 不同，这就是 GPU 生产 golden `rc` 的配置。

### GPU 算子回放与合成用例

GPU 一侧把抓到的算子输入重新喂给同一个 kernel、对 CPU 复现的舍入方式、对 FP64 真值比。用来证明 GPU 参照的数是可复现、可解释的，也是 TPU 一侧 L1 判法（R 比 1.2）的起点。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | GPU B200 × 8 |
|---|---|---|---|---|---|
| GEMM 与 RMSNorm 回放<br>`L1.gemm.router.replay`<br>`L1.gemm.router.upcast_fp32`<br>`L1.gemm.attn_proj.replay`<br>`L1.gemm.qkv.replay`<br>… | 路由、qkv、输出投影 GEMM 与各 RMSNorm 用抓到的输入重算，对抓到的输出逐位 | 第 0–3 层各算子<br>比较单元：32 个用例<br>聚合：逐用例比，数不一致的用例；逐用例算，只报不判 | 逐位｜参考：无 | GPU 单算子回放 | [逐位全中 ✅](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) |
| 舍入方式复现<br>`L1.emulate.gemm`<br>`L1.emulate.other_guesses` | 在 CPU 上按张量核的累加方式复现 GEMM，对 GPU 输出逐位 | 路由 logits、投影<br>比较单元：28 个用例<br>聚合：逐用例比，数不一致的用例；逐用例算，只报不判 | 逐位｜参考：无 | GPU 单算子回放 | [逐位全中 ✅](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) |
| R 比自检<br>`L1.rratio.clean`<br>`L1.rratio.drill` | 干净实现 R ≈ 1；多舍一次的实现 R 被 1.2 的门槛抓到（演练） | RMSNorm、路由 sigmoid<br>比较单元：4 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：无 | GPU 单算子回放 | [1（0.833×） ✅](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) |
| 逐元素函数<br>`L1.elem.exp.fp32`<br>`L1.elem.sigmoid.fp32`<br>`L1.elem.silu.fp32`<br>`L1.elem.rsqrt.fp32`<br>… | exp、sigmoid、silu、rsqrt 对正确舍入值的 ULP；加、乘、类型转换逐位 | 合成输入与真实输入<br>比较单元：33 个用例<br>聚合：逐用例算 ulp_max，取最差的用例；逐用例算，只报不判；逐用例算 bit_match_frac、cr_match_frac、nonfinite_mismatch，取最差的用例 | ulp_max ≤ 2｜参考：无 | GPU 单算子回放 | [2（1×） ✅](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) |
| 极端输入<br>`L1.extreme.rmsnorm`<br>`L1.extreme.rmsnorm.overflow`<br>`L1.extreme.sigmoid`<br>`L1.extreme.softmax` | RMSNorm、sigmoid、softmax 在极小、次正规、极大、全 −inf 等输入上对 FP64 | 合成输入<br>比较单元：39 个用例<br>聚合：逐用例算 ulp_max、nonfinite_mismatch，取最差的用例；逐用例算，只报不判；逐用例算 rel_L2、nonfinite_mismatch，取最差的用例 | ulp_max ≤ 2｜参考：无 | GPU 单算子回放 | [1（0.5×） ✅](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) |
| 路由平局<br>`L1.ties.synthetic`<br>`L1.ties.cpu_torch`<br>`L1.ties.real` | 精确平局时 top-k 的选择规则 | 合成平局与真实路由输入<br>比较单元：15 个用例<br>聚合：逐用例比，数不一致的用例；逐用例算，只报不判 | 逐位｜参考：无 | GPU 单算子回放 | [逐位全中 ✅](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) |
| 反向（VJP）<br>`L1.vjp.fp32`<br>`L1.vjp.bf16` | RMSNorm、路由打分函数的反向对 FP64 解析梯度 | 第 1 层<br>比较单元：7 个用例<br>聚合：逐用例算 rel_L2，取最差的用例 | rel_L2 ≤ 0.01｜参考：无 | GPU 单算子回放 | [0.00172（0.172×） ✅](alignments/00-gpu-noise-and-drills/results/phase1-l1.report.md) |
| 专家 GEMM 与注意力 kernel<br>`L1.kernel.experts.fwd.replay`<br>`L1.kernel.experts.bwd.replay`<br>`L1.kernel.experts.vs_loop`<br>`L1.kernel.experts.wgrad`<br>… | 分组专家 GEMM、注意力 kernel 的前向与反向回放 | 第 0–3 层<br>比较单元：257 个用例<br>聚合：逐用例比，数不一致的用例；逐用例算，只报不判；逐用例算 rel_L2，取最差的用例 | rel_L2 ≤ 1.00e-05｜参考：无 | GPU 单算子回放 | [3.50e-06（0.35×） ✅](alignments/00-gpu-noise-and-drills/results/phase1-kernels.report.md) |
| padding 与因果<br>`L1.poison.grouped_gemm`<br>`L1.poison.attn_padding`<br>`L1.causal.attn`<br>`L1.causal.attn_future_nan` | padding 区填 NaN / Inf 不污染有效输出；改动未来的 token 不影响当前输出 | 合成用例<br>比较单元：33 个用例<br>聚合：逐用例比，数不一致的用例；逐用例算，只报不判 | 逐位｜参考：无 | GPU 单算子回放 | [逐位全中 ✅](alignments/00-gpu-noise-and-drills/results/phase1-kernels.report.md) |
| TF32 开关<br>`TF32.switch.matrix`<br>`TF32.switch.override_off`<br>`TF32.router_bwd`<br>`TF32.in_run`<br>… | 各种 TF32 开关组合下 FP32 GEMM 实际走不走 TF32 | 开关矩阵、训练中的探针<br>比较单元：44 个用例<br>聚合：逐用例算，只报不判；逐用例算 rel_L2，取最差的用例；逐用例算 probe_rel_l2_*，取最差的用例 | rel_L2 ≤ 1.00e-05｜参考：无 | GPU 单算子回放 | [1.15e-06（0.115×） ✅](alignments/00-gpu-noise-and-drills/results/phase1-kernels.report.md) |
| MoE 分阶段<br>`L2.moe.expert_weights`<br>`L2.moe.stages.chain`<br>`L2.moe.stages.local`<br>`L2.moe.movement`<br>… | MoE 块按分发、专家 GEMM、激活、合并、共享专家逐段回放 | 第 1–3 层<br>比较单元：75 个用例<br>聚合：逐用例比，数不一致的用例；逐用例算，只报不判 | 逐位｜参考：无 | GPU 单算子回放 | [HF 原件转成 Megatron 布局后 1152 个专家权重与训练加载时的 sha256 全部相同 ✅](alignments/00-gpu-noise-and-drills/results/phase1-moe-stages.report.md) |

## 已知的覆盖缺口

下面几处这一版没有覆盖或只覆盖了一部分。前两条是适用范围上的限制；接着五条是对照同类 GPU→TPU 迁移项目的检查范围找出来的；后几条是这次验收自己的限制。它们不改变上面任何一格的判定（判定只针对比到的对象），但限定了结论覆盖的范围。

| 缺口 | 现状 | 补做的办法 |
|---|---|---|
| 80 层生产形状没有验证 | 全部检查都在截 4 层的模型上做；日常回归的 T3 生产形状档位待定（生产配置与切片资源未定） | 在 80 层、生产切分与数据管线上至少跑 L0 与 L4 |
| EP = 8 只验了 L4 的四个量与学习率 | TPU 改成和 GPU 一样的 EP = 8 时，只对 `rcc-a` 比了 loss、loss 后段、grad_norm、param_norm 与学习率；路由偏置曲线、健康信号与 L0–L3 只在 FSDP 8 上做 | EP = 8 下补跑路由偏置与健康信号，必要时补第 1 步 |
| 第 1 步专家参数只比 2 / 192 个 | GPU 抓数默认每层只留 0、1 号专家的参数张量（`janus/dump/megatron.py` 的 `DEFAULT_SKIP`），L0 的初始参数清单与 L4 的 param_norm 覆盖全部专家 | GPU 侧去掉这条跳过规则重抓第 1 步，TPU 状态读本来就是全量 |
| TPU 上没有单独的输出端算子检查 | 词嵌入、最后的 RMSNorm、输出层、交叉熵只在 L3 整网链路里整体比 | 用 golden 的 `final_norm.out` 等输入在 TPU 上单独回放，对 FP64 判 R |
| TPU 上没有算子级反向 | 反向只比到层（L2 各层反向）与第 1 步整网梯度 | 对各主要算子做随机或真实余切的反向回放，对 FP64 VJP 判 R |
| 逐元素函数、极端输入、平局、padding / 因果的合成用例只在 GPU 上做过 | v5p 只核对了门控激活（`L1.ops.act`），v7x 没有这一组 | 在 TPU 上跑同样的合成用例，单芯片即可 |
| grad_norm 没有参考档 | 参考档只有 loss、param_norm、算子与优化器几类线 | 并列报告 grad_norm 的平均相对误差 |
| 换基线的敏感度没有逐份算 | L4 只对 `rcc-a` 比；换一份当基线，r 最多挪动 1，但实际挪动多少没算，`rcc` 各份的逐步序列没有入库 | 入库 `rcc-a..j` 的 `metrics.jsonl`，以每一份当基线各算一遍 |
| v7x 当前配置 1000 步的逐步序列没有入库 | v7x 对 `rc` 的 loss 偏移只能用推导值（对齐报告第 3 节） | 入库 `t-v7x-on-1000` 的 `metrics.jsonl` |
| 第 1 步比较的 GPU golden 不开重算，TPU 被测开重算 | GPU 上开整层重算加路由回放后，第 1 步激活已和不重算的 `g1k` 不逐位（`remat-vs-g1k` 的 `L3.step1.act.bitwise`，26 / 46 个张量不同），这部分差没有单独进第 1 步的尺子；1000 步时 `rc` 对 `g1k` 的 loss 在噪声内 | 第 1 步改用开重算的 GPU golden 做参照与噪声 |
| TPU 侧的故障演练只做了梯度 NaN 一种 | 9 种埋错的演练都在 GPU 上做 | 在 TPU 上照 GPU 的故障矩阵埋错 |
| v7x 上 TPU 自己的执行噪声没有标定 | 日常回归的 TPU 执行噪声只在 v5p 上标定（第 1 步整步的 7 份扰动运行两平台都有） | 照 v5p 的做法在 v7x 上标定 |

## 未列入本页的结果文件

01、02 两个实践里下面这些结果文件不进清单：旧配置或旧判法的结果（被上面的行取代），以及诊断用的臂。保留它们是为了可追溯。

| 结果 | 是什么、为什么不在清单里 | 判定 |
|---|---|---|
| [`phase2-acceptance-rcb-on`](alignments/01-v5p-acceptance/results/phase2-acceptance-rcb-on.report.md) | v5p 对 `rcb-a` 的交叉核对，只报：正式第一层仍对 `rcc-a`（`phase2-acceptance-rcc-on`） | pass 6 |
| [`phase2-l1-ops-own`](alignments/01-v5p-acceptance/results/phase2-l1-ops-own.report.md) | v5p 单算子，打开 `dense_fp32_accumulation` 之前的配置：qkv 与第 0 层 fc1 超线。被 `phase2-l1-ops-own-on` 取代 | pass 7、fail 2 |
| [`phase2-l1-ops-wcoff`](alignments/01-v5p-acceptance/results/phase2-l1-ops-wcoff.report.md) | v5p 单算子的诊断臂：整个关掉窗口化 collective matmul，用来定位 qkv 超线的来源（台账 #6） | pass 8、fail 1 |
| [`phase2-l2-replay-bwd-on`](alignments/01-v5p-acceptance/results/phase2-l2-replay-bwd-on.report.md) | v5p 逐层反向的旧判法：噪声只取 GPU 第 1 步重跑 3 对，这把尺子对路由翻转的宽度是 0。改按对 FP64 的 R 判（`phase2-l2-replay-bwd-fp64-on`） | pass 1、fail 3 |
| [`phase2-l2-replay-bwd-on-xnoise`](alignments/01-v5p-acceptance/results/phase2-l2-replay-bwd-on-xnoise.report.md) | v5p 逐层反向按两侧执行噪声并集判的对照：隔离回放里两侧只改执行方式都不翻路由，并集同样量不到路由翻转，只报 | pass 1、fail 3 |
| [`phase2-l2-replay-rratio`](alignments/01-v5p-acceptance/results/phase2-l2-replay-rratio.report.md) | v5p 逐层前向，打开 `dense_fp32_accumulation` 之前：4 层都只差在 `attn.qkv.out`。被 `phase2-l2-replay-rratio-on` 取代 | pass 6、fail 4 |
| [`phase2-obs-grad`](alignments/01-v5p-acceptance/results/phase2-obs-grad.report.md) | v5p 观测构建（每一步都加抓数钩子）第 1 步梯度的标定：没过，所以观测构建的张量不当正式证据（L4「加抓数不改数值」一行） | fail 2、inconclusive 1 |
| [`phase2-rcb-vs-rcc`](alignments/01-v5p-acceptance/results/phase2-rcb-vs-rcc.report.md) | GPU 一侧的便宜等价修法 `rcb-a`（输出头 dgrad 用 BF16 张量核分段）对 `rcc-a` 的 1000 步比较：只说明 golden 可以更便宜地造，不是 TPU 的检查；正式基线仍是 `rcc-a`（案例 01「便宜的等价修法 `rcb`」行） | pass 8 |
| [`phase2-st1-gradfp32-xnoise`](alignments/01-v5p-acceptance/results/phase2-st1-gradfp32-xnoise.report.md) | 三个梯度 FP32 开关全开时的并集旧结果，同上；现行 `phase2-st1-gradfp32-gpuonly` | pass 4、fail 2、inconclusive 6 |
| [`phase2-st1-on-state-xnoise`](alignments/01-v5p-acceptance/results/phase2-st1-on-state-xnoise.report.md) | v5p 第 1 步整步按两侧执行噪声并集判的旧结果（2026-11-08 撤回：尺子只来自 GPU）；现行 `phase2-st1-on-state-gpuonly`，留作判法变更记录 | pass 4、fail 3、inconclusive 5 |
| [`phase2-st1-prod-state`](alignments/01-v5p-acceptance/results/phase2-st1-prod-state.report.md) | v5p 第 1 步整步更早的判法（GPU 3 对，被测是改配置前的 TPU 运行）；现行判法 `phase2-st1-on-state-gpuonly` | pass 4、fail 7、inconclusive 1 |
| [`phase2-v7x-acceptance-rcb`](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcb.report.md) | v7x 对 `rcb-a` 的交叉核对，只报：正式第一层仍对 `rcc-a`（`phase2-v7x-acceptance-rcc`）；grad_norm 的 MARE 1.09 倍待定，对 `rcc` 单份当 golden 时也在 0.70 – 1.11 之间起伏 | pass 5、inconclusive 1 |
| [`phase2-v7x-l1-ops-own`](alignments/02-v7x-acceptance/results/phase2-v7x-l1-ops-own.report.md) | v7x 单算子，打开 `dense_fp32_accumulation` 之前的配置。被 `phase2-v7x-l1-ops-own-on` 取代 | pass 7、fail 2 |
| [`phase2-v7x-l2-replay-bwd-on`](alignments/02-v7x-acceptance/results/phase2-v7x-l2-replay-bwd-on.report.md) | v7x 逐层反向的旧判法，同上。改按 `phase2-v7x-l2-replay-bwd-fp64-on` 判 | pass 1、fail 3 |
| [`phase2-v7x-l2-replay-bwd-on-xnoise`](alignments/02-v7x-acceptance/results/phase2-v7x-l2-replay-bwd-on-xnoise.report.md) | v7x 逐层反向按两侧执行噪声并集判的对照，同上，只报 | pass 1、fail 3 |
| [`phase2-v7x-l2-replay-rratio`](alignments/02-v7x-acceptance/results/phase2-v7x-l2-replay-rratio.report.md) | v7x 逐层前向，打开 `dense_fp32_accumulation` 之前。被 `phase2-v7x-l2-replay-rratio-on` 取代 | pass 6、fail 4 |
| [`phase2-v7x-st1-on-state-xnoise`](alignments/02-v7x-acceptance/results/phase2-v7x-st1-on-state-xnoise.report.md) | v7x 第 1 步整步的并集旧结果，同上；现行 `phase2-v7x-st1-on-state-gpuonly` | pass 4、fail 2、inconclusive 6 |
| [`phase2-v7x-st1-prod-state`](alignments/02-v7x-acceptance/results/phase2-v7x-st1-prod-state.report.md) | v7x 第 1 步整步更早的判法，同上；现行判法 `phase2-v7x-st1-on-state-gpuonly` | pass 3、fail 8、inconclusive 1 |
