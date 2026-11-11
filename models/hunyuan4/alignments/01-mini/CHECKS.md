[janus](../../../../README.md) › [混元 4](../../README.md) › [mini](README.md) ｜ 检查清单

# 混元 4 · mini（hy4-mini）· 检查清单

这一页回答：hy4-mini 的对齐比了哪些量，每一项的对象是哪些模块和张量，用什么指标、什么门槛判，现行配置下各是什么结果；模型的每个算子由哪条检查覆盖、哪些还没覆盖。

- 页面由 `models/hunyuan4/dashboard/build.py` 按 [`dashboard/checks.yaml`](dashboard/checks.yaml) 从结果文件生成，数字和判定都不手写；没有写成 janus 结果文件的原始 JSON（单算子回放、第 1 步诊断等）由 `build.py` 按写明出处的规则现读。
- **判定**：GPU（Megatron-Core，B200 × 8）是 golden，TPU（MaxText，v5p 8 个设备）是被测；R = rel_L2(TPU, CPU FP64) ÷ rel_L2(GPU, CPU FP64)。**语义**以 MaxText 的混元 4 实现为准（[`PROTOCOL.md`](../../PROTOCOL.md) 第 1.2 节）。
- 三列：**mini** 是现行比较（L4 是现行对齐配置（2026-11-11 起 C 口径：M36、M52、M58 都开，base2 关）的 TPU `tc-a`（同配置 5 份的第一份）1000 步对 GPU `g3-a`；L1–L3 是各项最新的回放，配置写在各行：L1 单算子回放（低阈值 QK-Clip 逐头最大 logit 除外）、L2 逐层、路由一致率第 8 / 30 / 100 步是 base2 开时测的，关 base2 后没有重测，见「已知的覆盖缺口」；TPU 抓数不改数值在两种配置上都有证据；L3 第 1 步梯度与优化器一步用 C 口径之前的 `tz-off-rep` 第 1 步全量抓数（C 口径改了第 1 步梯度，没重判））；**w300** 是同一运行的前 300 步对 GPU `g2-a`；**真实权重** 是 AdamW 时期截层真实权重的 L0 转换与第 0 层单算子，不依赖优化器，仍有效。历代与诊断代见[历代比较](../../dashboard/history.md)；运行代号见[对齐细节第 1 节「代号」](DETAILS.md#代号)。
- 结论与偏差台账在[对齐报告](ALIGNMENT-REPORT.md)；两边配置在 [`config-alignment.md`](config-alignment.md)；每一份运行的来历在 [`goldens.md`](goldens.md)。 每一级为什么这样比见[用户指南第 4 章](../../../../docs/guide/04-level-by-level.md)。

## 规模

- **检查项**：表里的一行，回答一个问题；**检查条数**：结果文件里的检查，一个检查 id 算一条（mini / w300 / 真实权重）。三个「判定」栏数的是检查项（行），不是检查条数；所以 L4 mini 栏的「✅ 8 🟡 3」（11 行）和 `t1zo-l4` 报告的「过 7、待定 2」（只含 1000 步那 9 条检查）口径不同。
- **张量级比较**：这些检查逐个比了多少个张量；**其他比较单元**：数据指纹、dtype 条目、逐步曲线的步数等。写「待结果」「未开始」「进行中」的格子不进计数。 一个格子可以含几条检查，所以比较单元的合计不等于格子数。

| 级 | 目的 | 检查项 | 检查条数 | 张量级比较 | 其他比较单元（mini） | mini 判定 | w300 判定 | 真实权重 判定 |
|---|---|---|---|---|---|---|---|---|
| [L0](#l0--地基设置数据权重配置) | 两边从同一套设置、同一份输入、同一份权重出发 | 9 | 14 / — / 22 | 4,071 / — / 1,280 | 65,000 个数据单元、424 项 dtype、1,000 步 | ✅ 7；只报 2 | — | ✅ 4；只报 2 |
| [L1](#l1--单算子) | 每个算子单独看，TPU 不比 GPU 离 FP64 真值远 | 35 | 42 / — / 10 | 1,891 / — / 10 | 30 步 | ✅ 19 ❌ 2；只报 7 | — | ✅ 6 |
| [L2](#l2--逐层) | 每层输入取参照的，单层前向、反向不比参照离 FP64 远 | 2 | 4 / — / — | 16 / — / — | — | ✅ 1；只报 1 | — | — |
| [L3](#l3--整网第-1-步) | 第 1 步的梯度与状态在 GPU 第 1 步参照的噪声内，超出的对参照定责（参照是 CPU FP64 整网重算，与 TPU 不同源） | 6 | 40 / — / — | 1,369 / — / — | — | ✅ 2 🟡 2；只报 2 | — | — |
| [L4](#l4--多步训练) | 300 步 / 1000 步的逐步曲线在 GPU 重跑的噪声内 | 13 | 20 / 10 / — | — | 139 个字段、10,800 步 | ✅ 11 🟡 2 | ✅ 6 🟡 2 | — |
| 合计 |  | 65 | 120 / 10 / 32 | 7,347 / 0 / 1,290 |  |  |  |  |

## 怎么读

### 一个数是怎么聚合出来的

表里每个数属于下面三种之一，每一行「对象」格最后一句写明是哪一种：

| 种类 | 怎么算 | 例 |
|---|---|---|
| 逐步标量 | 每步一个数，算被测对参照的逐步相对差，再在窗口内算带符号均值、MARE（平均绝对相对差）、最大单步三个判定量，各自对噪声上限 N 判，最差的决定结果 | L4 的 loss（全局批的平均 loss）、grad_norm（裁剪前全局梯度范数）、param_norm（更新后参数范数；现行比较两边比精确值） |
| 逐张量取最差 | 每个张量（或算子输出）拉直成一个向量，单独算一个数（如 R 比 = rel_L2(TPU, 参照) ÷ rel_L2(GPU, 参照)）；一条检查选中一批张量时取最差的那个，不取平均 | L1 单算子、L3 索引器参数梯度 |
| 诊断 | 只报不判，判定写成 🟡；说明里写实测 | 第 1 步逐张量梯度对 GPU 三份散布的比值、第 1 步 grad_norm、第 0 层 KD 的值 |

### 三种严格门槛

| 门槛 | 用在哪 | 怎么判 |
|---|---|---|
| 逐位 | L0 设置、数据、权重、配置；TPU 同设置重跑 | 逐位相同才过（设置核对是两边每一步都等于登记值） |
| R 比 ≤ 1.2 | L1 单算子；L3 第 1 步超出噪声的项对 CPU FP64 定责 | R = rel_L2(TPU, 参照) ÷ rel_L2(GPU, 参照)（GPU 是参照答案，TPU 是被测）。R ≤ 1.2 即 TPU 不比 GPU 离精确算术远 |
| 噪声上限 N | L4 的逐步曲线；L3 第 1 步梯度与状态 | N 只用 GPU。L4：同一代 GPU 重跑两两配对，每一对算一次判定量，每个判定量取各对的最大值；份数按窗口：前 300 步 8 份 28 对（`g1-*` / `g2-*`），1000 步现行 16 份 120 对（`g3-a … g3-p`，2026-11-11 起；之前 8 份 28 对）；L3：第 1 步参照两两，梯度现行 6 份 15 对（`g2s-a … g2s-f`，2026-11-11 起），优化器一步与更早各代 3 份 3 对（`g1s-*` / `g2s-a/b/c`）。r = 被测对基线的判定量 ÷ N：≤ 1 过，> 2 不过，之间待定。TPU 只改执行参数的扰动运行只作诊断 |

定义与依据见 [`docs/reference/judging.md`](../../../../docs/reference/judging.md) 与 [`PROTOCOL.md`](../../PROTOCOL.md) 第 3–5 节。

### 表里各栏

- **证据来源**：「生产程序」是正式训练运行的逐步记录；「单算子回放」是单独编译、把 GPU 抓到的输入喂给 TPU 的程序；「第 1 步诊断」是训练程序的诊断模式（同一个梯度累积函数算第 1 步、不更新）；「离线计算」在 CPU 上算。
- **判法**：「｜」前是严格档，决定验收；「参考」后是同类 GPU→TPU 迁移项目用过的经验门槛，只并列报告。
- **平台格子**：实测值，括号里是离严格门槛多远，然后是判定。点格子打开那份结果。
- **符号**：✅ 过　🟡 待定、诊断或进行中　❌ 不过　⬜ 未开始　— 不适用；参考档：参✅ 过　参☑️ 记录了的都过、有门槛项没记录　参❌ 不过　参❔ 所需指标未记录。
- 每张表下面写非 ✅ 格子的原因；已定责的写明谁接受、证据在哪。
- 检查名后标「（只报）」的行不在验收路径上（旧协议的结果、诊断、默认实现的对照、只统计的量）：格子照实写结果文件里的判定，但不进判定计数。
- 张量名用 HF 检查点的名字（`model.layers.{0..6}…`、`model.mtp_layers.0…`）。hy4-mini 第 0 层是稠密层，第 1–6 层是 MoE 层（16 个路由专家 top-4 + 1 个共享专家）；索引器第 0、1、5 层与 MTP 层自己算 top-k，第 2–4、6 层复用（IndexCache）。

## L0 · 地基：设置、数据、权重、配置

目的：证明两边从同一套设置、同一份输入、同一份权重出发。这一级不过，上面各级的差都说不清来源。

L0 共 9 项检查，检查条数 14 / 22（mini / 真实权重），张量级比较 4,071 / 1,280 次（同上列序）。全部要求逐位相同。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | mini | 真实权重 |
|---|---|---|---|---|---|---|
| 设置核对：逐步学习率<br>`L0.settings.lr` | 两边每一步记下的学习率都等于登记的常数 1e-5。学习率是设置的常数，不是观测量，所以归设置核对、不进 L4（`PROTOCOL.md` 变更记录 2026-11-07「报告口径」一行） | `lr`，TPU 被测与 GPU 基线各自第 1–1000 步。被测是现行比较 `tc-a`（C 口径，base2 关）<br>比较单元：1,000 步<br>聚合：逐步比，任一步超出容差就不过 | rtol ≤ 0、abs_ulp ≤ 0｜参考：无 | 生产程序（逐步记录） | [两边每一步都等于登记值 1e-05 ✅](results/tc-l0-settings.report.md) | — |
| 检查点张量名、形状、dtype（只报）<br>`L0.ckpt.names_shapes` | HF 目录里每个张量的名字、形状、dtype 逐个等于形状表与 dtype 表，参数量对上 | mini：hy4-mini 的随机初始化 HF 目录（`make_tiny_hf.py --config mini`，清单 sha256 登记在 `PROTOCOL.md` 第 1.1 节）；真实权重：hy4-preview 截层的 HF 目录（135 个张量）<br>聚合：逐条规则核，数不满足的条数 | 只报，不判｜参考：无 | 离线核对（CPU，检查点文件） | [HF 张量名、形状、dtype 逐个等于形状表与 hy4_map.hf_dtype ✅](results/hy4-mini-l0-names.results.json) | [HF 张量名、形状、dtype 逐个等于形状表与 hy4_map.hf_dtype ✅](results/hy4-l0-names.results.json) |
| 权重转换往返<br>`L0.ckpt.convert.bitwise`<br>`L0.ckpt.convert.dtype`<br>`L0.ckpt.source.dtype` | HF 权重转成 MaxText 检查点后，逐张量还原回 HF 对原件逐位，加转换前后两份 dtype 清单 | mini：hy4-mini 的 HF 目录（212 个张量）；真实权重阶段 A / B / C / D 各一份<br>比较单元：mini 212 个参数张量、424 项 dtype；真实权重 262 个参数张量、524 项 dtype<br>聚合：逐张量比 sha256，数不同的个数；逐项核 dtype，数不合规的个数 | 逐位｜参考：无 | 离线核对（CPU，检查点文件） | [0 / 212 不同 ✅](results/hy4-mini-l0-ckpt-convert.report.md) | [0 / 31 不同 ✅](results/hy4-a-l0-ckpt-convert.report.md) |
| RoPE 列去交错（只报）<br>`L0.ckpt.rope_deinterleave.bitwise` | 检查点的 q_pe / k_pe 交错存放，转换时去交错（偶数位接奇数位），逆置换逐位回到原件 | mini：hy4-mini（32 个张量）；真实权重阶段 A / B / C / D<br>比较单元：mini 32 个张量；真实权重 50 个张量<br>聚合：逐条规则核，数不满足的条数 | 只报，不判｜参考：无 | 离线核对（CPU，检查点文件） | [32/32 个带 RoPE 列的张量：正向 = 偶数位接奇数位、逆置换逐位回到原件 ✅](results/hy4-mini-l0-rope.results.json) | [6/6 个带 RoPE 列的张量：正向 = 偶数位接奇数位、逆置换逐位回到原件 ✅](results/hy4-a-l0-rope.results.json) |
| 负对照：不去交错必须判不一致<br>`L0.neg.no_deinterleave.roundtrip`<br>`L0.neg.no_deinterleave.rope`<br>`L0.neg.no_deinterleave.init_vs_gpu`<br>`L0.ckpt.rope.negative_control` | 不去交错的检查点按去交错的读法核，必须判不一致（证明上一行的核对能抓到错） | mini：hy4-mini 212 个张量不去交错，往返、RoPE 核对、对 GPU 初始参数清单三处都必须报不一致；真实权重：hy4-tiny CPU 冒烟（阶段 A 结构）<br>比较单元：mini 212 个张量；真实权重 1 条规则<br>聚合：逐用例看埋的错有没有被抓到；逐条规则核，数不满足的条数 | 逐位｜参考：无 | 离线核对（不去交错的负对照）<br>离线核对（CPU 冒烟，hy4-tiny） | [32 / 212 不同 ✅](results/hy4-mini-l0-negative-control.results.json) | [不去交错的检查点按去交错的读法核，判不一致（期望如此） ✅](../00-tiny/results/hy4-cpu-smoke.json) |
| 跨侧初始权重<br>`L0.init.tpu_loaded_eq_ckpt`<br>`L0.init.tpu_vs_gpu_mcore`<br>`L0.init.tpu_vs_gpu.ga-b`<br>`L0.init.tpu_vs_gpu.ga1-a`<br>… | TPU 载入后的初始权重对检查点、对 GPU 初始参数清单（还原成 GPU 名字、FP32、按 GPU 的分片分 rank）逐位 | mini：两条检查，TPU 载入后清单对检查点复算 0 / 1,647 不同（8 份 × 参数 + 路由偏置），TPU 对 GPU 初始参数清单 0 / 1,968 不同（8 rank × 246）；真实权重：阶段 A（对 `ga-b`、`ga1-a`，248 个）、阶段 B（对 `gb-a`，472 个）<br>比较单元：mini 3,615 个参数张量；真实权重 968 个参数张量<br>聚合：逐张量比 sha256，数不同的个数 | 逐位｜参考：无 | 离线核对（TPU 载入后清单对检查点、对 GPU 初始参数清单）<br>离线核对（CPU，检查点文件） | [0 / 3,615 不同 ✅](results/t1-l0-init-vs-gpu-mcore.results.json) | [0 / 968 不同 ✅](results/hy4-a-l0-init-vs-gpu.report.md) |
| 每步输入指纹<br>`L0.data.fingerprint` | TPU 回放的每一步、每个 rank、每个 microbatch 的输入，对 GPU 存下的输入 | mini：TPU `t1zg-a` 回放的 tokens、labels、loss_mask 对 GPU `g3-a` 存下的（台账 G9）；真实权重：`ga-a` 300 步 × 8 rank × 2 microbatch<br>比较单元：mini 65,000 个数据单元；真实权重 4,800 个数据单元<br>聚合：逐单元比 sha256，数不同的个数 | 逐位｜参考：无 | 离线核对（回放数据对 GPU 基线 g3-a）<br>离线核对（CPU，回放数据） | [0 / 65,000 不同 ✅](results/t1zg-l0-data.report.md) | [0 / 4,800 不同 ✅](results/hy4-a-replay-check-ga-a.txt) |
| Muon / Adam 分区<br>`L0.optim.partition` | 每个参数走 Muon 还是 Adam、Newton–Schulz 作用在哪个矩阵上，两边逐参数一致 | GPU `g3-a` 的 `optimizer_partition.json`（239 个参数：140 个 Muon / 99 个 Adam，台账 G5）逐个经权重转换映射找到 TPU 上装它的叶子（TPU 表 [`results/hy4-mini-muon-partition.md`](../../results/hy4-mini-muon-partition.md)）<br>聚合：逐字段比，数不同的个数 | 逐位｜参考：无 | 离线核对（GPU g3-a 的 optimizer_partition.json 对 TPU 优化器叶子，经权重转换映射） | [0 / 239 不同 ✅](results/t1-l0-muon-partition.results.json) | — |
| 配置逐项对照<br>`L0.config.items` | 两边训练配置里必须相同的语义设置逐项对照；另核 TPU 实际生效的精度开关等于 [`config-alignment.md`](config-alignment.md) 登记的对齐值（登记值抄在脚本里，改登记要同步改脚本） | 现行比较 `tc-a` 实际生效的 MaxText 配置对 `g3-a` 的 meta 与两边共同的 HF `config.json`：结构 37 项对 HF config、训练设置 20 项对 GPU meta、登记值 12 项、精度开关 39 项（含 `sa_use_base2_exp`、C 口径的 M52、M58）。同一脚本对 base2 开的上一代 `t1zg-a` 恰好报 1 项不同（`sa_use_base2_exp`），证明能抓到<br>聚合：逐字段比，数不同的个数 | 逐位｜参考：无 | 离线核对（tc-a 实际生效的 MaxText 配置对 g3-a meta、HF config.json 与登记的精度开关） | [0 / 108 不同 ✅](results/tc-l0-config.results.json) | — |

## L1 · 单算子

目的：把每个算子单独拿出来，看 TPU 和 GPU 是否离 FP64 真值一样近。整网的差是各算子的差叠起来的，这一级定位到具体算子。

L1 共 35 项检查，检查条数 42 / 10（mini / 真实权重），张量级比较 1,891 / 10 次（同上列序）。

mini 的单算子回放（`t1-l1-op-tpu-r3.json`，被测 `tw-opmini-d`，GPU 分母 Megatron-Core + TE）99 个算子观测点全过（同一种算子在每一层各算一个，加嵌入、输出头、交叉熵等，文件里的 `n_ops_with_R`；按算子种类合成下表的 15 条 `L1.mini3.*` 检查）；它是 base2 开的上一代对齐配置，关 base2 后没有重测，注意力整块（q_a 投影、RoPE、索引器打分与 top-k 等各行）对现行配置是否仍过没有核（PROTOCOL 变更记录 2026-11-11）；真实权重的阶段 A（第 0 层，GPU 分母是 HF 参照模型）只用第 1 步前向、不依赖优化器，仍有效，单列一表。

回放的注入覆盖全部输入（位置与关键字参数，janus `8cacde9` 起）：每个算子吃的都是 GPU 的同一个输入。

### 模型的每个算子由哪条检查覆盖

按前向顺序列出 hy4-mini 的算子，再列损失与优化器。算子名与语义按两边代码核过：MaxText（混元 4 集成分支）、`frameworks/megatron/hy4_mcore.py`（Megatron-Core 0.20）、HF `hy_v4` 参考实现。

- 「判定」列取前三列（有判定的检查）里最差的一条；三列都没有检查的写「尚未覆盖」。
- 「L3 第 1 步」一列是现行配置训练步的第 1 步梯度对 GPU 第 1 步参照 `g2s-*` 三份下这个算子的参数梯度（或流经它的梯度）：超出 GPU 噪声的张量要对 CPU FP64 参照定责（R ≤ 1.2），N = 0 且两边误差都小于 u = 2^-8 的按分辨率下限过；有超 2N 而未定责的判不过、只有 1–2N 未定责的判待定（规则见 `build.py` 的 `_step1_judge`）。
- L1、L3 两列的 R 比格子在判定后并排写「GPU a、TPU b → R」：a、b 是两边对参照的 rel_L2（决定 R 的那个算子或张量），R = b ÷ a。只看 R 看不出两边的误差是否都在一次 BF16 舍入（u ≈ 3.9e-3）以下。
- 「L4」一列是现行比较 `tc-a` 的专项曲线（1000 步）。
- 「说明」一列写更高一级在哪里间接覆盖、为什么还没单独覆盖。只在整网 L4 loss 里间接体现的，记「尚未覆盖」。
- 格子与上一行同一列相同时写「同 #k」（k 是第一次出现的行号）。
- 逐元素函数的 ulp 检查（L1 表「逐元素函数」）不进本表的判定：它不是 TPU 对 GPU 的比较；对用到这些函数的算子，TPU 对 GPU 的比较看 L1 与 L3 两列。

共 53 个算子：有判定的检查覆盖且全过 46 个，覆盖但有待定或不过的 7 个（其中已定责 2 个），尚未覆盖 0 个。

| # | 算子 | 做什么 | L1 单算子（hy4-mini 7 层 + MTP，现行开关） | L3 第 1 步（现行配置训练步梯度，超出 GPU 噪声的对 CPU FP64 定责） | L4 专项曲线（tc-a 1000 步） | 说明 | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | 词嵌入 | token id → 1024 维（FP32 表查出转 BF16） | [`L1.mini3.embed`](results/t1-l1-op-tpu-r3.json) ✅（两边对参照都是 0（逐位）） | [`L3.step1.ref.op.embed`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 4.18e-3、TPU 4.10e-3 → R 0.98） | — | L0 权重转换逐位；L4 loss 间接 | ✅ |
| 2 | iHC pre：归一化因子 | 4 路流拼平后 rsqrt(mean(x²) + eps)，无可学习权重；乘在混合矩阵乘的结果上 | [`L1.mini3.ihc_mix`](results/t1-l1-op-tpu-r3.json) ✅（GPU 1.17e-3、TPU 1.17e-3 → R 1） | — | — | 单算子回放比的是混合后的分支输入（`attn_hc.out` / `ffn_hc.out`），pre 的四步合在一起比 | ✅ |
| 3 | iHC pre：混合矩阵乘（hc_fn） | FP32 拼平的流 × hc_fn（TPU `hc_matmul_precision: highest`） | 同 #2 | [`L3.step1.ref.op.ihc`](results/t1zo-step1-train-grad-vs-fp64-n6.json) 🟡（GPU 6.11e-3、TPU 3.57e-2 → R 5.83） | — | 单算子回放同上。L3：hc_pre 的 hc_fn / hc_base / hc_scale 的梯度同时流经 pre 与 post；以现行比较 tz-off-rep 为准、GPU 参照 6 份 15 对：超 N 且 R > 1.2 计入未定责的 1 个（第 3 层注意力前 `hc_scale`，1.05 N，待定），按 D11 豁免规则单列（豁免规则与沿用待复核）；MTP enorm 另列，同样按 D11 豁免规则单列（豁免规则与沿用待复核；接受理由见对齐细节 D13）；对齐细节第 4 节注 2；GPU 3 份时未定责 5 个（3 个是关 base2 后新出现的），补到 6 份后其余 4 个在噪声内；第 3 层 `hc_scale` 属于 base2 开那一代单列的 6 个（单层回放（单行）6 个里 5 个 R ≤ 1.2，对齐细节 D11） | 🟡 |
| 4 | iHC pre：sigmoid | pre = sigmoid(logit · scale + base) + hc_eps | 同 #2 | 同 #3 | — | 同上 | 🟡 |
| 5 | iHC pre：加权求和 | 4 路流按 pre 加权求和，得到分支输入（TPU 默认用 HIGHEST einsum，HLO 核过编译后是 FP32 逐元素乘 + reduce；开关打开后逐路 FP32 乘加；GPU 是逐元素 FP32 乘加） | 同 #2 | 同 #3 | — | 同上（TPU 逐路 FP32 乘加，台账 M28） | 🟡 |
| 6 | iHC post：2·sigmoid | post = 2.0 · sigmoid(logit · scale + base) + hc_eps（`hc_post_magnitude` 2.0） | [`L1.mini3.ihc_writeback`](results/t1-l1-op-tpu-r3.json) ✅（GPU 1.65e-3、TPU 1.65e-3 → R 1） | 同 #3 | — | MaxText 不把 post 单独输出，单算子回放只在 hy4-mini 的 iHC 写回里间接比（post 不单独比）；L3 是 hc_pre 参数的梯度，同时含 pre 与 post 两条路 | 🟡 |
| 7 | iHC post：写回 | x_n + post_n · 分支输出（FP32 里算，转回流的精度）；iHC 没有 res_alpha，残差是恒等 | 同 #6 | — | — | 同上；L4 loss 间接 | ✅ |
| 8 | 输入 RMSNorm | 注意力前的 RMSNorm（TPU 单次舍入，`rmsnorm_single_rounding`） | [`L1.mini3.norm`](results/t1-l1-op-tpu-r3.json) ✅（GPU 2.35e-3、TPU 2.35e-3 → R 1） | [`L3.step1.ref.op.input_norm`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 4.03e-3、TPU 4.16e-3 → R 1.03） | — | — | ✅ |
| 9 | MLA q_a 投影 | x → q 潜空间（q_lora 512） | [`L1.mini3.attn`](results/t1-l1-op-tpu-r3.json) ✅（GPU 3.36e-3、TPU 3.09e-3 → R 0.919） | [`L3.step1.ref.op.q_a`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 1.06e-2、TPU 1.11e-2 → R 1.04） | — | 只在注意力整块里比 | ✅ |
| 10 | MLA q_norm | q 潜空间的 RMSNorm（eps 1e-5） | 同 #9 | [`L3.step1.ref.op.q_norm`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 1.11e-2、TPU 1.20e-2 → R 1.08） | — | 同上 | ✅ |
| 11 | MLA q_b 投影 | 潜空间 → 8 头 × (nope 192 + rope 64) | 同 #9 | [`L3.step1.ref.op.q_b`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 1.08e-2、TPU 1.13e-2 → R 1.04） | — | 同 #10 | ✅ |
| 12 | MLA kv_a 投影 | x → kv 潜空间（kv_lora 512）+ 共享的 k_pe（64） | 同 #9 | [`L3.step1.ref.op.kv_a`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 4.11e-3、TPU 3.97e-3 → R 0.966） | — | 同 #10 | ✅ |
| 13 | MLA kv_norm | kv 潜空间的 RMSNorm（eps 1e-5） | 同 #9 | [`L3.step1.ref.op.kv_norm`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 4.12e-3、TPU 4.24e-3 → R 1.03） | — | 同 #10 | ✅ |
| 14 | MLA kv_b 投影 | 潜空间 → 8 头 × (k_nope 192 + v 256) | 同 #9 | [`L3.step1.ref.op.kv_b`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 4.21e-3、TPU 4.10e-3 → R 0.974） | — | 同 #10 | ✅ |
| 15 | MLA RoPE | θ = 1e7，YaRN factor 1（= 普通 RoPE）。检查点交错存放：TPU 转换时去交错、跑拼接式；GPU 运行时去交错（`mla_rotary_interleaved`） | 同 #9 | — | — | L0 去交错逐位；只在注意力整块里比 | ✅ |
| 16 | 索引器 wq_b | q 潜空间 → 32 头 × 128 | 同 #9 | [`L3.step1.ref.op.indexer_wq_b`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 7.38e-3、TPU 7.35e-3 → R 0.996） | — | 前向只在注意力整块里比；第 1 步梯度含第 0、1、5 层与 MTP 层 | ✅ |
| 17 | 索引器 wk | x → k（128） | 同 #9 | [`L3.step1.ref.op.indexer_wk`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 6.88e-3、TPU 6.86e-3 → R 0.996） | — | 同上 | ✅ |
| 18 | 索引器 k_norm | k 的 LayerNorm（带 bias，不是 RMSNorm；eps 1e-6），两边都在 FP32 里算 | 同 #9 | [`L3.step1.ref.op.indexer_k_norm`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 1.86e-3、TPU 1.96e-3 → R 1.05） | — | 同 #17 | ✅ |
| 19 | 索引器 RoPE（每头后 64 维） | RoPE 放在每头的后 qk_rope_head_dim 维（`indexer_rope_position: last`；Megatron-Core 原版放前面，GPU 侧改了） | 同 #9 | [`L3.step1.ref.op.indexer_wq_b`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 7.38e-3、TPU 7.35e-3 → R 0.996）<br>[`L3.step1.ref.op.indexer_wk`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 6.88e-3、TPU 6.86e-3 → R 0.996） | — | 同上；L3 是流经 RoPE 的 wq_b、wk 的梯度 | ✅ |
| 20 | 索引器 weights_proj | x → 每头一个权重，乘 H^-0.5 · d^-0.5（BF16 权重、FP32 输出，台账 M27） | 同 #9 | [`L3.step1.ref.op.indexer_weights_proj`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 5.43e-3、TPU 5.22e-3 → R 0.962） | — | 同上 | ✅ |
| 21 | 索引器 ReLU 与按头加权求和 | 分数 = Σ_头 w_h · ReLU(q_h · k)（TPU `indexer_fp32_scores` 真 FP32） | [`L1.indexer_kd.inputs`](results/t1-layer0-kd-attribution.json) ✅（GPU 5.27e-3、TPU 5.17e-3 → R 0.98）<br>[`L1.mini3.attn`](results/t1-l1-op-tpu-r3.json) ✅（GPU 3.36e-3、TPU 3.09e-3 → R 0.919） | 同 #20 | — | KD 输入一条比的就是这个分数（第 0 层第 0 行）；L3 是流经它的 weights_proj 梯度 | ✅ |
| 22 | 索引器 top-k（full 层） | 每个 query 选 2048 个 key（第 0、1、5 层与 MTP 层） | [`L1.mini3.attn`](results/t1-l1-op-tpu-r3.json) ✅（GPU 3.36e-3、TPU 3.09e-3 → R 0.919） | — | — | 只在注意力整块里间接比；逐层不重合率统计见 L2 | ✅ |
| 23 | 索引器 top-k（shared 层复用） | 第 2–4、6 层复用本组第一层的 top-k（IndexCache） | 同 #22 | — | — | hy4-mini 逐算子回放里只在注意力整块里间接比（shared 层用块首层自选的掩码） | ✅ |
| 24 | 注意力 logits | q · k × softmax scale（被选中的 2048 个 key + 因果） | 同 #22 | — | — | 只在注意力整块里比；logits 的精度两边不同，见下方「已知的覆盖缺口」 | ✅ |
| 25 | 可学习 sink | 每头一个 FP32 logit，作为额外的一列键拼进 softmax，不带 value | 同 #22 | [`L3.step1.ref.op.sink`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 1.62e-2、TPU 1.55e-2 → R 0.957） | — | 同上 | ✅ |
| 26 | softmax | FP32（TPU 是 splash 的在线 softmax；现行对齐配置 base2 关，用 exp，2026-11-11 前各代用 exp2） | 同 #22 | — | — | 同 #25 | ✅ |
| 27 | 乘 V（PV） | 概率 × V | 同 #22 | — | — | 同 #25 | ✅ |
| 28 | 输出门投影 | x → 8 头 × 256（门控 MLA，读注意力的输入） | [`L1.mini3.gate_proj`](results/t1-l1-op-tpu-r3.json) ✅（GPU 1.66e-3、TPU 1.66e-3 → R 1） | [`L3.step1.ref.op.gate`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 5.86e-3、TPU 5.78e-3 → R 0.987） | — | 单算子回放比的是投影的输出（sigmoid 之前） | ✅ |
| 29 | 输出门 sigmoid 与相乘 | out × sigmoid(gate) | [`L1.mini3.attn`](results/t1-l1-op-tpu-r3.json) ✅（GPU 3.36e-3、TPU 3.09e-3 → R 0.919） | 同 #28 | — | 只在注意力整块里比；L3 是流经 sigmoid 的 linear_gate 梯度；sigmoid 的精度两边不同（下方逐元素函数表，对齐细节 D10） | ✅ |
| 30 | o_proj | 8 头 × 256 → 1024 | 同 #29 | [`L3.step1.ref.op.o_proj`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 4.13e-3、TPU 4.01e-3 → R 0.971） | — | 同上 | ✅ |
| 31 | FFN 前 RMSNorm | 注意力写回之后、FFN 之前的 RMSNorm | [`L1.mini3.norm`](results/t1-l1-op-tpu-r3.json) ✅（GPU 2.35e-3、TPU 2.35e-3 → R 1） | [`L3.step1.ref.op.post_norm`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 1.32e-2、TPU 1.37e-2 → R 1.04） | — | — | ✅ |
| 32 | 稠密 MLP（第 0 层） | silu(gate) · up，再 down（3072 宽，不截断） | [`L1.mini3.dense_mlp`](results/t1-l1-op-tpu-r3.json) ✅（GPU 3.73e-3、TPU 3.31e-3 → R 0.888） | [`L3.step1.ref.op.dense_mlp`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 5.30e-3、TPU 5.11e-3 → R 0.963） | — | — | ✅ |
| 33 | 路由 logits（FP32） | FP32 激活 × 路由权重（TPU 先把权重舍成 BF16，`routed_gate_bf16_rounding`；`router_matmul_precision: highest`） | [`L1.mini3.router_logits`](results/t1-l1-op-tpu-r3.json) ✅（GPU 1.13e-6、TPU 1.20e-7 → R 0.106） | [`L3.step1.ref.op.router`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 2.86e-2、TPU 3.19e-2 → R 1.12） | — | 阶段 A 没有 MoE；hy4-mini 单算子回放比了路由 logits；L3 是路由权重（`mlp.gate.weight`）的梯度 | ✅ |
| 34 | 路由 sigmoid 与偏置 | sigmoid 打分；路由偏置只参与选择，门控权重用不带偏置的分数 | [`L1.mini3.moe`](results/t1-l1-op-tpu-r3.json) ✅（GPU 4.93e-3、TPU 4.65e-3 → R 0.943） | — | [`L4.router.bias_mean、L4.router.bias_std`](results/t1zc-l4.report.md) 🟡 | 偏置的逐步更新由 L4 路由偏置曲线比；现行比较两个窗口都过（base2 开的上一代标准差略超、已接受，对齐细节 D23） | 🟡 已定责 |
| 35 | 路由 top-k | 16 个专家选 4 个（不分组） | 同 #34 | — | 同 #34 | 偏置按负载更新，负载由 top-k 决定，所以偏置曲线间接覆盖选择；逐层翻转统计见 L2；现行比较偏置均值、标准差都过（对齐细节 D23 是 base2 开的上一代） | 🟡 已定责 |
| 36 | 路由权重归一化与缩放 | w / (Σw + 1e-20) × 2.827 | 同 #34 | [`L3.step1.ref.op.router`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 2.86e-2、TPU 3.19e-2 → R 1.12） | — | 路由权重的梯度流经归一化与缩放（L3 同上一行）；前向没有单独比 | ✅ |
| 37 | 路由专家（含 SwiGLU 截断） | gate ≤ 10、up ∈ [−10, 10] 截断后 silu(gate) · up，再 down；路由权重 TPU 乘在 down 之后、GPU 乘在 down 之前 | [`L1.mini3.routed_experts`](results/t1-l1-op-tpu-r3.json) ✅（GPU 5.37e-3、TPU 4.66e-3 → R 0.869） | [`L3.step1.ref.op.experts`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 1.96e-2、TPU 1.98e-2 → R 1.01） | — | L3 只含 rank 0 本地的 0–3 号专家（GPU 抓数只有这 4 个，台账 G12）；路由权重乘的位置两边不同（舍入点不同，M40 对上） | ✅ |
| 38 | 共享专家 | silu(gate) · up，再 down（256 宽，不截断） | [`L1.mini3.shared_expert`](results/t1-l1-op-tpu-r3.json) ✅（GPU 3.70e-3、TPU 3.31e-3 → R 0.895） | [`L3.step1.ref.op.shared`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 5.18e-3、TPU 5.22e-3 → R 1.01） | — | — | ✅ |
| 39 | hc_head | 4 路流合成 1 路（归一化因子、hc_fn、sigmoid 加权） | [`L1.mini3.hc_head`](results/t1-l1-op-tpu-r3.json) ✅（GPU 1.56e-3、TPU 1.56e-3 → R 1） | [`L3.step1.ref.op.hc_head`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 1.39e-2、TPU 5.17e-2 → R 3.72） | — | TPU 先归一化再乘、GPU 先乘再乘因子：数学相同、舍入不同；L3 的 R 3.9 不定责：TPU 对 GPU 在 GPU 6 份的噪声内（r 0.32；3 份时 0.72），按噪声过。但这把尺子在这里被撑宽：第 1 步按主 loss / MTP 拆开，hc_head 三个参数加最后的 norm 这组（4 个张量）的梯度误差里，GPU 三份是 MTP 那部分的约 1/4 到 6/10 被丢掉（组中位数 β −0.26、−0.35、−0.61），TPU 是多了约 3/10（+0.31），GPU 三份之间的散布因此变大，「按噪声过」不能说明 R 3.9 是噪声（`results/mini-step1-grad-main-mtp-split.json`，[知识库](../../knowledge.md#mini-主-loss-共同偏高要点与出处)）。注意 TPU 的 β 取自诊断程序（`train_hy4.py --diag-step1`）的梯度，而这一行的 R 3.9 用的是训练步动量反推的梯度，两者分开编译；拿这个 β 解释 R 3.9 还缺一步，要用训练步梯度重算。GPU 为什么丢掉这部分没查（待查） | ✅ |
| 40 | 最后的 RMSNorm | hc_head 之后的 RMSNorm | [`L1.mini3.norm`](results/t1-l1-op-tpu-r3.json) ✅（GPU 2.35e-3、TPU 2.35e-3 → R 1） | [`L3.step1.ref.op.final_norm`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 3.76e-3、TPU 3.67e-3 → R 0.976） | — | — | ✅ |
| 41 | 输出头（FP32） | 1024 → 120832 维 logits，FP32 权重与输出 | [`L1.mini3.lm_head`](results/t1-l1-op-tpu-r3.json) ✅（GPU 2.56e-7、TPU 8.32e-8 → R 0.325） | [`L3.step1.ref.op.lm_head`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 3.72e-3、TPU 3.72e-3 → R 0.998） | — | 现行配置设 `logits_matmul_precision: highest`，TPU 也是真 FP32 矩阵乘（台账 M13）；GPU 的 dgrad 每 1024 个词分段累加（台账 G19） | ✅ |
| 42 | MTP enorm、hnorm | 下一个 token 的嵌入与主干隐状态（hc_head + 最后的 norm 之后）各一个 RMSNorm | [`L1.mini3.norm`](results/t1-l1-op-tpu-r3.json) ✅（GPU 2.35e-3、TPU 2.35e-3 → R 1） | [`L3.step1.ref.op.mtp_norms`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 3.87e-3、TPU 3.99e-3 → R 1.03） | — | MTP loss 曲线间接；L3 现行训练步梯度（tz-off-rep）下 `enorm` R 1.24（GPU 1.57e-3，TPU 1.94e-3，超 N 1.44 倍），两边都小于 u = 2^-8，另列、不进未定责计数，按 D11 豁免规则单列（豁免规则与沿用待复核；接受理由见对齐细节 D13）；base2 开的上一代 R 1.18 | ✅ |
| 43 | MTP 拼接与 eh_proj | [嵌入, 隐状态] 拼成 2048 维，投影回 1024 | [`L1.mini3.eh_proj`](results/t1-l1-op-tpu-r3.json) ✅（GPU 1.66e-3、TPU 1.66e-3 → R 1） | [`L3.step1.ref.op.mtp_eh_proj`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 3.71e-3、TPU 3.65e-3 → R 0.984） | — | 同上 | ✅ |
| 44 | MTP 的 transformer 层（注意力、MoE） | 一层 MoE 层（不带超连接；索引器自己算 top-k） | [`L1.mini3.attn`](results/t1-l1-op-tpu-r3.json) ✅（GPU 3.36e-3、TPU 3.09e-3 → R 0.919）<br>[`L1.mini3.moe`](results/t1-l1-op-tpu-r3.json) ✅（GPU 4.93e-3、TPU 4.65e-3 → R 0.943） | [`L3.step1.ref.op.mtp_layer`](results/t1zo-step1-train-grad-vs-fp64-n6.json) ✅（GPU 5.01e-2、TPU 5.50e-2 → R 1.1） | — | 其中的索引器算在上面索引器各行（L3 索引器各组含 MTP 层的张量）；前向在单算子回放里比了注意力整块与 MoE 整块 | ✅ |
| 45 | 主损失（交叉熵） | 逐 token 交叉熵，eod 位置不算（`loss_mask`），按全局有效 token 数平均 | [`L1.mini3.cross_entropy`](results/t1-l1-op-tpu-r3.json) ✅（GPU 3.33e-8、TPU 3.82e-8 → R 1.15） | — | [`L4.loss`](results/t1zc-l4.report.md) ✅<br>[`L4.loss.tail`](results/t1zc-l4.report.md) ✅ | — | ✅ |
| 46 | MTP 损失 | 预测下下个 token 的交叉熵 × 0.1（两边都读 MaxText 口径 `mtp_loss_mt`） | — | — | [`L4.mtp_loss、L4.mtp_loss.tail`](results/t1zc-l4.report.md) ✅ | — | ✅ |
| 47 | 索引器 KD 损失 | KL(教师注意力分布 ‖ 索引器分布)，各产出层平均 × 0.1 | [`L1.indexer_kd.inputs`](results/t1-layer0-kd-attribution.json) ✅（GPU 5.27e-3、TPU 5.17e-3 → R 0.98） | — | [`L4.indexer_loss`](results/t1zc-l4.report.md) ✅ | 1000 步与前 300 步都过（打开 M50 后，对齐细节 D1）；第 0 层 KD 值的诊断见 L1 表 | ✅ |
| 48 | Muon：Newton–Schulz | 5 步，系数 (3.4445, −4.7750, 2.0315)；归一化 x / max(‖x‖, 1e-7) | [`L1.optim_adam_r2.tpu_adam_acc`](results/t1-optim-replay-adam-acc-r2-tpu.json) ✅（GPU 1.36e-2、TPU 1.41e-2 → R 1.03） | — | — | L1 是优化器连续 8 步回放（对齐细节 D17）；另有 MaxText 单测对 Megatron-Core 的固定参照（台账 M15–M17） | ✅ |
| 49 | Muon：Nesterov | β = 0.95，方向 = β · mu_t + (1 − β) · g_t，不做偏差修正 | 同 #48 | — | — | 同上 | ✅ |
| 50 | Muon：更新缩放 | √max(1, fan_out / fan_in) | 同 #48 | — | — | 同上；两边公式相同（GPU `muon_scale_mode: shape_scaling`，台账 G18） | ✅ |
| 51 | Muon：分区 | 哪些参数走 Muon（iHC 矩阵走 Adam、gate / up 拼接正交化、输出门每头一个矩阵） | [`L0.optim.partition`](results/t1-l0-muon-partition.results.json) ✅ | — | — | L0 逐参数对拍 | ✅ |
| 52 | 标量参数的 Adam | 普通 Adam（不是 NAdam），β 0.9 / 0.95，ε 1e-10，权重衰减 0 | [`L1.optim_adam_r2.tpu_adam_acc`](results/t1-optim-replay-adam-acc-r2-tpu.json) ✅（GPU 1.36e-2、TPU 1.41e-2 → R 1.03） | — | — | 同上；另有 MaxText 单测对 torch Adam（台账 M16） | ✅ |
| 53 | QK-Clip | 每层每头 scale = min(1, 100 / 最大 logit)（最大 logit 只取选中且因果的位置，不含 sink），q_b / kv_b 的 nope 列 × √scale、rope 列 × scale；梯度累积时最大 logit 取最大 | [`L1.qkclip.formula`](results/t1-qk-clip-tpu.json) ✅<br>[`L1.qkclip.max_logit`](results/t1zo-qkclip-maxlogit.results.json) ❌（GPU 2.97e-3、TPU 4.61e-3 → R 1.55） | — | — | T1 的 τ 100 下从未触发；L1 是 τ 0.1 的 30 步对比：裁剪算法过（打开 M31、M33 之前的设置）；第 1 步逐头最大 logit 在 base2 开时过（对齐细节 D15），关 base2 的现行配置下不过（第 4 层 R 1.55，一个头差 −1.19e-2，D31 未定责）；L4 的 QK-Clip 全局最大 logit 曲线见 L4 表 `L4.qkclip.max_logit`（τ 100 下没触发，比的是记录值）；MaxText 单测见台账 M1、M18、M19，最大 logit 不含 sink 见 M25 | ❌ |

### 单算子回放（阶段 A 真实权重，第 1 步前向）

每个算子的输入换成 GPU `ga1-a` 第 1 步抓到的同一个输入，抓 TPU 这个算子的输出，对 GPU 侧 FP64 单算子回放存下的真值算 R。参考档的「rel_L2 对 GPU」是同一输入下 TPU 输出对 GPU 输出。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | 真实权重 |
|---|---|---|---|---|---|
| iHC 混合与 hc_head<br>`L1.ops.op.ihc_attn`<br>`L1.ops.op.ihc_ffn`<br>`L1.ops.op.hc_head` | 注意力前 / FFN 前的 iHC 混合（pre 的四步）与 hc_head | `attn_hc.out`、`ffn_hc.out`、`hc_head`<br>比较单元：3 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（TPU v5p，单独编译）＋离线 FP64 计算 | [GPU 2.08e-3、TPU 2.08e-3 → R 1 ✅｜参☑️](results/hy4-a-l1-op-tpu-fix.json) |
| RMSNorm<br>`L1.ops.op.input_norm`<br>`L1.ops.op.post_attn_norm`<br>`L1.ops.op.final_norm` | 输入、FFN 前、最后的 RMSNorm | `input_layernorm`、`post_attention_layernorm`、`final_norm`<br>比较单元：3 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（TPU v5p，单独编译）＋离线 FP64 计算 | [GPU 2.32e-3、TPU 1.66e-3 → R 0.713 ✅｜参❌](results/hy4-a-l1-op-tpu-fix.json) |
| 输出门投影<br>`L1.ops.op.gate_proj` | gate_proj 的输出（sigmoid 之前） | `self_attn.gate_proj`<br>比较单元：1 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（TPU v5p，单独编译）＋离线 FP64 计算 | [GPU 1.66e-3、TPU 1.66e-3 → R 1 ✅｜参☑️](results/hy4-a-l1-op-tpu-fix.json) |
| 注意力整块<br>`L1.ops.op.attn_block` | MLA 投影与 norm、RoPE、索引器自选 top-k、稀疏注意力与 sink、输出门、o_proj | `self_attn`（自选 top-k）<br>比较单元：1 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（TPU v5p，单独编译）＋离线 FP64 计算 | [GPU 3.47e-3、TPU 3.16e-3 → R 0.91 ✅｜参❌](results/hy4-a-l1-op-tpu-fix.json) |
| 稠密 MLP<br>`L1.ops.op.dense_mlp` | 第 0 层稠密 MLP | `mlp`<br>比较单元：1 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（TPU v5p，单独编译）＋离线 FP64 计算 | [GPU 2.33e-3、TPU 2.22e-3 → R 0.95 ✅｜参❌](results/hy4-a-l1-op-tpu-fix.json) |
| 输出头<br>`L1.ops.op.lm_head` | FP32 输出头 | `lm_head`<br>比较单元：1 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：rel_L2 ≤ 2^-10、归一化最大差 ≤ 2^-7 | 单算子回放（TPU v5p，单独编译）＋离线 FP64 计算 | [GPU 7.92e-7、TPU 1.44e-7 → R 0.182 ✅｜参☑️](results/hy4-a-l1-op-tpu-fix.json) |

### mini 的单算子（GPU 分母 Megatron-Core + TE）

每个算子的输入换成 GPU `g7-l1-a` 第 1 步 rank 0 第 0 个 microbatch 抓到的同一个输入（`op_replay_mini.py`），TPU 输出与 GPU 输出各对 GPU 一侧在 CPU 上的 FP64 重算（`mini-l1/fp64`）算 R。对象都是 hy4-mini 第 0–6 层与 MTP 层里的同一类算子，每类取最差的一层（逐用例算 R 比，取最差）。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | mini |
|---|---|---|---|---|---|
| RMSNorm<br>`L1.mini3.norm` | 输入、FFN 前、最后的 RMSNorm 与 MTP 的 enorm / hnorm / final norm | <br>比较单元：20 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 2.35e-3、TPU 2.35e-3 → R 1 ✅](results/t1-l1-op-tpu-r3.json) |
| 输出门投影<br>`L1.mini3.gate_proj` | gate_proj 的输出（sigmoid 之前） | <br>比较单元：8 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 1.66e-3、TPU 1.66e-3 → R 1 ✅](results/t1-l1-op-tpu-r3.json) |
| 注意力整块<br>`L1.mini3.attn` | MLA 投影与 norm、RoPE、索引器自选 top-k（共享层用块首层自选的掩码）、稀疏注意力与 sink、输出门、o_proj | <br>比较单元：8 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 3.36e-3、TPU 3.09e-3 → R 0.919 ✅](results/t1-l1-op-tpu-r3.json) |
| 稠密 MLP<br>`L1.mini3.dense_mlp` | 第 0 层稠密 MLP（SwiGLU 两次舍入） | <br>比较单元：1 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 3.73e-3、TPU 3.31e-3 → R 0.888 ✅](results/t1-l1-op-tpu-r3.json) |
| MoE 整块<br>`L1.mini3.moe` | 路由（sigmoid + 偏置选、归一化、缩放）、路由专家（含 clamp 10）、共享专家，自选路由 | <br>比较单元：7 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 4.93e-3、TPU 4.65e-3 → R 0.943 ✅](results/t1-l1-op-tpu-r3.json) |
| 路由 logits<br>`L1.mini3.router_logits` | 路由矩阵乘的输出（FP32，sigmoid 之前） | <br>比较单元：7 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 1.13e-6、TPU 1.20e-7 → R 0.106 ✅](results/t1-l1-op-tpu-r3.json) |
| 共享专家<br>`L1.mini3.shared_expert` | 共享专家整块（FP64 参照在 CPU 上用 HF 检查点的权重算） | <br>比较单元：7 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 3.70e-3、TPU 3.31e-3 → R 0.895 ✅](results/t1-l1-op-tpu-r3.json) |
| 路由专家<br>`L1.mini3.routed_experts` | MoE 整块减共享专家（GPU 一侧没单独抓，用 mlp.out − shared.out） | <br>比较单元：7 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 5.37e-3、TPU 4.66e-3 → R 0.869 ✅](results/t1-l1-op-tpu-r3.json) |
| iHC 混合<br>`L1.mini3.ihc_mix` | 4 路流按 pre 加权求和成分支输入（注意力前、FFN 前） | <br>比较单元：14 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 1.17e-3、TPU 1.17e-3 → R 1 ✅](results/t1-l1-op-tpu-r3.json) |
| iHC 写回<br>`L1.mini3.ihc_writeback` | 4 路流 + post × 分支输出（分支输出换成 GPU 的），MTP 是两次残差相加 | <br>比较单元：15 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 1.65e-3、TPU 1.65e-3 → R 1 ✅](results/t1-l1-op-tpu-r3.json) |
| hc_head<br>`L1.mini3.hc_head` | 最后的 4 路流合成一路 | <br>比较单元：1 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 1.56e-3、TPU 1.56e-3 → R 1 ✅](results/t1-l1-op-tpu-r3.json) |
| MTP 的 eh_proj<br>`L1.mini3.eh_proj` | cat(enorm, hnorm) 的投影 | <br>比较单元：1 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 1.66e-3、TPU 1.66e-3 → R 1 ✅](results/t1-l1-op-tpu-r3.json) |
| 输出头<br>`L1.mini3.lm_head` | FP32 输出头 | <br>比较单元：1 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 2.56e-7、TPU 8.32e-8 → R 0.325 ✅](results/t1-l1-op-tpu-r3.json) |
| 交叉熵<br>`L1.mini3.cross_entropy` | GPU 的 logits 进 MaxText 的交叉熵（逐 token） | <br>比较单元：1 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [GPU 3.33e-8、TPU 3.82e-8 → R 1.15 ✅](results/t1-l1-op-tpu-r3.json) |
| embedding<br>`L1.mini3.embed` | 查表 | <br>比较单元：1 个用例<br>聚合：同表头 | R 比 ≤ 1.2｜参考：无 | 单算子回放（TPU v5p，MaxText 2b005b3d，tw-opmini-d：tw-opmini-c 再开 M40）＋GPU 一侧 FP64 计算 | [两边对参照都是 0（逐位） ✅](results/t1-l1-op-tpu-r3.json) |

### mini 的其他 L1 检查（KD 输入、优化器回放、逐元素函数、QK-Clip）

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | mini |
|---|---|---|---|---|---|
| 索引器 KD 的输入<br>`L1.indexer_kd.inputs` | 第 0 层第 0 行（GPU rank 0 micro 0 = TPU 第 0 行）的教师 q、教师 k、索引器分数，对 CPU FP64（`step1_ref.py --forward-only`，`t1-l1-kd-qkclip-vs-fp64.json`）；TPU 是 t1z 设置的 `t1zdiag2`，GPU 是 `g1p-kdd` | 教师 q、教师 k、索引器分数（只比因果部分）<br>比较单元：3 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | 对 CPU FP64 的 R 比 ≤ 1.2｜参考：无 | 第 1 步诊断（TPU）＋CPU FP64 前向重算与离线 FP64 计算 | [GPU 5.27e-3、TPU 5.17e-3 → R 0.98 ✅](results/t1-layer0-kd-attribution.json) |
| 优化器连续 8 步回放（现行：M30（T1 开，M31）、M34）<br>`L1.optim_adam_r2.tpu_adam_acc` | GPU `g2o-a` 第 1–8 步的梯度与主权重喂 MaxText 优化器，对 FP64 参照（Adam 一支的除法、sqrt、1 − β^t 用精确写法，台账 M34） | rank 0 的 205 个张量 × 8 步<br>比较单元：1,640 个用例<br>聚合：逐步逐张量算 R，取最差 | R 比 ≤ 1.2｜参考：无 | 优化器 8 步回放（TPU v5p，MaxText 7c95fabe，再开 muon_adam_accurate 最终版，台账 M34）＋FP64 参照 | [GPU 1.36e-2、TPU 1.41e-2 → R 1.03 ✅](results/t1-optim-replay-adam-acc-r2-tpu.json) |
| 逐元素函数：XLA 默认实现（FP32）（只报）<br>`L1.elem.exp.fp32`<br>`L1.elem.log.fp32`<br>`L1.elem.sigmoid.fp32`<br>`L1.elem.silu.fp32`<br>… | exp、log、sigmoid、silu、rsqrt 对 FP64 正确舍入的 ulp；合成输入（均匀、对数均匀、[1, 4) 全部 FP32）与真实输入（GPU `g2s-a` 的路由 logits、稠密 MLP gate、RMSNorm 均方，TPU 第 0 层 KD 的分数与教师目标） | 每个函数 3–5 个用例<br>比较单元：15 个用例<br>聚合：逐函数取最差用例；只报（生产不开开关时的精度，判定看现行开关写法一行） | ulp_max ≤ 1｜参考：无 | 单算子回放（TPU v5p，合成与真实输入，单独编译）＋离线 FP64 计算 | [62（62×） ❌](results/tpu-elementwise-ulp.json) |
| 逐元素 sqrt、除法：XLA 默认实现（FP32）（只报）<br>`L1.elem.sqrt.fp32`<br>`L1.elem.div.fp32` | FP32 sqrt（对数均匀 1e-30–1e30）与除法（随机、Adam 形状）对 FP64 正确舍入的 ulp；另报 1 − β^t（t = 1..1000） | 3 个用例<br>比较单元：3 个用例<br>聚合：逐用例取最差；只报 | ulp_max ≤ 1｜参考：无 | 单算子回放（TPU v5p，合成与真实输入，单独编译）＋离线 FP64 计算 | [3（3×） ❌](results/tpu-elementwise-ulp.json) |
| 逐元素函数：现行开关写法（FP32）<br>`L1.elem.exp.fp32_mt`<br>`L1.elem.log.fp32_mt`<br>`L1.elem.sigmoid.fp32_mt`<br>`L1.elem.silu.fp32_mt`<br>… | `transcendental_accuracy_highest` 的 exp / log / sigmoid / silu（`maxtext.utils.accurate_math`，M32 + M37）与 `muon_adam_accurate` 的 sqrt、除法（M34），加 rsqrt（无开关）；同一批输入 | 每个函数 3–5 个用例<br>比较单元：18 个用例<br>聚合：逐函数取最差用例，按函数取门槛：exp ≤ 2、log ≤ 1、sqrt / 除法 ≤ 1、rsqrt ≤ 2 ulp（CUDA 文档）；sigmoid / silu ≤ 4（推断，同混元 3） | ulp_max ≤ 1｜参考：无 | 单算子回放（TPU v5p，合成与真实输入，单独编译）＋离线 FP64 计算 | [3（3×） ❌](results/tpu-elementwise-ulp.json) |
| 逐元素 log：GPU 实测与 TPU 精确写法（只报）<br>`L1.elem.log.gpu_b200`<br>`L1.elem.log.tpu_fdlibm`<br>`L1.elem.log.tpu_maxtext_m51` | FP32 log 对 FP64 正确舍入的 ulp：GPU B200 `torch.log` 实测；TPU 精确写法（e_logf.c 算法的 FP32 移植加精确除法）的独立实现与 MaxText 开关 `transcendental_log_freebsd`（M51，默认关，对齐配置未开） | 4 批输入：[0.5, 4) 全部 FP32、两段对数均匀、KD 真实输入<br>比较单元：12 个用例<br>聚合：逐批取最大 ulp，取最差的一批；只报 | ulp_max ≤ 1｜参考：无 | 单算子（GPU B200 torch.log 与 TPU v5p 三种写法，同一批输入，输入 sha256 两边逐个相同）＋离线 FP64 | [1（1×） ✅](results/t1-log-ulp.results.json) |
| 逐元素 relu、clamp、加乘与类型转换<br>`L1.elem.relu.fp32`<br>`L1.elem.clamp.fp32`<br>`L1.elem.basic` | relu、clamp（±10，路由专家 SwiGLU 的截断）、FP32 加乘、FP32 → BF16，必须逐位等于 numpy（次正规数另计） | 合成与真实输入<br>比较单元：7 个用例<br>聚合：逐用例比，数不一致的用例 | 逐位｜参考：无 | 单算子回放（TPU v5p，合成与真实输入，单独编译）＋离线 FP64 计算 | [逐位全中 ✅](results/tpu-elementwise-ulp.json) |
| log-sum-exp（KD 的 log_softmax）（只报）<br>`L1.elem.lse.fp32` | logsumexp 与 log_softmax 反推的 LSE 对 FP64 的 ulp；真实 KD 输入（4096 行）与三种合成行；另拆 exp、FP32 求和、log 三步 | 4 组行<br>比较单元：4 个用例<br>聚合：逐组算，只报不判 | 只报，不判｜参考：无 | 单算子回放（TPU v5p，合成与真实输入，单独编译）＋离线 FP64 计算 | [诊断：只报数，不判 ✅](results/tpu-elementwise-ulp.json) |
| FP32 超越函数精度开关（KD 第 0 行）（只报）<br>`L1.transcendental.kd_row0` | MaxText `transcendental_accuracy_highest`（台账 M32）关 / 开：第 0 层索引器 KD 第 0 行，TPU 图内对同一批输入的 FP64 重算 | 1 行，两组<br>比较单元：2 个用例<br>聚合：两组各一个相对差，只报不判 | 只报，不判｜参考：无 | 第 1 步诊断（TPU，开关关 / 开两组）＋离线 FP64 计算 | [第 0 层第 0 行 KD 对 FP64：开关关 −1.15e-3、开 +9.67e-5 🟡](results/t1-step1-transcendental-accuracy.json) |
| 低阈值 QK-Clip：裁剪的算法<br>`L1.qkclip.formula` | τ 0.1、30 步（GPU `g6-qk-a/b`，TPU `tw-qk-a`，打开 M31、M33 之前的设置）：TPU 裁剪后的 q / kv 上投影主权重 ≟ GPU 裁剪前的主权重 × TPU 自己记下的 η（Megatron-Core 的行规则，α 0.5）。门槛 1e-3 是推断（`analysis/qk_clip_compare.py` 开头）：残差只剩两边这一步优化器更新的差，一次更新约 lr·O(1) = 1e-5，裁剪改变量约 (1−η)·|W| ≈ 1e-1 × 6e-3，比值约 1e-4，取 10 倍余量；GPU 自己的同一量 1.7e-6 | 第 1 步 7 层 + MTP 的 q_b / kv_b<br>比较单元：16 个用例<br>聚合：逐条规则核，数不满足的条数 | resid_over_clip ≤ 1.00e-3｜参考：无 | 生产程序（TPU 30 步，τ 0.1）对 GPU g6-qk-a / b＋CPU FP64 前向重算 | [5.68e-4（0.568×） ✅](results/t1-qk-clip-tpu.json) |
| 低阈值 QK-Clip：逐头最大 logit<br>`L1.qkclip.max_logit` | 第 1 步逐层逐头的最大 logit（η 的分母）对 CPU FP64；TPU 是现行配置（base2 关）的 `tz-off-qk01`（base2 开的上一代 `mz-qk-a` 见 `t1-qkclip-maxlogit`），GPU 取 `g6-qk-a` / `b` 的中位数 | 8 层 × 8 头<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | R 比 ≤ 1.2｜参考：无 | 生产程序（TPU `tz-off-qk01`，现行 yml、base2 关，τ 0.1，30 步）对 GPU g6-qk-a / b＋CPU FP64 前向重算 | [GPU 2.97e-3、TPU 4.61e-3 → R 1.55 ❌](results/t1zo-qkclip-maxlogit.results.json) |
| 低阈值 QK-Clip：被裁的头（只报）<br>`L1.qkclip.clipped_heads_30` | 30 步逐步逐头被裁的集合，TPU 对 GPU-a，对照 GPU-b 对 GPU-a；TPU 是现行配置（base2 关）的 `tz-off-qk01`，GPU `g6-qk-a` / `b` | 30 步 × 64 头<br>比较单元：30 步<br>聚合：集合差的合计，只报不判 | 只报，不判｜参考：无 | 生产程序（TPU `tz-off-qk01`，现行 yml、base2 关，τ 0.1，30 步）对 GPU g6-qk-a / b＋CPU FP64 前向重算 | [只报：30 步逐步逐层逐头被裁集合的差 TPU 对 g6-qk-a 21、g6-qk-b 对 g6-qk-a 12；lm_loss MARE… 🟡](results/t1zo-qkclip-maxlogit.results.json) |
| 低阈值 QK-Clip：GPU 另裁 BF16 权重（只报）<br>`L1.qkclip.gpu_bf16_copy` | GPU 把 BF16 模型权重单独按 bf16(W_bf16 × 因子) 裁，不是 bf16(裁剪后主权重)；MaxText 只有 FP32 主权重（已知语义差） | 第 1–5 步 7 层 + MTP 的 q_b / kv_b<br>比较单元：72 个用例<br>聚合：两种 BF16 权重的差与 30 步 lm_loss 的散布，只报不判 | 只报，不判｜参考：无 | 生产程序（TPU 30 步，τ 0.1）对 GPU g6-qk-a / b＋CPU FP64 前向重算 | [GPU 另裁 BF16 模型权重：与 bf16(裁剪后主权重) 最多差 1 个 BF16 ulp，被裁行里至多 28.9% 的元素不同；30… 🟡](results/t1-qk-clip-tpu.json) |
| T0：hy4-tiny 逐层<br> | hy4-tiny（FP32、CPU）逐层前向 / 反向与第 1 步更新，两侧各对 FP64 | 7 层 + MTP | — | — | ⬜ 未开始（待做） |

非 ✅ 格子的原因：

- **逐元素函数：XLA 默认实现（FP32）**（mini ❌）：exp 105、log 62、sigmoid 83、silu 86 ulp，超过门槛（exp ≤ 2、log ≤ 1 是 CUDA 文档的；sigmoid / silu ≤ 4 是推断，同混元 3：expf 2 ulp 经一次加、一次除后的起点）（对齐细节 D14）。
- **逐元素 sqrt、除法：XLA 默认实现（FP32）**（mini ❌）：门槛都是 1 ulp（CUDA 默认 sqrtf、x / y 正确舍入）；TPU v5p 实测 sqrt 3、除法 2 ulp，1 − β^t 最多 108 ulp（只报）。M34 换成修正写法后见下一行（对齐细节 D17）。
- **逐元素函数：现行开关写法（FP32）**（mini ❌）：log 3 ulp 超过 CUDA logf 的 1 ulp 门槛，GPU B200 实测 1 ulp；MaxText 开关 M51（精确 log，默认关，对齐配置未开）同一批输入 ≤ 1 ulp（下一行，对齐细节 D14）；其余过：exp 1、sigmoid 3、silu 4、sqrt 1、除法 1、rsqrt 2 ulp（1 − β^t 2，只报）；silu 4 ulp 正好在推断门槛上，略大于 GPU B200 实测最大 3.4（`b200-sigmoid-silu-probe.json`）。
- **FP32 超越函数精度开关（KD 第 0 行）**（mini 🟡）：诊断：开关打开后 KD 第 0 行的偏低与 LSE 的 0.95 格偏差都消失（数字见格子），第 0 层 KD 的定责成立（对齐细节 D14）。
- **低阈值 QK-Clip：逐头最大 logit**（mini ❌）：未定责：关 base2 后第 4 层 R 1.55（最差一个头相对差 −1.19e-2，约 3 个 BF16 舍入单位）；base2 开时同一量过（第 4 层 0.94，最差层 1.05）。线索：同一第 1 步只把累积 2 换成累积 1（全局批不变），第 4 层 R 1.55 → 1.83（MTP 层 1.07 → 1.00，其余层都 ≤ 0.76），超出不是累积带来的（不累积时反而更大）（`t1zo-qkclip-accum-tau01.json` 的 `step1.per_layer`）；没有定责
- **低阈值 QK-Clip：被裁的头**（mini 🟡）：诊断：现行配置 30 步的集合差合计 TPU 对 a 21、b 对 a 12（1.75 倍）；GPU 只有一对，判不了 21 在不在 GPU 的散布内。base2 开的上一代 `mz-qk-a` 是 17（`t1-qkclip-maxlogit`），更早打开 M31、M33 之前的 `tw-qk-a` 是 15（第 1 步三方都裁同样的 47 个头，`t1-qk-clip-tpu`）。同表逐头最大 logit 第 4 层不过，方向一致。
- **低阈值 QK-Clip：GPU 另裁 BF16 权重**（mini 🟡）：诊断：GPU 的 BF16 权重与 bf16(主权重) 只差 1 个 BF16 ulp，下一次优化器更新就被主权重覆盖；30 步 lm_loss 的 TPU 对 GPU 在 GPU 两份的散布内，所以不加开关。

## L2 · 逐层

目的：每层输入取参照的，单独跑一层，把误差隔离到层。

L2 共 2 项检查。逐层隔离回放 `layer_replay.py`（台账 J12、J19）：GPU `g7-l1-a` 第 1 步每层的输入进 TPU 单层程序，比输出与输入梯度。现行结果是 TPU `tw-l2-g`（当时 T1 的全部开关，含 M40 / M41，base2 开，MaxText `3741e818`；关 base2 后没有重测）对 CPU FP64 单层重算（`frameworks/megatron/layer_ref.py`，`t1-l2-layer-vs-fp64.json`）。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | mini |
|---|---|---|---|---|---|
| 各层前向 / 反向对 CPU FP64<br>`L2.layer.out`<br>`L2.layer.in_grad` | 第 1 步，每层输入取 GPU 的，单层前向输出与反向的层输入梯度，TPU 与 GPU 各对 CPU FP64 单层重算的 R | hy4-mini 第 0–6 层与 MTP 层<br>比较单元：16 个用例<br>聚合：逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例 | 对 CPU FP64 的 R 比 ≤ 1.2｜参考：无 | 逐层回放（TPU v5p `tw-l2-g`，单层单独编译，吃 GPU g7-l1-a 每层的输入，台账 J19）＋CPU FP64 单层重算 | [GPU 1.23e-2、TPU 1.42e-2 → R 1.16 ✅](results/t1-l2-layer-vs-fp64.json) |
| 索引器 top-k 翻转与路由翻转（只报）<br>`L2.flips.routing`<br>`L2.flips.dsa_topk` | 逐层翻转率（路由）与不重合率（DSA top-k）对 CPU FP64 自选，按 FP64 分差分桶；判定用 L3 路由一致率的对真值口径（同一批数），这里把逐层、逐分差桶的统计摊开 | 第 1、8、30、100 步；路由 28 个（步, 层），DSA 16 个（步, 层）。TPU 抓数：四步都是 base2 开那一代（第 1 步 `tw-ra1b`，第 8、30、100 步 `t1zw-a` 那一代，含 M42b；生成脚本 `analysis/l2_flip_stats.py` 不取 base2 关的 `tw-ra1z`）；关 base2 的第 1 步判定见 L3 路由一致率，这里的逐层统计没有按 `tw-ra1z` 重算<br>聚合：每个（步, 层）算 R = TPU 对 FP64 的翻转率 ÷ GPU 三份的最大值，取最差 | R 比 ≤ 1.2｜参考：无 | 第 1、8、30、100 步抓数（同路由一致率）逐层、按 FP64 分差分桶 | [GPU 6.68e-4、TPU 6.67e-4 → R 0.999 ✅](results/t1-l2-flip-stats.results.json) |

## L3 · 整网第 1 步

目的：同一份权重、同一个 batch，整网跑第 1 步，比梯度与优化器状态。

L3 共 6 项检查，检查条数 40，张量级比较 1,369 次。

- TPU 的第 1 步来自训练程序的诊断模式（`train_hy4.py --diag-step1`：回放 GPU 第 1 步的 batch，用训练里同一个梯度累积函数，不更新）。
- TPU 用现行配置训练步实际用的第 1 步梯度（由 `tz-off-rep` 第 1 步全量抓数的动量反推：反推的舍入量级：Adam 参数上两条反推路径（由 m、由 v）逐元素互比，最大 1.68 个 FP32 单位舍入（约 1.0e-7），比 TPU 离 FP64 的距离（中位数 6.4e-3）小四个数量级以上；落盘与取回核对见 [`t1zo-step1-train-grad-file-check.json`](results/t1zo-step1-train-grad-file-check.json)，728 个张量的 sha256 全对上；base2 开的上一代 `mz-opt2` 与诊断程序 `mz-diag1` 只作旁证）对 GPU 第 1 步全量抓数参照 `g2s-a … g2s-f` 6 份（HF 名字与布局，台账 G15；注意力 q·k 在 FP32 里算，台账 G17；`g2s-d … f` 是 2026-11-11 用同一代码包、镜像与参数补跑的，`meta.json` 除运行名与启动时的 TF32 探针外逐字段相同，新旧两批之间没有批次差，[`g2s6-meta-vs-g2sa.json`](gpu-noise/results/g2s6-meta-vs-g2sa.json)）。
- 噪声只用这 6 份两两 15 对（协议第 5 节；2026-11-11 前是 `g2s-a/b/c` 三份 3 对，那时的判定见 [`t1zo-step1-train-grad-vs-fp64.json`](results/t1zo-step1-train-grad-vs-fp64.json) 与 PROTOCOL 变更记录同日「L3 尺子 6 份」一行）。逐张量 r = TPU 对最近一份 GPU 的 rel_L2 ÷ GPU 6 份两两的最大 rel_L2。
- **分辨率下限**（PROTOCOL 第 5 节）：GPU 第 1 步噪声 N = 0、且两边对参照的 rel_L2 都小于 BF16 单位舍入 u = 2^-8 的张量，记「低于分辨率（按下限过）」，不再用 R 判。有 5 个这样的张量（第 0、1 层索引器的 `k_norm`、`weights_proj` 等）。
- 超出噪声的张量对 CPU FP64 参照定责：GPU golden 同一套 Megatron-Core 代码与 loss 语义在 CPU 上全程 FP64 重算第 1 步（`frameworks/megatron/step1_ref.py`；逐算子核过没有 FP64 以外的浮点输出；同一代码的 CPU FP32 版离它的 rel_L2 中位数 1.34e-4），离散选择（DSA top-k、MoE 路由）由 FP64 自己选。R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64)（GPU 取 6 份的中位数）≤ 1.2 算已定责。有超 2N 而未定责的张量判不过，只有 1–2N 未定责的判待定。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | mini |
|---|---|---|---|---|---|
| 全部参数梯度对 GPU 第 1 步噪声，超出的对参照定责<br>`L3.step1.grads.vs_gpu` | 逐张量算 r 与 R（见本级开头），参照是 CPU FP64 | 205 个张量（HF 名字；路由专家只有 rank 0 的 0–3 号）<br>比较单元：205 个张量<br>聚合：逐张量算 r 与 R，数「超出 N 且 R > 1.2」的张量：有超 2N 的就不过（待定责），只有 1–2N 的待定 | 超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数 ≤ 0｜参考：无 | 训练步实际用的第 1 步梯度（由 `tz-off-rep` 第 1 步全量抓数的动量反推，`export_train_grads.py`；反推的舍入量级约 1e-7；结果文件自身做了落盘核对（`tpu_file_check.checked` true，依据 `t1zo-step1-train-grad-file-check.json`：全量抓数 50 个文件、728 个张量 sha256 全对上），未发现损坏）对 GPU 6 份第 1 步参照 g2s-a … g2s-f（两两 15 对；d–f 是 2026-11-11 同设置补跑的，`gpu-noise/results/g2s6-meta-vs-g2sa.json`）＋CPU FP64 整网重算 | [1 🟡](results/t1zo-step1-train-grad-vs-fp64-n6.json) |
| 按算子分组的参数梯度<br>`L3.step1.ref.op.embed`<br>`L3.step1.ref.op.ihc`<br>`L3.step1.ref.op.input_norm`<br>`L3.step1.ref.op.q_a`<br>… | 同上一行，按算子分组各判一次（分组见 `build.py` 的 `STEP1_OPS`），供「模型的每个算子由哪条检查覆盖」的 L3 列用 | 205 个张量分成 27 组<br>比较单元：205 个张量<br>聚合：每组同上一行的规则；格子写最差的一组，并排写这一组里 R 最大的张量两边对参照的 rel_L2 | R 比 ≤ 1.2｜参考：无 | 训练步实际用的第 1 步梯度（由 `tz-off-rep` 第 1 步全量抓数的动量反推，`export_train_grads.py`；反推的舍入量级约 1e-7；结果文件自身做了落盘核对（`tpu_file_check.checked` true，依据 `t1zo-step1-train-grad-file-check.json`：全量抓数 50 个文件、728 个张量 sha256 全对上），未发现损坏）对 GPU 6 份第 1 步参照 g2s-a … g2s-f（两两 15 对；d–f 是 2026-11-11 同设置补跑的，`gpu-noise/results/g2s6-meta-vs-g2sa.json`）＋CPU FP64 整网重算 | [GPU 6.11e-3、TPU 3.57e-2 → R 5.83 🟡](results/t1zo-step1-train-grad-vs-fp64-n6.json) |
| iHC post 门控梯度的条件数（只报）<br>`L3.step1.ihc.post_conditioning` | 每个 token 的 dL/dpost_k = Σ_d h[t,d]·g[t,k,d]（h 是本层 FFN 分支输出，g 是本层输出流的梯度），条件数 κ_k = Σ_t ｜项｜ ÷ ｜Σ_t 项｜；GPU `g2s-a` 第 1 步 rank 0 第 0 个 microbatch 的一条 4096 token 序列 | 第 1–3 层 FFN 一侧，各 4 路<br>比较单元：3 个用例<br>聚合：每层 4 路的 κ 范围，只报不判 | 只报，不判｜参考：无 | 离线计算（GPU 第 1 步抓数） | [κ：1 层 710–1359，2 层 25，3 层 93–96 🟡](results/t1-ihc-post-grad-conditioning.json) |
| 路由一致率（对 CPU FP64）<br>`L3.route_agree.s1`<br>`L3.route_agree.s8`<br>`L3.route_agree.s30`<br>`L3.route_agree.s100` | TPU 自选的 MoE 路由（每个 token 选中的专家集合）与 DSA top-k（每个 query 选中的位置集合）逐层对 CPU FP64 自选比；每层 R = TPU 的翻转率（DSA 为不重合率）÷ GPU 三份对 FP64 的最大值 | 第 1、8、30、100 步；路由第 1–6 层与 MTP 层，DSA 第 0、1、5 层与 MTP 层；第 8、30、100 步三方从 GPU 同一份重建状态起步；TPU 只取现行配置的抓数（第 1 步 base2 关的 `tw-ra1z`；base2 开的 `tw-ra1b` 不计，`build.py` 的 `ROUTE_AGREE_SKIP`）；第 8、30、100 步只有 base2 开那一代的抓数（`tw-ra8`、`tw-ra30`、`tw-ra100b`），关 base2 后没有重测。用到的抓数文件 sha256 都等于作业内记下的；最早的 `tw-ra1`、`tw-ra100` 两份拿到的文件里各有一段连续的零（约 2.1 MB、4.1 MB，结尾落在 256 MiB 的整数倍上），与作业内的 sha256 对不上，已由同设置重抓的 `tw-ra1b`、`tw-ra100b` 替代：两份重抓的文件 sha256 正好等于原来作业内记下的值，除那一段外逐字节相同。格子写四步里 R 最大的一格，现为第 8 步（base2 开的 `tw-ra8`，DSA，GPU 6.68e-4、TPU 6.67e-4，R 0.999），链接仍指第 1 步的文件；现行配置第 1 步（`tw-ra1z`）最差是 DSA 第 0 层，GPU 6.74e-4、TPU 6.69e-4，R 0.993；第 30、100 步最差 0.991、0.998。每步的读数在总览细节<br>比较单元：44 个用例<br>聚合：每步取路由与 DSA 各层里最差的 R；≤ 1.2 过、≤ 2 待定、> 2 不过（PROTOCOL 第 5 节，2026-11-09 起） | R 比 ≤ 1.2｜参考：无 | 第 1、8、30、100 步抓数（TPU 自选，只抓不灌）对 GPU 三份与 CPU FP64 自选 | [GPU 6.68e-4、TPU 6.67e-4 → R 0.999 ✅](results/t1-route-agree-s1.json) |
| Muon 动量、Adam 状态、更新后参数<br>`L3.step1.optim_arith.delta`<br>`L3.step1.optim_arith.adam_v` | 第 1 步优化器一步本身：各侧对「FP64 优化器作用在自己的梯度上」比（TPU 梯度由自己的动量反推，GPU 用同一次运行抓的梯度） | 更新后参数 205 个张量（98 个低于分辨率下限）、Adam 二阶动量 99 个（都低于分辨率下限，格子里的 R 是两次舍入之比、按下限过）；一阶动量与 Muon 动量按定义是 (1−β)·梯度，TPU 梯度又由一阶动量反推，比较是循环的，只报<br>比较单元：304 个张量<br>聚合：逐张量算 R 比，取最差的张量，对固定门槛判 | R 比 ≤ 1.2｜参考：无 | 第 1 步优化器一步：各侧对「FP64 优化器作用在自己的梯度上」比；TPU 是现行 `tz-off-rep` 第 1 步全量抓数（梯度由动量反推），GPU 是 g2s 三份的抓数，CPU 离线重算（`analysis/l3_optim_step1.py`） | [GPU 8.61e-8、TPU 1.38e-7 → R 1.61（99 / 99 个张量低于分辨率下限，按下限过） ✅](results/t1zo-l3-optim-step1.results.json) |
| 第 1 步更新端到端对 FP64（只报）<br>`L3.step1.optim.delta`<br>`L3.step1.optim.adam_m`<br>`L3.step1.optim.adam_v`<br>`L3.step1.optim.muon_momentum`<br>… | 更新后参数、Adam 一阶 / 二阶动量、Muon 动量各对 CPU FP64 端到端比（含梯度的差） | 同上一行<br>比较单元：608 个张量<br>聚合：逐张量算 R 比，取最差的张量，对固定门槛判；逐张量算更新量的 rel_L2，取最差的张量 | R 比 ≤ 1.2｜参考：无 | 第 1 步优化器一步：各侧对「FP64 优化器作用在自己的梯度上」比；TPU 是现行 `tz-off-rep` 第 1 步全量抓数（梯度由动量反推），GPU 是 g2s 三份的抓数，CPU 离线重算（`analysis/l3_optim_step1.py`） | [GPU 4.04e-4、TPU 7.03e-1 → R 1.74e+03 ❌](results/t1zo-l3-optim-step1.results.json) |

非 ✅ 格子的原因：

- **全部参数梯度对 GPU 第 1 步噪声，超出的对参照定责**（mini 🟡）：以现行比较 tz-off-rep 为准、GPU 参照 6 份 15 对：169 个超出 N、35 个超出 2N；超出 N 而 R > 1.2 的 2 个：第 3 层注意力前 `hc_scale` 计入未定责（1.05 N；GPU 1.19e-3，TPU 6.40e-3，R 5.36；只在 1–2N，所以待定），MTP `enorm`（1.44 N，R 1.24、两边都小于 u）另列；两个都按 D11 豁免规则单列（豁免规则与沿用待复核；enorm 的接受理由见 D13）。另有 11 个 iHC 张量 R > 1.2 但在噪声内（最大第 3 层 FFN 前 `hc_scale` R 5.83、0.85 N），按规则不定责。GPU 3 份 3 对时（2026-11-11 前）是 177 个超出 N、46 个超出 2N、5 个 iHC 未定责（超 2N 的第 6 层 FFN 前 `hc_base` 5.09 N），判不过；补到 6 份后这 5 个里 4 个落进噪声（第 6 层 FFN 前 `hc_base` 0.21 N、`hc_fn` 0.59 N，第 0 层注意力前 `hc_base` 0.60 N，第 2 层注意力前 `hc_scale` 0.74 N），关 base2 后新出现的 3 个都在其中。剩下的第 3 层 `hc_scale` 与 base2 开那一代（`mz-opt2`）单列的 6 个 iHC 张量是同一批（对齐细节 D11）。那次接受理由里的「落在病态小参数上」后来量了（GPU `g2s-a` 第 1 步一条序列的条件数 κ）：κ 与 R 只弱相关（名次相关 0.25）（`t1-ihc-param-grad-conditioning.json`，对齐细节 D11）。
- **按算子分组的参数梯度**（mini 🟡）：iHC（hc_pre）一组待定（第 3 层注意力前 `hc_scale` 1.05 N 计入未定责；GPU 3 份时这一组不过）；其余过（索引器 k_norm 一组按分辨率下限过；enorm 另列）。hc_scale 与 enorm 都按 D11 豁免规则单列（豁免规则与沿用待复核）。
- **iHC post 门控梯度的条件数**（mini 🟡）：诊断。κ 约 25–1400：每个 token 的项里 1e-4 量级的系统差，加总以后放大成 1e-2 量级的相对差（层号按 HF 检查点从 0 数）。第 1 层 κ 710–1359、第 3 层约 95，放大倍数够大；第 2 层只有约 25。
- **第 1 步更新端到端对 FP64**（mini ❌）：只报：未定责来自梯度（以现行比较 tz-off-rep 为准）。更新后参数 17 个（10 个 iHC 张量与第 1 层 `input_layernorm`、第 2 / 4 层 `q_a_layernorm`、索引器 `wk` / `wq_b`、`model.norm` 等），Adam 一阶动量 6 个、二阶动量 11 个（含 sink、enorm）；Adam 第 1 步约为 −lr·sign(g)，近零梯度元素翻符号就是整格更新。最差的是第 3 层注意力前 `hc_base`（8 个元素）：参照的更新量范数 3.0e-5 ≈ √8 × lr，TPU 对参照 rel_L2 0.703 ≈ 2 ÷ √8，按数值反推是 1 个元素的更新方向反了（文件没有存逐元素数）；它的一阶动量 TPU 在 GPU 三份的波动内（GPU 自己离参照也有 0.10）。Muon 动量 GPU 抓数里没有，只报 TPU 对 FP64（rel_L2 中位数 6.1e-3）。base2 开那一代（`mz-opt2`，`t1-l3-optim-step1`）是 17 / 7 / 13 个。

## L4 · 多步训练

目的：训练 300 步或 1000 步，看 TPU 被测对 GPU 基线的差是否落在 GPU 自己重跑的散布里。

L4 共 13 项检查，检查条数 20 / 10（mini / w300）。
- 现行比较（mini 列，目录 `t1zo-l4.yaml`，2026-11-11 起）：TPU 被测 `tz-off-rep`（T1 yml 的全部开关，含 M50，base2 关）对 GPU 基线 `g3-a`，噪声 `g3-a … g3-p` 16 份两两 120 对；1000 步过 7、待定 2（主 loss 两个窗口）。上一代 `t1zg-a`（base2 开）与诊断代见[历代比较](../../dashboard/history.md)。
- 同一运行的前 300 步（w300 列，目录 `t1zc300-l4.yaml`）对 GPU `g2-a`，噪声 `g2-a … g2-h` 两两 28 对：过 8、待定 1（grad_norm）。QK-Clip 一行的 w300 仍是 C 口径之前的 `t1zo300-l4-qkclip`。

- 噪声上限 N 只用 GPU，每个判定量取各对的最大值（协议第 3 节）。TPU 只改执行参数的扰动运行只作诊断。
- 每条序列算三个判定量：带符号均值、MARE、最大单步；三个都对 N 判，取最差。格子写最差那个判定量的 TPU 实测（括号里是名字、实测 ÷ N 的倍数，N 与判法栏不同时另写 N）。首步差只报。
- 学习率不在这一级：它是登记的常数，归 L0 设置核对。

### 现行比较

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | mini | w300 |
|---|---|---|---|---|---|---|
| loss<br>`L4.loss` | 全程逐步主 loss 的相对差 | `lm_loss`，第 1–1000 步（w300 列：第 1–300 步）<br>比较单元：mini 1,000 步；w300 300 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 8.95e-6）｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | 生产程序（逐步记录） | [8.75e-6（MARE，0.9769×N） ✅｜参✅](results/t1zc-l4.report.md) | [5.86e-6（MARE，0.872×N，N=6.72e-6） ✅｜参✅](results/t1zc300-l4.report.md) |
| loss 后段<br>`L4.loss.tail` | 第 101 步起 | `lm_loss`，第 101–1000 步（w300 列：第 101–300 步）<br>比较单元：mini 900 步；w300 200 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 9.63e-6）｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | 生产程序（逐步记录） | [9.30e-6（MARE，0.9661×N） ✅｜参✅](results/t1zc-l4.report.md) | [6.89e-6（MARE，0.815×N，N=8.46e-6） ✅｜参✅](results/t1zc300-l4.report.md) |
| MTP loss<br>`L4.mtp_loss`<br>`L4.mtp_loss.tail` | 逐步 MTP loss（两边都读 MaxText 口径 `mtp_loss_mt`），全程与后段 | `mtp_loss_mt`，全程与第 101 步起。1000 步 MARE 贴线：TPU 3.722e-5，N 3.725e-5（0.9993 倍）；N 从 8 份换 16 份没变（`gpu-noise/results/g3x16-noise-vs-8.json` 的 MARE 比 1.0），上限由某一对决定，换一份 GPU 就可能翻成待定<br>比较单元：mini 1,900 步；w300 500 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 3.72e-5）｜参考：无 | 生产程序（逐步记录） | [3.00e-5（MARE，0.805×N） ✅](results/t1zc-l4.report.md) | [1.37e-5（MARE，0.618×N，N=2.22e-5） ✅](results/t1zc300-l4.report.md) |
| 索引器 KD loss<br>`L4.indexer_loss` | 逐步索引器 KD loss（各产出层平均 × 0.1） | `indexer_loss`，全程<br>比较单元：mini 1,000 步；w300 300 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 3.90e-4）｜参考：无 | 生产程序（逐步记录） | [2.94e-4（MARE，0.755×N） ✅](results/t1zc-l4.report.md) | [2.37e-4（MARE，0.9407×N，N=2.52e-4） ✅](results/t1zc300-l4.report.md) |
| grad_norm<br>`L4.grad_norm` | 逐步裁剪前全局梯度范数；分母用参照（2026-11-10 起，原为两边均值） | `grad_norm`，全程<br>比较单元：mini 1,000 步；w300 300 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 2.93e-3）｜参考：无 | 生产程序（逐步记录） | [2.32e-3（MARE，0.793×N） ✅](results/t1zc-l4.report.md) | [3.88e-4（MARE，1.009×N，N=3.85e-4） 🟡](results/t1zc300-l4.report.md) |
| param_norm<br>`L4.param_norm` | 逐步更新后的参数范数，两边读精确值：GPU `param_norm_fp64`，TPU `param_norm_exact`（每 128 个元素一段 FP32 求平方和、FP64 合成） | 全程<br>比较单元：mini 1,000 步；w300 300 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（最大单步 的 N = 2.15e-5）｜参考：带符号均值 ≤ 5e-6、最大单步 ≤ 1.25e-5 | 生产程序（逐步记录） | [3.06e-6（最大单步，0.142×N） ✅｜参✅](results/t1zc-l4.report.md) | [3.37e-8（MARE，0.177×N，N=1.90e-7） ✅｜参✅](results/t1zc300-l4.report.md) |
| 路由偏置曲线<br>`L4.router.bias_mean`<br>`L4.router.bias_std` | 逐步每层路由偏置的均值、标准差 | 第 1–6 层与 MTP 层。均值、标准差各自对 N 判：1000 步标准差 MARE 的 N 5.78e-3；均值在 0 附近，逐步相对差被除零放大，N 很宽（1000 步 MARE 1.82、单步最大 705；前 300 步 1.52、43.6），这条对偏置均值几乎没有分辨力（改用绝对差或按标准差归一：待定，待拍板）；w300 格的 40.1 是偏置均值的单步最大相对差<br>聚合：每步每层一个统计量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步 | ≤ N 过，> 2N 不过（带符号均值，各层取绝对值最大 的 N = 1.04）｜参考：无 | 生产程序（逐步记录） | [1.36（带符号均值，各层取绝对值最大，1.31×N，最差 带符号均值 1.31×N） 🟡](results/t1zc-l4.report.md) | [38.4（最大单步，0.88×N，N=43.6） ✅](results/t1zc300-l4.report.md) |
| QK-Clip<br>`L4.qkclip.max_logit` | 每步全局最大注意力 logit（各层含 MTP 各头最大 logit 的最大值，不含 sink）的相对差；被裁头数见健康信号一行 | TPU `tz-off-qkq`（现行 yml、base2 关，1000 步；不是另跑的运行：`analysis/derive_max_logit.py` 给 `tz-off-qkc` 的 metrics 每行加一列全局最大 logit；`tz-off-qkc` 与主 L4 的被测 `tz-off-rep` 1000 步 47 个逐步量逐位相同，见 `t1zo-determinism`，所以是同一条轨迹）对 GPU `g3-a`，噪声 `g3` 16 份两两 120 对；w300 列对 `g2` 一组<br>比较单元：mini 1,000 步；w300 300 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 1.86e-3）｜参考：无 | 生产程序（TPU `tz-off-qkq`：`tz-off-qkc` 的 metrics 加一列全局最大注意力 logit；GPU `g3-*`）逐步记录 | [2.02e-3（MARE，1.091×N） 🟡](results/t1zo-l4-qkclip.report.md) | [4.54e-2（最大单步，1.046×N，N=4.34e-2） 🟡](results/t1zo300-l4-qkclip.report.md) |
| 健康信号<br>`L4.health.nonfinite`<br>`L4.health.skipped`<br>`L4.health.grad_clip_trigger`<br>`L4.health.qk_clip_trigger`<br>… | NaN / Inf、跳步、梯度裁剪是否触发、QK-Clip 被裁头数、MoE 是否丢 token、loss 尖峰（协议第 4 节） | TPU `tz-off-qkc`（现行 yml、base2 关）对 GPU `g3-a`，1000 步<br>比较单元：2,000 步<br>聚合：逐步看 NaN / Inf 与突跳 | 逐位｜参考：无 | 生产程序（TPU `tz-off-qkc`，现行 yml、base2 关，1000 步；GPU `g3-a`）逐步记录 | [1000 步：TPU 非有限值 0 处、GPU 0 处 ✅](results/t1zo-health.results.json) | — |

非 ✅ 格子的原因：

- **grad_norm**（w300 🟡）：1000 步过（MARE r 0.81）；前 300 步 MARE N 3.85e-4，TPU 4.74e-4，r 1.23，待定。base2 开的上一代同一项已定责、已接受（同一状态单步 6 次里 5 次 TPU 更近 FP64），那套证据没在关 base2 的状态上重做（对齐细节 D2）。
- **路由偏置曲线**（mini 🟡）：两个窗口都过：1000 步均值 r 0.20、标准差 0.77；前 300 步 0.92、0.67。base2 开的上一代标准差 1000 步 1.14 倍，已定责、已接受（对齐细节 D23）。
- **QK-Clip**（mini 🟡、w300 🟡）：待定：1000 步 MARE N 1.86e-3、TPU 2.02e-3，r 1.09；前 300 步单步最大 N 4.34e-2、TPU 4.54e-2，r 1.05（base2 开的上一代：1000 步 1.02、前 300 步带符号均值 1.77，已定责、已接受，对齐细节 D29；1.02 是 8 份 28 对尺子的读数，按现行 16 份尺子推算约 1.00（推断：上一代 MARE 1.85e-3 ÷ 现行 N 1.86e-3），所以 1.02 → 1.09 不全是关 base2 变差，一部分是换了尺子）。关 base2 后同一状态单步的定责证据没有重做

### 程序与参照

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | mini |
|---|---|---|---|---|---|
| TPU 同设置重跑逐位（现行配置）<br>`DET.tpu.rerun.current` | TPU 同一份代码、配置、数据跑两份 1000 步，`metrics.jsonl` 的 47 个逐步量逐位比；只作记录，不改变噪声的取法（协议第 3 节） | 现行 yml（base2 关）：`tz-off-qkc` 对 `tz-off-rep`（同一代码包、不同节点池）；base2 开的上一代 `mz-g1` 对 `t1zg-b` 见 `t1mz-determinism`<br>比较单元：47 个字段<br>聚合：逐步、逐张量比两次运行，数不可复现的项 | 逐位｜参考：无 | 生产程序（现行 yml、base2 关两份 1000 步：`tz-off-qkc` 对 `tz-off-rep`）逐步记录 | [0 / 47 不同 ✅](results/t1zo-determinism.report.md) |
| TPU 同设置重跑逐位（旧配置）<br>`DET.tpu.rerun` | 同上，旧配置两组各两份 100 步、45 个逐步量 | `t1det-a/b`、`t1det2-a/b`<br>比较单元：45 个字段<br>聚合：逐步、逐张量比两次运行，数不可复现的项 | 逐位｜参考：无 | 生产程序（TPU 同设置两份） | [0 / 45 不同 ✅](results/hy4-mini-tpu-determinism.json) |
| TPU 存档续跑逐位<br>`DET.tpu.resume` | 第 500 步把整个训练状态存档，新进程换到另一个同型号同拓扑的切片上续跑到 1000 步，对不断开跑的同一配置逐位比（逐步量与最终状态张量的 sha256） | 现行配置（MaxText `b87a55aa`）：`x4-full` 不断开、`x4-s1` 跑到第 500 步存档、`x4-s2` 续跑；`x4-full` 与 `tz-off-rep` 1000 步主 loss、MTP、KD、grad_norm、param_norm 逐位相同<br>比较单元：1,000 步<br>聚合：逐步、逐张量比两次运行，数不可复现的项 | 逐位｜参考：无 | 生产程序（TPU 现行配置 1000 步：不断开一份，存档续跑一份，续跑换切片） | [0 / 1,000 不同 ✅](results/t1-resume-bitwise.json) |
| TPU 抓数不改数值<br>`DET.tpu.dump_no_effect` | 开全量抓数的运行与不抓的运行，前 30 步两边都记的逐步量逐位比（步时除外） | `mz-dump30`（base2 开那一代 yml，30 步；现行配置上的同类证据是 `tz-off-rep`（第 1 步全量抓数）对 `tz-off-qkc`（不抓，另开只记录的 `qk_clip_log_per_head`）1000 步 47 个逐步量逐位相同，见 L4「同设置重跑逐位」；第 1、30 步全量抓数：更新前后 FP32 主权重、Adam / Muon 动量、BF16 参数、路由偏置）对 `t1zg-b`（同一 MaxText 提交 `fd30fbfb`，不抓）<br>比较单元：47 个字段<br>聚合：逐步、逐张量比两次运行，数不可复现的项 | 逐位｜参考：无 | base2 开那一代 yml 30 步两份（`mz-dump30` 开全量抓数，`t1zg-b` 不抓）逐步记录 | [0 / 47 不同 ✅](results/t1mz-dump-determinism.report.md) |

## 逐元素函数的精度

逐元素函数在 TPU 上做了 ulp 检查（L1 表「逐元素函数」各行，`results/tpu-elementwise-ulp.json`）：XLA 默认的 FP32 exp / log / sigmoid / silu 是近似实现，ulp_max 105 / 62 / 83 / 86，超过 CUDA 文档给 GPU 的门槛。T1 打开 `transcendental_accuracy_highest`（台账 M32、M33；sigmoid / silu 的写法 M37，反向 M39），下表写「开关写法」的各处改走它，ulp_max exp 1、log 3、sigmoid 3、silu 4（`L1.elem.*.fp32_mt`），只有 log 超 CUDA 文档的 1 ulp（D14，已定责、已接受）；注意力内核里的 exp 不在开关范围内。rsqrt、relu、clamp、加乘与类型转换过。下表按代码列出每个函数在两边的计算精度；「检查」一栏说的是检查做没做，不是两边是否相同。

出处：TPU 为 MaxText 混元 4 集成分支 `b87a55aa`（现行比较 tz-off-rep 所用）在 T1 配置（`hy4-mini-t1.yml` + 模型 yml：`transcendental_accuracy_highest: true`、`sa_use_base2_exp: false`，七个按位置覆盖项（M38）都不设、跟随总开关，`transcendental_log_freebsd` 不开）下的代码路径，2026-11-11 按 `b87a55aa` 逐处核过 `accurate_math` 的调用。GPU 为 `frameworks/megatron/hy4_mcore.py` 与 Megatron-Core 0.20；GPU 的 sigmoid / silu 是 PyTorch eager 的 FP32 opmath，B200 实测最大 3.4 ulp（台账 G26）。两边的 ulp 不是同一把尺子：GPU 是 2e7 个 [−30, 30] 均匀输入、误差 ÷ ulp 的小数（`results/b200-sigmoid-silu-probe.json`），TPU 是合成与真实输入两组、离正确舍入值的整数 ulp 距离（`tpu-elementwise-ulp.json`）；表里并列只说明量级，不说明同输入下谁更准。表里只比前向：反向（M39）只量了 TPU，FP32 开关写法 sigmoid 反向 ulp_max 4056、silu 99468（被 1 − y 的抵消主导，rel_L2 2.6e-7、1.6e-7，`tpu-elementwise-ulp.json` 的 `backward`），GPU 反向没测。Transformer Engine（TE）融合算子按 TE 2.20.2 源码核（TE 版本 2026-11-11 实测，见 `setup.md`「软件版本」）。`accurate_math` 的调用点（`b87a55aa`，`git grep -n accurate_math -- src/maxtext`）：`layers/attention_mla.py` 1187、1240、1278、1285–1315（索引器 KD）、1529（输出门）；`layers/mhc.py` 472–473（iHC pre / post）、663（hc_head）；`layers/moe.py` 796（路由打分）、1110（路由专家 silu）；`layers/linears.py` 96（稠密 MLP、共享专家 silu）；`layers/multi_token_prediction.py` 413、656；`utils/max_utils.py` 768–770、`utils/vocabulary_tiling.py` 183、233、`trainers/pre_train/train.py` 261（交叉熵）。

| 函数 | 用在哪 | TPU（MaxText） | GPU（Megatron + TE） | 两边 | 检查 |
|---|---|---|---|---|---|
| sigmoid | iHC pre / post、hc_head | FP32，开关写法（`mhc.py` `gates`） | FP32（`Hy4HyperConnection`，autocast 关） | dtype 相同；TPU 开关写法 3 ulp，GPU 实测 ≤ 3.4 ulp（两边 ulp 口径不同，见上） | 逐元素 ulp（`L1.elem.sigmoid.fp32_mt` ✅）；hy4-mini 单算子 iHC 混合、写回与 hc_head（✅） |
| sigmoid | 路由打分 | FP32（`float32_gate_logits`），开关写法 | FP32（Megatron-Core `moe_utils`） | dtype 相同；TPU 开关写法 3 ulp，GPU 实测 ≤ 3.4 ulp（两边 ulp 口径不同，见上） | 逐元素 ulp（`L1.elem.sigmoid.fp32_mt` ✅）；hy4-mini 单算子 MoE 整块里间接比（✅） |
| sigmoid | 输出门 | FP32 里用开关写法算、舍成 BF16 后乘（`attention_mla.py`） | BF16 输入、BF16 输出（`torch.sigmoid(gate)`，`hy4_mcore.py`） | **不同**：算的精度 FP32 对 BF16（未处理，对齐细节第 5 节 D10） | 只在注意力整块里间接比；L3 第 1 步 linear_gate 梯度 R ≤ 0.99（以现行比较 tz-off-rep 为准） |
| silu | 稠密 MLP、共享专家、路由专家 | FP32 开关写法算（三处都跟随总开关）、输出舍 BF16，乘积再舍一次（`mlp_activation_round_per_op`） | BF16 逐算子（非融合 SwiGLU） | 舍入点相同（按算子舍入；TPU 用开关对上，台账 M10）；TPU 开关写法 4 ulp，GPU 实测 ≤ 3.4 ulp（两边 ulp 口径不同，见上） | 逐元素 ulp（`L1.elem.silu.fp32_mt` ✅）；稠密 MLP 的 L1（阶段 A 与 hy4-mini，✅）；专家在 hy4-mini 的共享专家、路由专家里间接比（✅） |
| clamp | 路由专家 SwiGLU（gate ≤ 10、up ∈ [−10, 10]） | FP32 里对 BF16 值截断（精确），边界处梯度照 torch.clamp（`mlp_activations_limit_inclusive_grad`） | BF16 `clamp` | 相同（截断不产生舍入） | TPU 上逐位（`L1.elem.clamp.fp32` ✅）；路由专家在 hy4-mini 单算子里间接比（✅） |
| rsqrt | 各 RMSNorm | FP32；scale 舍 BF16，输出只舍一次（`rmsnorm_single_rounding`） | TE 融合 RMSNorm：按 TE 2.20.2 源码（`rmsnorm_fwd_kernels.cuh`），FP32 里求均方、`rsqrtf`，乘 γ 后输出只舍一次（没设 `NVTE_NORM_FWD_USE_CUDNN` 时走这条内核，镜像的环境变量未核） | 舍入点相同（TPU 用开关对上，台账 M9）；rsqrt 两边都是近似实现 | L1 三个 RMSNorm（阶段 A，GPU 分母是 HF 写法） |
| rsqrt | iHC、hc_head 的归一化因子 | FP32 | FP32 | 相同；hc_head 先归一化再乘（TPU）对先乘再乘因子（GPU），舍入点不同 | L1 iHC 混合与 hc_head（阶段 A） |
| rsqrt | 索引器 k_norm（LayerNorm） | FP32 | FP32 输入（Megatron-Core `dsa.py`，`dsa_indexer_k_norm_fp32`）；按 TE 2.20.2 源码，LayerNorm 内核在 FP32 里求均值、方差，用 `rsqrtf`（CUDA 文档 ≤ 2 ulp），输出 FP32，出来后舍 BF16 一次（TE 版本按镜像推断；没设 `NVTE_NORM_FWD_USE_CUDNN` 时走这条内核，镜像的环境变量未核） | dtype 相同；rsqrt 两边都是近似实现 | L3 索引器梯度 |
| exp / softmax | 注意力 | FP32 在线 softmax，sink 作初始最大值；现行对齐配置 `sa_use_base2_exp: false` 用 exp（2026-11-11 起），之前各代用 exp2 | FP32 `torch.softmax`，sink 拼成一列 | dtype 相同、算法不同；注意力内核里的 exp 不在开关范围内，TPU 用默认的近似 exp（D14） | 只在注意力整块里间接比 |
| exp / softmax | KD 教师分布 | FP32，开关写法（`accurate_math.softmax`） | FP32（Megatron-Core `dsa_masking`） | dtype 相同；TPU 开关写法 exp 1 ulp，GPU ≤ 2 ulp（CUDA 文档） | 逐元素 ulp（`L1.elem.exp.fp32_mt` ✅）；L1 KD 输入（R ≤ 1.2） |
| log_softmax / log-sum-exp | 主损失、MTP 损失 | FP32，exp / log 走开关写法（交叉熵见 MaxText `94007054`） | FP32 | dtype 相同；TPU exp 1、log 3 ulp，GPU log ≤ 1 ulp（log 已接受，D14） | hy4-mini 单算子交叉熵（R：开关关时 2.95，开时 1.15，台账 M33，✅）；L4 loss、MTP loss |
| log_softmax / log-sum-exp | 索引器 KD（学生） | FP32，开关写法（`accurate_math.log_softmax`） | FP32 `torch.log_softmax` | dtype 相同；第 0 层 KD 的 LSE 对 FP64：开关关时平均低 0.95 格，开后 +0.025 格（`t1-step1-transcendental-accuracy.json`） | L1 KD 值（诊断）；L4 KD 过（M50 后，D1） |
| log(max(p, 1e-10)) | 索引器 KD（教师目标） | FP32，log 走开关写法，p = 0 的项不计（`indexer_kd_mcore_log`） | 同（Megatron-Core `dsa_indexer_loss`） | 写法相同（TPU 用开关对上，台账 M20）；TPU log 3 ulp，GPU ≤ 1 ulp（D14，已接受） | 逐元素 ulp（`L1.elem.log.fp32_mt` ❌，D14）；L1 KD 值（诊断） |
| relu | 索引器分数 | FP32（`indexer_fp32_scores`） | FP32 | 相同（TPU 用开关对上，台账 M23） | L1 KD 输入 |
| 除法 / 归一化 | 路由权重 w / (Σw + 1e-20)；KD 的 L1 归一化 / max(Σ, 1e-10)；NS 前 / max(‖x‖, 1e-7) | FP32 | FP32 | dtype 相同 | 路由权重在 hy4-mini 单算子 MoE 整块里间接比；NS 与 KD 有 MaxText 单测（台账 M17、M20）；Adam 的 m̂ / (√v̂ + eps) 里 TPU 的 FP32 开方与除法不精确，由 M34 修正（对齐细节 D17） |
| sqrt | QK-Clip 的 √scale；Muon 缩放；索引器 H^-0.5 · d^-0.5 | FP32 | FP32 | 相同（Muon 缩放两边都是 √max(1, fan_out / fan_in)，台账 G18） | QK-Clip 的 √scale：τ 100 下从未触发（scale 恒为 1）；τ 0.1 的 30 步里由「低阈值 QK-Clip：裁剪的算法」间接核到 1e-3；Muon 缩放、索引器缩放没有单独检查 |
| softplus | — | 混元 4 不用 | 不用（Megatron-Core 只有 `sqrtsoftplus` 路由打分一支，没选） | — | 不适用 |

## 已知的覆盖缺口

下面几处没有覆盖，或者读代码时发现两边写法不同、还没核实；它们不改变上面任何一格的判定，但限定了结论的范围。按级的缺口与补法见[对齐细节第 4 节「覆盖缺口」](DETAILS.md#覆盖缺口)。逐张量的比较在第 0、1 步（路由一致率另有第 8、30、100 步，优化器回放 8 步）；多步只比标量曲线。

| 缺口 | 现状 | 补做的办法 |
|---|---|---|
| iHC post 与写回没有单独比 | MaxText 不把 post 单独输出 | 加观测点，或在逐层回放里比流的状态 |
| MoE 路由的归一化与缩放没有单独的观测点 | 路由 logits、路由专家、共享专家与 MoE 整块都比过 | 加观测点 |
| 输出门 sigmoid 两边写法不同 | GPU 在 BF16 上算，TPU 在 FP32 里算再舍 BF16（对齐细节 D10），第 1 步梯度上没看到影响 | 看要不要加开关对上 |
| 现行配置的扰动运行只有 1000 步 3 份 | 以现行比较 tz-off-rep 为准：只改执行方式的 `tz-off-nolhs`、`tz-off-vm32`、`tz-off-vm64`（主 loss r 1.06、0.93、0.95，`gpu-noise/results/mini-base2-loo.json`），只作诊断；前 300 步已对全部 9 项逐项算了 r 与组内极差（`gpu-noise/results/exec-spread-300.json`、`exec-spread-300-g3.json` 的 `per_item_range`），1000 步除主 loss 外各项没有逐项看 | 需要时再补 |
| GPU 一侧自身的噪声与演练 | 只有 L4 尺子用的重跑两两（前 300 步 28 对、1000 步 120 对）；GPU 自比、确定性与故障演练没做（[`gpu-noise/`](gpu-noise/README.md)） | 照 gpu-noise 的计划补 |
| L1、L2、部分 L3 不在现行配置上 | 都是 base2 开时测的：L1 单算子回放（MaxText `2b005b3d`；只有低阈值 QK-Clip 逐头最大 logit 已在关 base2 上重测）、L2 逐层（`tw-l2-g`，没有 M42b、M50；第 2–4 层单层回放 `t1zg-l2-param-grads.json`）、路由一致率第 8 / 30 / 100 步。另外 L1 索引器 KD 输入是 t1z 设置、低阈值 QK-Clip 裁剪算法是 t1v 设置（`tw-qk-a`） | 在现行配置上重跑 |
| L3 优化器一步的噪声仍是 3 对 | 第 1 步梯度的尺子 2026-11-11 补到 `g2s-a … g2s-f` 6 份 15 对；优化器一步（`t1zo-l3-optim-step1`）的 N 仍用 `g2s-a/b/c` 三份的抓数（这项两边都低于 u 的按下限过，判定没有用到 N） | 需要时用 `g2s-d … f` 的全量抓数（已在 POC 桶里）重算 |

## 未列入本页的结果文件

01 实践里下面这些 janus 格式的结果文件不进清单；原始 JSON（探针、抓数核对等）由报告与案例直接引用，不在这里列。原因相同的并成一行。

| 结果 | 是什么、为什么不在清单里 |
|---|---|
| [`t1-l0-config`](results/t1-l0-config.results.json) | base2 开的上一代（`t1zg-a`）的配置逐项对照（69 项，不同 0，不含精度开关）；mini 列用现行比较的 `tc-l0-config` |
| [`t1-l0-settings`](results/t1-l0-settings.report.md)、[`t1s-l0-settings`](results/t1s-l0-settings.report.md)、[`t1t-l0-settings`](results/t1t-l0-settings.report.md)、[`t1u-l0-settings`](results/t1u-l0-settings.report.md)、[`t1v-l0-settings`](results/t1v-l0-settings.report.md)、[`t1w-l0-settings`](results/t1w-l0-settings.report.md)、[`t1z-l0-settings`](results/t1z-l0-settings.report.md)、[`t1zk-l0-settings`](results/t1zk-l0-settings.report.md) | 历代的设置核对（学习率逐步等于 1e-5，各代都过）；mini 列用现行比较的 `tc-l0-settings` |
| [`t1-l1-op-tpu`](results/t1-l1-op-tpu.json)、[`t1-l1-op-tpu-r2`](results/t1-l1-op-tpu-r2.json) | 没有 M40 的逐算子回放（第一版与 tw-opmini-c）；现行是含 M40 的 `t1-l1-op-tpu-r3` |
| [`t1-l2-flip-stats-diag`](results/t1-l2-flip-stats.results.json) | 同 `t1-l2-flip-stats`，总览用的只统计写法（状态记为诊断） |
| [`t1-l3-optim-step1`](results/t1-l3-optim-step1.results.json) | 上一代（base2 开，`mz-opt2`）的第 1 步优化器一步；2026-11-11 起现行换成 `t1zo-l3-optim-step1`（同一脚本、同一 GPU 与 FP64 输入，TPU 换成 `tz-off-rep` 的第 1 步全量抓数；用同一流程重跑 `mz-opt2` 与这份结果逐项相同） |
| [`t1-l4`](results/t1-l4.report.md)、[`t1-l4-nozm`](results/t1-l4-nozm.report.md)、[`t1s-l4`](results/t1s-l4.report.md)、[`t1t-l4`](results/t1t-l4.report.md)、[`t1u-l4`](results/t1u-l4.report.md)、[`t1v-l4`](results/t1v-l4.report.md)、[`t1w-l4`](results/t1w-l4.report.md)、[`t1z-l4`](results/t1z-l4.report.md)、[`t1zk-l4`](results/t1zk-l4.report.md)、[`t1zr-l4`](results/t1zr-l4.report.md)、[`t1zr300-l4`](results/t1zr300-l4.report.md)、[`t1zu-l4`](results/t1zu-l4.report.md)、[`t1zu300-l4`](results/t1zu300-l4.report.md) | 历代比较，判定见[历代比较](../../dashboard/history.md) |
| [`t1-optim-replay-adam-acc-tpu`](results/t1-optim-replay-adam-acc-tpu.json)、[`t1-optim-replay-tpu`](results/t1-optim-replay-tpu.json) | 优化器回放的旧版本（配置原样 / muon_adam_accurate 第一版）；现行是最终版 `t1-optim-replay-adam-acc-r2-tpu` |
| [`t1-qkclip-maxlogit`](results/t1-qkclip-maxlogit.results.json) | 上一代（base2 开，`mz-qk-a`）的低阈值 QK-Clip 第 1 步逐头最大 logit；2026-11-11 起现行换成 `t1zo-qkclip-maxlogit` |
| [`t1-step1-diag-tpu`](results/t1-step1-diag-tpu.json) | t1n 一代的第 1 步诊断（grad_norm、开关打开前的逐算子散布），只作诊断，已被之后各代的第 1 步取代 |
| [`t1-step1-grad-vs-ref`](results/t1-step1-grad-vs-ref.json)、[`t1t-step1-grad-vs-ref`](results/t1t-step1-grad-vs-ref.json)、[`t1v-step1-grad-vs-ref`](results/t1v-step1-grad-vs-ref.json)、[`t1z-step1-grad-vs-ref`](results/t1z-step1-grad-vs-ref.json) | 第 1 步梯度对 TPU 一侧产出的参照，与被测同源、不作定责；正式比较是对应的 `*-step1-grad-vs-fp64`（CPU FP64） |
| [`t1mz-determinism`](results/t1mz-determinism.report.md) | 上一代（base2 开）同设置重跑逐位（`mz-g1` 对 `t1zg-b`）；2026-11-11 起现行换成 `t1zo-determinism` |
| [`t1mz-health`](results/t1mz-health.results.json) | 上一代（base2 开，`mz-g1`）的健康信号；2026-11-11 起现行换成 `t1zo-health` |
| [`t1mz-l4-qkclip`](results/t1mz-l4-qkclip.report.md) | 上一代（base2 开，`mz-g1q`）的 L4 QK-Clip 曲线；2026-11-11 起现行换成 `t1zo-l4-qkclip` |
| [`t1mz300-l4-qkclip`](results/t1mz300-l4-qkclip.report.md) | 上一代（base2 开，`mz-g1q`）的 L4 QK-Clip 曲线前 300 步；2026-11-11 起现行换成 `t1zo300-l4-qkclip` |
| [`t1pr-gpu-self`](results/t1pr-gpu-self.report.md) | 生产配方臂 GPU 两份自比，对现行尺子量这套配方下 GPU 自己的波动；诊断 |
| [`t1pr-l4`](results/t1pr-l4.report.md) | 生产配方臂（学习率 2e-5 cosine warmup 200、YaRN 8、路由偏置零均值开）TPU 对 GPU，尺子借现行配置的 GPU 16 份（近似）；不在现行对齐配置上，不进签字判定 |
| [`t1pr-l4-gpu2`](results/t1pr-l4-gpu2.report.md) | 生产配方臂 TPU 对 GPU，尺子是同配方 GPU 两份的一对（只有一对，判不了）；不进签字判定 |
| [`t1pr-l4-gpu8`](results/t1pr-l4-gpu8.report.md) | 生产配方臂 TPU 对 GPU，尺子是同配方 GPU 8 份两两 28 对；不在现行对齐配置上，不进签字判定 |
| [`t1s-step1-grad-vs-fp64`](results/t1s-step1-grad-vs-fp64.json)、[`t1t-step1-grad-vs-fp64`](results/t1t-step1-grad-vs-fp64.json)、[`t1v-step1-grad-vs-fp64`](results/t1v-step1-grad-vs-fp64.json) | 历代的第 1 步整网；现行一代是 `t1zo-step1-train-grad-vs-fp64` |
| [`t1t-step1-ihc-experiments`](results/t1t-step1-ihc-experiments.json) | iHC 梯度三种 TPU 写法对 TPU 一侧产出的参照的比较，不作定责；M28 对 CPU FP64 的效果见 t1t / t1v-step1-grad-vs-fp64 |
| [`t1z-step1-grad-vs-fp64`](results/t1z-step1-grad-vs-fp64.json) | 更早一代（t1z 设置）的第 1 步整网；现行是 `t1zo-step1-train-grad-vs-fp64-n6`，base2 开的上一代是 `t1zg-step1-train-grad-vs-fp64` |
| [`t1zg-l0-config-precision`](results/t1zg-l0-config-precision.results.json) | 负对照：同一脚本加精度开关后对 base2 开的上一代 `t1zg-a` 核 106 项，恰好 1 项不同（`sa_use_base2_exp`），证明新加的核对能抓到 base2 |
| [`t1zg-step1-grad-vs-fp64`](results/t1zg-step1-grad-vs-fp64.json) | 旁证：base2 开那一代配置的诊断程序（`--diag-step1`）梯度；诊断程序与训练步分开编译，梯度相对差中位数 2.9e-4，那一代正式判定用训练步梯度 `t1zg-step1-train-grad-vs-fp64` |
| [`t1zg-step1-train-grad-vs-fp64`](results/t1zg-step1-train-grad-vs-fp64.json) | 上一代现行（base2 开，`mz-opt2`）训练步第 1 步梯度对 CPU FP64；2026-11-11 起现行换成 `t1zo-step1-train-grad-vs-fp64`（2026-11-11 尺子补到 6 份后是 `-n6`），判定见[历代比较](../../dashboard/history.md) |
| [`t1zga-l4`](results/t1zga-l4.report.md) | 上一代现行（`t1zg-a`，base2 开）的 1000 步 / 前 300 步比较；2026-11-11 起对齐配置关 base2，现行换成 `t1zo-l4`，判定见[历代比较](../../dashboard/history.md) |
| [`t1zga300-l4`](results/t1zga300-l4.report.md) | 上一代现行（`t1zg-a`，base2 开）的 1000 步 / 前 300 步比较；2026-11-11 起对齐配置关 base2，现行换成 `t1zo300-l4`，判定见[历代比较](../../dashboard/history.md) |
| [`t1zgb-l4`](results/t1zgb-l4.report.md) | 上一代 t1zg-a（base2 开）的重跑，M50 修订版（MaxText fd30fbfb）的重跑 t1zg-b；判定与 t1zga-l4 相同 |
| [`t1zla-l4`](results/t1zla-l4.report.md)、[`t1zla300-l4`](results/t1zla300-l4.report.md) | 验证：`t1zl-a`（`t1zg-a` 再开 M51 精确 log）1000 步 / 前 300 步；主 loss 变差，M51 没进对齐配置；历代比较，判定见[历代比较](../../dashboard/history.md) |
| [`t1zo-l4`](results/t1zo-l4.report.md) | 上一代现行（`tz-off-rep`，C 口径之前：M52、M58 不开）的 1000 步比较（过 7、待定 2）；2026-11-11 起对齐配置改 C 口径，现行换成 `t1zc-l4`，判定见[历代比较](../../dashboard/history.md) |
| [`t1zo-step1-train-grad-vs-fp64`](results/t1zo-step1-train-grad-vs-fp64.json) | 同一被测（`tz-off-rep` 训练步第 1 步梯度）对 GPU 第 1 步参照 `g2s-a/b/c` 三份 3 对的判定（❌：177 个超出 N、5 个 iHC 未定责）；2026-11-11 起尺子补到 6 份 15 对，现行换成 `t1zo-step1-train-grad-vs-fp64-n6`（PROTOCOL 变更记录同日「L3 尺子 6 份」一行，新旧判定并列），判定见[历代比较](../../dashboard/history.md) |
| [`t1zo300-l4`](results/t1zo300-l4.report.md) | 上一代现行（`tz-off-rep`，C 口径之前）的前 300 步比较（过 7、待定 2）；2026-11-11 起现行换成 `t1zc300-l4`，判定见[历代比较](../../dashboard/history.md) |
| [`t1zp-l4`](results/t1zp-l4.report.md)、[`t1zp300-l4`](results/t1zp300-l4.report.md) | 验证：生产建议组合整体跑一次（`x5-prod`，对齐细节「生产配置建议」）1000 步 / 前 300 步；不是对齐配置，判定见[历代比较](../../dashboard/history.md) |
| [`t1zs-l4`](results/t1zs-l4.report.md)、[`t1zs300-l4`](results/t1zs300-l4.report.md)、[`t1zt-l4`](results/t1zt-l4.report.md)、[`t1zt300-l4`](results/t1zt300-l4.report.md)、[`t1zub-l4`](results/t1zub-l4.report.md)、[`t1zub300-l4`](results/t1zub300-l4.report.md)、[`t1zuc-l4`](results/t1zuc-l4.report.md)、[`t1zuc300-l4`](results/t1zuc300-l4.report.md)、[`t1zud-l4`](results/t1zud-l4.report.md)、[`t1zud300-l4`](results/t1zud300-l4.report.md)、[`t1zv-l4`](results/t1zv-l4.report.md)、[`t1zv300-l4`](results/t1zv300-l4.report.md)、[`t1zw-l4`](results/t1zw-l4.report.md)、[`t1zw300-l4`](results/t1zw300-l4.report.md) | 诊断代与上一代的 1000 步 / 前 300 步比较（设置见 [`goldens.md`](goldens.md)）；历代比较，判定见[历代比较](../../dashboard/history.md) |
| [`t1zw-l0-settings`](results/t1zw-l0-settings.report.md) | base2 开那一代（`t1zw-a`）的设置核对（学习率逐步等于 1e-5，过）；mini 列用现行比较的 `tc-l0-settings` |
| [`t1zz-l4`](results/t1zz-l4.report.md)、[`t1zz300-l4`](results/t1zz300-l4.report.md) | 验证关 base2 的 `t1zz-a`（`t1zw-a` 再关 `sa_use_base2_exp`）1000 步 / 前 300 步；当时没采用（根源一览第 24 行）。2026-11-11 起关 base2 成为现行，现行 `tz-off-rep` 是 `t1zz-a` 加 M50 的重跑，主 loss、MTP、grad_norm 1000 步逐位相同，判定见 `t1zo-l4` |
| [`tq-alloff-l4`](results/tq-alloff-l4.report.md)、[`tq-base-l4`](results/tq-base-l4.report.md)、[`tq-base2-l4`](results/tq-base2-l4.report.md)、[`tq-m10-l4`](results/tq-m10-l4.report.md)、[`tq-m12-l4`](results/tq-m12-l4.report.md)、[`tq-m13-l4`](results/tq-m13-l4.report.md)、[`tq-m23-l4`](results/tq-m23-l4.report.md)、[`tq-m25-l4`](results/tq-m25-l4.report.md)、[`tq-m26-l4`](results/tq-m26-l4.report.md)、[`tq-m27-l4`](results/tq-m27-l4.report.md)、[`tq-m28-l4`](results/tq-m28-l4.report.md)、[`tq-m31-l4`](results/tq-m31-l4.report.md)、[`tq-m33-l4`](results/tq-m33-l4.report.md)、[`tq-m34-l4`](results/tq-m34-l4.report.md)、[`tq-m36-l4`](results/tq-m36-l4.report.md)、[`tq-m40-l4`](results/tq-m40-l4.report.md)、[`tq-m41-l4`](results/tq-m41-l4.report.md)、[`tq-m8-l4`](results/tq-m8-l4.report.md)、[`tq-m9-l4`](results/tq-m9-l4.report.md) | 单开关 A/B 300 步（以 `t1zu-a` 设置为底每次只关一个；`tq-alloff` 15 个全关）；只作诊断，代价与效果见对齐细节第 6 节 |
| [`tzoff-l0-config`](results/tzoff-l0-config.results.json) | 上一代现行（`tz-off-rep`）的配置逐项对照（106 项，不同 0）；2026-11-11 起 mini 列用 C 口径的 `tc-l0-config`（108 项） |
| [`tzoff-l0-settings`](results/tzoff-l0-settings.report.md) | 上一代现行（`tz-off-rep`）的设置核对（学习率逐步等于 1e-5，过）；2026-11-11 起 mini 列用 C 口径的 `tc-l0-settings` |
