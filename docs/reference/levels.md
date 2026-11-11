[janus](../../README.md) › [文档](../README.md) › 参考 · 分级流程 ｜ 操作步骤见[第 4 章 逐级对齐](../guide/04-level-by-level.md)

# 分级流程：L0–L4

从「输入是不是同一份」查起，一级一级往上到「多步训练是不是同一条轨迹」。低一级没过，不进高一级。

这一页讲每一级为什么这样设计、跨平台时怎么判、不过时先查什么、用哪些检查类型实现。每一级具体比了哪些模块和张量、混元 3 的结果，在[检查清单](../../models/hunyuan3/CHECKS.md)；怎么跑在[第 4 章](../guide/04-level-by-level.md)。

| 级 | 问的问题 | 跨平台怎么判 |
|---|---|---|
| L0 地基 | 两边拿到的数据、权重、配置是不是同一份 | 逐位相同 |
| L1 单算子 | 同一个算子，TPU 是否不比 GPU 离 FP64 远 | FP64 R 比；优化器 FP64 重放 |
| L2 逐层 | 每层输入相同时，层内各观测点、反向、参数梯度 | 前向与反向 FP64 R 比；参数梯度对噪声 |
| L3 第 1 步整网 | 整网第 1 步的输出、全部梯度、更新后状态 | 输出 FP64 R 比；梯度与状态对 GPU 第 1 步重跑的噪声，超出的对 FP64 定责（2026-11-08 起；混元 3 当时用两侧执行噪声的并集） |
| L4 多步 | 多步训练的 loss、梯度范数、参数范数、路由偏置是不是同一条轨迹 | 对最终 golden 重跑的噪声 |

尺子为什么这样选，见 [`judging.md`](judging.md#哪一级用哪把尺子)「哪一级用哪把尺子」。

## 为什么低一级不过，高一级的结论不可信

每一级的比较都假设它的输入两边相同。输入不同，比出来的差就混进了上游的差，既判不了过不过，也定位不了。

- L0 不过（比如权重转换错了一个转置），L2 每一层都会不同：看到「第 0 层就超线」，分不清是这层算错了还是权重本来就不一样。
- L1 不过（比如路由 GEMM 的精度两边不同），L2 的路由选中集合会翻转，翻转又改变专家输出，误差一路放大到后面所有层。
- L4 的 loss 是所有差的总和：L4 超线只能说明有问题，说不出问题在哪；L4 过了也不能说明没问题（[`principles.md`](principles.md)「loss 只能证伪」）。

误差还会放大：第一阶段 GPU 自比，第 1 步反向的激活梯度从第 3 层的 1e-5 涨到第 0 层的 4.3e-3（rel_L2）。越往上游，一个小差越难从放大后的结果里倒推出来。

## 为什么逐层、逐张量只在第一步比

逐层比较要求两边权重相同。训练一步以后两边的权重就各自走了：第一阶段两次 GPU 默认设置运行走到第 1000 步，激活 rel_L2 最大 0.15，路由选中集合一致的 token 只剩 34%。这时候比激活，比的是两份不同的模型。

所以逐层、逐张量的比较只在第一步做；要把误差隔离到单层，再加逐层隔离（每层的输入取参照的）。多步只比 loss 与统计量。

步号：Megatron 的迭代号从 1 起，GPU 抓数里的第一步是 `step_1`。两边步号起点不同时，换算写进 `meta.json` 的 `steps.convention`，janus 按它对齐。

---

## L0 · 地基：数据、权重、配置

**做什么**

- 数据：每个 rank、每个 microbatch 的 tokens / labels / loss_mask / position_ids 记 sha256，逐步比；另比每步全局非 padding token 数、首步 loss。
- 权重：载入后全部参数逐位比（全量，不抽样），包括 FP32 主权重和浮点 buffer（如 expert_bias）；权重转换做往返（HF 转成框架格式再转回 HF）后逐张量比；dtype 按规则查。
- 配置：模型、优化器、调度器、并行、数据、版本逐项对照；「必须等于某个值」的项（dropout = 0）两边都断言。
- 结构：看实例化出来的模块树（路由层数、共享专家、q / k 归一化），不看配置文件。

**为什么这么设计**

- 主权重单独比：FP32 主权重上小于 BF16 半格的错，舍成 BF16 后看不见。BF16 参数逐位相同、第 1 步前向也逐位相同，第 2 步才分叉（第一阶段演练 `f-weight`，[`phase1-faults.md`](../../models/hunyuan3/alignments/00-gpu-noise-and-drills/faults/phase1-faults.md)）。
- dropout 要「等于 0」，不是「两边相同」：两边一起开着 dropout 也是「相同」，但 RNG 不同，结果必然不同。
- 配置读运行时实际生效的值：第一阶段发现调度器的默认权重衰减 0.033 覆盖了配置里的 0.1，只看配置文件发现不了。
- 往返对上只能证明转换可逆。映射表本身错了（两个方向错得互逆）往返查不出，要靠 L2 逐层比。

**准出**：全部逐位相同；配置对照零差异，或每一项差异都写明理由。

**不过时先查**：依次查数据管线（分词器、packing、EOS 是否计 loss、读取顺序与种子）、权重转换（转置、拆分、拼接、dtype）、配置默认值是否被别处覆盖。

**检查类型**：[`data_fingerprint`](checks/data_fingerprint.md)（输入指纹、续训数据位置）、[`checkpoint_equal`](checks/checkpoint_equal.md) 与 [`bitwise_equal`](checks/bitwise_equal.md)（参数、主权重、buffer、权重转换）、[`dtype_manifest`](checks/dtype_manifest.md)、[`config_diff`](checks/config_diff.md)、[`artifact_check`](checks/artifact_check.md)（模型结构）、[`coverage`](checks/coverage.md)（往返张量集合）。数据加载器两边一致、权重直接互转这两项生产能力的设计与状态见 [`design/0002-l0-data-weights.md`](../dev/design/0002-l0-data-weights.md)。

## L1 · 单算子：真实输入回放与 FP64 定责

**做什么**

- 从训练中抓下某个算子的真实输入，单独回放这个算子：先和训练时抓下的输出比（同平台应当逐位），再和 FP64 比。
- 跨平台：每一侧的真值都从这一侧自己抓到的算子输入在 FP64 里算，比 R = 被测离自己的真值 ÷ GPU 离自己的真值。上游的差不进来，只看算子本身。
- 降精度复现：在 CPU 上模拟一边的舍入方式（如 GPU 张量核的累加方式），逐位命中对方的输出。
- 优化器：用读出的 FP32 主权重、梯度、实际生效的超参，在 FP64 里重放一步 AdamW，比更新量和二阶动量。
- 路由：同一份输入、权重、expert_bias，在 FP64 里重算 top-k，看选中集合翻了没有。

**为什么这么设计**

- 回放先对训练逐位：证明回放的就是训练里那个程序。第一阶段发现，配置 `moe_router_dtype=fp32` 时 Megatron 的路由 GEMM 实际是「BF16 入、FP32 出」，不是「先升 FP32 再乘」；后一种写法与训练逐位相同率只有 0.002%–0.05%（`phase1-l1` 的 `L1.gemm.router.upcast_fp32`）。照想当然的写法去对，参照本身就错了。
- 用 FP64 定责、不用 GPU 当标准答案：两边离 FP64 一样远是平台固有；TPU 远是 TPU 的问题；TPU 近、GPU 远说明 GPU 写法本身就粗（[`principles.md`](principles.md#对齐的目标以-gpu-为参照)）。
- 优化器比更新量，不比更新后的参数：参数比更新量大约三个数量级，比参数会把更新错藏进舍入里。
- 降精度复现命中，说明两边的差完全由已知的舍入方式决定，可以定责为平台固有。

**准出**：被测算子离 FP64 不比 GPU 远（R ≤ 1.2），或降精度复现逐位命中；优化器重放的更新量 rel_L2 ≤ 5e-3（GPU 实测单步 1.79e-3、连续 8 步 3.37e-3）。5e-3 只对混元 3 的 AdamW 成立；优化器本身离 FP64 就远时（混元 4 的 Muon（BF16 Newton–Schulz）GPU 自己离 FP64 1.2%–13%（R 最差的那个张量，第 6 步第 0 层 `weights_proj`，GPU 读数 1.36e-2）），整个优化器回放（含 Adam 管的小张量）改按 R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64) ≤ 1.2 判（混元 4 [检查清单](../../models/hunyuan4/alignments/01-mini/CHECKS.md) `L1.optim_adam_r2`）。

**不过时先查**：先查回放写法是不是训练走的那条代码路径，再查两边的精度设置（矩阵乘默认精度、TF32），最后查是否多舍入了一次。R ≈ 1.41 是多一次同量级舍入的特征。

**检查类型**：[`op_replay`](checks/op_replay.md)（单算子回放、R 比、初等函数、降精度复现、极端输入、平局、VJP）、[`optimizer_step`](checks/optimizer_step.md)、[`discrete_agreement`](checks/discrete_agreement.md)（`reference: fp64_replay`）。GPU 回放要 torch 与 Transformer Engine，由脚本 [`l1_ops.py`](../../models/hunyuan3/frameworks/megatron/l1_ops.py) 直接调 `op_replay` 写结果；TPU 侧由 [`fwd_rratio.py --mode ops`](../../models/hunyuan3/analysis/fwd_rratio.py) 在 CPU 上算真值与 R。全部用例与第一阶段结论见 [`design/0003-l1-op-replay.md`](../dev/design/0003-l1-op-replay.md)。

## L2 · 单层：逐层、逐子模块

**做什么**

- 第一步，同一份权重、同一个 batch，比每层每个子模块（qkv、q / k 归一化、核心注意力、输出投影、MLP 前归一化、路由、共享专家、MLP 输出、层输出）的前向输出、层输入的梯度、逐参数梯度、路由选中集合。
- 逐层隔离：第 k 层的输入用参照的 `layers.k.in`，单独编译这一层。TPU 侧由 `layer_replay.py` 做；GPU 侧叫逐层注入（训练脚本 `--inject-from <参照运行>`）。
- 串跑找台阶：整网一起跑，找第一个从逐位相同变成不同的点。适用于两边本应逐位的场合，如只换并行方式。
- 跨平台路由回放（把参照选中的专家喂给被测，切断选专家的累积影响）：混元 4 第 1 步已有「灌 GPU 的选择」的诊断，只用来拆差，不进判定（[差异清单 G12](tpu-vs-gpu-differences.md#g12-第-1-步正式判定用-tpu-自选的路由灌-gpu-的选择只作诊断)）。它和 GPU 生产 golden [`rc`](../glossary.md#rc) 里的「路由回放」不是一回事：后者是同一次运行里，反向重算沿用自己首次前向选中的专家，两侧都已实现（[第 2 章 2.2](../guide/02-gpu-golden.md#22-为什么最终-golden-是生产程序只修一处)）。

**为什么这么设计**

- 整网串跑时，上一层的小差会改变下一层的输入，看到第 3 层超线说明不了第 3 层有错。隔离把每层的输入钉成同一份，超线的那层就是出错的那层。第一阶段演练（`phase1-inject`）：第 2 层权重埋错，注入后只有第 2 层的输出不同。
- 专家并行分发后的缓冲区（`*.mlp.experts.out`）不能跨运行逐位比：它的行数取决于别的 rank 发来多少 token。比 token 空间的 `mlp.out`。
- 跨平台前向不对噪声判：GPU 默认设置下第 1 步前向逐位可复现，两份 GPU 之间的噪声上限是 0（[`regression.md`](regression.md) 第五节），改用 FP64 R 比；路由按「被测对 FP64 的翻转率不超过 GPU 的 1.2 倍」判。
- 跨平台反向也用 FP64 R 比：隔离回放里两侧只改执行方式路由都不翻，GPU 重跑与 TPU 执行扰动的并集都量不到路由翻转（[`judging.md`](judging.md#第-1-步张量两侧执行噪声的并集)）。

**准出**：GPU 对 GPU 时各项在噪声参照内；TPU 对 GPU 时前向、反向 R ≤ 1.2，路由翻转率在线内，参数梯度在 GPU 第 1 步噪声内或超出部分已对 FP64 定责（混元 3 当时用两侧执行噪声的并集）。

**不过时先查**：先找哪一层第一个超线，对这一层做隔离，确认超线是本层的；再查这一层的权重映射（转置、q / k 归一化的分组布局）；最后回到 L1 回放这一层的算子。

**检查类型**：[`tensor_tolerance`](checks/tensor_tolerance.md)（GPU 自比的前向、反向、参数梯度）、[`op_replay`](checks/op_replay.md)（跨平台的 FP64 R 比）、[`discrete_agreement`](checks/discrete_agreement.md)（路由）、[`bitwise_equal`](checks/bitwise_equal.md)（串跑找台阶、逐层注入、截断前缀、只换并行方式）、[`replica_consistency`](checks/replica_consistency.md)（DP 副本）。

### 隔离融合与逐算子执行（TPU 侧）

TPU 是整图编译。「哪一层先不同」可能不是算子本身不同，而是编译器把它和前后的算子融在了一起。

下面都是测试模型或玩具模型上的结论。

#### 融合怎么改了数值、怎么隔开

跨算子融合会改求和。DP 与 FSDP 第一处不逐位是注意力后 RMSNorm，原因是它的平方和被融进了输出投影 matmul 的输出端，两边操作数的布局不同。RMSNorm 本身两边逐位相同。

要隔开就加 `jax.lax.optimization_barrier`：在待测模块的边界插一行，或把前后两段拆成两个 `jax.jit` 调用。前半段的结果与不加时逐位相同，后半段的融合被切断。开了 `--xla_tpu_aggressive_opt_barrier_removal=true` 时 barrier 会被移除，排障时设成 `false`。barrier 本身改了程序，隔开后的结论只用于定位，不当生产图的数值（同 [`docs/reference/dump.md`](dump.md) 第 2、3 条）。

同一个算子在前向与重算里也可能编得不同。同一个 matmul 在前向和 `jax.checkpoint` 重算里拿到的片上内存不同，后端因此选了不同的分块（`window_config`），输入逐位相同、输出仍差 1–2 ULP。没有全局开关能让两份逐位一致，要逐位就保存、不重算。

#### 逐算子执行能做什么、不能做什么

逐算子执行（`jax.disable_jit()`）不是绕过 XLA。每个原语仍由 XLA 单独编译并做 SPMD 分区。它把所有融合连同布局一起去掉，所以能在单层上验证「差来自融合」，但定位不到具体哪一步。它和 jit 的结果本身就不同，也不代表更准。

整模型逐算子执行目前跑不起来。MaxText 训练步在 `disable_jit` 下有四处阻塞：

1. 复数逐元素运算「复制 × 切分」时 libtpu 编译崩溃，yarn RoPE 会撞上。
2. `shard_map` 内的原语每次调用都重新编译，megablox 与注意力 kernel 都在 `shard_map` 里。
3. `shard_map` 内的 remat offload 报错。
4. megablox `gmm` 不接受 Python int 的 `group_offset`。

绕开之后，重算策略、offload 这类只改执行方式的改动在逐算子下能逐位不变，jit 下不能。这是验证扰动臂「不改数学」的一条路，等上游修好再用。

没有不经 XLA 编译的 TPU 执行路径，只能在 jit 和逐算子之间选。直接执行 HLO 的需求尚未解决。

## L3 · 第一步整网

**做什么**：同一份检查点、同一个 batch、关 dropout，第一步的：

- 逐 token loss、logits（rel_L2 与 KL）；
- 全部参数梯度（rel_L2、cosine、范数比、标准差比、零元素比例、逐元素相对误差 p99）与梯度 ULP 分布（只诊断）；
- 更新后的 FP32 主权重、Adam 两个动量、更新后的 expert_bias；
- 「假 FP32」：标着 FP32、低 16 位却几乎全是 0 的张量；
- 覆盖率：该比的张量都比到了。

**为什么这么设计**

- 梯度配范数比：cosine 对整体缩放完全看不见，梯度整体乘 1/8 时 cosine 仍是 1。第一阶段演练 `f-gscale`（抓数之前把一个参数的梯度乘 1/8）靠范数比报警。
- 验证更新后参数要挑学习率大于 0 的一步：预热第一步学习率为 0 时参数不动，什么也验证不了。
- 首步 loss 差单独给门槛，不进噪声判定：GPU 自比时第一步前向确定，差恒为 0。
- 跨平台的梯度与状态，噪声只用 GPU 第 1 步重跑；整步里近似平局的路由翻转、舍入位置的差 GPU 重跑量不到，超出的部分逐张量用 FP64 整步参照定责，看谁离精确更远（[`judging.md`](judging.md#第-1-步张量两侧执行噪声的并集)，2026-11-08 起；混元 3 当时用「GPU 重跑 ∪ TPU 只改执行参数的重跑」）。
- 跨平台的 logits 与逐 token loss 用整网只前向的回放比 FP64 R：加钩子会改 TPU 的融合（第 4 章 4.8）。

**准出**：上面各项过；目录写了 `first_step_tol` 时首步 loss 差在线内（没写就只报不判）；不要求跨平台逐位；超出噪声的项已用 FP64 定责并确认不累积。

**不过时先查**：依次查 loss 的约定（平均方式、loss_mask、EOS）、embedding 与输出层（是否共享、转置）、梯度缩放（范数比偏离是否接近某个整数倍，如 microbatch 数），最后回到 L1 做优化器重放。

**检查类型**：[`tensor_tolerance`](checks/tensor_tolerance.md)、[`op_replay`](checks/op_replay.md)（对 FP64）、[`ulp_distribution`](checks/ulp_distribution.md)（只诊断）、[`fake_fp32`](checks/fake_fp32.md)、[`coverage`](checks/coverage.md)。

## L4 · 多步训练

**做什么**：同一份检查点起步、同一个数据顺序，跑几百到上千步，逐步比：

- loss 的首步差、带符号均值、MARE、最大单步：对事先指定的一份基线 golden 算，其余几份 golden 只用来造噪声带（[`judging.md`](judging.md#golden-怎么用几份只造噪声带比较只对一份)）；
- 学习率（逐步逐位或 ULP 容差）：它是登记的常数，归 L0 设置核对，不计入 L4 的过 / 不过 / 待定，检查 id 仍是 `L4.lr`；
- 梯度范数（裁剪前）、参数范数；
- 每层 expert_bias 的统计量、专家负载的变异系数；
- 健康信号：非有限值、突跳。

**为什么这么设计**

- 平均会盖住尖峰，所以最大单步同报。
- 带符号均值是判定量：随机的差正负抵消，留下的是固定方向的偏移。
- 路由偏置单独比：它直接反映偏置更新有没有按设计走。第一阶段 `f-bias`（偏置更新率置 0）时，第一步更新后的偏置和逐步偏置统计都报警。
- 健康检查看梯度范数，不能只看 loss：第一阶段在第 5 步的一个梯度元素里埋 NaN，梯度范数第 5 步就非有限，loss 那一步还正常；框架没有跳过这次更新，第 6 步 loss 才变 NaN，那时参数已经坏了（[`phase1-nan.yaml`](../../models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/phase1-nan.yaml) 的注释）。
- golden 自己有系统偏差时，先修掉偏差得到最终 golden，再分两层：第一层对最终 golden、必须过；第二层对生产 golden、写成已解释差异（[`judging.md`](judging.md#golden-本身有系统偏差时修掉偏差再按一般规矩比)）。

**准出**：loss、梯度范数、参数范数、路由统计对基线 golden 在噪声内（同设置噪声参照至少 3 对；签字前至少 6 份 15 对，长窗口要更多，混元 4 的 1000 步用 16 份 120 对，见 [`judging.md`](judging.md#参照要几对)）；健康信号无报警；学习率逐步在容差内（归 L0 设置核对）。

**不过时先查**：依次查学习率曲线与实际生效的权重衰减、路由偏置更新率、梯度通信与主权重 dtype，最后回到 L3 看第一步的梯度。

**检查类型**：[`multi_golden`](checks/multi_golden.md)（loss）、[`loss_series`](checks/loss_series.md)（梯度范数、参数范数；`exact` 时做确定性重跑）、[`scalar_series`](checks/scalar_series.md)（学习率）、[`router_stats`](checks/router_stats.md)、[`health_signals`](checks/health_signals.md)、[`overlap_replay`](checks/overlap_replay.md)（存档续训对连续训练）、[`determinism`](checks/determinism.md)（加抓数不改数值）。

分项 loss（aux / z）没有检查：混元 3 的 golden 与 TPU 配置都把辅助损失、z-loss 系数设成 0（[`../../models/hunyuan3/config-alignment.md`](../../models/hunyuan3/config-alignment.md)「辅助损失 / z-loss」行）。开了以后每项加一条。

---

## 换了软硬件版本怎么办

升级 jax / libtpu、换芯片、换镜像，都当作一次新的被测：先跑日常回归（[第 7 章](../guide/07-daily-regression.md)）的回归门，它对上一次通过的 TPU 运行比，数学不变的改动要求逐位相同；再按需重跑 L2–L4。换 libtpu 基本不会逐位，而且不能假定只换了求和顺序：混元 4 上 0.0.46 → 0.0.52，「RMSNorm 多舍一次」从 r 1.06 变成 5.50。做法是新版本先建 golden、和旧 golden 比，再把已知敏感项（混元 4 是 TPU 自比的六件，加上对齐配置里补 GPU 舍入点的 M36、M52、M58）重测一遍（[`libtpu-052.md`](../../models/hunyuan4/alignments/04-tpu-self/libtpu-052.md)），并扫一遍编译器静默算错的特征（[差异清单 F8](tpu-vs-gpu-differences.md#f8-jit-静默算错dus-和读同一缓冲的算子被合进一颗多输出-fusion)）。回归门的设计、数值变了以后要重跑哪几级，见 [`regression.md`](regression.md#九升级-jax--libtpu--kernel-库)。

---

[回文档索引](../README.md) ｜ [回用户指南](../guide/README.md)
