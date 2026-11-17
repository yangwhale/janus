[janus](../../../../README.md) › [混元 3](../../README.md) › [实践 04](README.md) ｜ 知识库

# 混元 3 · 实践 04 · 要点与出处

给人读的是 [README](README.md)。这一页按问题列要点，每条带出处，方便直接回答「CP 会不会改数值」这类问题。
「阶梯实验」一节是现行结论；其余各节出自阶梯实验之前的排查（[`history.md`](history.md)），条件是真实权重截 4 层、FSDP 基线。

## 阶梯实验（16 层随机初始化、纯数据并行 8 基线、恒定学习率）

| 问题 | 要点 | 出处 |
|---|---|---|
| 纯数据并行的基线有没有精度问题？ | 有：词嵌入反向的 scatter-add 每卡 4096 个下标，没过排序门槛（15104），BF16 里每加一次舍一次，第 1 步词嵌入梯度离 FP64 2.57e-2，是 FSDP 的 6.0 倍，整网 1.96 倍 | [`results/ladder/step1-lbd-L0.json`](results/ladder/step1-lbd-L0.json)、[`hlo-audit.json`](results/ladder/hlo-audit.json) |
| CP 臂呢？ | DP4 × CP2 每卡 8192 个下标，同样不排序：All-Gather / Ring 7.1 倍、Ulysses 11.6 倍；DP2 × CP4 与 USP 每卡 16384 个，排序，和 FSDP 一样 | 同上 |
| 怎么修？ | `embedding_shard_over_data=True`（hunyuan@5d1b0e95 起）：所有臂 32768 个下标、排序，整网 0.99–1.03 倍 | [`step1-lbd-L2.json`](results/ladder/step1-lbd-L2.json) |
| 关 excess precision 呢？ | 更糟：排过序的 scatter 退回每加一次舍一次；FSDP 的词嵌入梯度离 FP64 8.52e-2（开着时 4.30e-3），CP4 臂 5.76e-2 | [`step1-lbd-L0p.json`](results/ladder/step1-lbd-L0p.json) |
| CP 还有别的额外舍入吗？ | 有但很小：不开 `dense_fp32_accumulation` 时 CP 的切分让 q/k/v、输出投影、稠密 MLP、共享专家、norm 的权重梯度跨 context 轴 BF16 求和（15–26 个 BF16 all-reduce，基线 4 个），对基线只远 1–2%；L4 后对基线 0.99–1.00 倍 | [`hlo-audit.json`](results/ladder/hlo-audit.json)、[`step1-lbd-L4.json`](results/ladder/step1-lbd-L4.json) |
| 修完以后长跑能看出来吗？ | 看不出：6 级 × 7 个 CP 臂后段 r 都 ≤ 1.62；只改片上内存上限就能让后段偏 1e-3–5e-3（相对），修复不让它收窄 | [`ruler-lbd-*.json`](results/ladder/)、[`chaos-lbd.json`](results/ladder/chaos-lbd.json) |
| 分叉在哪？ | 只改编译开关的运行第 55–79 步分开，之前相对差 2e-5 量级；学习率 3e-4 时第 55–56 步有一次 loss 尖峰 | [`chaos-lbd.json`](results/ladder/chaos-lbd.json)、[`chaos-lad.json`](results/ladder/chaos-lad.json) |
| 第 1 步和学习率有关吗？ | 无关：两组学习率的同名运行第 1 步 206 个一阶动量逐位相同 | README 第 3 节 |
| 自己的配置过不过排序门槛？ | 每卡下标数 = 序列长 × 每卡微批 × CP 度，和表行数的 1/8 比（libtpu 0.0.46）；别的版本离线编译看那次 scatter 有没有 `indices_are_sorted=true`，[`tools/hlo_audit.py`](tools/hlo_audit.py) | README 第 2 节 |

## CP 方案的数值

| 问题 | 要点 | 出处 |
|---|---|---|
| All-Gather、Ring、Ulysses 会不会让梯度变差？ | 不会。第 1 步整网梯度离 FP64 的距离是不开 CP 基线的 0.988–1.015 倍（CP2 / CP4，负载均衡开 / 关），只改片上内存上限 0.995–0.996 | [`results/step1-cp-vs-fp64.json`](results/step1-cp-vs-fp64.json) |
| USP 呢？ | 多一次 BF16 舍入：q/k/v 投影权重梯度离 FP64 远 2.9 倍（v 块 3 倍），其余张量不变；关负载均衡不变 | 同上；[`results/step1-qkv-split.json`](results/step1-qkv-split.json) |
| USP 的舍入在哪？ | 编译后的训练步里多出一个 all-reduce，结果是 `bf16[4096,8,128]` / `bf16[4096,64,128]`（k、q、v 投影权重梯度），在 BF16 里跨卡求和 | README 第 3 节；HLO 由 `XLA_DUMP=1` 的 `i10-st-usp4x` / `i10-st-bx` 得出 |
| 为什么只有 USP？ | `DenseGeneral` 先把权重转 BF16 再乘，权重梯度在语义上要舍成 BF16；平时紧跟着转回 FP32，被 XLA 省舍入消掉。USP 的切分在这中间插了一次跨卡求和，消不掉 | README 第 3 节；`maxtext` `linears.py` 的 `kernel_grad_fp32` 说明 |
| 怎么证明是这一处？ | 基线关掉省舍入（`--xla_allow_excess_precision=false`）同样远 3 倍；USP 让 q/k/v 走 `kernel_grad_fp32`（先用诊断补丁，后用正式开关，两者逐位相同）回到基线 | README 第 3 节表 |
| 正式开关是什么、怎么用？ | `qkv_proj_wgrad_fp32=True`（提交 `c9f5ccdf`，默认关）：Attention 的 query / key / value（或融合 qkv_proj）权重梯度以 FP32 落地，前向不变。关着时混元 3 FSDP 与 USP 的程序与基底相同；开着时 USP 程序与诊断补丁逐字节相同，v5p 第 1 步逐位相同，q/k/v 一组 R 2.89 → 1.09 | README 第 3 节；[`results/usp-switch-hlo.json`](results/usp-switch-hlo.json)、[`results/step1-usp-switch.json`](results/step1-usp-switch.json) |
| 混元 4 要不要这个开关？ | 不要：混元 4 是 MLA + sink + DSA indexer，MaxText 的 USP 只支持 global 注意力、不支持 sink 与 indexer，混元 4 开不了 USP | README 第 3 节 |
| 修了在长跑上看得出来吗？ | 看不出来：E1 3000 步，USP 开开关对不开，带符号均值 / MARE 的 r 0.15 / 0.80（尺子是 5 份只改编译开关的运行）；最大单步 2.94 只来自不开开关那条第 1598 步那一步（轨迹敏感的一批）。这次舍入比轨迹分叉小得多，只能靠第 1 步对 FP64 查 | [`results/e1-tpu-ruler-usp-switch.json`](results/e1-tpu-ruler-usp-switch.json) |
| Ring 的第 1 步 loss 偏吗？ | 不明显：离 FP64 +6.2e-5（负载均衡关时 +1.8e-5），基线 −1.8e-5，Ulysses CP4 −9.6e-5，都在 ±1e-4；GPU golden 是 −1.46e-4 | [`results/step1-cp-vs-fp64.json`](results/step1-cp-vs-fp64.json) |
| `train_hy3.py` 能回放 CP 吗？ | 本实践起可以：放宽分片行数检查，补上负载均衡重排 | [`setup.md`](setup.md#train_hy3py-回放支持上下文并行本实践加的) |

## base2

| 问题 | 要点 | 出处 |
|---|---|---|
| base2 开在混元 3 上偏哪边？ | 截 4 层第 1 步 loss 比 FP64 高 6.9e-4（关时 −1.8e-5），整网梯度远 1.76 倍；混元 4 上是偏低，方向随模型 | [`results/step1-cp-vs-fp64.json`](results/step1-cp-vs-fp64.json)；[混元 4 · 06](../../../hunyuan4/alignments/06-rw3/README.md) |
| Ring 对默认基线一开始就偏，是 CP 的问题吗？ | 不是：Ring 内核不走 base2，基线走。FineWeb 第 0 步 Ring 对 base2 开的基线 −7.55e-4，对关 base2 的基线 −8.3e-5 | [`results/series/`](results/series/) 的 `RSB`、`RSBnb`、`RSRG2` |
| base2 不一致对长跑有什么影响？ | 分叉提前到第 24 步（一致时 E1 第 72–82 步，学习率恒定 1e-5 时 89–95 步）；学习率恒定 1e-5 时后段只剩约 +1e-4 | [`results/shape-B.json`](results/shape-B.json)、[`results/shape-E1.json`](results/shape-E1.json) |

## 三段形态与后段偏移

| 问题 | 要点 | 出处 |
|---|---|---|
| 三段形态在什么条件下出现？ | 真实权重 + 学习率先升到 1e-4 再余弦降到 0；只改编译开关也出现。随机初始化只出前两段；学习率恒定 1e-5 没有中段 | [`results/shape-A.json`](results/shape-A.json)、[`results/shape-B.json`](results/shape-B.json)、[`results/shape-E1.json`](results/shape-E1.json) |
| 起跳步由什么决定？ | 跟学习率走：随机初始化峰值 3e-5 / 预热 300 → 275 步，预热 150 → 215，峰值 3e-4 → 105（10-29）；本实践 v5p 峰值 3e-4 → 105–125 | 原问题 10-29；[`results/shape-A.json`](results/shape-A.json) |
| 后段固定偏移是不是 CP 固有的？ | 不是，跟着轨迹走：同一个 Ring 臂换编译开关，偏移会变号 | README 第 6 节 |
| 同程序重跑会不会差？ | 不会：3000 步逐位相同；16 片存档续跑也逐位相同 | [`results/shape-A.json`](results/shape-A.json)（`rerun`）；README 第 7 节 |
| 判 CP 有没有问题该和谁比？ | 和同调度、只改编译开关的几条 TPU 运行比（本实践 5 条，两两 10 对），不能拿单次偏移和 0 比 | README 第 5 节；[`cp_tpu_ruler.py`](../../analysis/cp_tpu_ruler.py) |
| 已经排除的「波动变小」原因 | 学习率衰减到小、数据重复、单批次、小数据多轮、各臂从同一存档续训 | README 第 7 节 |

## 资源与限制

| 问题 | 要点 |
|---|---|
| 真实权重截 4 层要几片 v5p？ | FSDP 8 放得下（每设备约 51 GB），单步 1.46 s；纯 DP 放不下 |
| 抢占式节点被收回怎么办？ | 运行从头重跑，不拼接（本实践长跑被收回 3 次；最后一次是 USP 修法那条，按收回前的 1510 步判） |
| 为什么没有 32 芯片、80 层？ | 本项目最多 16 芯片；真实权重只有截 4 层这一份检查点 |
