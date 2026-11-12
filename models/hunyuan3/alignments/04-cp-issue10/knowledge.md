[janus](../../../../README.md) › [混元 3](../../README.md) › [实践 04](README.md) ｜ 知识库

# 混元 3 · 实践 04 · 要点与出处

给人读的是 [README](README.md)。这一页按问题列要点，每条带出处，方便直接回答「CP 会不会改数值」这类问题。

## CP 方案的数值

| 问题 | 要点 | 出处 |
|---|---|---|
| All-Gather、Ring、Ulysses 会不会让梯度变差？ | 不会。第 1 步整网梯度离 FP64 的距离是不开 CP 基线的 0.988–1.015 倍（CP2 / CP4，负载均衡开 / 关），只改片上内存上限 0.995–0.996 | [`results/step1-cp-vs-fp64.json`](results/step1-cp-vs-fp64.json) |
| USP 呢？ | 多一次 BF16 舍入：q/k/v 投影权重梯度离 FP64 远 2.9 倍（v 块 3 倍），其余张量不变；关负载均衡不变 | 同上；[`results/step1-qkv-split.json`](results/step1-qkv-split.json) |
| USP 的舍入在哪？ | 编译后的训练步里多出一个 all-reduce，结果是 `bf16[4096,8,128]` / `bf16[4096,64,128]`（k、q、v 投影权重梯度），在 BF16 里跨卡求和 | README 第 3 节；HLO 由 `XLA_DUMP=1` 的 `i10-st-usp4x` / `i10-st-bx` 得出 |
| 为什么只有 USP？ | `DenseGeneral` 先把权重转 BF16 再乘，权重梯度在语义上要舍成 BF16；平时紧跟着转回 FP32，被 XLA 省舍入消掉。USP 的切分在这中间插了一次跨卡求和，消不掉 | README 第 3 节；`maxtext` `linears.py` 的 `kernel_grad_fp32` 说明 |
| 怎么证明是这一处？ | 基线关掉省舍入（`--xla_allow_excess_precision=false`）同样远 3 倍；USP 打开 `kernel_grad_fp32`（诊断补丁）回到基线 | README 第 3 节表 |
| 修了在长跑上看得出来吗？ | 看不出来：E1 前 1510 步，USP 加补丁对不加补丁 r 0.80（尺子是 5 份只改编译开关的运行）；这次舍入比轨迹分叉小得多，只能靠第 1 步对 FP64 查 | [`results/e1-tpu-ruler-usp-fix-1510.json`](results/e1-tpu-ruler-usp-fix-1510.json) |
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
| 起跳步由什么决定？ | 跟学习率走：随机初始化峰值 3e-5 / 预热 300 → 275 步，预热 150 → 215，峰值 3e-4 → 105（10-29）；本实践 v5p 峰值 3e-4 → 105–125 | 一次实际排查；[`results/shape-A.json`](results/shape-A.json) |
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
