[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 09

# 案例 09：路由偏置均值 r 19.1——协议要求两边零均值更新，原版 Megatron-Core 没有这个更新

> **状态：已修（台账 G14：对齐配置里 TPU 关 `routed_bias_zero_mean_update`；生产建议两边都开）。** [根源一览第 1 行](../ROOT-CAUSES.md#r1)；对齐报告 [D4](../alignments/01-mini/DETAILS.md#d4)。已修的是本案 r 19.1 那个协议设置问题；以现行比较 `tc-a`（C 口径）为准，1000 步路由偏置均值带符号均值又是待定（N 1.04，TPU 1.36，r 1.31，未定责；C 口径另 5 份 0.36–0.80），和这里的协议问题不是一回事，见 [mini 对齐细节](../alignments/01-mini/DETAILS.md) L4 一行。

## 现象

T1 的 GPU 对 TPU 比较（GPU `g1-a` 对 TPU `t1-a`，提交 `5d36f99`）里，路由偏置均值不过：带符号均值 N 0.799，差 15.3，r 19.1。路由偏置标准差两边相同（提交 `5d36f99` 的说明、台账 G14），判定待定。

## 机制

[`PROTOCOL.md`](../PROTOCOL.md) 原来写两边都对路由偏置做零均值更新（每步把偏置更新减去均值）。原版 Megatron-Core 0.20 没有这个选项，GPU 一直不减均值；TPU 开着。于是两边的偏置只差一个逐步累积的整体平移：均值漂开，标准差不变。整体平移不改 top-k，门控权重用不带偏置的分数（PROTOCOL 变更记录，标「推断」），所以路由本身两边不受影响。

## 证据

TPU 设 `routed_bias_zero_mean_update: false` 重跑 6 份 `t1n-*`（`t1n-b` 与 `t1n-a` 逐位相同；提交 `0a39dd5`）。GPU `g1-a` 对 TPU 基线，噪声 `g1` 两两 28 对；GPU 当时是被测，写「差」：

| 检查 | 修前 `t1-l4`（`t1-a`） | 修后 `t1-l4-nozm`（`t1n-a`） |
|---|---|---|
| 路由偏置均值 | 不过：带符号均值 N 0.799、差 15.3，r 19.1；MARE N 0.800、差 15.3，r 19.1 | 过：带符号均值差 0.135，r 0.17；MARE 差 0.251，r 0.31 |
| 路由偏置标准差 | 待定 | 过 |
| 主 loss | 不过：最大单步 N 3.74e-5、差 2.68e-4，r 7.17 | 不过：同左 |

出处：[`t1-l4.results.json`](../alignments/01-mini/results/t1-l4.results.json)、[`t1-l4-nozm.results.json`](../alignments/01-mini/results/t1-l4-nozm.results.json)、[`t1-l4-nozm.report.md`](../alignments/01-mini/results/t1-l4-nozm.report.md)。索引器 KD loss、grad_norm 两次都不过，这一改没有让它们动出可分辨的量：带符号均值的变化都小于它自身的标准误（结果文件的 `signed_mean_se`）。KD：+4.80e-4 → +4.89e-4，变化 9.1e-6，标准误修前 5.7e-5、修后 6.5e-5；grad_norm：−1.72e-4 → −2.22e-4，变化 5.1e-5，标准误修前 9.7e-5、修后 1.04e-4（N 分别是 5.59e-5、8.29e-5）。MARE 与最大单步变化更小（KD r 2.39 → 2.49、0.95 → 0.99；grad_norm r 1.71 → 1.75、1.13 → 1.14）。

## 结论

- **修法**：对齐配置里 TPU 设 `routed_bias_zero_mean_update: false`，GPU 保持原版（台账 **G14**）。PROTOCOL 第 1.2 节「路由」行改为「路由偏置不做零均值更新」；旧结果 `t1-l4.results.json` 原样保留。
- **定责依据**：
  - 读代码，对照写明的语义标准：优化器与 loss 细节以原版 Megatron-Core 为准；原版没有路由偏置零均值更新，协议原来的设置 GPU 做不到（G14）。
  - 只改 TPU 一边后两边的差收拢：偏置均值的带符号均值 15.3 → 0.135（N 0.799），r 19.1 → 0.17；MARE 15.3 → 0.251（N 0.800），r 19.1 → 0.31；同一次改动里主 loss 的差不变（N 3.74e-5，差 2.68e-4），不是这一处的问题。
- **生产建议：两边都开零均值更新**。关掉它只是对齐设置。
  - 符号式固定步长的更新（DeepSeek 的 loss-free balancing，arXiv 2408.15664，DeepSeek-V3 采用）每步各专家更新之和一般不为 0，偏置的共同分量会漂移。例如 GLM-5.2 公开 checkpoint 第 6 层的 `e_score_correction_bias` 均值 15.6、标准差 8.71e-2（GLM-5.2 实践的 goldens）。
  - 减均值不改变 top-k，但防止漂移、保住分辨率：偏置相加若在 BF16 里做，15 附近 1 ulp 是 0.0625，已与标准差同量级。Qwen 的 ID Balancing（arXiv 2609.39137）同样把偏置拉回零均值。
  - 混元 4 的生产配方里 expert bias 与零均值更新都开；MaxText 一侧在我们修改过的 MaxText 里加入。
  - GPU 一侧原版 Megatron-Core 没有这个选项。janus 的 GPU 训练入口已加同写法的 `--bias-zero-mean`（[改动台账](../CHANGES-LEDGER.md) G35）；生产配方臂里两边都开着各跑 1000 步，同配方 GPU 8 份尺子下 TPU 9 项全过（诊断，不进签字；TPU 只 1 份，三项一起改没单开；8 份尺子是看过借现行尺子的结果后才定的，借现行尺子时主 loss、梯度范数超），见[根源一览第 1 行](../ROOT-CAUSES.md#r1)。用原版 Megatron-Core 时仍要在偏置更新后加一行减均值。
  - 与硬件无关，没有证据表明它对 TPU 特别有利。
