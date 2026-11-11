[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 02

# 案例 02：第 0 层索引器 KD 偏低约 1.2e-3——TPU 近似的 FP32 exp / log 让 log-sum-exp 偏低约 1 格，被 KL ≪ log-sum-exp 的病态放大

> **状态：已修（台账 M32，T1 由 M33 打开，`t1z-*` 起）。** 1000 步第 301 步以后 TPU 转为偏高，不在本案范围，见对齐报告 [D1](../alignments/01-mini/DETAILS.md#d1)。

## 现象

- 第 1 步 KD：TPU `t1n-*` / `t1np-*` 8.47e-6，GPU `g1-*` 8.48e-6，相对差 −9.2e-4；TPU 各份运行一样（[`t1-layer0-kd-attribution.json`](../alignments/01-mini/results/t1-layer0-kd-attribution.json) `step1_kd_all_runs`）。
- 同一行（GPU rank 0 micro 0 = TPU 第 0 行）第 0 层 KD：TPU 图内对同一输入的 FP64 重算相对差 −1.155e-3（同上 `kd_value_row0_layer0`）。
- 300 步 L4（`t1v-l4`）索引器 KD 带符号均值：N 1.38e-4，TPU −5.06e-4，r 3.67，不过；此前各代同样不过、TPU 都偏低。

## 机制

KL 每行约 1.1e-4 nat，log-sum-exp 约 7.1：KD 是两个大项相减剩下的小量。FP32 里 lse 1 格（ulp）是 4.73e-7，等于 KL 的 4.31e-3。TPU 默认的 FP32 exp / log 是近似实现（v5p 上 ulp_max exp 105、log 62），让 lse 平均比 FP64 低 0.953 格（只有 8.6% 的行偏正；CPU torch f32 −4e-3 格）。单独编译的 logsumexp 复现平均偏低 0.95 格，来自 exp（输入近 0 时平均偏低约 2.5 格）与 log（约 0.6 格），求和不贡献（[`tpu-elementwise-ulp.json`](../alignments/01-mini/results/tpu-elementwise-ulp.json) `lse_claim_check`）。lse 偏低 → log q 偏高 → KD 偏低。病态（[精度边界](../../../docs/reference/precision-boundaries.md)第 12 类）放大了近似超越函数的偏差（第 13 类）。

量级一致、没有逐项对上：按 0.95 格 × 4.31e-3 推算 KD 应偏低约 4.1e-3，图内实测 −1.155e-3，同号，差的部分仓库里没有拆开。

梯度不受影响：KD 对索引器分数的梯度是 q − p，lse 偏 1 格只让 q 整体差 4.7e-7（相对），远在 BF16 以下。这一处只影响 KD loss 这个记录值。

## 证据

M32 开关前后（第 1 步「关」是 `t1diag15`，「开」是 t1v 设置加开关，两组设置不完全相同）：

| 量 | GPU | TPU 关 → 开 |
|---|---|---|
| 第 0 层第 0 行 KD 对 FP64 的相对差 | +1.84e-4 | −1.155e-3 → +9.7e-5 |
| 第 0 层 lse 对 FP64 的带符号平均偏差（FP32 ulp） | — | −0.953 → +0.025 |
| 300 步 KD 带符号均值（`t1v-l4` → `t1z-l4`） | N 1.38e-4 | −5.06e-4 → −1.89e-6，r 3.67 → 0.01 |

出处：[`t1-step1-transcendental-accuracy.json`](../alignments/01-mini/results/t1-step1-transcendental-accuracy.json)、[`t1v-l4.results.json`](../alignments/01-mini/results/t1v-l4.results.json)、[`t1z-l4.results.json`](../alignments/01-mini/results/t1z-l4.results.json)。修后 MARE：N 2.52e-4，TPU 2.62e-4，r 1.04（待定）。

- **不在 KD 的输入**：第 0 层教师 q / 教师 k / 索引器分数对 CPU FP64 的 rel_L2，GPU 3.55e-3 / 3.33e-3 / 5.27e-3，TPU（M23 开，`t1diag15`）3.44e-3 / 3.15e-3 / 5.22e-3，R 0.97 / 0.94 / 0.99（[`t1-l1-kd-qkclip-vs-fp64.json`](../alignments/01-mini/results/t1-l1-kd-qkclip-vs-fp64.json)）；开 M23 前后 KD 的差不变（台账 M23）。
- **不在 lse 之前**：同一行逐项对 FP64，教师目标 p 相对误差 1.73e-7，索引器对数概率 7.6e-8；整体 CPU 上 JAX f32 +2.9e-4、torch f32 +8e-6，TPU 图内 −1.155e-3（`kd_value_row0_layer0`，诊断开关 M24 记下的中间量）。
- **不在 eod 行**：差在第 0 行这一行上就已出现（台账 G16）。
- **不在舍入开关**：[案例 01](01-step1-grad-rounding-switches.md) 的开关前后第 0 层 KD 只在第 5 位有效数字上动（[`t1-step1-precision-switches.json`](../alignments/01-mini/results/t1-step1-precision-switches.json)）。
- **梯度不受影响**：20 个索引器梯度张量对 CPU FP64 的 R 0.81–1.05，最大是第 5 层 `weights_proj`（GPU 6.81e-3，TPU 7.14e-3）（[`t1s-step1-grad-vs-fp64.json`](../alignments/01-mini/results/t1s-step1-grad-vs-fp64.json)）。

## 结论

- **修法**：台账 M32 `transcendental_accuracy_highest`，exp / log 用 `accuracy=HIGHEST`（1 / 3 ULP），默认关；T1 由 M33 打开。M23 `indexer_fp32_scores`（T1 开）对本差无效。诊断：M24 `debug_sow_indexer`、G16 `--diag-kd-dump`、J6（[`CHANGES-LEDGER.md`](../CHANGES-LEDGER.md)）。
- **定责依据**：对 CPU FP64——同一批输入，TPU 的 KD 与 lse 偏离 FP64（−1.155e-3、−0.953 格），开 M32 后 +9.7e-5、+0.025 格；KD 输入两边离 FP64 一样近（R 0.94–0.99）。只改 TPU 一边后 300 步差收拢（r 3.67 → 0.01）。`t1z` 同时开了 M31，归到 M33 的依据是第 1 步只开 M32 的诊断。
- **生产建议**：见[根源一览第 13 行](../ROOT-CAUSES.md#r13)；对齐报告 [D14](../alignments/01-mini/DETAILS.md#d14)。
