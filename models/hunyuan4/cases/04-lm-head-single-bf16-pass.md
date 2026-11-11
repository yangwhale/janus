[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 04

# 案例 04：TPU 的「FP32 输出头」只做了一遍 BF16——加 `logits_matmul_precision: highest`

> **状态：已修（台账 M13，T1 开，`t1t-*` 起）。** 代价步时 +24%，生产看情况（取舍，C）。

## 现象

- GPU 输出头是真 FP32：FP32 权重 × FP32 隐状态，TF32 关（[`PROTOCOL.md`](../PROTOCOL.md) 第 1.2 节「输出头」行）。TPU 在 `t1n-*`、`t1s-*` 两代只开了 `logits_dot_in_fp32`。
- 300 步单开关 A/B：只关它（`tq-m13`）时主 loss 最大单步 N 4.53e-5、TPU 2.73e-4，r 6.03，不过。
- 第 1 步量不到它（[案例 13](13-weight-side-switch-step1-blind-spot.md)）。

## 机制

MaxText 的 `logits_dot_in_fp32` 只把操作数转成 FP32，矩阵乘精度仍跟全局 `matmul_precision`；TPU 默认精度下对 FP32 操作数也只做一遍 BF16，等于先把 FP32 权重舍成 BF16 再乘。要真 FP32，得单独给输出头设 HIGHEST。

第 1 步看不出：起训时权重和输入都在 BF16 格点上，HIGHEST 多出的几遍贡献为 0（[案例 13](13-weight-side-switch-step1-blind-spot.md)）。

## 证据

300 步单开关 A/B（以 T1 yml 全开为底，只差 M13；对 `g2-a`，第 1–300 步）：

| 检查 | 判定量 | N | 关（`tq-m13`） | 开（`tq-base`） |
|---|---|---|---|---|
| 主 loss | 最大单步 | 4.53e-5 | TPU 2.73e-4，r 6.03（不过） | TPU 2.75e-5，r 0.61 |
| param_norm | 带符号均值 | 1.90e-7 | TPU −4.71e-7，r 2.48（不过） | TPU −1.09e-7，r 0.57 |

出处：[`tq-m13-l4.results.json`](../alignments/01-mini/results/tq-m13-l4.results.json)、[`tq-base-l4.results.json`](../alignments/01-mini/results/tq-base-l4.results.json)。

- 训练后权重的重放（[案例 13](13-weight-side-switch-step1-blind-spot.md)）：换上第 8 步的 FP32 主权重，输出头 logits 对 FP64（FP32 权重）开时 8.3e-8，关时 2.73e-3。
- 第 1 步开关前后（`tw-sw-base` / `tw-sw-m13`）输出头 logits 前向逐位相同；`lm_head.weight` 梯度 TPU 对 GPU 都是 1.81e-3（GPU 三份之间 1.01e-3）。
- 第 1 步 `lm_head.weight` 梯度对 CPU FP64（`t1diag15`）：GPU 3.66e-3、TPU 3.63e-3，R 0.99（[`t1s-step1-grad-vs-fp64.json`](../alignments/01-mini/results/t1s-step1-grad-vs-fp64.json)）。只把输出头设成 HIGHEST 不改变第 1 步逐张量结论（提交 `0a39dd5`，没有单独的结果文件）；那里的普遍偏大来自舍入位置（[案例 01](01-step1-grad-rounding-switches.md)）。

## 结论

- **修法**：台账 M13 `logits_matmul_precision`（提交 `352014df`），默认跟 `matmul_precision`；T1 设 `highest`（[`hy4-mini-t1.yml`](../frameworks/maxtext/hy4-mini-t1.yml)）。GPU 一侧输出头 dgrad 的分段累加另见台账 G19（[`CHANGES-LEDGER.md`](../CHANGES-LEDGER.md)）。
- **定责依据**：读代码，对照的语义标准是「GPU 输出头是真 FP32 矩阵乘」（台账 G1）；只改 TPU 一边后两边的差收拢：主 loss 最大单步 r 6.03 → 0.61、param_norm 带符号均值 r 2.48 → 0.57（上表）。
- **生产建议**：看情况（取舍，C）。代价：hy4-mini 步时 0.3854 → 0.4767 s（+24%），hy4-mid 2.2610 → 2.8125 s。见[根源一览第 5 行](../ROOT-CAUSES.md#r5)、对齐报告 [D8](../alignments/01-mini/DETAILS.md#d8)。`t1t-*` 一代同时改了 M13、M25、M26、M27 与 GPU 的 G17，合在一起的前后数字见[案例 06](06-gpu-attn-scores-bf16.md)。
