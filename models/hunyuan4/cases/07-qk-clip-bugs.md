[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 07

# 案例 07：QK-Clip 三处取法不对——扫描层共用缩放、梯度累积把最大 logit 求和、最大 logit 含 sink

> **状态：三处都已修（M1、M19 修 MaxText 的 bug，无开关；M25 开关，T1 开）。** T1 的 τ 100 下 QK-Clip 从未触发（两边最大 logit 约 0.14），M1、M19 推断不改 T1 的数值；M25 只关它的 300 步对照里主 loss 带符号均值 0.48 → 0.93 倍（推断：编译结果变了）。[根源一览第 8、12 行](../ROOT-CAUSES.md#r8)；对齐报告 [D5](../alignments/01-mini/DETAILS.md#d5)、[D8](../alignments/01-mini/DETAILS.md#d8)。

## 现象

- **A. 扫描层下各层共用一个缩放（M1）**：同一个扫描块里的几层用的是同一组逐头缩放。
- **B. 梯度累积时最大 logit 被求和（M19）**：记下的最大 logit 约为真值的累积步数倍，于是裁得过多（T1 累积 2）。
- **C. 最大 logit 含 sink（M25）**：TPU flash（tokamax splash）带 sink 时，记下的是 max(sink, max q·k)，原版 Megatron-Core 的统计不含 sink。

## 机制

- A：MaxText 对扫描块的 max_logits [层, batch, 头] 连扫描维一起取了最大，同一块的各层因此共用一个缩放。
- B：`gradient_accumulation.py` 对各 microbatch 的 aux 整棵树求和，没有区分「取最大」的统计量。
- C：flash 内核的 running max 从 sink 起算，统计量里带上了 sink。

## 证据

- **A**：TPU 检查脚本 `qk_clip_scan_check.py`（台账 J1）在 hy4-tiny（1 稠密层 + 10 MoE 层，τ 取 0.001）上实测，修前结果在 [`hy4-tiny11-qk-clip-scan.json`](../results/hy4-tiny11-qk-clip-scan.json)。扫描块 `moe_layers/layers_0` 的两层，实测每头 √scale 都是 0.3355 / 0.2889 / 0.3084 / 0.3196；按各层自己的最大 logit 算，第 1 层应为 0.3437 / 0.3494 / 0.3084 / 0.3375，第 2 层应为 0.3355 / 0.2889 / 0.3558 / 0.3196。4 个扫描块的实测缩放都等于两层期望的逐头最小值，即 logit 小的层跟着 logit 大的层多裁了；不扫描的层都对。单测 `qk_clip_scan_test.py` 修前 4 条全不过；修后 hy4-tiny 11 层每层每头 √scale 与期望一致（rtol 1e-5）。
- **B**：单测 `mcore_compat_switches_test.py`（`reduce_microbatch_aux`）确认 max_logits 取最大、其余 aux 求和；`gradient_accumulation_nnx_test` 下 hy4-tiny CPU 梯度累积 2 训练跑通。TPU 上「累积 2 对不累积」的最大 logit 对比还没做。
- **C**：单测 `qk_clip_max_logit_exclude_sink_test.py` 对 float64 参照通过，并覆盖 logit 全为负时不报 0。由此引出的 NaN（最大 logit 为负时开负数的根号，M35）单测改前 NaN、改后通过；Megatron-Core 同一情形的 CPU 复现见台账 G25。
- **L4**：B 没有修前 / 修后的数（T1 下不触发）；C（M25）只关它的 300 步对照见上。M1、M19 在 T1 所有运行里都已生效；M25 只在 `t1t-*` 起生效，其前后变化与其他几处混在一起（[案例 06](06-gpu-attn-scores-bf16.md)）。

## 结论

| 台账 | 修法 | 开关 / T1 | 提交 |
|---|---|---|---|
| M1 | 扫描块的 max_logits 只对 batch 取最大，缩放按 `param_scan_axis` 广播回 kernel | 无（修 bug） | `f10b4d9a` |
| M19 | 梯度累积时 max_logits 按 microbatch 取最大，其余 aux 仍求和 | 无（修 bug） | `27d82457` |
| M25 | 在内核外按「因果 ∩ 索引器 top-k ∩ 同段」重算缩放后 q·k 的最大值（只作统计） | `qk_clip_max_logit_exclude_sink`，默认关，T1 开（`t1t-*` 起）；须与 M35 一起用 | `3e6d4c9a` |

相关的两处对齐：QK-Clip 分母不加 1e-6（M18，见[案例 08](08-muon-mcore-switches.md)）；MTP 层的 QK-Clip 由 GPU 补胶水（G7）。

定责依据（都是读代码，对照写明的语义标准）：

- **A（M1）**：标准是「QK-Clip 按层、按头裁」；实测缩放等于两层期望的逐头最小值，修后单测每层每头一致（rtol 1e-5）。
- **B（M19）**：标准是「最大 logit 取最大、不求和」；单测 `reduce_microbatch_aux`。
- **C（M25）**：标准是原版 Megatron-Core 的统计口径（不含 sink）；单测对 float64 参照通过。
