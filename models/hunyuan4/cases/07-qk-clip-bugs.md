[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 07

# 案例 07：QK-Clip 三处取法不对——扫描层共用缩放、梯度累积把最大 logit 求和、最大 logit 含 sink

> **状态：三处都已修（M1、M19 修 MaxText 的 bug，无开关；M25 开关，T1 开）。** T1 的 τ 100 下 QK-Clip 从未触发（两边最大 logit 约 0.14），M1、M19 推断不改 T1 的数值；M25 只关它的 300 步对照里主 loss 带符号均值 0.48 → 0.93 倍（推断：编译结果变了；升幅 0.45 在只改执行方式的波动线 1.07 以内，单看曲线归不到开关）。[根源一览第 8、12 行](../ROOT-CAUSES.md#r8)；对齐报告 [D5](../alignments/01-mini/DETAILS.md#d5)、[D8](../alignments/01-mini/DETAILS.md#d8)。

## 现象

- **A. 扫描层下各层共用一个缩放（M1）**：同一个扫描块里的几层用的是同一组逐头缩放。
- **B. 梯度累积时最大 logit 被求和（M19）**：记下的最大 logit 约为真值的累积步数倍，于是裁得过多（T1 累积 2）。
- **C. 最大 logit 含 sink（M25）**：TPU flash（tokamax splash）带 sink 时，记下的是 max(sink, max q·k)，原版 Megatron-Core 的统计不含 sink。

## 机制

- A：MaxText 对扫描块的 max_logits [层, batch, 头] 连扫描维一起取了最大，同一块的各层因此共用一个缩放。
- B：`gradient_accumulation.py` 对各 microbatch 的 aux 整棵树求和，没有区分「取最大」的统计量。
- C：flash 内核的 running max 从 sink 起算，统计量里带上了 sink。

## 证据

- **A**：TPU 检查脚本 `qk_clip_scan_check.py`（台账 J1）在 hy4-tiny（1 稠密层 + 10 MoE 层，τ 取 0.001）上实测，修前结果在 [`hy4-tiny11-qk-clip-scan.json`](../results/hy4-tiny11-qk-clip-scan.json)。扫描块 `moe_layers/layers_0` 的两层，实测每头 √scale 都是 0.3355 / 0.2889 / 0.3084 / 0.3196；按各层自己的最大 logit 算，第 1 层应为 0.3437 / 0.3494 / 0.3084 / 0.3375，第 2 层应为 0.3355 / 0.2889 / 0.3558 / 0.3196。4 个扫描块的实测缩放都等于两层期望的逐头最小值，即 logit 小的层跟着 logit 大的层多裁了。修后同一脚本复跑（MaxText `b87a55aa`），11 层每层每头的实测 √scale 与期望逐位相同（[`hy4-tiny11-qk-clip-scan-postfix.json`](../results/hy4-tiny11-qk-clip-scan-postfix.json)）；不扫描的层都对。单测 `qk_clip_scan_test.py` 修前 4 条全不过；修后 hy4-tiny 11 层每层每头 √scale 与期望一致（rtol 1e-5）。
- **B**：单测 `mcore_compat_switches_test.py`（`reduce_microbatch_aux`）确认 max_logits 取最大、其余 aux 求和；`gradient_accumulation_nnx_test` 下 hy4-tiny CPU 梯度累积 2 训练跑通。
- **B 的 TPU 实测（2026-11-11）**：现行配置（base2 关）把阈值调到 0.1，让 QK-Clip 真触发；累积 2（每设备微批 1，现行）对累积 1（每设备微批 2），全局批 16 不变，回放同一份 GPU batch，各 100 步，记逐层逐头最大 logit、缩放 η、被裁头数、loss。结果在 [`t1zo-qkclip-accum-tau01.json`](../alignments/01-mini/results/t1zo-qkclip-accum-tau01.json)：
  - 第 1 步 64 个头的最大 logit，累积 2 ÷ 累积 1 在 0.997–1.003 之间，35 个逐位相同。M19 修之前那种「各 microbatch 求和」会让这个比值接近 2，这里没有。
  - 前 30 步两者之差与 GPU 两份重跑（`g6-qk-a` / `b`，只差 GPU 自己的求和顺序）之差同一量级：逐头最大 logit 的相对 L2，中位数 6.8e-3 对 6.6e-3（1.04 倍）；单头最大相对差 7.8e-2 对 5.8e-2（1.35 倍）；loss 的平均相对差 1.9e-6 对 2.8e-6（0.71 倍）。被裁头数前 30 步合计 205 对 208。
  - 100 步里 53 步有头被裁，多在前 50 步；之后最大 logit 回到阈值以下，很少再裁。
  - 判定：两种累积只差求和顺序。阈值 0.05 再测一遍，同样的结论（[`t1zo-qkclip-accum-tau005.json`](../alignments/01-mini/results/t1zo-qkclip-accum-tau005.json)；这次对照用的还是阈值 0.1 的 GPU 那一对）。
  - 附带：第 4 层第 1 步离 FP64 的 R（对齐细节 D31），累积 2 是 1.55，累积 1 是 1.83。可见这一层的偏差不是累积带来的。
- **C**：单测 `qk_clip_max_logit_exclude_sink_test.py` 对 float64 参照通过，并覆盖 logit 全为负时不报 0。由此引出的 NaN（最大 logit 为负时开负数的根号，M35）单测改前 NaN、改后通过；Megatron-Core 同一情形的 CPU 复现见台账 G25。
- **L4**：B 没有修前 / 修后的 L4 数（τ 100 下不触发；低阈值的 100 步实测见上一条）；C（M25）只关它的 300 步对照见上。M1、M19 在 T1 所有运行里都已生效；M25 只在 `t1t-*` 起生效，其前后变化与其他几处混在一起（[案例 06](06-gpu-attn-scores-bf16.md)）。

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
