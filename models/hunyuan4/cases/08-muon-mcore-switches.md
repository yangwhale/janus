[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 08

# 案例 08：Muon 与标量 Adam 的写法与原版 Megatron-Core 不同——MaxText 用开关对上

> **状态：已对齐（M15–M18、M30 / M31、M34，默认关、T1 开；更新缩放两边一致，不改）。** [根源一览第 7、10、17 行](../ROOT-CAUSES.md#r7)；对齐报告 [D17](../alignments/01-mini/DETAILS.md#d17)。

## 现象

MaxText（optax 0.2.8）与原版 Megatron-Core（emerging_optimizers 0.3.0）在优化器上有四处写法不同（台账 M15–M18）：

| # | 量 | MaxText 原来 | 原版 Megatron-Core | 开关（台账） |
|---|---|---|---|---|
| 1 | Nesterov 方向 | optax 的写法（台账只记新写法「不做偏差修正」） | β·mu_t + (1−β)·g_t，不做偏差修正 | `muon_mcore_nesterov`（M15，只支持 `optax_muon`） |
| 2 | 标量参数的 Adam 分支 | 随 Muon 的 Nesterov 变成 NAdam | 普通 Adam | `muon_adam_plain`（M16） |
| 3 | Newton–Schulz 前的归一化 | x / (‖x‖ + eps)，eps 取 `adam_eps` | x / max(‖x‖, 1e-7) | `muon_mcore_ns_normalize`（M17） |
| 4 | QK-Clip 缩放的分母 | min(1, τ / (max_logit + 1e-6)) | min(1, τ / max_logit) | `qk_clip_mcore_denominator`（M18） |

GPU 对 TPU 的连续 8 步优化器回放（GPU `g2o-a` 第 1–8 步，TPU `tw-optim8`）另有两处不同：

- Muon 增量对 GPU 的最大相对差 1.28e-1。
- Adam 管的几个小张量离 FP64 比 GPU 远：第 5 步 `hc_head_base` 主权重增量对 FP64，GPU 1.49e-6，TPU 5.70e-6，R 3.82。

## 机制

- 四处写法：MaxText 用的是 optax 的 Muon 组合，在 Nesterov 写法、标量 Adam、NS 归一化、QK-Clip 分母上与原版 Megatron-Core 的实现不同（优化器语义）。
- Muon 增量：GPU（emerging_optimizers）的 Newton–Schulz 在 BF16 张量上迭代，TPU 留在 FP32。
- Adam 小张量：TPU 的 FP32 除法、sqrt 与偏差修正 1 − β^t 不精确（`tw-elem3`：2 / 3 / 108 ULP）；GPU 的除法与 sqrt 正确舍入。

## 证据

- **单测**（MaxText `tests/unit/mcore_compat_switches_test.py`；固定参照 `tests/assets/muon_mcore/` 由 Megatron-Core `685fb8b` 的 TensorParallelMuon 在同一组梯度上逐步生成）：
  - M15–M17 全开时，每步更新对原版的相对差最大 1.3e-6；三个全关时 1.3e-1。
  - M16：对 torch Adam 逐步相对差 ≤ 1e-5；NAdam 时 5.6e-1。
  - M17：单独开时影响约 1e-6，属舍入量级。
  - M18：τ 1e-6、max logit 1e-5 时缩放 0.1（关时 1/11）；τ = 100 时相对影响 ≤ 1e-8。
- **更新缩放（G18，核对）**：GPU 运行时 `muon_scale_mode: shape_scaling`（`g1-a` 的 meta.json），即 √max(1, 输出维 / 输入维)；TPU 不设 `muon_consistent_rms`，同为 √max(1, fan_out / fan_in)；参照 `gen_ref.py` 也用 `shape_scaling`。两边一致。
- **回放**：[`t1-optim-replay-tpu.json`](../alignments/01-mini/results/t1-optim-replay-tpu.json)（M30 前后）、[`t1-optim-replay-adam-acc-tpu.json`](../alignments/01-mini/results/t1-optim-replay-adam-acc-tpu.json)（`tw-optim8b`，M34 后，含同一回放在 CPU 上的对照）、[`t1-optim-replay-adam-acc-r2-tpu.json`](../alignments/01-mini/results/t1-optim-replay-adam-acc-r2-tpu.json)（最终版 `tw-optim8c`）。
  - M30 / M31 后 Muon 增量对 GPU 的最大相对差 1.28e-1 → 1.6e-2。
  - M34 后同一张量 TPU 5.70e-6 → 1.49e-6（GPU 1.49e-6），R 3.82 → 1.00；`tw-optim8c` 里原来超线的 11 项与 GPU 逐位相同。
- **L4**：这些开关在 T1 所有运行里都开着，没有关着它们的 L4 运行，因此没有 L4 前后对比。

## 结论

- **修法**：M15–M18 在 MaxText 提交 `27d82457`（新模块 `optimizers/muon_mcore_compat.py`，三个 Muon 开关全关时仍调 optax 的 muon）；Newton–Schulz 走 BF16 用 `muon_mcore_ns_bf16`（M30，T1 开见 M31）；Adam 精确算术用 `muon_adam_accurate`（M34，T1 / T2 开）。四个开关的写法见 [`PROTOCOL.md`](../PROTOCOL.md) 第 1.2 节「Nesterov 的写法」「NS 归一化」「标量参数的 Adam」「QK-Clip」行。更新缩放（G18）两边同一公式，不需要开关。
- **定责依据**：
  - M15–M18、M30：读代码，对照写明的语义标准——优化器写法以原版 Megatron-Core 为准，GPU 保持原版（[`PROTOCOL.md`](../PROTOCOL.md) 变更记录）。M30 关时 TPU 离 FP64 更近、离 GPU 更远，所以是照 GPU 写法对齐，不是 TPU 算错。
  - M34：对 CPU FP64 的 R。GPU 1.49e-6，TPU 5.70e-6，R 3.82；同一回放在 CPU 上没有这个差（绝对值见 `t1-optim-replay-adam-acc-tpu.json`），所以错在 TPU 的 FP32 算术；修后 TPU 1.49e-6，R 1.00。
