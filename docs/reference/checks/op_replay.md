[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `op_replay`

# op_replay

**用途**：L1 单算子。把一个算子的输出（GPU 回放、合成输入上的运算、故意改坏的实现）同时对三样东西比：

- FP64 真值；
- 应当逐位相同的另一份结果；
- CPU 模拟的舍入方式。

整套 L1 的做法见 [`../../dev/design/0003-l1-op-replay.md`](../../dev/design/0003-l1-op-replay.md)。

## 输入

**输入**：用例列表，每个用例一个字典：

| 键 | 必填 | 含义 |
|---|---|---|
| `name` | 是 | 用例名，如 `layers.1.router.logits` |
| `out` | 是 | 被测输出，保持原始 dtype（FP32 / BF16 / FP16，或整数下标） |
| `fp64` | | FP64 真值（同形状 float64） |
| `bitref` | | 应当逐位相同的结果：训练时抓的输出、CPU 上的同一运算 |
| `emulate` | | CPU 模拟 GPU 舍入方式算出的结果 |
| `ref` | | 参照实现的输出，用于 R-ratio |
| `expect` | | 整数输出（top-k 下标）的期望，按行比集合 |
| `extra` | | 原样写进明细（实现说明、开关实测值、输入区间等） |

## 指标

**逐用例指标**（明细一行一个用例）：

- 给了 `fp64`：
  - `rel_l2`（对 FP64，只在两边都有限的位置算）；
  - `cr_match_frac`——对「FP64 真值一次 RNE 舍到输出 dtype」逐位相同的元素比例；
  - `ulp_max`、`ulp_p999` 与分桶 `ulp_hist`（0 / 1 / 2–4 / 5–16 / 17+）——对正确舍入值的整数 ULP 距离（正确舍入本身是 0，与 CUDA 手册的 ULP 口径一致）；
  - `nonfinite_mismatch`——非有限值位置或取值与正确舍入值不同的元素数（NaN 对 NaN 算相同）；
  - `cross_zero`。
- 给了 `bitref`：`bit_match_frac`。给了 `emulate`：`emulate_match_frac`。
- 给了 `ref`：`r_ratio = rel_l2(被测, FP64) / rel_l2(参照, FP64)`，另报 `rel_l2_test`、`rel_l2_ref`、`rel_l2_test_ref`、`cosine_test`、`cosine_ref`。
- 给了 `vsref`（默认取 `ref`）：参考档统计 `vs_ref_rel_l2`、`vs_ref_cosine`、`vs_ref_oot_frac`、`vs_ref_mean_diff`、`vs_ref_norm_max`。
  - 来自 [`janus/metrics.py`](../../../janus/metrics.py) 的 `reference_stats`，被测对参照的同名输出，只报不判。
  - `vsref_out` 给了就用它代替 `out`（同一输入上的另一份被测输出，例如 `layer_replay.py --l1` 在 golden 输入上的算子输出）。
  - `vsref_note` 写它从哪来。
  - 两侧算子输入不同的用例不要给 `vsref`：差里混着上游。
- 给了 `expect`：`expect_match_frac`（选中集合相同的行比例）、`rows_differ`、`rows_differ_first`。

**汇总**：汇总名就是门槛里写的名字。

- `rel_l2`、`ulp_max`、`ulp_p999`、`r_ratio` 取最大。
- `vs_ref_*` 取最差（cosine 取最小、其余取最大）。
  - 另报有对照的用例数 `vs_ref_cases` 与 rel_L2 最大的用例名 `vs_ref_worst`。
- `*_match_frac` 取最小。
- `nonfinite_mismatch` 求和。

## 判据、输出、边界

**判据**：

- 无门槛 → 只诊断，判 `pass`（目录里配 `hard: false`）。
- `exact` →
  - 给了 `bitref` / `emulate` / `expect` 的，这些比例必须是 1；
  - 都没给时 `cr_match_frac` 必须是 1；
  - `nonfinite_mismatch` 必须是 0。
- 数值 → 写在汇总名上；`*_match_frac` 是下限，其余是上限。必须写 `source`。
- `params.expect: fail`（演练）：用例是故意改坏的实现。
  - 门槛判 `fail` 时本检查 `pass`，判 `pass` 时本检查 `fail`。
  - 原判定记在 `metrics.drill_judged`。
- `params.worst_by`：`worst` 按哪个汇总指标排。

**输出**：`metrics` 是上面的汇总加 `cases`；`worst` 列最差 5 个用例；明细一个用例一行。

**边界**：

- 没有用例 → `error`。
- `bitref` / `emulate` 与 `out` 的 dtype 或形状不同 → `error`。
- FP64 真值本身离数学真值约 1e-16（相对），舍到 FP32 / BF16 时只有落在格点中点 1e-16 以内才可能舍错，不处理。

## 实现

**实现**：[`janus/checks/ops.py`](../../../janus/checks/ops.py)。

- FP64 舍入、R-ratio、GEMM 舍入模拟、打平规则、解析 VJP 在 [`janus/analysis/ops.py`](../../../janus/analysis/ops.py)（纯 numpy）。
- GPU 回放要 torch 与 Transformer Engine，不进 `janus run`。由 `models/<模型>/frameworks/megatron/` 下的脚本调本类型并写同一格式的 `results.json`：
  - `l1_ops.py`（单算子）
  - `l1_kernels.py`（TF32、专家分组 GEMM 与注意力回放、poison 与因果性）
  - `moe_stages.py`（MoE 分阶段）
  - `causal_model.py`（整网因果性）
  - `ckpt_special.py`（特殊值存 / 恢复）
  - 写法见 `../../dev/design/0003-l1-op-replay.md`、[`../../dev/design/0004-l1-gpu-kernels.md`](../../dev/design/0004-l1-gpu-kernels.md)。
- TPU 侧逐层隔离回放的前向用本类型的 R-ratio 判：[`models/hunyuan3/analysis/fwd_rratio.py`](../../../models/hunyuan3/analysis/fwd_rratio.py)（真值来自同目录的 `fp64_forward.py`）。
