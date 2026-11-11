[janus](../../../README.md) › [用户指南](../../guide/README.md) › [第 4 章 逐级对齐](../../guide/04-level-by-level.md) › 检查类型规格 ｜ 每级用哪些类型见 [L0](../levels.md) · [L1](../levels.md) · [L2](../levels.md) · [L3](../levels.md) · [L4](../levels.md)

# 检查类型规格

- 每种类型一份，实现与规格同 PR 提交、同 PR 修改。
- 共用约定见 [`../result-schema.md`](../result-schema.md)（状态、门槛的四种写法、退出码）。

## 类型一览

| 类型 | 规格 | 实现 |
|---|---|---|
| `bitwise_equal` | [bitwise_equal.md](bitwise_equal.md) | [`janus/checks/tensor.py`](../../../janus/checks/tensor.py) |
| `tensor_tolerance` | [tensor_tolerance.md](tensor_tolerance.md) | `janus/checks/tensor.py` |
| `ulp_distribution` | [ulp_distribution.md](ulp_distribution.md) | `janus/checks/tensor.py` |
| `discrete_agreement` | [discrete_agreement.md](discrete_agreement.md) | [`janus/checks/discrete.py`](../../../janus/checks/discrete.py) |
| `coverage` | [coverage.md](coverage.md) | [`janus/checks/meta.py`](../../../janus/checks/meta.py) |
| `poison_selftest` | [poison_selftest.md](poison_selftest.md) | `janus/checks/meta.py` |
| `loss_series` | [loss_series.md](loss_series.md) | [`janus/checks/series.py`](../../../janus/checks/series.py) |
| `multi_golden` | [multi_golden.md](multi_golden.md) | `janus/checks/series.py` |
| `scalar_series` | [scalar_series.md](scalar_series.md) | `janus/checks/series.py` |
| `noise_verdict` | [noise_verdict.md](noise_verdict.md) | `janus/checks/series.py` |
| `optimizer_step` | [optimizer_step.md](optimizer_step.md) | [`janus/checks/extra.py`](../../../janus/checks/extra.py) |
| `router_stats` | [router_stats.md](router_stats.md) | `janus/checks/extra.py` |
| `determinism` | [determinism.md](determinism.md) | `janus/checks/extra.py` |
| `config_diff` | [config_diff.md](config_diff.md) | `janus/checks/extra.py` |
| `data_fingerprint` | [data_fingerprint.md](data_fingerprint.md) | `janus/checks/extra.py` |
| `checkpoint_equal` | [checkpoint_equal.md](checkpoint_equal.md) | `janus/checks/extra.py` |
| `replica_consistency` | [replica_consistency.md](replica_consistency.md) | `janus/checks/extra.py` |
| `overlap_replay` | [overlap_replay.md](overlap_replay.md) | `janus/checks/extra.py` |
| `artifact_check` | [artifact_check.md](artifact_check.md) | `janus/checks/extra.py` |
| `run_complete` | [run_complete.md](run_complete.md) | `janus/checks/extra.py` |
| `health_signals` | [health_signals.md](health_signals.md) | `janus/checks/extra.py` |
| `dtype_manifest` | [dtype_manifest.md](dtype_manifest.md) | `janus/checks/extra.py` |
| `fake_fp32` | [fake_fp32.md](fake_fp32.md) | `janus/checks/extra.py` |
| `op_replay` | [op_replay.md](op_replay.md) | [`janus/checks/ops.py`](../../../janus/checks/ops.py) |

## 门槛判定（`janus/checks/base.py`）

- `exact`：逐位相同才 `pass`。
- 数值：每个指标都在门槛内才 `pass`；必须写 `source`。
  - 越大越好的指标（`cosine`、`zero_frac`、`set_frac`、`order_frac`、`*_match_frac`）门槛是下限，其余是上限。
  - 也可写 `{min, max}`。
- `calibrated`：
  - 越小越好的指标 N = 噪声参照最大值、r = x / N；越大越好的先换成「离 1 的距离」。
  - r ≤ 1 `pass`（参照不足 3 对时只给 `inconclusive`），r > 2 `fail`，其间 `inconclusive`。
  - 多份 golden 时用 `params.baseline` 指定只对一份比。
  - 不写则取对各份里最差的一份，和正式的比较协议不同（[第 2 章 2.4](../../guide/02-gpu-golden.md#24-噪声带与基线)）。
- `pending`（待标定）：按「没有噪声参照」判，最多 `inconclusive`。
  - 被测值照样算出来写进 `metrics`。
  - 非有限值、单步硬上限这类硬底线照样判 `fail`。
  - 只能用在 `loss_series`、`router_stats`（目录校验会拦）。
  - 结果里 `threshold.kind` 写 `pending`。
  - 用于噪声参照还没标定的容差门，例如日常回归里「会改数值 · 执行级」时才用的那几条。
