[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `tensor_tolerance`

# tensor_tolerance

**用途**：跨平台、跨执行方式的张量比较——算子输出、逐层激活、逐张量梯度。

**输入**：

- `[(名字, 被测, 参照)]`。
- `metrics` 从 `rel_l2`、`cosine`、`mean_shift`、`max_abs`、`out_of_tol_ratio`、`norm_ratio`、`std_ratio`、`zero_frac_diff`、`rel_err_p99`、`kl` 中选（定义见 [`../metrics.md`](../metrics.md)）。
- `params.atol`、`params.rtol`（只用于 `out_of_tol_ratio`）。

**判据**：每个张量拉直算指标（FP64），取全体最差值判门槛。

- `cosine` 取最小。
- `norm_ratio`、`std_ratio` 取 |比值 − 1| 的最大（偏大偏小都算，数值门槛也写在偏离量上）。
- 其余取最大（`kl` 是每个张量各自的行平均，再取张量间最大）。
- 门槛可写数值或 `calibrated`；不接受 `exact`（用 `bitwise_equal`）。

**输出**：

- `<指标>_max`（`cosine_min`、`norm_ratio_dev_max`、`std_ratio_dev_max`）、`tensors`。
- 另有参考档统计 `oot_frac_max`、`mean_diff_max`、`norm_max_max`。
  - 来自 [`janus/metrics.py`](../../../janus/metrics.py) 的 `reference_stats`，只报不判，明细里是 `ref_*`。
  - `params.no_reference_stats: true` 关掉。
- `worst` 按第一个指标列最差 5 个；明细每个张量一行。

**边界**：任一张量非有限值位置不一致 → `fail`；选空 → `error`；未知指标 → `error`。
