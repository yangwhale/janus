[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `ulp_distribution`

# ulp_distribution

**用途**：区分「只差末位」与「真算错」；定位 0 ULP 比例在哪个张量掉下来。默认只诊断（`hard: false`）。

**输入**：`[(名字, 被测, 参照)]`，两边同一种原始浮点 dtype。

**判据**：

- 无门槛或 `calibrated` → 只报分布、判 `pass`。
- `exact` → 最大 ULP 为 0 且无跨零才 `pass`。
- 数值门槛写在 `p999`、`max`（上限）、`zero_frac`（下限）上。
- 非有限值位置不一致，在有门槛时直接 `fail`。

**输出**：

- `zero_frac_min`。
- `one_frac`（全体元素合起来恰好差 1 ULP 的比例）。
- `buckets`（全体元素合起来的分桶占比：0 / 1 / 2-4 / 5-8 / 9-16 / 17+，按各张量元素数加权）。
- `p999_max`、`max_max`、`cross_zero`、`nonfinite_mismatch`。
- 明细每个张量一行，含 `zero_frac`、`one_frac`、`buckets`、分位数；`worst` 按 `zero_frac` 升序。

**边界**：dtype 不一致 → `error`；跨零元素单列，不进分位数。
