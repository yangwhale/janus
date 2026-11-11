[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `noise_verdict`

# noise_verdict

**用途**：已经算好的被测值对噪声参照下结论（其他类型的 `calibrated` 内部也用同一规则）。

**输入**：`values`（指标 → 被测值，越小越好）、`noise`（指标 → 噪声参照值列表）。

**判据**：逐指标 r = x / max(噪声)：r ≤ 1 `pass`（参照不足 3 条只给 `inconclusive`），r > 2 `fail`，其间 `inconclusive`；多个指标取最严。

**输出**：被测值；`r1`（MARE）、`r2`（带符号均值）若在指标中。
