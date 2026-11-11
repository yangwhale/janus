[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `bitwise_equal`

# bitwise_equal

**用途**：该逐位相同的地方——token、权重、确定性重跑、抓数不改数值。

**输入**：`[(名字, 被测, 参照)]`。

**判据**：dtype、形状相同，且逐字节相同才算相等；全部相等 `pass`，否则 `fail`。门槛固定为 `exact`。

**输出**：`tensors`、`differ`；明细每个张量 `equal`、`differ_bytes`（或 dtype / 形状不同的原因）。

**边界**：选空 → `error`；空张量视为相等；NaN 按位模式比（同位模式的 NaN 相等）。
