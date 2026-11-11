[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `overlap_replay`

# overlap_replay

**用途**：重启前后的重叠步逐位比，排查硬件静默错误、验证重放。

**输入**：重启前、后的 {步号: {字段: 值}}；可选字段清单。

**判据**：重叠步所有字段逐位相同 → `pass`；否则报第一处不同的步与字段。没有重叠步 → `error`。
