[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `replica_consistency`

# replica_consistency

**用途**：数据并行副本、多级存档副本是否一致。

**输入**：{副本名: 数组或校验和}，至少 2 个。

**判据**：全部校验和相同 → `pass`；否则 `fail` 并按校验和分组列出。
