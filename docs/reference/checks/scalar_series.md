[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `scalar_series`

# scalar_series

**用途**：逐步标量逐值对拍，如学习率。

**输入**：两条序列；`params.window`（必填）、`rtol`（默认 0）、`abs_ulp`（默认 2）、`dtype`（默认 float32）、`source`。

**判据**：每步 \|t − r\| ≤ rtol·\|r\| + abs_ulp·ulp(r)，ulp 按 dtype 数格（防止 FP32 warmup 舍入误判）；任一步超或非有限 → `fail`。

**输出**：`steps`、`bad_steps`、`first_bad`。
