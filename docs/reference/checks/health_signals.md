[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `health_signals`

# health_signals

**用途**：长跑健康：非有限值、突跳（单步先于 loss 报警）。

**输入**：{字段: {步号: 值}}（如 lm_loss、raw_grad_norm、param_norm）；`params.window`（默认 20）、`spike_ratio`（默认 10）、`source`。

**判据**：任一非有限值 → `fail`；某步超过前 window 步中位数的 spike_ratio 倍 → `fail`。

**边界**：选中的字段里有一步数据都没有的（拼错、或这次运行没记）→ `error`，不判「健康」。
