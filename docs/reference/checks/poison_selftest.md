[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `poison_selftest`

# poison_selftest

**用途**：工具自证——证明一条检查真的能抓到问题。

**输入**：目标类型 `target_type`、`[(名字, _, 参照)]`、目标类型的参数、注入方式（默认全部）。

**判据**：先用干净数据（被测 = 参照）跑目标检查，必须 `pass`；再逐种注入，每种都必须判 `fail` 或 `error`。全部满足 `pass`，否则 `fail`。

**注入方式**：`flip_bit`（中间翻一位）、`nan`（置一个 NaN；整数张量改成 +1）、`shift`（错一格）、`scale`（乘 1.5；整数 +1）。

**说明**：门槛放得太松的检查会在这里现形。实现自证时就发现过 `ulp_distribution` 对 NaN 注入不报警，已修。
