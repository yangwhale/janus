[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `router_stats`

# router_stats

**用途**：路由偏置更新、负载均衡——逐层统计序列对比。

**输入**：{统计名: {步号: 值}}（如 `router.bias_std.<层>`）两份；`params.window` 等同 `loss_series`。

**判据**：每个统计按 `loss_series` 的判定量（带符号均值、MARE、最大单步）比，取最差判门槛；参照里有、被测里缺的统计 → `error`。
