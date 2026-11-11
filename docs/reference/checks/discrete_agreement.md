[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `discrete_agreement`

# discrete_agreement

**用途**：离散结果——路由选中的专家、哈希编号等整数输出。

**输入**：`[(名字, 被测, 参照)]`；`params.mode`：`exact`（默认，逐元素比）或 `topk`（`[行, k]` 下标）；`topk` 时可给 `params.scores = {名字: (被测分数, 参照分数)}`。

**判据**：`exact` 门槛要求全部相同；数值门槛写在 `exact_match_frac` / `set_frac` / `order_frac` 上（下限）；`calibrated` 按「离 1 的距离」对噪声参照判。

**输出**：`exact_match_frac_min` 或 `set_frac_min`、`order_frac_min`；`topk` 给了分数时，明细里有不一致行第 k 与 k+1 名分差的最大值与中位数（说明是不是「几乎打平」）。

**边界**：形状不同 → `error`；选空 → `error`。
