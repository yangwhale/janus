[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `loss_series`

# loss_series

**用途**：被测对一份参照（golden 或 LKG）的逐步 loss。

## 输入、判据

**输入**：两条序列（步号 → 值），参数如下。

- `params.window`（必填，`"a:b"`）。
- `denominator`（`golden` / `test` / `mean`）。
  - `mean`：两边均值 \|t + r\| / 2，交换两边只差符号，给逐步涨落大的量用。
  - 见 [`judging.md`](../judging.md#两处测量口径被测与参照的量要一样准统计量要对称)「两处测量口径」。
- `select.test_field`（被测改读另一列，golden 与噪声对仍读 `field`）。
  - 派生列 `param_norm_from_groups` 由 `group_sq.*` 现算。
- `first_step`（首步步号，`janus run` 取自 `meta.steps.first`）。
- `first_step_tol` + `first_step_source`（首步单独判，可选）。
- `hard_limit.max_step`（单步硬上限，可选）。
- `signed_mean_se_k`（可选，见下）。

**判据**：判定量取绝对值：首步差 `d0`、带符号均值、MARE、最大单步。

- 顺序：
  1. 非有限值 → `fail`；
  2. 首步超 `first_step_tol` → `fail`；
  3. 单步超硬上限 → `fail`；
  4. 再按门槛判。
- `calibrated` 只判带符号均值、MARE、最大单步（首步差是单个样本，GPU 自比时常为 0，不能当尺子）。
- `exact` 要求四个量都是 0。

## 可选机制

**带符号均值的下限**（`signed_mean_se_k = k`，默认关）：

- `calibrated` 时带符号均值的上限取 max(噪声参照的最大值, k × 噪声对标准误的最大值)。
- 防止几条参照恰好都很小、上限过紧。
- 报告里 `threshold.noise.floor` 写出实际下限。

**步号对齐**：

- `janus run` 取被测序列时按被测 `meta.steps.convention`（`golden_step = step + k`）把步号换算成 golden 步号，`first_step` 也换算。
  - `scalar_series`、`multi_golden`、`router_stats` 同样处理。
  - 噪声参照对是同一侧的两次运行，不换算。
- 张量类检查的 `select.step` 目前两边用同一个步号，不换算。

**步号偏移探测**（`auto_offset: [lo, hi]`，可选）：

- 怎么探：被测首步（换算后）的值依次与参照第 首步 + δ 步比，δ ∈ [lo, hi]。
  - 取 |Δ| 最小的 δ（并列取 |δ| 最小，0 优先）。
  - `metrics.offset_probe` 列出每个 δ 的 |Δ|。
- δ ≠ 0 即说明约定与数据不一致 → `fail`。
  - 消息里报探测到的偏移（写成被测原始步号：`golden_step = step + k + δ`）与约定。
- 顺序：探测在首步检查与窗口统计之前做。
  - 首步检查（`first_step_tol`）也在窗口统计之前，窗口因错位缺步时先报首步差，不报 error。
- 前提：探测靠首步 loss 区分相邻步，首步差要远小于相邻两步 loss 的差（GPU 自比首步差为 0，满足）。
  - 被测本身首步就不对（埋错、换框架）时探测结果不可信，这时首步检查已经先红。

**噪声参照剔重与留一法**（`calibrated`，`noise_from_pairs`）：

- 剔重：窗口内逐步完全相等的参照对是同一条轨迹的重复（如两次确定性运行）。
  - 它的噪声为 0，会把上限拉低、把参照条数虚增。
  - 剔除并在 `metrics.noise_dropped` 里列出；剔除后不足 3 条，最多给 `inconclusive`。
- 留一法互判：对每条保留下来的参照对，用其余参照对的最大值当上限算 r（带符号均值、MARE、最大单步各一个）。
  - 列在 `metrics.noise_loo`，只报不判。
  - 某一条 r 明显大于 1，说明上限由它一条撑着，参照本身不齐。

## 输出、待标定、边界

**输出**：

- `d0`、`signed_mean`（带符号原值）、`mare`、`max_step`、`max_step_at`、`window`、`denominator`、`nonfinite_steps`、`signed_mean_se`。
- `first_differ_step`：窗口内第一个两边不逐位相同的步。
  - 一边缺这一步也算；全部相同为空。
  - 日常回归二分的判据从这里取。
- `calibrated` 时 `r1`（MARE 比值）、`r2`（带符号均值比值）。

另：`noise_dropped`、`noise_loo`（`calibrated` 时）、`offset_probe`（开 `auto_offset` 时）。

**待标定**（`threshold: pending`）：按 `calibrated` 但没有噪声参照判，最多 `inconclusive`，判定量照样写出；非有限值、首步、单步硬上限照样判 `fail`。

**边界**：窗口内缺步 → `error`；缺 `window` → `error`。
