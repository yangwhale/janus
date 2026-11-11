[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `multi_golden`

# multi_golden

**用途**：被测对多份 golden 的逐步序列。正式验收用单 golden 模式（下文）：只对一份基线比，其余 golden 只组噪声带。

**输入**：

- 被测序列、`{golden 名: 序列}`；参数同 `loss_series`。
- 噪声参照：目录 `runs.noise` 的运行对，没给就用 golden 两两组合。
- 参数 `noise_from: golden` 时不读 `runs.noise`，固定用 golden 两两组合。
  - 用途：同一份目录里 `runs.noise` 装的是另一种噪声（例如日常回归的 TPU 执行噪声，给对 LKG 的容差门用）时，对 golden 的这一条不跟着换尺子。

**单 golden 模式**（`params.baseline: <golden 名>`，正式验收用）：

- 被测只对这一份指定的 golden 算判定量。
- 其余 golden 只用来组噪声参照（没给 `runs.noise` 时取全部 golden 两两组合），不取平均、不取各份里的最差。
- 首步差与步号偏移探测也只对它做；`in_band_frac` 仍按全部 golden 算（参考）。
- 名字不在 `golden` 列表里报 `error`。
- `golden` 只列一份、噪声对写在 `runs.noise` 里，效果相同。
- 为什么这样比、代价是什么，见 [`judging.md`](../judging.md#golden-怎么用几份只造噪声带比较只对一份)「golden 怎么用」。

**判据**：

- 对每份 golden（单 golden 模式下只有基线这一份）算 `loss_series` 的判定量，取最差值判门槛。
- `calibrated` 的上限来自噪声参照。
- 另报 `in_band_frac`：被测每一步是否落在各 golden 的最小、最大值之间（参考，不参与判定）。
- `auto_offset` 对每份 golden 各探测一次，任一份上偏移不为 0 即 `fail`。
- 首步检查取各 golden 首步差的最大值，在窗口统计之前判。
- 噪声参照同样剔除逐位相同的参照对、报留一法互判（见 `loss_series`）。

**输出**：

- `<量>_max`（单 golden 模式下就是对基线的值）、`baseline`（单 golden 模式）、`goldens`、`in_band_frac`、`noise_dropped`、`noise_loo`、`offset_probe`（开探测时）。
- 明细每份 golden 一行。

**噪声参照要同一种设置**：带符号均值只在两条运行设置相同时围绕 0 随机分布。

- 「设置」包括确定性开关和是否用分布式优化器。
- 第一阶段 9 次运行实测（[`../../../models/hunyuan3/alignments/00-gpu-noise-and-drills/README.md`](../../../models/hunyuan3/alignments/00-gpu-noise-and-drills/README.md) 第 6 条）：
  - 「分布式优化器开 + 默认设置」这一组对其余三组都有持续偏移（|均值| / 标准误最大 9.1）；
  - 其余各组之间、各组内部没有（≤ 3.3）；
  - 早先只用这一组配一份确定性运行做参照，上限从 1.54e-5 被抬到 4.99e-5。
- 参照少时可加 `signed_mean_se_k`（见 `loss_series`）防上限过紧。
