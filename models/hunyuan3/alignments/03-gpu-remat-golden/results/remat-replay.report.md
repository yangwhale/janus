# janus 报告：hunyuan3-4l / remat-replay

- 被测：`rv-det`；golden：`r-remat`
- janus 0.0.1；目录 `/data/rc/cat/remat-replay.yaml`（sha256 c61444dc23df…）
- 结果：pass 3、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `DET.replay.vs_first` | bitwise_equal | L2 | pass | tensors=57, differ=0 | 全部逐位相同 |
| `DET.replay.fwd_vs_remat` | bitwise_equal | L2 | pass | tensors=57, differ=0 | 全部逐位相同 |
| `DET.replay.loss_vs_remat` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 |  |

数字全部来自 `results.json`，本报告不手写数字。
