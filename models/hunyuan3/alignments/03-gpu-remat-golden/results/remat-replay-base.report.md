# janus 报告：hunyuan3-4l / remat-replay-base

- 被测：`rv-det`；golden：`d-a`
- janus 0.0.1；目录 `/data/rc/cat/remat-replay-base.yaml`（sha256 752e2e2f2a15…）
- 结果：pass 0、fail 4、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `DET.replay.recompute_vs_baseline` | bitwise_equal | L2 | fail | tensors=57, differ=34 | 34 / 57 个张量不同 |
| `DET.replay.fwd_vs_baseline` | bitwise_equal | L2 | fail | tensors=57, differ=34 | 34 / 57 个张量不同 |
| `DET.replay.grad_vs_baseline` | bitwise_equal | L3 | fail | tensors=50, differ=50 | 50 / 50 个张量不同 |
| `DET.replay.loss_vs_baseline` | loss_series | L4 | fail | d0=7.486e-07, signed_mean=-1.358e-06, mare=1.358e-06, max_step=1.358e-06 |  |

数字全部来自 `results.json`，本报告不手写数字。
