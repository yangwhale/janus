# janus 报告：hunyuan3-4l / phase1-spike

- 被测：`f-spike`；golden：`d-a`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-spike.yaml`（sha256 b7bb8f00ec2e…）
- 结果：pass 0、fail 2、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.health` | health_signals | L4 | fail | fields=2, problems=1 | grad_norm@16=387（中位数 19.1 的 20.2 倍） |
| `L4.loss.vs_golden` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=2.725e-01, mare=2.725e-01, max_step=1.012e+00 |  |

数字全部来自 `results.json`，本报告不手写数字。
