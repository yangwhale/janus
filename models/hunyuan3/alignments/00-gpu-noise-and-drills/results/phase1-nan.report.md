# janus 报告：hunyuan3-4l / phase1-nan

- 被测：`f-nan`；golden：`d-a`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-nan.yaml`（sha256 e6fa3611bbcd…）
- 结果：pass 1、fail 1、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.health.grad_norm` | health_signals | L4 | fail | fields=1, problems=1 | grad_norm@5=非有限 |
| `L4.health.loss` | health_signals | L4 | pass | fields=1, problems=0 | 健康 |

数字全部来自 `results.json`，本报告不手写数字。
