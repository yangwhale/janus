# janus 报告：hunyuan3-4l / phase2-v7x-t-nan8

- 被测：`t-v7x-nan8`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-b200x8-to-v7x4-4layers/catalogs/phase2-v7x-t-nan8.yaml`（sha256 f45466fb9c54…）
- 结果：pass 0、fail 2、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.health.grad_norm` | health_signals | L4 | fail | fields=1, problems=1 | grad_norm@5=非有限 |
| `L4.health.loss` | health_signals | L4 | fail | fields=1, problems=1 | lm_loss@6=非有限 |

数字全部来自 `results.json`，本报告不手写数字。
