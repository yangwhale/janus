# janus 报告：hunyuan3-4l / phase2-t-nan8

- 被测：`t-nan8`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-b200x8-to-v5p8-4layers/catalogs/phase2-t-nan8.yaml`（sha256 9bf617b6d0d6…）
- 结果：pass 0、fail 2、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.health.grad_norm` | health_signals | L4 | fail | fields=1, problems=1 | grad_norm@5=非有限 |
| `L4.health.loss` | health_signals | L4 | fail | fields=1, problems=1 | lm_loss@6=非有限 |

数字全部来自 `results.json`，本报告不手写数字。
