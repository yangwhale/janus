# janus 报告：hunyuan3-4l / phase1-tp-cp

- 被测：`p-cp2`；golden：`p-ep4f`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-tp-cp.yaml`（sha256 df6b2613cf6d…）
- 结果：pass 2、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L3.loss.step1` | loss_series | L3 | pass | d0=2.911e-06, signed_mean=-1.408e-06, mare=4.110e-06, max_step=5.517e-06 | 在门槛内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=4.590e-05, signed_mean=1.050e-05, mare=2.667e-05, max_step=3.717e-05 | 在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
