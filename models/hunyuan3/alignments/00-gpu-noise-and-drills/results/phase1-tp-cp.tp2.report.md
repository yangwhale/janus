# janus 报告：hunyuan3-4l / phase1-tp-cp

- 被测：`p-tp2`；golden：`p-ep4f`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-tp-cp.yaml`（sha256 df6b2613cf6d…）
- 结果：pass 2、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L3.loss.step1` | loss_series | L3 | pass | d0=4.991e-06, signed_mean=1.909e-06, mare=4.881e-06, max_step=6.790e-06 | 在门槛内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=2.372e-04, signed_mean=1.966e-04, mare=1.966e-04, max_step=2.716e-04 | 在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
