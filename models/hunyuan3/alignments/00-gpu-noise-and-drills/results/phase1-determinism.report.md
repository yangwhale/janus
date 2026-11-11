# janus 报告：hunyuan3-4l / phase1-determinism

- 被测：`rd-det-b`；golden：`rd-golden-1`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-determinism.yaml`（sha256 4c1a87a47afe…）
- 结果：pass 5、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss.exact` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 |  |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |
| `L3.step1.bitwise` | bitwise_equal | L3 | pass | tensors=24, differ=0 | 全部逐位相同 |
| `L3.step1.coverage` | coverage | L3 | pass |  | 全部比到 |
| `L4.step1000.bitwise` | bitwise_equal | L4 | pass | tensors=24, differ=0 | 全部逐位相同 |

数字全部来自 `results.json`，本报告不手写数字。
