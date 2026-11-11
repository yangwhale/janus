# janus 报告：hunyuan4-mini / t1v-l4

- 被测：`t1v-a`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1v-l4.yaml`（内容指纹 7cf9449118f1…）
- 结果：pass 7、fail 1、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.139e-06, signed_mean_max=2.449e-06, mare_max=6.460e-06, max_step_max=2.430e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.139e-06, signed_mean_max=2.006e-06, mare_max=7.089e-06, max_step_max=2.430e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-8.142e-07, signed_mean=8.479e-07, mare=1.231e-05, max_step=6.380e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-8.142e-07, signed_mean=5.666e-07, mare=1.651e-05, max_step=6.380e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-4.295e-04, signed_mean=-5.060e-04, mare=5.395e-04, max_step=1.275e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-2.667e-04, signed_mean=-1.976e-05, mare=5.368e-04, max_step=1.751e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=8.252e-09, signed_mean=-1.661e-07, mare=1.681e-07, max_step=3.224e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.299e+00, mare_max=1.299e+00, max_step_max=4.187e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.011e-03, mare_max=5.925e-03, max_step_max=2.682e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
