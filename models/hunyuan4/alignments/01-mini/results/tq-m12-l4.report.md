# janus 报告：hunyuan4-mini / tq-m12-l4

- 被测：`tq-m12`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m12-l4.yaml`（内容指纹 21e69f23d22f…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.221e-06, signed_mean_max=1.493e-06, mare_max=6.528e-06, max_step_max=3.202e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.221e-06, signed_mean_max=1.308e-06, mare_max=7.555e-06, max_step_max=3.202e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.443e-06, signed_mean=-5.940e-06, mare=1.219e-05, max_step=5.999e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.443e-06, signed_mean=-8.544e-06, mare=1.681e-05, max_step=5.999e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-5.974e-05, signed_mean=5.067e-05, mare=2.210e-04, max_step=8.215e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.010e-04, signed_mean=4.031e-05, mare=5.174e-04, max_step=1.654e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=-3.659e-09, signed_mean=-2.010e-07, mare=2.010e-07, max_step=4.265e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.691e-02, mare_max=1.377e-01, max_step_max=1.402e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.097e-03, mare_max=4.869e-03, max_step_max=2.900e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
