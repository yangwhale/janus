# janus 报告：hunyuan4-mini / tq-m9-l4

- 被测：`tq-m9`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m9-l4.yaml`（内容指纹 92e2c97270b3…）
- 结果：pass 6、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.indexer_loss`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=1.058e-06, signed_mean_max=3.736e-06, mare_max=6.285e-06, max_step_max=2.775e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=3.706e-06, mare_max=6.939e-06, max_step_max=2.775e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=-1.607e-06, mare=1.123e-05, max_step=6.313e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=-2.063e-06, mare=1.545e-05, max_step=6.313e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.126e-04, signed_mean=2.092e-04, mare=2.651e-04, max_step=9.235e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.663e-04, signed_mean=1.220e-04, mare=5.038e-04, max_step=1.767e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-2.695e-09, signed_mean=-1.505e-07, mare=1.505e-07, max_step=3.059e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.857e-01, mare_max=5.028e-01, max_step_max=3.489e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.011e-03, mare_max=5.765e-03, max_step_max=2.748e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
