# janus 报告：hunyuan4-mini / tq-m40-l4

- 被测：`tq-m40`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m40-l4.yaml`（内容指纹 91cfc515929e…）
- 结果：pass 5、fail 1、inconclusive 3、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss.tail`、`L4.indexer_loss`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=1.953e-06, signed_mean_max=6.303e-06, mare_max=7.744e-06, max_step_max=2.818e-05 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=1.953e-06, signed_mean_max=6.724e-06, mare_max=8.206e-06, max_step_max=2.818e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=1.730e-06, signed_mean=1.959e-06, mare=1.174e-05, max_step=5.605e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=1.730e-06, signed_mean=1.768e-06, mare=1.546e-05, max_step=5.605e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.402e-04, signed_mean=1.580e-04, mare=2.602e-04, max_step=9.713e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-5.202e-04, signed_mean=1.314e-04, mare=4.937e-04, max_step=1.878e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.362e-09, signed_mean=-1.600e-07, mare=1.600e-07, max_step=3.986e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.056e+00, mare_max=1.073e+00, max_step_max=3.838e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.121e-03, mare_max=6.247e-03, max_step_max=2.758e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
