# janus 报告：hunyuan4-mini / t1zuc-l4

- 被测：`t1zu-c`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zuc-l4.yaml`（内容指纹 48c207562c6d…）
- 结果：pass 4、fail 0、inconclusive 5、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.indexer_loss`、`L4.router.bias_mean`、`L4.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=3.744e-06, signed_mean_max=7.204e-06, mare_max=1.068e-05, max_step_max=5.857e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=3.744e-06, signed_mean_max=7.476e-06, mare_max=1.116e-05, max_step_max=5.857e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=1.425e-06, signed_mean=4.028e-06, mare=3.353e-05, max_step=1.717e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=1.425e-06, signed_mean=4.198e-06, mare=3.679e-05, max_step=1.717e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.585e-04, signed_mean=2.512e-04, mare=3.783e-04, max_step=1.734e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-5.723e-04, signed_mean=2.935e-04, mare=2.539e-03, max_step=4.917e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=3.180e-09, signed_mean=-1.683e-06, mare=2.091e-06, max_step=1.250e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=7.875e-01, mare_max=7.902e-01, max_step_max=9.813e+00, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=1.736e-03, mare_max=4.907e-03, max_step_max=2.758e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
