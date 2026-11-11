# janus 报告：hunyuan4-mini / tq-m25-l4

- 被测：`tq-m25`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m25-l4.yaml`（内容指纹 6077c6859a96…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.mtp_loss`、`L4.mtp_loss.tail`、`L4.indexer_loss`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=2.803e-06, mare_max=5.830e-06, max_step_max=2.641e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=2.730e-06, mare_max=6.695e-06, max_step_max=2.641e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=2.646e-06, signed_mean=2.348e-06, mare=1.161e-05, max_step=1.245e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=2.646e-06, signed_mean=3.136e-06, mare=1.594e-05, max_step=1.245e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.139e-04, signed_mean=1.968e-04, mare=2.658e-04, max_step=1.113e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.709e-04, signed_mean=4.441e-05, mare=4.813e-04, max_step=1.456e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-4.422e-09, signed_mean=-1.331e-07, mare=1.331e-07, max_step=3.182e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.260e-01, mare_max=1.778e-01, max_step_max=4.206e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=5.558e-04, mare_max=4.763e-03, max_step_max=2.350e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
