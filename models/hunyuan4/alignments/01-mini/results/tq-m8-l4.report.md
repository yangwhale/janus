# janus 报告：hunyuan4-mini / tq-m8-l4

- 被测：`tq-m8`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m8-l4.yaml`（内容指纹 6c253ea3f2a4…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.indexer_loss`、`L4.grad_norm`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=1.058e-06, signed_mean_max=1.170e-06, mare_max=6.760e-06, max_step_max=3.182e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=2.137e-06, mare_max=8.153e-06, max_step_max=3.182e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=3.361e-06, mare=1.399e-05, max_step=9.439e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=3.794e-06, mare=1.887e-05, max_step=9.439e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.126e-04, signed_mean=1.652e-04, mare=2.702e-04, max_step=9.661e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.628e-04, signed_mean=1.503e-04, mare=5.323e-04, max_step=1.733e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-5.048e-09, signed_mean=8.357e-08, mare=1.275e-07, max_step=6.703e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=1.306e+00, mare_max=1.308e+00, max_step_max=4.535e+01, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.233e-03, mare_max=5.251e-03, max_step_max=2.176e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
