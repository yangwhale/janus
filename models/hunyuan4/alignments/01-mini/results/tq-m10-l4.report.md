# janus 报告：hunyuan4-mini / tq-m10-l4

- 被测：`tq-m10`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m10-l4.yaml`（内容指纹 a9e327598227…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.indexer_loss`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=3.339e-07, mare_max=6.472e-06, max_step_max=2.295e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=4.338e-07, mare_max=7.662e-06, max_step_max=2.295e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.036e-06, signed_mean=-4.815e-08, mare=1.191e-05, max_step=8.836e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.036e-06, signed_mean=3.493e-08, mare=1.642e-05, max_step=8.836e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.292e-04, signed_mean=2.005e-04, mare=2.727e-04, max_step=9.603e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-6.097e-04, signed_mean=1.745e-05, mare=4.860e-04, max_step=1.515e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-5.172e-09, signed_mean=-9.505e-08, mare=9.505e-08, max_step=1.890e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=8.559e-02, mare_max=2.557e-01, max_step_max=4.034e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=9.911e-04, mare_max=4.941e-03, max_step_max=2.129e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
