# janus 报告：hunyuan4-mini / tq-m28-l4

- 被测：`tq-m28`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m28-l4.yaml`（内容指纹 59932d99c03d…）
- 结果：pass 6、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.indexer_loss`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=8.139e-07, signed_mean_max=3.773e-06, mare_max=6.444e-06, max_step_max=2.828e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=8.139e-07, signed_mean_max=3.787e-06, mare_max=7.221e-06, max_step_max=2.828e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=3.053e-06, signed_mean=-5.769e-06, mare=1.244e-05, max_step=8.208e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=3.053e-06, signed_mean=-8.157e-06, mare=1.702e-05, max_step=8.208e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=9.234e-05, signed_mean=1.006e-04, mare=2.554e-04, max_step=9.437e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-5.465e-04, signed_mean=7.702e-05, mare=5.037e-04, max_step=1.493e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-6.825e-10, signed_mean=-1.731e-07, mare=1.731e-07, max_step=3.875e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=6.849e-01, mare_max=6.849e-01, max_step_max=2.791e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.326e-03, mare_max=4.404e-03, max_step_max=2.656e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
