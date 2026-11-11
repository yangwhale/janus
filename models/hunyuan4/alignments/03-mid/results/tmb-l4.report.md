# janus 报告：hunyuan4-mid / tmb-l4

- 被测：`tm-b`；golden：`gm-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/03-mid/catalogs/tmb-l4.yaml`（内容指纹 d569e4162e8b…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.indexer_loss`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=1.517e-05, signed_mean_max=6.410e-06, mare_max=1.625e-05, max_step_max=5.941e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=1.517e-05, signed_mean_max=4.947e-06, mare_max=1.647e-05, max_step_max=5.941e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-9.891e-06, signed_mean=1.691e-05, mare=2.118e-05, max_step=7.607e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-9.891e-06, signed_mean=2.429e-05, mare=2.601e-05, max_step=7.607e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.323e-04, signed_mean=3.607e-04, mare=3.986e-04, max_step=1.404e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-8.564e-04, signed_mean=7.433e-05, mare=6.773e-04, max_step=3.669e-03 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=3.382e-09, signed_mean=-1.603e-07, mare=1.609e-07, max_step=3.271e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.525e-01, mare_max=2.593e-01, max_step_max=6.349e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.342e-03, mare_max=2.027e-03, max_step_max=9.221e-03, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
