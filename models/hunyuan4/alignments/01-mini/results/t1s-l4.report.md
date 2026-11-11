# janus 报告：hunyuan4-mini / t1s-l4

- 被测：`g1-a`；golden：`t1s-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1s-l4.yaml`（内容指纹 bf05a176fe8e…）
- 结果：pass 4、fail 3、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss.tail`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=6.511e-07, signed_mean_max=6.927e-06, mare_max=2.202e-05, max_step_max=2.618e-04 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=6.511e-07, signed_mean_max=3.903e-06, mare_max=8.284e-06, max_step_max=3.235e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=1.934e-06, signed_mean=9.301e-06, mare=1.565e-05, max_step=9.644e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=1.934e-06, signed_mean=1.339e-05, mare=2.020e-05, max_step=9.644e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=8.852e-04, signed_mean=8.945e-04, mare=9.571e-04, max_step=3.593e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | fail | d0=4.282e-04, signed_mean=-2.282e-04, mare=6.347e-04, max_step=2.475e-03 | 超过噪声参照两倍 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=2.991e-07, signed_mean=3.801e-07, mare=3.865e-07, max_step=8.708e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.150e-01, mare_max=2.012e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.950e-03, mare_max=5.507e-03, max_step_max=2.241e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
