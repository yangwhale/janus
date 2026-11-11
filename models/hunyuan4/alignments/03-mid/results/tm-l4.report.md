# janus 报告：hunyuan4-mid / tm-l4

- 被测：`tm-a`；golden：`gm-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/03-mid/catalogs/tm-l4.yaml`（内容指纹 6a3678043459…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.mtp_loss`、`L4.mtp_loss.tail`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=1.896e-05, signed_mean_max=4.649e-06, mare_max=1.637e-05, max_step_max=6.599e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.896e-05, signed_mean_max=3.003e-06, mare_max=1.707e-05, max_step_max=5.813e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=-8.175e-06, signed_mean=6.965e-06, mare=1.655e-05, max_step=8.812e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=-8.175e-06, signed_mean=1.140e-05, mare=2.028e-05, max_step=8.812e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.237e-04, signed_mean=2.630e-04, mare=3.343e-04, max_step=1.180e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-9.464e-04, signed_mean=-7.374e-06, mare=7.053e-04, max_step=4.294e-03 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=2.468e-09, signed_mean=-9.030e-08, mare=9.215e-08, max_step=2.301e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.078e-01, mare_max=2.267e-01, max_step_max=1.660e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.696e-03, mare_max=2.546e-03, max_step_max=1.034e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
