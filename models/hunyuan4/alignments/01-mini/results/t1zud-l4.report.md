# janus 报告：hunyuan4-mini / t1zud-l4

- 被测：`t1zu-d`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zud-l4.yaml`（内容指纹 384a95fd22a2…）
- 结果：pass 6、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=3.174e-06, signed_mean_max=2.944e-06, mare_max=8.999e-06, max_step_max=4.889e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=3.174e-06, signed_mean_max=3.055e-06, mare_max=9.610e-06, max_step_max=4.889e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=2.743e-06, mare=3.249e-05, max_step=2.126e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=3.045e-06, mare=3.578e-05, max_step=2.126e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.128e-04, signed_mean=1.262e-04, mare=3.301e-04, max_step=1.520e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-5.061e-04, signed_mean=8.718e-05, mare=2.560e-03, max_step=4.501e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=1.885e-09, signed_mean=-2.209e-06, mare=2.594e-06, max_step=1.558e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=5.900e-01, mare_max=6.689e-01, max_step_max=6.619e+00, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.044e-03, mare_max=4.888e-03, max_step_max=2.400e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
