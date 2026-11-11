# janus 报告：hunyuan4-mini / tq-m23-l4

- 被测：`tq-m23`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m23-l4.yaml`（内容指纹 7eff43d25bf2…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.indexer_loss`、`L4.grad_norm`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=9.766e-07, signed_mean_max=3.249e-06, mare_max=6.596e-06, max_step_max=2.285e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=9.766e-07, signed_mean_max=2.227e-06, mare_max=6.798e-06, max_step_max=2.285e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=-3.297e-06, mare=1.283e-05, max_step=8.714e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=-5.863e-06, mare=1.732e-05, max_step=8.714e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.243e-04, signed_mean=2.051e-04, mare=2.713e-04, max_step=1.028e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-5.044e-04, signed_mean=5.160e-05, mare=4.851e-04, max_step=1.493e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-5.833e-09, signed_mean=-7.461e-08, mare=8.040e-08, max_step=1.743e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=1.621e+00, mare_max=1.621e+00, max_step_max=4.535e+01, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.749e-03, mare_max=4.028e-03, max_step_max=2.053e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
