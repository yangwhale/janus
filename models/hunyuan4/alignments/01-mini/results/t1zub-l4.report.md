# janus 报告：hunyuan4-mini / t1zub-l4

- 被测：`t1zu-b`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zub-l4.yaml`（内容指纹 d01f6b65f71c…）
- 结果：pass 4、fail 0、inconclusive 5、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.indexer_loss`、`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=3.581e-06, signed_mean_max=2.626e-06, mare_max=1.014e-05, max_step_max=5.030e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=3.581e-06, signed_mean_max=2.604e-06, mare_max=1.077e-05, max_step_max=5.030e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=1.018e-07, signed_mean=8.245e-06, mare=3.444e-05, max_step=1.999e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=1.018e-07, signed_mean=9.380e-06, mare=3.789e-05, max_step=1.999e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.359e-04, signed_mean=5.465e-05, mare=3.962e-04, max_step=1.830e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.985e-04, signed_mean=-2.326e-04, mare=2.609e-03, max_step=7.535e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=3.778e-09, signed_mean=-6.194e-06, mare=6.240e-06, max_step=1.839e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.442e-01, mare_max=4.620e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.560e-03, mare_max=4.758e-03, max_step_max=2.377e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
