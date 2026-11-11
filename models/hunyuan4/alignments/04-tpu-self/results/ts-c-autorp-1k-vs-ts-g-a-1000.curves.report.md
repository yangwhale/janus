# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-c-autorp-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=3.583e-06, mare=8.683e-06, max_step=6.789e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=4.226e-06, mare=9.211e-06, max_step=6.789e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.029e-05, mare=3.130e-05, max_step=1.846e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.108e-05, mare=3.431e-05, max_step=1.846e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-5.107e-05, mare=2.992e-04, max_step=1.706e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=7.049e-04, mare=2.760e-03, max_step=7.693e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-3.703e-06, mare=3.707e-06, max_step=2.519e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=8.584e-02, mare_max=1.394e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.654e-03, mare_max=4.777e-03, max_step_max=2.154e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
