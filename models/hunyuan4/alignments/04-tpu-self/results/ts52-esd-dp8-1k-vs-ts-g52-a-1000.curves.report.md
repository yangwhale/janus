# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a-1000

- 被测：`ts52-esd-dp8-1k`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a-1000.yaml`（内容指纹 aa60356a0036…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=-9.766e-07, signed_mean=1.480e-06, mare=8.205e-06, max_step=5.717e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=-9.766e-07, signed_mean=1.878e-06, mare=8.700e-06, max_step=5.717e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=1.401e-05, mare=3.207e-05, max_step=1.675e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=1.521e-05, mare=3.513e-05, max_step=1.675e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-5.985e-05, signed_mean=-1.352e-04, mare=3.305e-04, max_step=1.645e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=7.131e-05, signed_mean=4.540e-04, mare=2.845e-03, max_step=5.525e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=2.355e-07, mare=1.889e-06, max_step=1.013e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.269e-01, mare_max=2.216e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.169e-03, mare_max=5.012e-03, max_step_max=2.794e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
