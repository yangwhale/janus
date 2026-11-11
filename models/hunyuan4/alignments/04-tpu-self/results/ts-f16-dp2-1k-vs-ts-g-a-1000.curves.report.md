# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-f16-dp2-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=4.883e-07, signed_mean=2.243e-06, mare=8.065e-06, max_step=5.167e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=4.883e-07, signed_mean=2.510e-06, mare=8.577e-06, max_step=5.167e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=3.053e-07, signed_mean=1.321e-05, mare=3.101e-05, max_step=1.662e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=3.053e-07, signed_mean=1.483e-05, mare=3.403e-05, max_step=1.662e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.909e-05, signed_mean=-9.542e-05, mare=3.107e-04, max_step=1.560e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-1.915e-04, signed_mean=3.305e-04, mare=2.913e-03, max_step=6.765e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-2.804e-10, signed_mean=-4.111e-07, mare=1.162e-06, max_step=7.002e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.319e-01, mare_max=2.334e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.582e-03, mare_max=5.175e-03, max_step_max=2.756e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
