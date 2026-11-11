# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-d-square-1ks`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.970e-06, mare=8.138e-06, max_step=4.753e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.294e-06, mare=8.621e-06, max_step=4.753e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=3.648e-06, mare=3.043e-05, max_step=1.680e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=4.453e-06, mare=3.324e-05, max_step=1.680e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=5.924e-06, mare=2.998e-04, max_step=1.397e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=1.245e-05, signed_mean=3.154e-04, mare=2.808e-03, max_step=5.952e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-3.350e-06, mare=3.359e-06, max_step=1.815e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.213e-01, mare_max=1.958e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.824e-03, mare_max=4.818e-03, max_step_max=2.514e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
