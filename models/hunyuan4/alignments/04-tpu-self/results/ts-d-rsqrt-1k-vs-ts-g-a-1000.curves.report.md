# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-d-rsqrt-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=4.069e-07, signed_mean=2.592e-06, mare=8.085e-06, max_step=4.486e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=4.069e-07, signed_mean=3.154e-06, mare=8.556e-06, max_step=4.486e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-7.125e-07, signed_mean=3.008e-06, mare=2.853e-05, max_step=1.634e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-7.125e-07, signed_mean=3.638e-06, mare=3.124e-05, max_step=1.634e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=5.127e-05, signed_mean=7.238e-05, mare=3.102e-04, max_step=1.339e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-8.262e-05, signed_mean=4.585e-04, mare=2.475e-03, max_step=4.261e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-6.527e-07, mare=1.254e-06, max_step=9.024e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.208e-01, mare_max=2.766e-01, max_step_max=1.346e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.771e-03, mare_max=5.362e-03, max_step_max=2.556e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
