# janus 报告：hunyuan4-mini / tpu-self-v7x-1k-vs-gpu16

- 被测：`ts-h-v7-1k3`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/v7x-1k-vs-gpu16.yaml`（内容指纹 fbcb674c5c6d…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=2.523e-06, signed_mean_max=2.531e-06, mare_max=8.265e-06, max_step_max=4.048e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=2.523e-06, signed_mean_max=2.608e-06, mare_max=8.678e-06, max_step_max=4.048e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=1.730e-06, signed_mean=-2.912e-06, mare=3.088e-05, max_step=1.814e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=1.730e-06, signed_mean=-3.237e-06, mare=3.394e-05, max_step=1.814e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-6.328e-06, signed_mean=-2.366e-04, mare=3.660e-04, max_step=1.846e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-4.870e-04, signed_mean=-2.653e-04, mare=2.435e-03, max_step=5.242e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=2.846e-09, signed_mean=8.141e-07, mare=1.085e-06, max_step=9.612e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=6.746e-01, mare_max=6.783e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.137e-03, mare_max=4.628e-03, max_step_max=2.485e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
