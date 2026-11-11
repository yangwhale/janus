# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-d-rsqrt-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=4.069e-07, signed_mean=-1.521e-06, mare=4.993e-06, max_step=1.991e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=4.069e-07, signed_mean=-1.027e-06, mare=5.585e-06, max_step=1.991e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-7.125e-07, signed_mean=1.658e-06, mare=1.254e-05, max_step=6.387e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=5.127e-05, signed_mean=4.526e-05, mare=2.271e-04, max_step=9.570e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-8.262e-05, signed_mean=-8.824e-05, mare=3.770e-04, max_step=1.957e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.178e-08, mare=4.629e-08, max_step=2.882e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.259e-01, mare_max=2.821e-01, max_step_max=1.346e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.893e-03, mare_max=6.614e-03, max_step_max=2.556e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
