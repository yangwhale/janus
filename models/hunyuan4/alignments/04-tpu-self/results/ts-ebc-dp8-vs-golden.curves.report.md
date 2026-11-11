# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-ebc-dp8`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=-1.418e-06, mare=5.384e-06, max_step=3.141e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=-1.850e-06, mare=6.560e-06, max_step=3.141e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=2.767e-06, mare=1.204e-05, max_step=7.569e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.738e-05, signed_mean=3.345e-06, mare=2.094e-04, max_step=9.837e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.087e-04, signed_mean=-1.465e-05, mare=3.577e-04, max_step=2.024e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-4.617e-08, mare=6.337e-08, max_step=2.993e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.283e-01, mare_max=2.299e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.389e-03, mare_max=5.939e-03, max_step_max=2.873e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
