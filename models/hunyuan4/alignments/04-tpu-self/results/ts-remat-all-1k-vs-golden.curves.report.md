# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-remat-all-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=4.637e-07, mare=6.160e-06, max_step=2.465e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=1.613e-06, mare=7.388e-06, max_step=2.465e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.018e-07, signed_mean=3.735e-06, mare=1.255e-05, max_step=6.627e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=2.231e-05, signed_mean=9.543e-06, mare=2.210e-04, max_step=1.221e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=9.491e-05, signed_mean=2.947e-05, mare=3.639e-04, max_step=1.636e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-5.793e-08, mare=9.810e-08, max_step=5.931e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.421e-01, mare_max=3.613e-01, max_step_max=9.768e-01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.111e-03, mare_max=5.609e-03, max_step_max=2.361e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
