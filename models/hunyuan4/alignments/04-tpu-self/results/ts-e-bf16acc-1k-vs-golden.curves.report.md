# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-e-bf16acc-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-4.069e-07, signed_mean=1.371e-06, mare=5.932e-06, max_step=3.013e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-4.069e-07, signed_mean=1.421e-06, mare=7.033e-06, max_step=3.013e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.018e-07, signed_mean=-1.175e-06, mare=1.310e-05, max_step=7.530e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-1.739e-04, signed_mean=-9.592e-05, mare=2.195e-04, max_step=1.066e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=6.350e-05, signed_mean=-1.094e-05, mare=3.716e-04, max_step=1.616e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-7.350e-08, mare=8.327e-08, max_step=2.116e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.836e-01, mare_max=3.141e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.592e-03, mare_max=5.438e-03, max_step_max=2.513e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
