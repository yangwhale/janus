# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-c-nolhs-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-5.384e-07, mare=5.422e-06, max_step=2.562e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-3.439e-07, mare=6.498e-06, max_step=2.562e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.243e-06, mare=1.142e-05, max_step=5.445e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.056e-05, mare=2.222e-04, max_step=8.539e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-2.249e-07, signed_mean=-6.830e-05, mare=3.503e-04, max_step=1.551e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-4.367e-08, mare=4.817e-08, max_step=1.770e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.621e-01, mare_max=4.687e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.583e-03, mare_max=6.052e-03, max_step_max=2.715e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
