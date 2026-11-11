# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-fx-dp8-embf32`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=-1.794e-06, mare=5.326e-06, max_step=2.435e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=-2.096e-06, mare=6.011e-06, max_step=2.435e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=-1.048e-06, mare=1.290e-05, max_step=7.085e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.738e-05, signed_mean=2.866e-05, mare=2.197e-04, max_step=9.162e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.116e-04, signed_mean=-9.265e-05, mare=3.788e-04, max_step=2.003e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=4.175e-08, mare=4.601e-08, max_step=1.684e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.937e-01, mare_max=3.102e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=3.030e-03, mare_max=5.788e-03, max_step_max=2.205e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
