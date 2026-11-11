# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-c-nocopy`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=4.534e-07, mare=5.765e-06, max_step=2.921e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.146e-06, mare=6.772e-06, max_step=2.921e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-3.764e-06, mare=1.364e-05, max_step=8.494e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-6.821e-05, mare=2.390e-04, max_step=8.755e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-4.639e-05, mare=3.736e-04, max_step=1.582e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.695e-07, mare=1.742e-07, max_step=6.940e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.065e-01, mare_max=3.147e-01, max_step_max=2.693e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.908e-03, mare_max=5.716e-03, max_step_max=2.469e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
