# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-e-approx-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=-2.520e-06, mare=5.699e-06, max_step=2.342e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=-4.135e-06, mare=6.707e-06, max_step=2.342e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-5.089e-07, signed_mean=1.065e-06, mare=1.171e-05, max_step=6.178e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-4.934e-06, signed_mean=-1.129e-05, mare=2.198e-04, max_step=8.144e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-4.806e-05, signed_mean=-3.802e-05, mare=3.457e-04, max_step=1.513e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=7.387e-08, mare=7.421e-08, max_step=2.777e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.164e-01, mare_max=2.359e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.205e-03, mare_max=4.747e-03, max_step_max=2.154e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
