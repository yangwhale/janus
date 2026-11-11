# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-d-square`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.940e-06, mare=5.750e-06, max_step=2.911e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.419e-06, mare=6.741e-06, max_step=2.911e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.649e-06, mare=1.266e-05, max_step=8.444e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.701e-05, mare=2.089e-04, max_step=1.041e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=1.245e-05, signed_mean=-5.919e-05, mare=3.779e-04, max_step=1.577e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-4.146e-08, mare=7.150e-08, max_step=2.726e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.172e-01, mare_max=3.012e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.351e-03, mare_max=5.888e-03, max_step_max=2.514e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
