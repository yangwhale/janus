# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-f16-dp2-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=4.883e-07, signed_mean=9.122e-07, mare=6.091e-06, max_step=2.498e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=4.883e-07, signed_mean=1.452e-06, mare=7.416e-06, max_step=2.498e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=3.053e-07, signed_mean=4.116e-06, mare=1.372e-05, max_step=6.948e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.909e-05, signed_mean=-8.755e-05, mare=2.494e-04, max_step=1.089e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.915e-04, signed_mean=-7.064e-05, mare=3.669e-04, max_step=1.689e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-2.804e-10, signed_mean=-1.057e-07, mare=1.110e-07, max_step=4.032e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.535e-01, mare_max=3.585e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.696e-03, mare_max=6.505e-03, max_step_max=2.756e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
