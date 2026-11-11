# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-ep-base`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=-1.449e-07, mare=5.455e-06, max_step=2.505e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=1.187e-08, mare=6.352e-06, max_step=2.505e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-7.125e-07, signed_mean=-1.438e-06, mare=1.272e-05, max_step=9.202e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.813e-05, signed_mean=6.770e-06, mare=2.115e-04, max_step=8.567e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-6.275e-05, signed_mean=-3.140e-05, mare=3.694e-04, max_step=1.688e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-7.763e-08, mare=8.252e-08, max_step=2.649e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.284e-01, mare_max=2.617e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.117e-03, mare_max=6.550e-03, max_step_max=2.630e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
