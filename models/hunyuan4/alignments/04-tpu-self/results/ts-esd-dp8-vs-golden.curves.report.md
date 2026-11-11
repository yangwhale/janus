# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-esd-dp8`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=-1.117e-06, mare=5.853e-06, max_step=2.466e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=-1.195e-06, mare=6.713e-06, max_step=2.466e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=5.306e-06, mare=1.597e-05, max_step=9.398e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.716e-05, signed_mean=9.587e-06, mare=2.402e-04, max_step=9.411e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-1.032e-04, signed_mean=-3.823e-05, mare=3.917e-04, max_step=1.719e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-5.447e-09, mare=4.437e-08, max_step=1.391e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.211e-01, mare_max=3.058e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.721e-03, mare_max=5.063e-03, max_step_max=2.361e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
