# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-mb2`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=8.139e-07, signed_mean=6.157e-07, mare=5.680e-06, max_step=2.244e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=8.139e-07, signed_mean=1.107e-06, mare=6.751e-06, max_step=2.244e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.221e-06, signed_mean=1.129e-06, mare=1.290e-05, max_step=6.483e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-1.373e-05, signed_mean=-6.913e-06, mare=2.286e-04, max_step=9.869e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-1.277e-04, signed_mean=-1.424e-05, mare=3.971e-04, max_step=1.608e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-8.337e-08, mare=8.567e-08, max_step=2.689e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.073e-01, mare_max=3.135e-01, max_step_max=2.693e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.585e-03, mare_max=6.674e-03, max_step_max=2.521e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
