# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-f-ltpu45`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.407e-06, mare=5.666e-06, max_step=2.635e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.301e-06, mare=6.833e-06, max_step=2.635e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-4.018e-06, mare=1.215e-05, max_step=1.067e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.150e-05, mare=2.236e-04, max_step=1.241e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=5.698e-06, signed_mean=-4.248e-05, mare=3.798e-04, max_step=1.754e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.005e-07, mare=1.085e-07, max_step=2.891e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.793e-01, mare_max=3.092e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=4.021e-03, mare_max=5.411e-03, max_step_max=2.154e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
