# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-remat-min-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=-1.155e-06, mare=5.773e-06, max_step=2.447e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=-1.478e-06, mare=6.780e-06, max_step=2.447e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-3.053e-07, signed_mean=7.911e-06, mare=1.488e-05, max_step=8.108e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-2.832e-05, signed_mean=-2.746e-05, mare=2.390e-04, max_step=1.182e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-1.769e-05, signed_mean=-1.027e-04, mare=3.794e-04, max_step=1.741e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.329e-08, mare=5.064e-08, max_step=1.864e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.543e-01, mare_max=2.566e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.138e-03, mare_max=5.554e-03, max_step_max=2.309e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
