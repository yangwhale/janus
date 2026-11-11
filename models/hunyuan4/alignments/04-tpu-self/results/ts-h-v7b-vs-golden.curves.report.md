# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-h-v7b`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=-5.391e-07, mare=5.938e-06, max_step=2.218e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=-1.988e-07, mare=6.939e-06, max_step=2.218e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-5.089e-07, signed_mean=-1.846e-06, mare=1.208e-05, max_step=6.951e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=6.103e-05, signed_mean=1.713e-05, mare=2.003e-04, max_step=8.431e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=3.524e-05, signed_mean=5.821e-06, mare=3.924e-04, max_step=1.706e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=-3.436e-08, signed_mean=-6.109e-08, mare=7.657e-08, max_step=2.262e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.003e-01, mare_max=3.594e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.920e-03, mare_max=5.284e-03, max_step_max=2.673e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
