# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-e-idx32`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.628e-07, signed_mean=1.634e-06, mare=5.293e-06, max_step=4.202e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.628e-07, signed_mean=2.216e-06, mare=6.375e-06, max_step=4.202e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=1.018e-07, signed_mean=-7.395e-07, mare=1.248e-05, max_step=7.944e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=9.010e-06, signed_mean=-4.707e-05, mare=2.436e-04, max_step=1.071e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-4.086e-05, signed_mean=-5.422e-05, mare=3.654e-04, max_step=1.395e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-6.294e-08, mare=6.742e-08, max_step=2.750e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.636e-01, mare_max=3.019e-01, max_step_max=1.346e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.267e-03, mare_max=5.249e-03, max_step_max=2.663e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
