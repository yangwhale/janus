# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-dp8`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 5、fail 2、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=7.325e-07, signed_mean=5.582e-06, mare=8.835e-06, max_step=3.704e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | fail | d0=7.325e-07, signed_mean=9.222e-06, mare=1.113e-05, max_step=3.704e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=2.537e-06, mare=1.391e-05, max_step=8.045e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.727e-05, signed_mean=2.882e-05, mare=2.341e-04, max_step=1.085e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | fail | d0=-5.941e-04, signed_mean=-2.599e-04, mare=5.226e-04, max_step=1.836e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-7.346e-08, mare=1.001e-07, max_step=6.900e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=5.483e-01, mare_max=5.694e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.228e-03, mare_max=5.736e-03, max_step_max=2.424e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
