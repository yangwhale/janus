# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-dp8-embf32-ga1`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.indexer_loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-7.325e-07, signed_mean=-7.485e-07, mare=5.161e-06, max_step=2.102e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-7.325e-07, signed_mean=-8.813e-08, mare=5.967e-06, max_step=2.102e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.221e-06, signed_mean=3.685e-06, mare=1.391e-05, max_step=8.968e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=-2.606e-05, signed_mean=6.773e-06, mare=2.546e-04, max_step=1.148e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-8.164e-05, signed_mean=2.230e-05, mare=3.539e-04, max_step=1.389e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-5.398e-08, mare=6.912e-08, max_step=4.499e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.712e-01, mare_max=3.952e-01, max_step_max=1.402e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.566e-03, mare_max=6.126e-03, max_step_max=2.948e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
