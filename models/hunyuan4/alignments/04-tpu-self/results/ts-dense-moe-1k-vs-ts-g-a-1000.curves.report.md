# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-dense-moe-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 8、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=2.830e-06, mare=7.992e-06, max_step=5.189e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=3.147e-06, mare=8.495e-06, max_step=5.189e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=1.018e-07, signed_mean=-1.682e-05, mare=3.188e-05, max_step=1.582e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=1.018e-07, signed_mean=-1.824e-05, mare=3.488e-05, max_step=1.582e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-2.102e-05, signed_mean=-2.033e-04, mare=3.493e-04, max_step=1.723e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=3.194e-05, signed_mean=5.546e-04, mare=2.833e-03, max_step=8.889e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=3.415e-07, mare=1.556e-06, max_step=1.210e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.332e-01, mare_max=4.526e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.266e-03, mare_max=5.396e-03, max_step_max=2.614e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
