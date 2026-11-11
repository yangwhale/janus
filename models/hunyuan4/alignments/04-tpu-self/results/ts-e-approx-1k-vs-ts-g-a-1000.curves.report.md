# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-e-approx-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 8、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=2.036e-06, mare=8.880e-06, max_step=6.304e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=2.178e-06, mare=9.454e-06, max_step=6.304e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-5.089e-07, signed_mean=6.789e-06, mare=3.084e-05, max_step=1.555e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-5.089e-07, signed_mean=7.880e-06, mare=3.378e-05, max_step=1.555e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-4.934e-06, signed_mean=-1.710e-04, mare=3.483e-04, max_step=1.874e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.806e-05, signed_mean=6.561e-04, mare=2.729e-03, max_step=6.700e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.266e-06, mare=1.433e-06, max_step=1.343e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.159e-01, mare_max=2.493e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.267e-03, mare_max=4.761e-03, max_step_max=2.446e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
