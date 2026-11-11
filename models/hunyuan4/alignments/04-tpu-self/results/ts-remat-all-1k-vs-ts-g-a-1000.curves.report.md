# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-remat-all-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 6、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`L4.mtp_loss`、`L4.mtp_loss.tail`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=2.564e-06, mare=7.522e-06, max_step=4.621e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=3.050e-06, mare=7.945e-06, max_step=4.621e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=-1.018e-07, signed_mean=2.642e-05, mare=3.811e-05, max_step=2.232e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=-1.018e-07, signed_mean=2.950e-05, mare=4.198e-05, max_step=2.232e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=2.231e-05, signed_mean=-5.179e-05, mare=3.079e-04, max_step=1.457e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=9.491e-05, signed_mean=9.226e-05, mare=3.044e-03, max_step=5.654e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.367e-06, mare=2.379e-06, max_step=1.393e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.135e-01, mare_max=2.351e-01, max_step_max=9.768e-01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.418e-03, mare_max=5.137e-03, max_step_max=2.705e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
