# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-c-nolhs-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=3.173e-06, mare=9.101e-06, max_step=7.235e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=3.625e-06, mare=9.744e-06, max_step=7.235e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.272e-06, mare=2.814e-05, max_step=1.537e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.810e-06, mare=3.084e-05, max_step=1.537e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-8.606e-05, mare=2.937e-04, max_step=1.299e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-2.249e-07, signed_mean=5.237e-04, mare=2.671e-03, max_step=4.943e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.054e-06, mare=2.096e-06, max_step=1.953e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.528e-01, mare_max=3.548e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.151e-03, mare_max=5.097e-03, max_step_max=2.715e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
