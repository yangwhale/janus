# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-f-ltpu45-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=3.593e-06, mare=8.612e-06, max_step=5.388e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=4.032e-06, mare=9.195e-06, max_step=5.388e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.822e-06, mare=3.041e-05, max_step=1.677e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.638e-06, mare=3.329e-05, max_step=1.677e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=9.781e-05, mare=3.122e-04, max_step=1.605e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=5.698e-06, signed_mean=7.342e-04, mare=2.968e-03, max_step=6.134e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-2.130e-06, mare=2.140e-06, max_step=2.222e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.160e-01, mare_max=2.492e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.275e-03, mare_max=5.105e-03, max_step_max=2.234e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
