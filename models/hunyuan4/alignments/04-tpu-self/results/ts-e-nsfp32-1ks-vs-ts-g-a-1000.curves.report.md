# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-e-nsfp32-1ks`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=8.313e-07, mare=7.992e-06, max_step=4.187e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.008e-06, mare=8.376e-06, max_step=4.187e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=3.447e-07, mare=2.923e-05, max_step=1.900e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=8.028e-07, mare=3.192e-05, max_step=1.900e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.407e-04, mare=3.440e-04, max_step=1.784e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=3.522e-04, mare=2.848e-03, max_step=6.685e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.902e-06, mare=2.906e-06, max_step=1.869e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.124e-01, mare_max=3.271e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.355e-03, mare_max=5.047e-03, max_step_max=2.692e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
