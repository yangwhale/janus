# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-ebc-dp8-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 8、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=1.952e-06, mare=8.475e-06, max_step=6.223e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=2.227e-06, mare=9.076e-06, max_step=6.223e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=1.189e-05, mare=3.143e-05, max_step=1.958e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=1.345e-05, mare=3.448e-05, max_step=1.958e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.738e-05, signed_mean=-8.318e-05, mare=3.091e-04, max_step=1.522e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-1.087e-04, signed_mean=5.538e-04, mare=3.117e-03, max_step=6.201e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-2.681e-06, mare=2.686e-06, max_step=2.046e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.683e-01, mare_max=2.232e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.255e-03, mare_max=4.927e-03, max_step_max=2.873e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
