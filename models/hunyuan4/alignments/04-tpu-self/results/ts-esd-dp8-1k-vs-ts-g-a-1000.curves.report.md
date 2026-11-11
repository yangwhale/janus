# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-esd-dp8-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 8、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=3.023e-06, mare=8.229e-06, max_step=4.240e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=3.461e-06, mare=8.681e-06, max_step=4.240e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=1.072e-05, mare=3.302e-05, max_step=2.003e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=1.218e-05, mare=3.620e-05, max_step=2.003e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.716e-05, signed_mean=-6.076e-05, mare=3.229e-04, max_step=1.805e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-1.032e-04, signed_mean=1.370e-04, mare=3.375e-03, max_step=8.036e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-2.515e-06, mare=2.526e-06, max_step=1.777e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.583e-01, mare_max=1.945e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.596e-03, mare_max=4.915e-03, max_step_max=2.361e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
