# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-f16-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | inconclusive | d0=-8.952e-07, signed_mean=5.117e-06, mare=9.573e-06, max_step=5.198e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | loss_series | L4 | inconclusive | d0=-8.952e-07, signed_mean=5.594e-06, mare=1.025e-05, max_step=5.198e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-2.036e-07, signed_mean=1.342e-07, mare=3.099e-05, max_step=1.776e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-2.036e-07, signed_mean=2.426e-07, mare=3.413e-05, max_step=1.776e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=6.521e-05, signed_mean=-6.014e-05, mare=2.875e-04, max_step=1.536e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-1.682e-04, signed_mean=5.647e-04, mare=3.210e-03, max_step=7.544e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=1.298e-08, signed_mean=-2.606e-06, mare=2.638e-06, max_step=2.294e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.885e-01, mare_max=2.081e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=9.022e-04, mare_max=4.869e-03, max_step_max=2.365e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
