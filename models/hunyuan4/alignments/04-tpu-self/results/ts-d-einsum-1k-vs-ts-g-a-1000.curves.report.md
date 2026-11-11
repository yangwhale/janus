# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-d-einsum-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 4、fail 0、inconclusive 5、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`、`L4.param_norm`、`L4.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=2.032e-06, mare=8.492e-06, max_step=9.399e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=2.380e-06, mare=8.965e-06, max_step=9.399e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-3.053e-07, signed_mean=5.015e-06, mare=3.069e-05, max_step=1.625e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-3.053e-07, signed_mean=6.085e-06, mare=3.344e-05, max_step=1.625e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-1.094e-05, signed_mean=7.744e-05, mare=3.124e-04, max_step=1.601e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-3.374e-06, signed_mean=5.570e-04, mare=3.192e-03, max_step=1.026e-01 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-3.937e-06, mare=3.958e-06, max_step=2.299e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.290e-01, mare_max=3.347e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=2.163e-03, mare_max=5.136e-03, max_step_max=2.513e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
