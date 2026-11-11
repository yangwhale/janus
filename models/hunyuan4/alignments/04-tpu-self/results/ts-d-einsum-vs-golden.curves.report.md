# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-d-einsum`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 6、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`C.indexer_loss`、`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.015e-06, mare=6.064e-06, max_step=2.888e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-9.551e-07, mare=6.990e-06, max_step=2.888e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-3.053e-07, signed_mean=-9.917e-06, mare=1.523e-05, max_step=7.860e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=-1.094e-05, signed_mean=5.874e-05, mare=2.623e-04, max_step=1.199e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-3.374e-06, signed_mean=-3.127e-05, mare=4.243e-04, max_step=1.747e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.754e-08, mare=4.159e-08, max_step=1.836e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=5.183e-01, mare_max=5.354e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.458e-03, mare_max=5.742e-03, max_step_max=2.513e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
