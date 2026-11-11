# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-mb2-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=8.139e-07, signed_mean=2.227e-06, mare=7.826e-06, max_step=4.414e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=8.139e-07, signed_mean=2.513e-06, mare=8.300e-06, max_step=4.414e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-1.221e-06, signed_mean=7.346e-06, mare=2.948e-05, max_step=1.351e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-1.221e-06, signed_mean=8.549e-06, mare=3.227e-05, max_step=1.351e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-1.373e-05, signed_mean=-1.972e-04, mare=3.655e-04, max_step=1.798e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-1.277e-04, signed_mean=3.492e-04, mare=2.711e-03, max_step=6.634e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-3.984e-06, mare=3.985e-06, max_step=1.790e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.413e-01, mare_max=3.578e-01, max_step_max=2.693e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.409e-03, mare_max=5.523e-03, max_step_max=2.521e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
