# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-h-v7-1k3`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=1.172e-06, mare=7.207e-06, max_step=3.829e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=1.436e-06, mare=7.569e-06, max_step=3.829e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-5.089e-07, signed_mean=5.191e-06, mare=2.979e-05, max_step=1.536e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-5.089e-07, signed_mean=6.028e-06, mare=3.269e-05, max_step=1.536e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=6.103e-05, signed_mean=-1.100e-04, mare=3.043e-04, max_step=1.749e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=3.524e-05, signed_mean=1.837e-04, mare=2.659e-03, max_step=5.912e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-3.436e-08, signed_mean=3.303e-07, mare=9.560e-07, max_step=4.630e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.483e-01, mare_max=2.796e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=8.989e-04, mare_max=4.599e-03, max_step_max=2.673e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
