# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-remat-min-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=1.729e-06, mare=7.699e-06, max_step=4.755e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=-1.058e-06, signed_mean=1.974e-06, mare=8.135e-06, max_step=4.755e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-3.053e-07, signed_mean=1.680e-05, mare=3.541e-05, max_step=1.604e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-3.053e-07, signed_mean=1.888e-05, mare=3.894e-05, max_step=1.604e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-2.832e-05, signed_mean=-3.947e-05, mare=2.961e-04, max_step=1.749e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-1.769e-05, signed_mean=2.275e-04, mare=2.801e-03, max_step=4.767e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-7.982e-07, mare=1.179e-06, max_step=8.683e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.437e-01, mare_max=2.466e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.280e-03, mare_max=5.041e-03, max_step_max=2.369e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
