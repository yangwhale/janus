# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-g52-a`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=-4.069e-07, signed_mean=1.189e-06, mare=7.866e-06, max_step=4.646e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=-4.069e-07, signed_mean=1.297e-06, mare=8.368e-06, max_step=4.646e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-1.221e-06, signed_mean=-7.458e-07, mare=3.012e-05, max_step=1.557e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-1.221e-06, signed_mean=-5.530e-07, mare=3.300e-05, max_step=1.557e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=2.306e-05, signed_mean=-1.348e-05, mare=3.015e-04, max_step=1.261e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-1.144e-04, signed_mean=3.032e-04, mare=2.815e-03, max_step=5.763e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-9.553e-07, mare=1.370e-06, max_step=1.495e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.627e-01, mare_max=2.825e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.169e-03, mare_max=5.019e-03, max_step_max=2.624e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
