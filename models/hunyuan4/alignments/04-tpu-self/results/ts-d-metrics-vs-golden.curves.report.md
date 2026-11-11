# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-d-metrics`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=2.442e-07, signed_mean=-1.609e-06, mare=5.515e-06, max_step=2.067e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=2.442e-07, signed_mean=-9.960e-07, mare=5.866e-06, max_step=1.858e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.234e-06, mare=1.246e-05, max_step=6.637e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-3.540e-06, signed_mean=-7.571e-05, mare=2.303e-04, max_step=8.759e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=2.024e-05, signed_mean=-5.427e-05, mare=3.442e-04, max_step=1.240e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.596e-08, mare=5.833e-08, max_step=2.421e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.920e-01, mare_max=3.119e-01, max_step_max=2.693e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.602e-03, mare_max=5.643e-03, max_step_max=2.873e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
