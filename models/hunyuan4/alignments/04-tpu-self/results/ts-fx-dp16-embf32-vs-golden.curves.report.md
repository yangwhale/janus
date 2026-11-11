# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-fx-dp16-embf32`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-8.139e-07, signed_mean=3.389e-07, mare=5.726e-06, max_step=2.333e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-8.139e-07, signed_mean=1.069e-06, mare=6.590e-06, max_step=2.333e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=5.089e-07, signed_mean=-1.771e-06, mare=1.324e-05, max_step=7.646e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-1.555e-05, signed_mean=-1.970e-05, mare=2.398e-04, max_step=1.142e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.247e-04, signed_mean=-7.274e-06, mare=3.624e-04, max_step=1.459e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-7.044e-08, mare=7.343e-08, max_step=2.982e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.791e-01, mare_max=2.918e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=3.050e-03, mare_max=6.234e-03, max_step_max=2.513e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
