# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-c-nowe`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=4.883e-07, signed_mean=-3.306e-07, mare=5.183e-06, max_step=1.940e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=4.883e-07, signed_mean=-7.901e-07, mare=5.814e-06, max_step=1.940e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=3.053e-07, signed_mean=9.125e-06, mare=1.445e-05, max_step=8.076e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.866e-05, signed_mean=1.607e-05, mare=2.139e-04, max_step=1.211e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.146e-04, signed_mean=-9.911e-05, mare=3.729e-04, max_step=1.537e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-9.796e-08, mare=9.823e-08, max_step=3.867e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.548e-01, mare_max=4.560e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.403e-03, mare_max=5.521e-03, max_step_max=2.361e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
