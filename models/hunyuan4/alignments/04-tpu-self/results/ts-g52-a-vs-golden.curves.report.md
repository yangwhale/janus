# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-g52-a`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-4.069e-07, signed_mean=-1.720e-06, mare=5.449e-06, max_step=2.564e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-4.069e-07, signed_mean=-2.674e-06, mare=6.512e-06, max_step=2.564e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.221e-06, signed_mean=6.736e-08, mare=1.371e-05, max_step=6.698e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=2.306e-05, signed_mean=-3.471e-07, mare=2.229e-04, max_step=9.461e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.144e-04, signed_mean=-8.295e-05, mare=3.441e-04, max_step=1.689e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=4.158e-08, mare=4.948e-08, max_step=2.597e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.072e-01, mare_max=2.269e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=3.053e-03, mare_max=5.590e-03, max_step_max=2.513e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
