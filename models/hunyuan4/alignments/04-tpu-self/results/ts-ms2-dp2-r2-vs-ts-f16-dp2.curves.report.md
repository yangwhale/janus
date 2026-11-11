# janus 报告：hunyuan4-mini / tpu-self-vs-ts-f16-dp2

- 被测：`ts-ms2-dp2-r2`；golden：`ts-f16-dp2`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-f16-dp2.yaml`（内容指纹 fc00f77cf766…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.384e-06, signed_mean=-1.556e-06, mare=5.963e-06, max_step=2.415e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.384e-06, signed_mean=-1.438e-06, mare=7.356e-06, max_step=2.415e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.120e-06, signed_mean=-2.424e-06, mare=1.245e-05, max_step=7.158e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-6.886e-05, signed_mean=7.995e-05, mare=2.382e-04, max_step=9.987e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=7.169e-05, signed_mean=6.172e-05, mare=3.618e-04, max_step=1.445e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=2.804e-10, signed_mean=7.450e-08, mare=7.538e-08, max_step=2.533e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.907e-01, mare_max=5.004e-01, max_step_max=2.442e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.806e-03, mare_max=6.717e-03, max_step_max=2.782e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
