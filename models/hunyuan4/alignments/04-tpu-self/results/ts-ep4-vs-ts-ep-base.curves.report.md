# janus 报告：hunyuan4-mini / tpu-self-vs-ts-ep-base

- 被测：`ts-ep4`；golden：`ts-ep-base`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-ep-base.yaml`（内容指纹 a041ba5e9666…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=5.091e-07, mare=4.884e-06, max_step=2.253e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=6.200e-07, mare=6.164e-06, max_step=2.253e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=2.036e-07, signed_mean=1.922e-06, mare=1.120e-05, max_step=7.471e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-5.041e-06, signed_mean=-1.985e-05, mare=1.991e-04, max_step=9.856e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=6.500e-05, signed_mean=5.114e-05, mare=3.135e-04, max_step=1.663e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-2.670e-12, signed_mean=4.247e-08, mare=4.580e-08, max_step=1.102e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.430e-01, mare_max=4.343e-01, max_step_max=1.121e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.175e-03, mare_max=5.097e-03, max_step_max=2.843e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
