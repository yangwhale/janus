# janus 报告：hunyuan4-mini / tpu-self-vs-ts52-ep-base

- 被测：`ts52-ep8-r2`；golden：`ts52-ep-base`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts52-ep-base.yaml`（内容指纹 decc9f5caafc…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=8.952e-07, signed_mean=1.051e-06, mare=6.267e-06, max_step=2.921e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=8.952e-07, signed_mean=1.628e-06, mare=7.982e-06, max_step=2.921e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-4.071e-07, signed_mean=-2.087e-06, mare=1.192e-05, max_step=7.196e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=9.118e-06, signed_mean=3.731e-05, mare=2.215e-04, max_step=8.952e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-6.830e-05, signed_mean=-8.614e-05, mare=3.714e-04, max_step=1.770e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-3.445e-10, signed_mean=-2.931e-08, mare=3.698e-08, max_step=1.210e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=5.464e-01, mare_max=5.868e-01, max_step_max=9.813e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=8.219e-04, mare_max=6.073e-03, max_step_max=2.613e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
