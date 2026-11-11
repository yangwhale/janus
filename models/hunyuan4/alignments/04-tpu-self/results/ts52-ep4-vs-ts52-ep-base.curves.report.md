# janus 报告：hunyuan4-mini / tpu-self-vs-ts52-ep-base

- 被测：`ts52-ep4`；golden：`ts52-ep-base`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts52-ep-base.yaml`（内容指纹 decc9f5caafc…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=8.952e-07, signed_mean=-3.464e-07, mare=6.138e-06, max_step=2.773e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=8.952e-07, signed_mean=-1.404e-06, mare=7.597e-06, max_step=2.773e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-4.071e-07, signed_mean=4.014e-07, mare=1.096e-05, max_step=5.994e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=9.118e-06, signed_mean=1.616e-05, mare=1.913e-04, max_step=7.319e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-6.822e-05, signed_mean=-1.489e-05, mare=3.227e-04, max_step=1.596e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-2.670e-12, signed_mean=6.473e-08, mare=7.324e-08, max_step=5.391e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.309e-01, mare_max=3.664e-01, max_step_max=2.093e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.066e-03, mare_max=4.401e-03, max_step_max=2.424e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
