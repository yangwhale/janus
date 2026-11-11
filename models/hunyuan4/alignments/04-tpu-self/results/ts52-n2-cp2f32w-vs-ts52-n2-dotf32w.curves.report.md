# janus 报告：hunyuan4-mini / tpu-self-vs-ts52-n2-dotf32w

- 被测：`ts52-n2-cp2f32w`；golden：`ts52-n2-dotf32w`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts52-n2-dotf32w.yaml`（内容指纹 5da2154afc65…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=1.953e-06, signed_mean=-5.459e-07, mare=5.065e-06, max_step=2.473e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=1.953e-06, signed_mean=-8.173e-07, mare=6.076e-06, max_step=2.473e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=1.018e-06, signed_mean=1.781e-08, mare=1.125e-05, max_step=8.123e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-1.931e-06, signed_mean=4.908e-06, mare=1.977e-04, max_step=8.444e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=1.429e-04, signed_mean=3.158e-05, mare=3.059e-04, max_step=1.393e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=7.123e-08, mare=8.278e-08, max_step=3.625e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.501e-01, mare_max=1.727e-01, max_step_max=3.489e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.320e-03, mare_max=5.280e-03, max_step_max=2.722e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
