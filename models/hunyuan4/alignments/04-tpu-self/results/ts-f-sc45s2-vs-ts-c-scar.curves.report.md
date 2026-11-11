# janus 报告：hunyuan4-mini / tpu-self-vs-ts-c-scar

- 被测：`ts-f-sc45s2`；golden：`ts-c-scar`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-c-scar.yaml`（内容指纹 0c29dd7a3ecf…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.402e-06, mare=4.856e-06, max_step=1.927e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.473e-06, mare=5.634e-06, max_step=1.927e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-4.139e-06, mare=1.308e-05, max_step=8.936e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.017e-05, mare=2.094e-04, max_step=9.215e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=5.773e-06, signed_mean=5.574e-05, mare=3.466e-04, max_step=2.225e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-3.556e-09, mare=3.511e-08, max_step=1.597e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.370e+00, mare_max=1.480e+00, max_step_max=2.791e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.546e-03, mare_max=6.178e-03, max_step_max=2.763e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
