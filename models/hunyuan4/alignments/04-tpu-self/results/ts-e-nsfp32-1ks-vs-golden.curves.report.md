# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-e-nsfp32-1ks`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.361e-06, mare=6.559e-06, max_step=3.299e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.651e-06, mare=7.578e-06, max_step=3.299e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.953e-06, mare=1.342e-05, max_step=7.771e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.120e-04, mare=2.199e-04, max_step=9.982e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-8.869e-05, mare=3.775e-04, max_step=2.000e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-7.332e-08, mare=8.587e-08, max_step=2.958e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.578e-01, mare_max=2.678e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.401e-03, mare_max=5.912e-03, max_step_max=2.692e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
