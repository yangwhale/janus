# janus 报告：hunyuan4-mini / tpu-self-vs-ts-ep-base

- 被测：`ts-ep8`；golden：`ts-ep-base`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-ep-base.yaml`（内容指纹 a041ba5e9666…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=-1.430e-06, mare=4.596e-06, max_step=2.749e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=-1.098e-06, mare=5.453e-06, max_step=2.749e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=2.036e-07, signed_mean=2.387e-06, mare=1.182e-05, max_step=1.029e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-5.041e-06, signed_mean=-3.754e-05, mare=1.918e-04, max_step=7.557e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=6.500e-05, signed_mean=-4.080e-05, mare=2.924e-04, max_step=1.879e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-3.445e-10, signed_mean=-2.931e-08, mare=5.043e-08, max_step=2.869e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.134e-01, mare_max=1.225e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.093e-03, mare_max=4.292e-03, max_step_max=2.420e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
