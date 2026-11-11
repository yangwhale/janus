# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-f-sc45-1ks`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.043e-07, mare=5.203e-06, max_step=2.493e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.343e-07, mare=5.860e-06, max_step=2.493e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.331e-06, mare=1.232e-05, max_step=7.437e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.110e-05, mare=2.254e-04, max_step=1.159e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=5.773e-06, signed_mean=-1.427e-05, mare=3.553e-04, max_step=1.288e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-6.441e-08, mare=6.930e-08, max_step=2.299e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=5.327e-01, mare_max=5.333e-01, max_step_max=6.051e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.960e-03, mare_max=5.561e-03, max_step_max=2.711e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
