# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`mz-g1`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=0.000e+00, mare_max=0.000e+00, max_step_max=0.000e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=0.000e+00, mare_max=0.000e+00, max_step_max=0.000e+00, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
