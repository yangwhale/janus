# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a-1000

- 被测：`ts-g52-c`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a-1000.yaml`（内容指纹 aa60356a0036…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=0.000e+00, mare=0.000e+00, max_step=0.000e+00 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=0.000e+00, mare_max=0.000e+00, max_step_max=0.000e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=0.000e+00, mare_max=0.000e+00, max_step_max=0.000e+00, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
