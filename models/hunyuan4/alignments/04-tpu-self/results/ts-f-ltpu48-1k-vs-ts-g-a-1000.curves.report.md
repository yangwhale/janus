# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-f-ltpu48-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=-2.442e-07, signed_mean=3.794e-06, mare=8.128e-06, max_step=4.842e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=-2.442e-07, signed_mean=4.182e-06, mare=8.627e-06, max_step=4.842e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=7.493e-06, mare=2.912e-05, max_step=1.784e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=8.652e-06, mare=3.189e-05, max_step=1.784e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-8.581e-07, signed_mean=-6.156e-05, mare=2.921e-04, max_step=1.398e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=9.971e-06, signed_mean=3.319e-04, mare=2.687e-03, max_step=5.678e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.689e-06, mare=2.692e-06, max_step=1.692e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.921e-01, mare_max=3.246e-01, max_step_max=8.069e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.314e-03, mare_max=4.870e-03, max_step_max=2.798e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
