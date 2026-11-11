# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-remat-full-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=3.800e-06, mare=8.011e-06, max_step=4.146e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=4.287e-06, mare=8.490e-06, max_step=4.146e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.036e-07, signed_mean=-1.123e-06, mare=2.885e-05, max_step=1.688e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.036e-07, signed_mean=-8.127e-07, mare=3.149e-05, max_step=1.688e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-3.432e-06, signed_mean=-1.678e-04, mare=3.307e-04, max_step=1.400e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-7.047e-06, signed_mean=4.611e-04, mare=2.680e-03, max_step=4.843e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.825e-06, mare=1.838e-06, max_step=1.691e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.517e-01, mare_max=3.724e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.003e-03, mare_max=5.111e-03, max_step_max=2.513e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
