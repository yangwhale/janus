# janus 报告：hunyuan4-mini / tpu-self-vs-ts-c-scar-1k-1000

- 被测：`ts-f-sc45-1ks`；golden：`ts-c-scar-1k`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-c-scar-1k-1000.yaml`（内容指纹 7de9790e802e…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=9.707e-07, mare=7.223e-06, max_step=4.096e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=9.389e-07, mare=7.656e-06, max_step=4.096e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.603e-05, mare=3.327e-05, max_step=1.618e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.774e-05, mare=3.657e-05, max_step=1.618e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.649e-05, mare=3.025e-04, max_step=1.336e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=5.773e-06, signed_mean=3.224e-04, mare=2.518e-03, max_step=6.842e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-5.005e-09, mare=1.133e-06, max_step=1.156e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.592e-01, mare_max=6.936e-01, max_step_max=2.791e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=7.308e-04, mare_max=5.394e-03, max_step_max=2.763e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
