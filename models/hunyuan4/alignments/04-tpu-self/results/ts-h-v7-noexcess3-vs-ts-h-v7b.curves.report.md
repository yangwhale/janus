# janus 报告：hunyuan4-mini / tpu-self-vs-ts-h-v7b

- 被测：`ts-h-v7-noexcess3`；golden：`ts-h-v7b`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-h-v7b.yaml`（内容指纹 3e97e2842d7e…）
- 结果：pass 7、fail 1、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=6.511e-07, signed_mean=-3.244e-07, mare=6.359e-06, max_step=2.750e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=6.511e-07, signed_mean=-2.860e-07, mare=7.457e-06, max_step=2.750e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.527e-06, signed_mean=-3.185e-06, mare=1.325e-05, max_step=8.573e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=3.529e-05, signed_mean=-4.234e-05, mare=2.125e-04, max_step=7.996e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | fail | d0=-5.993e-04, signed_mean=-3.403e-04, mare=4.971e-04, max_step=2.138e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=8.006e-08, mare=1.145e-07, max_step=4.486e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.696e-01, mare_max=5.951e-01, max_step_max=1.121e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.229e-03, mare_max=5.001e-03, max_step_max=2.471e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
