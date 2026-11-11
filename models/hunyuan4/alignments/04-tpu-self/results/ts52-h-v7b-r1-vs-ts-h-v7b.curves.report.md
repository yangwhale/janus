# janus 报告：hunyuan4-mini / tpu-self-vs-ts-h-v7b

- 被测：`ts52-h-v7b-r1`；golden：`ts-h-v7b`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-h-v7b.yaml`（内容指纹 3e97e2842d7e…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-3.255e-07, signed_mean=1.518e-07, mare=5.544e-06, max_step=2.811e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-3.255e-07, signed_mean=-4.139e-07, mare=6.591e-06, max_step=2.811e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-6.107e-07, signed_mean=2.829e-06, mare=1.256e-05, max_step=6.237e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-7.615e-05, signed_mean=-9.256e-05, mare=2.262e-04, max_step=9.211e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-6.207e-05, signed_mean=-5.931e-05, mare=3.530e-04, max_step=1.833e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=4.957e-08, mare=5.971e-08, max_step=2.403e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=6.814e-01, mare_max=6.814e-01, max_step_max=7.009e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.561e-03, mare_max=5.030e-03, max_step_max=2.049e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
