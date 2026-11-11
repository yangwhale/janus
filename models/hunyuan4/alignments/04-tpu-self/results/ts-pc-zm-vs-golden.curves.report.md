# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-pc-zm`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 1、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=6.109e-07, mare=5.549e-06, max_step=2.994e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.534e-06, mare=6.271e-06, max_step=2.994e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-9.755e-07, mare=1.310e-05, max_step=7.585e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.022e-05, mare=2.231e-04, max_step=8.531e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-3.188e-05, mare=3.483e-04, max_step=2.087e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-8.648e-08, mare=1.046e-07, max_step=5.977e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | fail | signed_mean_max=3.137e+00, mare_max=3.137e+00, max_step_max=1.296e+02, stats=7 | 超过噪声参照两倍 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.688e-03, mare_max=4.724e-03, max_step_max=2.119e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
