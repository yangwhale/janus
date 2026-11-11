# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-s-mtp1e-3`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 1、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.271e-06, mare=5.898e-06, max_step=2.299e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.480e-06, mare=7.062e-06, max_step=2.299e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | fail | d0=1.000e-03, signed_mean=9.806e-04, mare=9.806e-04, max_step=1.032e-03 | 超过噪声参照两倍 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.142e-05, mare=2.058e-04, max_step=1.014e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=1.200e-05, signed_mean=7.595e-06, mare=3.451e-04, max_step=1.368e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-4.895e-08, mare=5.325e-08, max_step=1.901e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.363e-01, mare_max=4.813e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.364e-03, mare_max=5.212e-03, max_step_max=2.340e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
