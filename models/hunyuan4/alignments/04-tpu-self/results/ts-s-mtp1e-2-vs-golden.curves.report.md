# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-s-mtp1e-2`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 3、fail 5、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=1.859e-05, mare=1.933e-05, max_step=6.398e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=2.725e-05, mare=2.735e-05, max_step=6.398e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | fail | d0=1.000e-02, signed_mean=9.818e-03, mare=9.818e-03, max_step=1.000e-02 | 超过噪声参照两倍 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-5.868e-05, mare=2.309e-04, max_step=1.038e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | fail | d0=1.296e-04, signed_mean=4.797e-04, mare=6.014e-04, max_step=2.323e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-5.855e-07, mare=5.855e-07, max_step=1.910e-06 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.102e-01, mare_max=4.296e-01, max_step_max=1.346e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=3.034e-03, mare_max=5.647e-03, max_step_max=2.562e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
