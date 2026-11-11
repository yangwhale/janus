# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-s-loss2`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 1、fail 7、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-5.734e-04, mare=5.734e-04, max_step=1.478e-03 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-8.057e-04, mare=8.057e-04, max_step=1.478e-03 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=8.673e-03, mare=8.673e-03, max_step=2.384e-02 | 超过噪声参照两倍 |
| `C.indexer_loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=1.248e-03, mare=1.838e-03, max_step=8.255e-03 | 超过噪声参照两倍 |
| `C.grad_norm` | loss_series | L4 | fail | d0=9.878e-01, signed_mean=9.603e-01, mare=9.603e-01, max_step=9.904e-01 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=2.731e-05, mare=2.731e-05, max_step=9.288e-05 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=7.982e-01, mare_max=8.004e-01, max_step_max=1.324e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | fail | signed_mean_max=1.684e-02, mare_max=1.697e-02, max_step_max=3.891e-02, stats=7 | 超过噪声参照两倍 |

数字全部来自 `results.json`，本报告不手写数字。
