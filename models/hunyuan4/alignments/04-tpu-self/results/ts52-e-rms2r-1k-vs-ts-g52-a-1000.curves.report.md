# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a-1000

- 被测：`ts52-e-rms2r-1k`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a-1000.yaml`（内容指纹 aa60356a0036…）
- 结果：pass 7、fail 2、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | fail | d0=4.069e-07, signed_mean=1.711e-05, mare=1.747e-05, max_step=5.665e-05 | 超过噪声参照两倍 |
| `L4.loss.tail` | loss_series | L4 | fail | d0=4.069e-07, signed_mean=1.848e-05, mare=1.881e-05, max_step=5.665e-05 | 超过噪声参照两倍 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=3.053e-07, signed_mean=2.231e-05, mare=3.617e-05, max_step=2.111e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=3.053e-07, signed_mean=2.472e-05, mare=3.983e-05, max_step=2.111e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-3.218e-06, signed_mean=-9.933e-05, mare=3.003e-04, max_step=1.915e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-2.189e-05, signed_mean=6.673e-06, mare=2.680e-03, max_step=6.305e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.792e-06, mare=2.841e-06, max_step=1.927e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.084e-01, mare_max=3.104e-01, max_step_max=4.034e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.174e-03, mare_max=4.769e-03, max_step_max=2.270e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
