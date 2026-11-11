# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-e-eps`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 1、fail 7、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=-9.522e-06, signed_mean=-5.853e-05, mare=7.224e-05, max_step=2.731e-04 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=-9.522e-06, signed_mean=-9.611e-05, mare=9.627e-05, max_step=2.731e-04 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | fail | d0=3.226e-05, signed_mean=1.463e-03, mare=1.463e-03, max_step=3.536e-03 | 超过噪声参照两倍 |
| `C.indexer_loss` | loss_series | L4 | fail | d0=4.199e-02, signed_mean=4.720e-02, mare=4.720e-02, max_step=5.125e-02 | 超过噪声参照两倍 |
| `C.grad_norm` | loss_series | L4 | fail | d0=-2.838e-03, signed_mean=-2.637e-03, mare=2.638e-03, max_step=5.473e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=2.316e-06, mare=2.549e-06, max_step=9.627e-06 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.848e-01, mare_max=4.365e-01, max_step_max=7.202e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | fail | signed_mean_max=3.804e-02, mare_max=3.820e-02, max_step_max=7.374e-02, stats=7 | 超过噪声参照两倍 |

数字全部来自 `results.json`，本报告不手写数字。
