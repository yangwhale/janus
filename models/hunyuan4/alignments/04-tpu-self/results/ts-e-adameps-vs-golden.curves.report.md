# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-e-adameps`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 1、fail 6、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=7.357e-04, mare=7.357e-04, max_step=1.326e-03 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=9.804e-04, mare=9.804e-04, max_step=1.326e-03 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=1.370e-04, mare=1.371e-04, max_step=4.828e-04 | 超过噪声参照两倍 |
| `C.indexer_loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=2.275e-03, mare=2.277e-03, max_step=5.419e-03 | 超过噪声参照两倍 |
| `C.grad_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-4.678e-03, mare=4.679e-03, max_step=7.703e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-7.836e-05, mare=7.845e-05, max_step=2.704e-04 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.699e-01, mare_max=3.712e-01, max_step_max=1.090e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.158e-03, mare_max=5.753e-03, max_step_max=2.154e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
