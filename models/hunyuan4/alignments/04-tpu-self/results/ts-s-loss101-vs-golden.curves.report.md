# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-s-loss101`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 3、fail 4、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.573e-05, mare=1.641e-05, max_step=5.885e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-2.185e-05, mare=2.200e-05, max_step=5.885e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=1.723e-04, mare=1.723e-04, max_step=4.884e-04 | 超过噪声参照两倍 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=4.095e-05, mare=2.333e-04, max_step=1.076e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | fail | d0=9.752e-03, signed_mean=9.378e-03, mare=9.378e-03, max_step=1.090e-02 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=3.369e-07, mare=3.369e-07, max_step=9.694e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.007e-01, mare_max=4.356e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.518e-03, mare_max=5.682e-03, max_step_max=2.367e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
