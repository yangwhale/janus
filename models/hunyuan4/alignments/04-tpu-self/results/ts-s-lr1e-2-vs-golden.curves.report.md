# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-s-lr1e-2`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 1、fail 6、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.388e-03, mare=1.388e-03, max_step=2.411e-03 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.830e-03, mare=1.830e-03, max_step=2.411e-03 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-2.938e-04, mare=2.938e-04, max_step=9.681e-04 | 超过噪声参照两倍 |
| `C.indexer_loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.939e-03, mare=1.943e-03, max_step=3.961e-03 | 超过噪声参照两倍 |
| `C.grad_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.074e-03, mare=1.499e-03, max_step=7.557e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=4.757e-05, mare=4.757e-05, max_step=1.417e-04 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=5.477e-01, mare_max=5.609e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.393e-03, mare_max=6.295e-03, max_step_max=2.599e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
