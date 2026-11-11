# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-s-mtp1e-1`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 1、fail 6、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=1.445e-04, mare=1.446e-04, max_step=3.848e-04 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=2.033e-04, mare=2.033e-04, max_step=3.848e-04 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | fail | d0=1.000e-01, signed_mean=9.803e-02, mare=9.803e-02, max_step=9.999e-02 | 超过噪声参照两倍 |
| `C.indexer_loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-3.013e-04, mare=4.392e-04, max_step=1.955e-03 | 超过噪声参照两倍 |
| `C.grad_norm` | loss_series | L4 | fail | d0=1.381e-03, signed_mean=5.644e-03, mare=5.644e-03, max_step=1.528e-02 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-3.191e-06, mare=3.191e-06, max_step=7.939e-06 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.549e-01, mare_max=4.867e-01, max_step_max=1.324e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=4.028e-03, mare_max=7.460e-03, max_step_max=3.171e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
