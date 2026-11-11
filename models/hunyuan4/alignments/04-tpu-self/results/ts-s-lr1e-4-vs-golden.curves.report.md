# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-s-lr1e-4`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 4、fail 2、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.param_norm`、`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.235e-05, mare=1.281e-05, max_step=3.971e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.597e-05, mare=1.612e-05, max_step=3.971e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-7.499e-06, mare=1.439e-05, max_step=8.013e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-8.398e-05, mare=2.379e-04, max_step=1.070e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-6.566e-05, mare=3.758e-04, max_step=1.305e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=3.207e-07, mare=3.210e-07, max_step=8.836e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.222e-01, mare_max=3.222e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.326e-03, mare_max=4.953e-03, max_step_max=2.513e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
