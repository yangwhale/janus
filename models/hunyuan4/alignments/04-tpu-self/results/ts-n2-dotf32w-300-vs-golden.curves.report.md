# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-n2-dotf32w-300`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-5.697e-07, signed_mean=1.997e-06, mare=5.722e-06, max_step=3.072e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-5.697e-07, signed_mean=3.040e-06, mare=6.856e-06, max_step=3.072e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.018e-07, signed_mean=-1.930e-06, mare=1.435e-05, max_step=7.754e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=6.200e-05, signed_mean=-9.407e-05, mare=2.007e-04, max_step=8.092e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.499e-07, signed_mean=7.461e-05, mare=3.444e-04, max_step=1.507e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-9.344e-08, mare=9.620e-08, max_step=2.308e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.281e-01, mare_max=2.628e-01, max_step_max=9.768e-01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.170e-03, mare_max=5.454e-03, max_step_max=2.513e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
