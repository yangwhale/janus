# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-h-v7f16c`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-2.686e-06, signed_mean=-1.193e-06, mare=5.280e-06, max_step=2.605e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-2.686e-06, signed_mean=-1.492e-06, mare=6.061e-06, max_step=2.605e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-5.089e-07, signed_mean=3.147e-06, mare=1.272e-05, max_step=6.855e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=4.054e-05, signed_mean=1.555e-05, mare=2.191e-04, max_step=1.106e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-9.071e-05, signed_mean=-9.999e-05, mare=3.567e-04, max_step=1.332e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=-7.571e-09, signed_mean=-6.176e-08, mare=7.032e-08, max_step=3.814e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.540e-01, mare_max=2.713e-01, max_step_max=3.026e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.043e-03, mare_max=5.407e-03, max_step_max=2.643e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
