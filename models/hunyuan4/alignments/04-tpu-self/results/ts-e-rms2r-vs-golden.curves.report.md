# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-e-rms2r`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.309e-06, mare=5.726e-06, max_step=2.317e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.777e-06, mare=6.791e-06, max_step=2.317e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-7.333e-06, mare=1.399e-05, max_step=1.099e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.699e-05, mare=2.208e-04, max_step=1.084e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=4.273e-06, signed_mean=4.562e-06, mare=3.500e-04, max_step=1.856e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.242e-07, mare=1.249e-07, max_step=4.708e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.270e-01, mare_max=3.286e-01, max_step_max=2.828e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.264e-03, mare_max=6.025e-03, max_step_max=2.293e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
