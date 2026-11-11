# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-dp8-ga1`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 3、fail 3、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.param_norm`、`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=-7.325e-07, signed_mean=1.110e-05, mare=1.253e-05, max_step=5.091e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=-7.325e-07, signed_mean=1.643e-05, mare=1.701e-05, max_step=5.091e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.221e-06, signed_mean=-4.291e-06, mare=1.347e-05, max_step=9.814e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-2.606e-05, signed_mean=-8.557e-05, mare=2.380e-04, max_step=9.101e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | fail | d0=-1.220e-03, signed_mean=-3.131e-04, mare=5.928e-04, max_step=2.391e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=-1.131e-07, signed_mean=-3.578e-07, mare=3.590e-07, max_step=1.157e-06 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.300e-01, mare_max=3.300e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.688e-03, mare_max=5.938e-03, max_step_max=2.198e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
