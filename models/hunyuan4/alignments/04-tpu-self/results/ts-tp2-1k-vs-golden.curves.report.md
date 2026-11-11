# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-tp2-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=-8.139e-08, signed_mean=-6.049e-07, mare=7.305e-06, max_step=2.720e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=-5.664e-07, mare=7.229e-06, max_step=2.720e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-2.545e-06, signed_mean=7.552e-06, mare=1.411e-05, max_step=8.753e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.888e-05, signed_mean=2.503e-05, mare=2.287e-04, max_step=1.028e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.319e-05, signed_mean=-2.065e-05, mare=3.515e-04, max_step=1.587e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-2.247e-09, signed_mean=-1.074e-07, mare=1.162e-07, max_step=5.133e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.775e-01, mare_max=3.848e-01, max_step_max=1.317e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.918e-03, mare_max=5.321e-03, max_step_max=2.203e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
