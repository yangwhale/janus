# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-remat-full-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=9.208e-07, mare=5.766e-06, max_step=2.159e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=1.688e-06, mare=6.811e-06, max_step=2.159e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=2.036e-07, signed_mean=-1.810e-06, mare=1.245e-05, max_step=9.159e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-3.432e-06, signed_mean=-1.018e-04, mare=2.088e-04, max_step=7.625e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-7.047e-06, signed_mean=-5.907e-05, mare=3.824e-04, max_step=1.674e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-8.181e-08, mare=1.009e-07, max_step=3.422e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.994e-01, mare_max=3.076e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.648e-03, mare_max=5.939e-03, max_step_max=2.513e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
