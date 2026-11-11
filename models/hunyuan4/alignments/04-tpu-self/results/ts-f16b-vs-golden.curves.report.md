# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-f16b`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-8.952e-07, signed_mean=1.040e-06, mare=6.081e-06, max_step=2.373e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-8.952e-07, signed_mean=1.169e-06, mare=7.388e-06, max_step=2.373e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-2.036e-07, signed_mean=-2.025e-06, mare=1.218e-05, max_step=6.838e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=6.521e-05, signed_mean=9.067e-06, mare=1.949e-04, max_step=9.729e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.682e-04, signed_mean=-9.717e-06, mare=3.411e-04, max_step=1.769e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=1.298e-08, signed_mean=4.791e-09, mare=2.428e-08, max_step=9.846e-08 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.783e-01, mare_max=1.940e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.552e-03, mare_max=5.720e-03, max_step_max=2.249e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
