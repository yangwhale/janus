# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-c-vmem`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=-3.643e-08, mare=5.429e-06, max_step=2.492e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=-4.424e-07, mare=6.515e-06, max_step=2.492e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-4.071e-07, signed_mean=5.461e-07, mare=1.108e-05, max_step=7.142e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.287e-06, signed_mean=3.747e-05, mare=2.049e-04, max_step=1.127e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=1.649e-06, signed_mean=-7.701e-05, mare=3.416e-04, max_step=1.709e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.172e-08, mare=4.926e-08, max_step=2.081e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.964e-01, mare_max=2.979e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.948e-03, mare_max=5.393e-03, max_step_max=2.461e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
