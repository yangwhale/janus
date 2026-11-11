# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-e-swiglu-1ks`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=-8.138e-07, mare=6.260e-06, max_step=3.112e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=-1.442e-06, mare=7.719e-06, max_step=3.112e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-5.089e-07, signed_mean=9.071e-06, mare=1.375e-05, max_step=9.639e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.770e-05, signed_mean=7.535e-05, mare=2.153e-04, max_step=1.122e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.374e-04, signed_mean=-4.386e-05, mare=3.793e-04, max_step=1.890e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.360e-08, mare=4.062e-08, max_step=1.766e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.040e-01, mare_max=3.128e-01, max_step_max=1.077e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.155e-03, mare_max=5.547e-03, max_step_max=2.390e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
