# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-f-ltpu47`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-2.442e-07, signed_mean=4.637e-08, mare=5.302e-06, max_step=2.065e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-2.442e-07, signed_mean=-6.095e-08, mare=6.149e-06, max_step=2.065e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=5.539e-06, mare=1.335e-05, max_step=8.717e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-8.581e-07, signed_mean=1.450e-05, mare=2.044e-04, max_step=9.302e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=9.971e-06, signed_mean=-2.635e-05, mare=3.721e-04, max_step=1.795e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-7.195e-08, mare=8.186e-08, max_step=3.691e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=5.274e-01, mare_max=5.314e-01, max_step_max=8.069e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.676e-03, mare_max=5.657e-03, max_step_max=2.798e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
