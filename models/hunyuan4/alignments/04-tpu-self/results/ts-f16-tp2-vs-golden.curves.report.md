# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-f16-tp2`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=-4.069e-07, signed_mean=-7.221e-07, mare=8.364e-06, max_step=3.230e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-4.069e-07, signed_mean=-9.474e-07, mare=8.179e-06, max_step=3.230e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-2.443e-06, signed_mean=5.626e-06, mare=1.354e-05, max_step=7.877e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.019e-05, signed_mean=-2.505e-05, mare=2.299e-04, max_step=1.177e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-2.744e-05, signed_mean=-6.983e-05, mare=3.827e-04, max_step=1.420e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=1.100e-08, signed_mean=6.239e-09, mare=2.791e-08, max_step=9.808e-08 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.324e-01, mare_max=4.421e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.351e-03, mare_max=5.387e-03, max_step_max=3.135e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
