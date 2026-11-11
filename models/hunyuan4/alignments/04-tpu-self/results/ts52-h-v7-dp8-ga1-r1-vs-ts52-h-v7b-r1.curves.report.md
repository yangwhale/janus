# janus 报告：hunyuan4-mini / tpu-self-vs-ts52-h-v7b-r1

- 被测：`ts52-h-v7-dp8-ga1-r1`；golden：`ts52-h-v7b-r1`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts52-h-v7b-r1.yaml`（内容指纹 6ba43723648b…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-9.766e-07, signed_mean=-1.771e-06, mare=5.706e-06, max_step=2.739e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-9.766e-07, signed_mean=-2.246e-06, mare=7.040e-06, max_step=2.739e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.018e-06, signed_mean=6.019e-06, mare=1.346e-05, max_step=8.370e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=3.057e-05, signed_mean=6.239e-05, mare=2.246e-04, max_step=7.813e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-2.362e-05, signed_mean=8.365e-05, mare=3.416e-04, max_step=1.960e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-9.691e-08, signed_mean=9.098e-08, mare=9.456e-08, max_step=4.033e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=5.530e-01, mare_max=5.556e-01, max_step_max=1.849e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.144e-03, mare_max=5.214e-03, max_step_max=2.275e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
