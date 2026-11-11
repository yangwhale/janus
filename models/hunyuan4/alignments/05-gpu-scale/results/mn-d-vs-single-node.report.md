# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`mn-d`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=-1.708e-08, mare=4.944e-06, max_step=2.137e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=-4.528e-07, mare=5.758e-06, max_step=2.137e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=1.736e-06, mare=1.300e-05, max_step=8.040e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-1.384e-05, signed_mean=7.129e-05, mare=2.314e-04, max_step=8.230e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-5.944e-05, signed_mean=3.575e-05, mare=3.039e-04, max_step=1.615e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=6.226e-11, signed_mean=2.509e-08, mare=3.464e-08, max_step=3.183e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.976e-01, mare_max=2.463e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=9.161e-04, mare_max=5.850e-03, max_step_max=2.884e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
