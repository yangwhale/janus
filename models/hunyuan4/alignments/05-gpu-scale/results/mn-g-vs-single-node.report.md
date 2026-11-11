# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`mn-g`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-2.442e-06, signed_mean=1.918e-07, mare=5.446e-06, max_step=2.833e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-2.442e-06, signed_mean=1.329e-07, mare=6.731e-06, max_step=2.833e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=4.071e-07, signed_mean=1.910e-06, mare=1.321e-05, max_step=5.680e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=7.293e-06, signed_mean=1.195e-05, mare=2.223e-04, max_step=1.003e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-9.171e-05, signed_mean=1.136e-04, mare=3.463e-04, max_step=1.926e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=5.327e-09, signed_mean=3.359e-08, mare=4.642e-08, max_step=2.918e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=8.350e-01, mare_max=8.465e-01, max_step_max=1.332e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.098e-03, mare_max=5.751e-03, max_step_max=3.099e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
