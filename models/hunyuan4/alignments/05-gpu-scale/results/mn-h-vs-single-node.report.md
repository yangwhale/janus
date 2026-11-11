# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`mn-h`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-4.883e-07, signed_mean=-1.495e-06, mare=5.357e-06, max_step=2.254e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-4.883e-07, signed_mean=-1.977e-06, mare=6.442e-06, max_step=2.254e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=5.089e-07, signed_mean=-3.892e-06, mare=1.165e-05, max_step=5.983e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-1.534e-05, signed_mean=4.348e-05, mare=1.969e-04, max_step=8.906e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-8.902e-05, signed_mean=7.199e-05, mare=3.148e-04, max_step=1.418e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=3.730e-09, signed_mean=5.017e-08, mare=5.338e-08, max_step=2.775e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.033e-01, mare_max=4.617e-01, max_step_max=7.009e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.227e-03, mare_max=4.241e-03, max_step_max=2.350e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
