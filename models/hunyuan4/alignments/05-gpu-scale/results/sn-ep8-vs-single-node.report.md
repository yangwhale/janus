# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`sn-ep8`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.709e-06, signed_mean=-1.353e-06, mare=5.837e-06, max_step=3.044e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.709e-06, signed_mean=-6.886e-07, mare=6.777e-06, max_step=3.044e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=2.406e-06, mare=1.166e-05, max_step=6.199e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.544e-05, signed_mean=5.038e-05, mare=1.731e-04, max_step=6.872e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-3.019e-05, signed_mean=7.378e-05, mare=2.962e-04, max_step=1.416e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=6.268e-10, signed_mean=-1.391e-08, mare=5.046e-08, max_step=2.086e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.756e-01, mare_max=1.756e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=8.548e-04, mare_max=5.345e-03, max_step_max=2.924e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
