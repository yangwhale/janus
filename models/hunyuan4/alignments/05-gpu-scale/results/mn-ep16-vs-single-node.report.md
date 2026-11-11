# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`mn-ep16`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.302e-06, signed_mean=9.755e-07, mare=5.679e-06, max_step=2.503e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.302e-06, signed_mean=1.304e-06, mare=7.165e-06, max_step=2.503e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=2.036e-07, signed_mean=3.167e-06, mare=1.140e-05, max_step=7.170e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-2.789e-06, signed_mean=2.670e-05, mare=2.116e-04, max_step=8.341e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-5.337e-05, signed_mean=1.970e-05, mare=3.083e-04, max_step=1.573e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=1.044e-09, signed_mean=-5.627e-08, mare=6.175e-08, max_step=2.019e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.152e-01, mare_max=1.201e-01, max_step_max=5.043e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=6.056e-04, mare_max=4.349e-03, max_step_max=2.440e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
