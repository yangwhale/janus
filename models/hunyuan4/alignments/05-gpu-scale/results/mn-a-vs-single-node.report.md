# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`mn-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 5、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`C.loss`、`C.loss.tail`、`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=-1.302e-06, signed_mean=-3.209e-06, mare=6.457e-06, max_step=3.546e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | inconclusive | d0=-1.302e-06, signed_mean=-5.064e-06, mare=8.215e-06, max_step=3.546e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=5.089e-07, signed_mean=2.038e-06, mare=1.402e-05, max_step=7.774e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-5.899e-06, signed_mean=8.585e-05, mare=2.243e-04, max_step=9.054e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=1.319e-06, signed_mean=-3.187e-05, mare=3.391e-04, max_step=1.427e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=-3.013e-09, signed_mean=1.728e-07, mare=1.770e-07, max_step=7.679e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.018e-01, mare_max=2.878e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.201e-03, mare_max=6.175e-03, max_step_max=2.751e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
