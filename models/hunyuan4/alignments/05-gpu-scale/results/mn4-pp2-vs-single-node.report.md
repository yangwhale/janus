# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`mn4-pp2`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.465e-06, signed_mean=-1.451e-06, mare=6.445e-06, max_step=2.522e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.465e-06, signed_mean=-2.424e-06, mare=7.810e-06, max_step=2.522e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=1.018e-06, signed_mean=7.597e-06, mare=1.430e-05, max_step=6.086e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.898e-05, signed_mean=5.608e-05, mare=2.226e-04, max_step=1.001e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-5.751e-05, signed_mean=5.610e-05, mare=3.257e-04, max_step=1.523e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=4.050e-09, signed_mean=1.450e-07, mare=1.455e-07, max_step=6.518e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=7.763e-01, mare_max=7.763e-01, max_step_max=7.009e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.667e-03, mare_max=5.653e-03, max_step_max=2.751e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
