# janus 报告：hunyuan4-mini / t1zc-l4

- 被测：`tc-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zc-l4.yaml`（内容指纹 54809b00dd73…）
- 结果：pass 8、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=3.011e-06, signed_mean_max=3.530e-06, mare_max=8.746e-06, max_step_max=4.753e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=3.011e-06, signed_mean_max=3.941e-06, mare_max=9.303e-06, max_step_max=4.753e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-7.125e-07, signed_mean=8.427e-07, mare=3.000e-05, max_step=1.646e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-7.125e-07, signed_mean=7.557e-07, mare=3.285e-05, max_step=1.646e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-1.283e-04, signed_mean=6.815e-05, mare=2.943e-04, max_step=1.282e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-3.970e-05, signed_mean=-1.442e-04, mare=2.320e-03, max_step=4.645e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.073e-09, signed_mean=-2.311e-07, mare=6.483e-07, max_step=3.057e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=1.356e+00, mare_max=1.389e+00, max_step_max=8.411e+00, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.274e-03, mare_max=4.887e-03, max_step_max=2.453e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
