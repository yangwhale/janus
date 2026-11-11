# janus 报告：hunyuan4-mini / t1zu-l4

- 被测：`t1zu-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zu-l4.yaml`（内容指纹 4ef4c521dfd0…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=2.848e-06, signed_mean_max=2.448e-06, mare_max=8.780e-06, max_step_max=4.805e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=2.848e-06, signed_mean_max=2.642e-06, mare_max=9.325e-06, max_step_max=4.805e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.341e-06, signed_mean=9.373e-06, mare=3.196e-05, max_step=2.136e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.341e-06, signed_mean=1.042e-05, mare=3.516e-05, max_step=2.136e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.310e-04, signed_mean=5.685e-05, mare=3.338e-04, max_step=1.469e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-5.149e-04, signed_mean=3.259e-04, mare=2.579e-03, max_step=3.874e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-5.058e-10, signed_mean=-1.883e-06, mare=1.970e-06, max_step=1.066e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.846e-01, mare_max=4.119e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.426e-03, mare_max=4.931e-03, max_step_max=2.440e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
