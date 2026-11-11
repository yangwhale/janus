# janus 报告：hunyuan4-mini / t1z-l4

- 被测：`t1z-a`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1z-l4.yaml`（内容指纹 c1545e5b832f…）
- 结果：pass 6、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`L4.indexer_loss`、`L4.grad_norm`、`L4.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=1.720e-06, mare_max=6.094e-06, max_step_max=2.409e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=3.893e-07, mare_max=6.386e-06, max_step_max=2.409e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-6.107e-07, signed_mean=2.708e-06, mare=1.192e-05, max_step=7.505e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-6.107e-07, signed_mean=3.336e-06, mare=1.602e-05, max_step=7.505e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=5.555e-04, signed_mean=-1.890e-06, mare=2.618e-04, max_step=1.023e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-2.589e-04, signed_mean=3.206e-05, mare=5.038e-04, max_step=1.487e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=7.789e-09, signed_mean=-7.541e-08, mare=1.014e-07, max_step=2.418e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=9.598e-01, mare_max=1.012e+00, max_step_max=3.838e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=1.542e-03, mare_max=6.849e-03, max_step_max=2.392e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
