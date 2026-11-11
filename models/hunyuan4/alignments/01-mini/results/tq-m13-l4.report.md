# janus 报告：hunyuan4-mini / tq-m13-l4

- 被测：`tq-m13`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m13-l4.yaml`（内容指纹 d6d4ed6959d7…）
- 结果：pass 5、fail 2、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss.tail`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=1.058e-06, signed_mean_max=9.331e-06, mare_max=2.400e-05, max_step_max=2.732e-04 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=1.058e-06, signed_mean_max=6.864e-06, mare_max=1.114e-05, max_step_max=3.210e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=-4.558e-07, mare=1.587e-05, max_step=8.480e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=-9.601e-07, mare=2.050e-05, max_step=8.480e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.126e-04, signed_mean=1.087e-04, mare=2.372e-04, max_step=1.105e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.495e-04, signed_mean=1.221e-04, mare=6.277e-04, max_step=2.532e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | fail | d0=-4.988e-09, signed_mean=-4.713e-07, mare=4.723e-07, max_step=9.800e-07 | 超过噪声参照两倍 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=9.952e-01, mare_max=1.046e+00, max_step_max=3.838e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.306e-03, mare_max=5.611e-03, max_step_max=2.516e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
