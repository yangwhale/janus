# janus 报告：hunyuan4-mini / tq-m27-l4

- 被测：`tq-m27`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m27-l4.yaml`（内容指纹 f6f6a52f617c…）
- 结果：pass 7、fail 1、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=1.664e-06, mare_max=5.796e-06, max_step_max=2.407e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=6.977e-07, mare_max=6.148e-06, max_step_max=2.407e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=3.452e-06, mare=1.228e-05, max_step=7.727e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=5.430e-06, mare=1.664e-05, max_step=7.727e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=1.126e-04, signed_mean=-3.250e-04, mare=5.638e-04, max_step=2.562e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.628e-04, signed_mean=4.754e-05, mare=4.990e-04, max_step=1.680e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-5.048e-09, signed_mean=-1.011e-07, mare=1.129e-07, max_step=2.248e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=5.724e-01, mare_max=6.172e-01, max_step_max=9.813e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.184e-03, mare_max=4.485e-03, max_step_max=2.682e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
