# janus 报告：hunyuan4-trunc / t2d1c-l4

- 被测：`tdc-a`；golden：`gd3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/02-real-weight/catalogs/t2d1c-l4.yaml`（内容指纹 f9ab94720da7…）
- 结果：pass 7、fail 1、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=4.831e-05, signed_mean_max=4.004e-07, mare_max=1.782e-05, max_step_max=1.338e-04 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=4.831e-05, signed_mean_max=2.689e-06, mare_max=1.735e-05, max_step_max=1.338e-04 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-1.512e-04, signed_mean=-2.438e-05, mare=5.580e-05, max_step=3.146e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-1.512e-04, signed_mean=-2.415e-05, mare=5.515e-05, max_step=2.410e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=1.144e-04, signed_mean=1.279e-03, mare=2.425e-03, max_step=1.664e-02 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=3.830e-04, signed_mean=1.011e-03, mare=1.385e-02, max_step=4.544e-01 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-2.031e-09, signed_mean=-3.188e-08, mare=4.275e-08, max_step=1.401e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=4.018e-04, mare_max=1.401e-03, max_step_max=9.011e-03, stats=1 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.102e-05, mare_max=3.039e-04, max_step_max=1.323e-03, stats=1 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
