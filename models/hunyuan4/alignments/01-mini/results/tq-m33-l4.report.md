# janus 报告：hunyuan4-mini / tq-m33-l4

- 被测：`tq-m33`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m33-l4.yaml`（内容指纹 8c651f8b89a2…）
- 结果：pass 7、fail 1、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.872e-06, signed_mean_max=2.990e-06, mare_max=5.925e-06, max_step_max=3.121e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.872e-06, signed_mean_max=3.019e-06, mare_max=6.593e-06, max_step_max=3.121e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.036e-06, signed_mean=-5.576e-06, mare=1.225e-05, max_step=6.821e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.036e-06, signed_mean=-8.148e-06, mare=1.697e-05, max_step=6.821e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-8.611e-04, signed_mean=-4.069e-04, mare=4.915e-04, max_step=1.294e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-5.287e-04, signed_mean=1.743e-04, mare=5.294e-04, max_step=2.005e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=3.342e-10, signed_mean=-6.719e-08, mare=8.416e-08, max_step=1.752e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.541e-01, mare_max=4.572e-01, max_step_max=8.411e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.619e-03, mare_max=4.982e-03, max_step_max=2.500e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
