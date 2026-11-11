# janus 报告：hunyuan4-rw3 / rw3-l4

- 被测：`rfx-on`；golden：`rw1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/06-rw3/catalogs/rw3-l4.yaml`（内容指纹 90a594773a3d…）
- 结果：pass 2、fail 3、inconclusive 4、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`、`L4.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=8.783e-05, signed_mean_max=3.454e-05, mare_max=3.931e-05, max_step_max=1.630e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=8.783e-05, signed_mean_max=2.597e-05, mare_max=3.127e-05, max_step_max=1.155e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | fail | d0=-2.286e-04, signed_mean=-7.631e-05, mare=9.054e-05, max_step=3.897e-04 | 超过噪声参照两倍 |
| `L4.mtp_loss.tail` | loss_series | L4 | fail | d0=-2.286e-04, signed_mean=-7.549e-05, mare=8.917e-05, max_step=3.897e-04 | 超过噪声参照两倍 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=1.821e-05, signed_mean=3.798e-04, mare=6.615e-04, max_step=2.807e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-2.459e-03, signed_mean=-4.351e-05, mare=8.734e-03, max_step=6.829e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-3.687e-09, signed_mean=3.765e-09, mare=9.508e-09, max_step=3.095e-08 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.814e-02, mare_max=3.839e-02, max_step_max=2.029e-01, stats=3 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=1.058e-03, mare_max=1.463e-03, max_step_max=6.358e-03, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
