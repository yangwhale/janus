# janus 报告：hunyuan4-rw3 / rw3-l4-b2r250

- 被测：`irf-a`；golden：`rw1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/06-rw3/catalogs/rw3-l4-b2r250.yaml`（内容指纹 38ac5f2ef9ef…）
- 结果：pass 6、fail 1、inconclusive 1、error 1；**有硬检查失败**

**证据不足（inconclusive）**：`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.429e-06, signed_mean_max=3.262e-06, mare_max=1.210e-05, max_step_max=4.883e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.429e-06, signed_mean_max=3.981e-06, mare_max=1.330e-05, max_step_max=4.883e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-4.944e-05, signed_mean=1.882e-05, mare=6.969e-05, max_step=3.171e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-4.944e-05, signed_mean=2.449e-05, mare=7.184e-05, max_step=2.283e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-4.853e-04, signed_mean=-4.176e-04, mare=4.513e-04, max_step=1.195e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=6.522e-05, signed_mean=-4.998e-04, mare=2.811e-03, max_step=2.143e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | error |  | 取数或参数错误：窗口内缺步: [0, 1, 2, 3, 4]… |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=5.970e-02, mare_max=7.018e-02, max_step_max=1.584e-01, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.316e-03, mare_max=1.757e-03, max_step_max=6.573e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
