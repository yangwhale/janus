# janus 报告：hunyuan4-rw3 / rw3-l4-b2r250

- 被测：`irf-b`；golden：`rw1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/06-rw3/catalogs/rw3-l4-b2r250.yaml`（内容指纹 38ac5f2ef9ef…）
- 结果：pass 5、fail 1、inconclusive 2、error 1；**有硬检查失败**

**证据不足（inconclusive）**：`L4.mtp_loss.tail`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.826e-06, signed_mean_max=2.405e-06, mare_max=1.174e-05, max_step_max=4.342e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.826e-06, signed_mean_max=9.761e-07, mare_max=1.271e-05, max_step_max=4.342e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-7.498e-05, signed_mean=1.734e-05, mare=8.115e-05, max_step=4.337e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=-7.498e-05, signed_mean=1.001e-05, mare=6.870e-05, max_step=3.312e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-4.917e-04, signed_mean=-4.648e-04, mare=4.882e-04, max_step=1.611e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-1.227e-05, signed_mean=-2.875e-04, mare=2.910e-03, max_step=2.795e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | error |  | 取数或参数错误：窗口内缺步: [0, 1, 2, 3, 4]… |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=2.375e-02, mare_max=4.320e-02, max_step_max=3.172e-01, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=3.539e-04, mare_max=9.449e-04, max_step_max=3.572e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
