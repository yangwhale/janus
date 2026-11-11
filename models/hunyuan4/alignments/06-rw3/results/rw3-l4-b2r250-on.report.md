# janus 报告：hunyuan4-rw3 / rw3-l4-b2r250

- 被测：`b2r-on`；golden：`rw1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/06-rw3/catalogs/rw3-l4-b2r250.yaml`（内容指纹 38ac5f2ef9ef…）
- 结果：pass 2、fail 5、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.indexer_loss`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=1.563e-04, signed_mean_max=5.984e-05, mare_max=6.328e-05, max_step_max=1.630e-04 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | fail | d0_max=1.563e-04, signed_mean_max=2.598e-05, mare_max=3.168e-05, max_step_max=1.066e-04 | 超过噪声参照两倍 |
| `L4.mtp_loss` | loss_series | L4 | fail | d0=-1.396e-04, signed_mean=-8.091e-05, mare=1.033e-04, max_step=3.255e-04 | 超过噪声参照两倍 |
| `L4.mtp_loss.tail` | loss_series | L4 | fail | d0=-1.396e-04, signed_mean=-7.948e-05, mare=1.039e-04, max_step=2.809e-04 | 超过噪声参照两倍 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=-5.494e-04, signed_mean=2.663e-05, mare=3.615e-04, max_step=1.516e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=1.426e-03, signed_mean=-8.087e-05, mare=3.352e-03, max_step=2.923e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | fail | d0=-8.732e-10, signed_mean=-3.538e-09, mare=3.538e-09, max_step=5.828e-09 | 超过噪声参照两倍 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=6.689e-02, mare_max=6.762e-02, max_step_max=2.029e-01, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.620e-03, mare_max=1.956e-03, max_step_max=6.358e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
