# janus 报告：hunyuan4-rw3 / rw3-l4-b2r250

- 被测：`b2r-off`；golden：`rw1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/06-rw3/catalogs/rw3-l4-b2r250.yaml`（内容指纹 38ac5f2ef9ef…）
- 结果：pass 4、fail 3、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.mtp_loss.tail`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=5.881e-05, signed_mean_max=3.902e-05, mare_max=4.053e-05, max_step_max=1.025e-04 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | fail | d0_max=5.881e-05, signed_mean_max=2.507e-05, mare_max=2.758e-05, max_step_max=9.512e-05 | 超过噪声参照两倍 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-1.100e-04, signed_mean=-2.393e-05, mare=8.672e-05, max_step=4.554e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=-1.100e-04, signed_mean=-5.019e-05, mare=7.286e-05, max_step=2.605e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-6.730e-04, signed_mean=-3.528e-04, mare=4.047e-04, max_step=1.018e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=1.537e-03, signed_mean=-1.391e-04, mare=3.201e-03, max_step=2.425e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-8.971e-10, signed_mean=-8.468e-10, mare=9.692e-10, max_step=1.911e-09 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=9.823e-02, mare_max=1.018e-01, max_step_max=2.729e-01, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.860e-03, mare_max=2.124e-03, max_step_max=6.285e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
