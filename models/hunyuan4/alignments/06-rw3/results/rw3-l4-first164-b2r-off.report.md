# janus 报告：hunyuan4-rw3 / rw3-l4-first164

- 被测：`b2r-off`；golden：`rw1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/06-rw3/catalogs/rw3-l4-first164.yaml`（内容指纹 258c783fc76e…）
- 结果：pass 1、fail 5、inconclusive 3、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.mtp_loss`、`L4.grad_norm`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=5.881e-05, signed_mean_max=5.228e-05, mare_max=5.228e-05, max_step_max=1.025e-04 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | fail | d0_max=5.881e-05, signed_mean_max=4.008e-05, mare_max=4.008e-05, max_step_max=9.512e-05 | 超过噪声参照两倍 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=-1.100e-04, signed_mean=-1.680e-05, mare=9.931e-05, max_step=4.554e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | fail | d0=-1.100e-04, signed_mean=-6.732e-05, mare=8.629e-05, max_step=2.352e-04 | 超过噪声参照两倍 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-6.730e-04, signed_mean=-4.060e-04, mare=4.267e-04, max_step=9.796e-04 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=1.537e-03, signed_mean=-5.912e-05, mare=1.744e-03, max_step=1.293e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | fail | d0=-8.971e-10, signed_mean=-1.057e-09, mare=1.076e-09, max_step=1.911e-09 | 超过噪声参照两倍 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=1.151e-01, mare_max=1.206e-01, max_step_max=2.729e-01, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.983e-03, mare_max=2.307e-03, max_step_max=6.285e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
