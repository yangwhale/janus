# janus 报告：hunyuan4-rw3 / rw3-l4

- 被测：`rfx-off`；golden：`rw1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/06-rw3/catalogs/rw3-l4.yaml`（内容指纹 90a594773a3d…）
- 结果：pass 3、fail 0、inconclusive 6、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.mtp_loss`、`L4.mtp_loss.tail`、`L4.indexer_loss`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=5.881e-05, signed_mean_max=2.221e-05, mare_max=2.806e-05, max_step_max=1.025e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=5.881e-05, signed_mean_max=1.804e-05, mare_max=2.453e-05, max_step_max=1.024e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=-1.100e-04, signed_mean=-4.484e-05, mare=7.401e-05, max_step=4.554e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=-1.100e-04, signed_mean=-5.151e-05, mare=7.030e-05, max_step=2.837e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=-6.730e-04, signed_mean=-2.744e-04, mare=5.687e-04, max_step=2.548e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=1.537e-03, signed_mean=7.548e-04, mare=8.244e-03, max_step=6.244e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-8.971e-10, signed_mean=3.003e-09, mare=7.988e-09, max_step=3.613e-08 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=4.644e-02, mare_max=4.734e-02, max_step_max=2.729e-01, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=3.762e-04, mare_max=1.517e-03, max_step_max=6.285e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
