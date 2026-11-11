# janus 报告：hunyuan4-trunc / t2d1z-l4

- 被测：`td3-a`；golden：`gd2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/02-real-weight/catalogs/t2d1z-l4.yaml`（内容指纹 720376e1cb43…）
- 结果：pass 2、fail 3、inconclusive 4、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss.tail`、`L4.indexer_loss`、`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=8.489e-05, signed_mean_max=1.922e-05, mare_max=2.682e-05, max_step_max=1.072e-04 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=8.489e-05, signed_mean_max=4.415e-06, mare_max=1.557e-05, max_step_max=4.693e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | fail | d0=-3.066e-04, signed_mean=-1.040e-04, mare=1.152e-04, max_step=4.587e-04 | 超过噪声参照两倍 |
| `L4.mtp_loss.tail` | loss_series | L4 | fail | d0=-3.066e-04, signed_mean=-1.103e-04, mare=1.223e-04, max_step=4.015e-04 | 超过噪声参照两倍 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=-6.483e-06, signed_mean=-1.510e-04, mare=7.973e-04, max_step=4.621e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=4.372e-04, signed_mean=-4.262e-04, mare=4.255e-03, max_step=5.845e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=-1.991e-09, signed_mean=1.968e-09, mare=5.404e-09, max_step=1.843e-08 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=8.628e-05, mare_max=8.800e-04, max_step_max=5.971e-03, stats=1 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.403e-04, mare_max=2.639e-04, max_step_max=1.156e-03, stats=1 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
