# janus 报告：hunyuan4-mini / t1zk-l4

- 被测：`t1zk-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zk-l4.yaml`（内容指纹 b3f9ba4506c9…）
- 结果：pass 4、fail 2、inconclusive 3、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.mtp_loss`、`L4.mtp_loss.tail`、`L4.indexer_loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=2.848e-06, signed_mean_max=7.987e-06, mare_max=1.276e-05, max_step_max=7.078e-05 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | fail | d0_max=2.848e-06, signed_mean_max=8.460e-06, mare_max=1.355e-05, max_step_max=7.078e-05 | 超过噪声参照两倍 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=-9.160e-07, signed_mean=-1.486e-05, mare=3.744e-05, max_step=1.901e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=-9.160e-07, signed_mean=-1.667e-05, mare=4.112e-05, max_step=1.901e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=5.738e-04, signed_mean=2.770e-04, mare=4.036e-04, max_step=1.746e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-3.110e-04, signed_mean=-4.662e-05, mare=2.465e-03, max_step=4.013e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=1.233e-08, signed_mean=-5.956e-07, mare=1.562e-06, max_step=6.284e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.340e-01, mare_max=4.445e-01, max_step_max=8.411e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.200e-03, mare_max=5.269e-03, max_step_max=2.751e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
