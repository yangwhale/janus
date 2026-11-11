# janus 报告：hunyuan4-mini / t1zt-l4

- 被测：`t1zt-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zt-l4.yaml`（内容指纹 f07443c852d5…）
- 结果：pass 5、fail 1、inconclusive 3、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=3.581e-06, signed_mean_max=5.281e-06, mare_max=9.747e-06, max_step_max=5.088e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=3.581e-06, signed_mean_max=5.535e-06, mare_max=1.032e-05, max_step_max=5.088e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=1.018e-07, signed_mean=-6.979e-06, mare=3.200e-05, max_step=1.637e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=1.018e-07, signed_mean=-7.620e-06, mare=3.519e-05, max_step=1.637e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.359e-04, signed_mean=1.209e-04, mare=3.242e-04, max_step=1.383e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.948e-04, signed_mean=-5.995e-05, mare=2.506e-03, max_step=8.037e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=4.049e-09, signed_mean=-2.534e-06, mare=2.728e-06, max_step=1.116e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | fail | signed_mean_max=1.273e+00, mare_max=1.294e+00, max_step_max=6.619e+00, stats=7 | 超过噪声参照两倍 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.224e-03, mare_max=5.269e-03, max_step_max=2.620e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
