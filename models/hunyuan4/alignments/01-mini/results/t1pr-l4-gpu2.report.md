# janus 报告：hunyuan4-mini / t1pr-l4-gpu2

- 被测：`p1t-a`；golden：`p1g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1pr-l4-gpu2.yaml`（内容指纹 93bc5268e3f0…）
- 结果：pass 0、fail 3、inconclusive 6、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`、`L4.param_norm`、`L4.router.bias_mean`、`L4.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=8.139e-08, signed_mean_max=1.381e-05, mare_max=1.495e-05, max_step_max=7.368e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=8.139e-08, signed_mean_max=1.515e-05, mare_max=1.625e-05, max_step_max=7.368e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | fail | d0=2.443e-06, signed_mean=1.003e-05, mare=3.344e-05, max_step=1.986e-04 | 超过噪声参照两倍 |
| `L4.mtp_loss.tail` | loss_series | L4 | fail | d0=2.443e-06, signed_mean=1.116e-05, mare=3.689e-05, max_step=1.986e-04 | 超过噪声参照两倍 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-1.291e-04, signed_mean=-2.987e-04, mare=3.782e-04, max_step=1.783e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-5.420e-04, signed_mean=2.958e-04, mare=7.170e-03, max_step=6.987e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=-2.040e-09, signed_mean=-4.273e-06, mare=4.277e-06, max_step=1.435e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=3.864e-06, mare_max=4.555e-06, max_step_max=1.843e-05, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=8.528e-04, mare_max=5.262e-03, max_step_max=2.988e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
