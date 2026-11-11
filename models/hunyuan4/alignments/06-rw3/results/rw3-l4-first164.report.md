# janus 报告：hunyuan4-rw3 / rw3-l4-first164

- 被测：`trw-e`；golden：`rw1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/06-rw3/catalogs/rw3-l4-first164.yaml`（内容指纹 258c783fc76e…）
- 结果：pass 1、fail 6、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.grad_norm`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=1.563e-04, signed_mean_max=8.676e-05, mare_max=8.702e-05, max_step_max=1.630e-04 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | fail | d0_max=1.563e-04, signed_mean_max=4.903e-05, mare_max=4.969e-05, max_step_max=1.066e-04 | 超过噪声参照两倍 |
| `L4.mtp_loss` | loss_series | L4 | fail | d0=-1.396e-04, signed_mean=-8.487e-05, mare=1.085e-04, max_step=3.255e-04 | 超过噪声参照两倍 |
| `L4.mtp_loss.tail` | loss_series | L4 | fail | d0=-1.396e-04, signed_mean=-8.765e-05, mare=1.178e-04, max_step=2.809e-04 | 超过噪声参照两倍 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-5.494e-04, signed_mean=-1.673e-04, mare=2.884e-04, max_step=6.626e-04 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=1.426e-03, signed_mean=-2.835e-04, mare=1.660e-03, max_step=9.718e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | fail | d0=-8.732e-10, signed_mean=-2.792e-09, mare=2.792e-09, max_step=4.559e-09 | 超过噪声参照两倍 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=9.408e-02, mare_max=9.519e-02, max_step_max=2.029e-01, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.187e-03, mare_max=2.403e-03, max_step_max=6.358e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
