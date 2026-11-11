# janus 报告：hunyuan4-mini / t1zs-l4

- 被测：`t1zs-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zs-l4.yaml`（内容指纹 8226649b1319…）
- 结果：pass 3、fail 0、inconclusive 6、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.mtp_loss`、`L4.mtp_loss.tail`、`L4.indexer_loss`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=3.581e-06, signed_mean_max=4.517e-06, mare_max=8.213e-06, max_step_max=4.189e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=3.581e-06, signed_mean_max=4.643e-06, mare_max=8.601e-06, max_step_max=4.189e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=1.018e-07, signed_mean=-3.056e-05, mare=3.984e-05, max_step=2.135e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=1.018e-07, signed_mean=-3.383e-05, mare=4.386e-05, max_step=2.135e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.359e-04, signed_mean=3.001e-04, mare=3.950e-04, max_step=1.588e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-4.948e-04, signed_mean=-1.501e-04, mare=2.582e-03, max_step=4.694e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=4.049e-09, signed_mean=-9.315e-07, mare=1.109e-06, max_step=7.207e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=9.297e-01, mare_max=9.303e-01, max_step_max=1.121e+01, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.444e-03, mare_max=4.682e-03, max_step_max=2.806e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
