# janus 报告：hunyuan4-mini / t1zv-l4

- 被测：`t1zv-b`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zv-l4.yaml`（内容指纹 81d0ad3bd2c4…）
- 结果：pass 6、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=2.848e-06, signed_mean_max=4.985e-06, mare_max=1.073e-05, max_step_max=5.120e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=2.848e-06, signed_mean_max=5.182e-06, mare_max=1.141e-05, max_step_max=5.120e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.341e-06, signed_mean=-1.283e-05, mare=3.259e-05, max_step=2.101e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.341e-06, signed_mean=-1.416e-05, mare=3.583e-05, max_step=2.101e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.310e-04, signed_mean=1.246e-04, mare=3.339e-04, max_step=1.639e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-5.183e-04, signed_mean=-6.012e-05, mare=2.316e-03, max_step=4.549e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=1.866e-10, signed_mean=-1.895e-06, mare=2.453e-06, max_step=1.263e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=8.052e-01, mare_max=8.835e-01, max_step_max=6.619e+00, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.264e-03, mare_max=4.456e-03, max_step_max=2.411e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
