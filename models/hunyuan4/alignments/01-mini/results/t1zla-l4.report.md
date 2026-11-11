# janus 报告：hunyuan4-mini / t1zla-l4

- 被测：`t1zl-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zla-l4.yaml`（内容指纹 9e023eb26045…）
- 结果：pass 4、fail 0、inconclusive 5、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.mtp_loss`、`L4.mtp_loss.tail`、`L4.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=2.604e-06, signed_mean_max=5.442e-06, mare_max=9.537e-06, max_step_max=5.568e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=2.604e-06, signed_mean_max=5.705e-06, mare_max=1.006e-05, max_step_max=5.568e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=2.239e-06, signed_mean=-2.902e-05, mare=3.921e-05, max_step=2.111e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=2.239e-06, signed_mean=-3.207e-05, mare=4.310e-05, max_step=2.111e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-6.736e-05, signed_mean=-1.165e-04, mare=3.195e-04, max_step=1.575e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-5.223e-04, signed_mean=-1.614e-05, mare=2.522e-03, max_step=5.765e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=2.215e-10, signed_mean=-1.166e-06, mare=1.531e-06, max_step=7.787e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=9.031e-02, mare_max=1.629e-01, max_step_max=8.079e-01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=1.604e-03, mare_max=4.306e-03, max_step_max=2.334e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
