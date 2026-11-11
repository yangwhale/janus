# janus 报告：hunyuan4-mini / t1zr-l4

- 被测：`t1zr-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zr-l4.yaml`（内容指纹 03a290871ecb…）
- 结果：pass 6、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=3.500e-06, signed_mean_max=2.868e-06, mare_max=8.668e-06, max_step_max=4.538e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=3.500e-06, signed_mean_max=2.814e-06, mare_max=9.117e-06, max_step_max=4.538e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=1.730e-06, signed_mean=-2.844e-06, mare=3.201e-05, max_step=1.854e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=1.730e-06, signed_mean=-3.158e-06, mare=3.519e-05, max_step=1.854e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.719e-04, signed_mean=1.265e-04, mare=3.102e-04, max_step=1.619e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-5.214e-04, signed_mean=1.054e-04, mare=2.366e-03, max_step=4.245e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=5.413e-09, signed_mean=-1.176e-06, mare=1.531e-06, max_step=6.995e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=5.909e-01, mare_max=6.249e-01, max_step_max=6.619e+00, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.059e-03, mare_max=5.003e-03, max_step_max=2.541e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
