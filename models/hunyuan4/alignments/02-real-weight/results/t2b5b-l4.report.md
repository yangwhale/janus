# janus 报告：hunyuan4-trunc / t2b5b-l4

- 被测：`tb5b-a`；golden：`gb3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/02-real-weight/catalogs/t2b5b-l4.yaml`（内容指纹 737c140469b5…）
- 结果：pass 4、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=1.450e-04, signed_mean_max=8.022e-06, mare_max=2.913e-05, max_step_max=1.359e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.450e-04, signed_mean_max=1.714e-05, mare_max=2.411e-05, max_step_max=9.742e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-1.222e-03, signed_mean=2.082e-04, mare=9.253e-04, max_step=6.045e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=3.094e-03, signed_mean=5.866e-04, mare=9.650e-03, max_step=8.296e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=-1.920e-09, signed_mean=-5.998e-08, mare=5.998e-08, max_step=1.919e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.909e-03, mare_max=1.263e-02, max_step_max=1.428e-01, stats=1 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=7.133e-05, mare_max=1.123e-03, max_step_max=6.050e-03, stats=1 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
