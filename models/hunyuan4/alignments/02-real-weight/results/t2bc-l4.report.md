# janus 报告：hunyuan4-trunc / t2bc-l4

- 被测：`tbc-a`；golden：`gb3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/02-real-weight/catalogs/t2bc-l4.yaml`（内容指纹 6f1427bf99ef…）
- 结果：pass 6、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.indexer_loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=6.022e-05, signed_mean_max=1.235e-05, mare_max=2.847e-05, max_step_max=8.674e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=6.022e-05, signed_mean_max=1.866e-05, mare_max=2.665e-05, max_step_max=8.674e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=-2.093e-03, signed_mean=-7.975e-04, mare=1.081e-03, max_step=6.642e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=3.211e-03, signed_mean=2.672e-04, mare=7.754e-03, max_step=5.973e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.833e-09, signed_mean=-3.150e-08, mare=3.200e-08, max_step=7.742e-08 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.502e-02, mare_max=2.112e-02, max_step_max=1.525e-01, stats=1 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=4.286e-04, mare_max=9.960e-04, max_step_max=4.755e-03, stats=1 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
