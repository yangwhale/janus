# janus 报告：hunyuan4-trunc / t2bz-l4

- 被测：`tb3-b`；golden：`gb2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/02-real-weight/catalogs/t2bz-l4.yaml`（内容指纹 27fae3e4f37f…）
- 结果：pass 2、fail 1、inconclusive 4、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.indexer_loss`、`L4.param_norm`、`L4.router.bias_mean`、`L4.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=1.515e-04, signed_mean_max=2.557e-05, mare_max=3.753e-05, max_step_max=1.528e-04 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.515e-04, signed_mean_max=1.214e-06, mare_max=1.667e-05, max_step_max=6.378e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=-1.314e-03, signed_mean=1.716e-04, mare=7.194e-04, max_step=2.774e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=3.025e-03, signed_mean=-1.263e-05, mare=5.059e-03, max_step=2.788e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=-1.888e-09, signed_mean=-1.677e-08, mare=1.677e-08, max_step=4.570e-08 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=1.311e-01, mare_max=1.342e-01, max_step_max=3.116e-01, stats=1 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=1.830e-03, mare_max=3.219e-03, max_step_max=7.747e-03, stats=1 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
