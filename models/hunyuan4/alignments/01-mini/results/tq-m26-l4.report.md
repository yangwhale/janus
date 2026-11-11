# janus 报告：hunyuan4-mini / tq-m26-l4

- 被测：`tq-m26`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m26-l4.yaml`（内容指纹 5a6aa2cc1e83…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.indexer_loss`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=7.325e-07, signed_mean_max=1.746e-06, mare_max=5.389e-06, max_step_max=2.953e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=7.325e-07, signed_mean_max=7.012e-07, mare_max=5.750e-06, max_step_max=2.953e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.748e-06, signed_mean=-3.488e-06, mare=1.322e-05, max_step=6.794e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.748e-06, signed_mean=-5.456e-06, mare=1.835e-05, max_step=6.794e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.167e-04, signed_mean=1.471e-04, mare=2.364e-04, max_step=1.177e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.766e-04, signed_mean=3.575e-05, mare=4.813e-04, max_step=1.572e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-5.241e-09, signed_mean=-1.001e-08, mare=6.313e-08, max_step=2.228e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.144e+00, mare_max=1.195e+00, max_step_max=4.187e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.305e-03, mare_max=4.852e-03, max_step_max=2.682e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
