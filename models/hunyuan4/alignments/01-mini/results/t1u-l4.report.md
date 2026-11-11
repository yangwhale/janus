# janus 报告：hunyuan4-mini / t1u-l4

- 被测：`t1u-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1u-l4.yaml`（内容指纹 0021d7d41be3…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.indexer_loss`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=3.337e-06, signed_mean_max=4.733e-07, mare_max=7.622e-06, max_step_max=4.235e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=3.337e-06, signed_mean_max=5.674e-07, mare_max=8.098e-06, max_step_max=4.235e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-3.053e-07, signed_mean=8.879e-06, mare=3.159e-05, max_step=1.553e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-3.053e-07, signed_mean=9.698e-06, mare=3.465e-05, max_step=1.553e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=-8.537e-04, signed_mean=3.029e-04, mare=6.739e-04, max_step=2.334e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-5.251e-04, signed_mean=-1.904e-04, mare=2.592e-03, max_step=5.245e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=2.940e-09, signed_mean=-4.296e-06, mare=4.296e-06, max_step=1.575e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.774e-01, mare_max=3.838e-01, max_step_max=4.034e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.221e-03, mare_max=4.780e-03, max_step_max=3.099e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
