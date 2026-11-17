# janus 报告：hunyuan4-mid / tmc-l4

- 被测：`tmc-a`；golden：`gm-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/03-mid/catalogs/tmc-l4.yaml`（内容指纹 7261578b8e3f…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.mtp_loss`、`L4.mtp_loss.tail`、`L4.indexer_loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=2.187e-05, signed_mean_max=1.241e-06, mare_max=1.636e-05, max_step_max=6.540e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=2.187e-05, signed_mean_max=6.002e-07, mare_max=1.740e-05, max_step_max=6.540e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=-1.080e-05, signed_mean=2.283e-05, mare=2.454e-05, max_step=9.293e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=-1.080e-05, signed_mean=3.111e-05, mare=3.142e-05, max_step=9.293e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.362e-04, signed_mean=2.975e-04, mare=3.289e-04, max_step=1.272e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-1.637e-04, signed_mean=-4.337e-05, mare=6.159e-04, max_step=4.007e-03 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=1.244e-09, signed_mean=-5.876e-08, mare=6.552e-08, max_step=2.149e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.431e-01, mare_max=1.577e-01, max_step_max=1.982e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=7.492e-04, mare_max=2.016e-03, max_step_max=8.262e-03, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
