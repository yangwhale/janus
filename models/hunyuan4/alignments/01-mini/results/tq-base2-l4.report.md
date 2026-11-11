# janus 报告：hunyuan4-mini / tq-base2-l4

- 被测：`tq-base2`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-base2-l4.yaml`（内容指纹 58bd7b08e52e…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=2.360e-06, signed_mean_max=3.728e-06, mare_max=6.302e-06, max_step_max=3.157e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=2.360e-06, signed_mean_max=4.750e-06, mare_max=7.565e-06, max_step_max=3.157e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.036e-07, signed_mean=-6.987e-07, mare=1.179e-05, max_step=1.031e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.036e-07, signed_mean=-4.873e-07, mare=1.629e-05, max_step=1.031e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=5.931e-05, signed_mean=1.272e-04, mare=2.445e-04, max_step=9.800e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.919e-04, signed_mean=1.134e-04, mare=5.245e-04, max_step=1.860e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=-4.745e-09, signed_mean=-2.238e-07, mare=2.238e-07, max_step=4.773e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.165e-01, mare_max=4.740e-01, max_step_max=2.965e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=6.169e-04, mare_max=5.312e-03, max_step_max=2.326e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
