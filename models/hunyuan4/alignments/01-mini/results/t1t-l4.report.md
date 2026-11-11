# janus 报告：hunyuan4-mini / t1t-l4

- 被测：`t1t-c`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1t-l4.yaml`（内容指纹 d1daaf4d0d07…）
- 结果：pass 4、fail 1、inconclusive 4、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=1.546e-06, signed_mean_max=2.605e-06, mare_max=6.875e-06, max_step_max=3.104e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=1.546e-06, signed_mean_max=3.725e-06, mare_max=8.676e-06, max_step_max=3.104e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.780e-06, mare=1.389e-05, max_step=1.156e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.021e-06, mare=1.912e-05, max_step=1.156e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-8.721e-04, signed_mean=-4.264e-04, mare=4.857e-04, max_step=1.167e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.730e-04, signed_mean=5.788e-05, mare=5.158e-04, max_step=2.228e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=-1.602e-09, signed_mean=-3.044e-07, mare=3.045e-07, max_step=7.563e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=6.172e-01, mare_max=6.187e-01, max_step_max=3.489e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.258e-03, mare_max=5.449e-03, max_step_max=2.713e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
