# janus 报告：hunyuan4-mini / t1-l4-nozm

- 被测：`g1-a`；golden：`t1n-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1-l4-nozm.yaml`（内容指纹 0816769ace65…）
- 结果：pass 2、fail 3、inconclusive 4、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss.tail`、`L4.mtp_loss`、`L4.mtp_loss.tail`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=7.325e-07, signed_mean_max=7.775e-06, mare_max=2.423e-05, max_step_max=2.681e-04 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=7.325e-07, signed_mean_max=5.388e-06, mare_max=1.120e-05, max_step_max=3.481e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=4.071e-07, signed_mean=1.398e-05, mare=1.883e-05, max_step=1.325e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=4.071e-07, signed_mean=2.097e-05, mare=2.470e-05, max_step=1.325e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=9.365e-04, signed_mean=4.888e-04, mare=5.568e-04, max_step=1.345e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | fail | d0=2.582e-04, signed_mean=-2.225e-04, mare=6.451e-04, max_step=2.640e-03 | 超过噪声参照两倍 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=1.546e-07, signed_mean=4.167e-07, mare=4.206e-07, max_step=9.166e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.353e-01, mare_max=2.513e-01, max_step_max=3.601e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=9.212e-04, mare_max=6.087e-03, max_step_max=2.756e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
