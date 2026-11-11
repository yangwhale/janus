# janus 报告：hunyuan4-mini / t1zo-l4

- 被测：`tz-off-rep`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zo-l4.yaml`（内容指纹 514f61624bd4…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=4.151e-06, signed_mean_max=7.212e-06, mare_max=1.144e-05, max_step_max=6.868e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=4.151e-06, signed_mean_max=7.618e-06, mare_max=1.213e-05, max_step_max=6.868e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-1.018e-07, signed_mean=1.061e-05, mare=3.722e-05, max_step=2.143e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-1.018e-07, signed_mean=1.156e-05, mare=4.094e-05, max_step=2.143e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-1.211e-04, signed_mean=-9.367e-05, mare=2.985e-04, max_step=1.400e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-5.347e-04, signed_mean=3.375e-04, mare=2.372e-03, max_step=3.756e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.121e-09, signed_mean=-3.531e-07, mare=1.598e-06, max_step=1.188e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.056e-01, mare_max=2.235e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.206e-03, mare_max=4.441e-03, max_step_max=2.486e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
