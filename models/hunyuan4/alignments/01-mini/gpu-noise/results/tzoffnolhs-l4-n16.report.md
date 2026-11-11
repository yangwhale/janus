# janus 报告：hunyuan4-mini / tzoffnolhs-l4-n16

- 被测：`tz-off-nolhs`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/gpu-noise/catalogs/tzoffnolhs-l4-n16.yaml`（内容指纹 2ba5edd13088…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=4.151e-06, signed_mean_max=2.960e-06, mare_max=9.459e-06, max_step_max=3.769e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=4.151e-06, signed_mean_max=3.100e-06, mare_max=1.005e-05, max_step_max=3.769e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-1.018e-07, signed_mean=-1.795e-05, mare=3.396e-05, max_step=1.915e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-1.018e-07, signed_mean=-1.976e-05, mare=3.739e-05, max_step=1.915e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-1.211e-04, signed_mean=-2.273e-04, mare=3.489e-04, max_step=1.532e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-5.330e-04, signed_mean=-2.304e-04, mare=2.679e-03, max_step=6.665e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-8.229e-10, signed_mean=-1.426e-06, mare=2.139e-06, max_step=8.700e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.518e-01, mare_max=4.747e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.449e-03, mare_max=4.759e-03, max_step_max=2.343e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
