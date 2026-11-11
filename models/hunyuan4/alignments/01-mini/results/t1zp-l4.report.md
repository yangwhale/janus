# janus 报告：hunyuan4-mini / t1zp-l4

- 被测：`x5-prod`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zp-l4.yaml`（内容指纹 688f28b0d710…）
- 结果：pass 8、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.indexer_loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.221e-06, signed_mean_max=5.774e-07, mare_max=7.440e-06, max_step_max=4.735e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.221e-06, signed_mean_max=7.029e-07, mare_max=7.707e-06, max_step_max=4.735e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=1.425e-06, signed_mean=-2.458e-05, mare=3.637e-05, max_step=1.835e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=1.425e-06, signed_mean=-2.729e-05, mare=3.997e-05, max_step=1.835e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=3.134e-04, signed_mean=-1.509e-04, mare=4.564e-04, max_step=2.651e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-2.828e-04, signed_mean=3.641e-04, mare=2.505e-03, max_step=6.190e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=1.993e-08, signed_mean=2.892e-06, mare=2.956e-06, max_step=9.178e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=5.602e-01, mare_max=6.416e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.280e-03, mare_max=4.953e-03, max_step_max=2.392e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
