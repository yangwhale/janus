# janus 报告：hunyuan4-mini / tq-m34-l4

- 被测：`tq-m34`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-m34-l4.yaml`（内容指纹 9714ad79c834…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.indexer_loss`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=2.600e-06, mare_max=6.223e-06, max_step_max=2.415e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.058e-06, signed_mean_max=2.161e-06, mare_max=6.991e-06, max_step_max=2.415e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=-4.795e-06, mare=1.319e-05, max_step=6.978e-05 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.646e-06, signed_mean=-6.783e-06, mare=1.821e-05, max_step=6.978e-05 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | inconclusive | d0=1.126e-04, signed_mean=1.768e-04, mare=2.695e-04, max_step=1.285e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.628e-04, signed_mean=3.717e-05, mare=4.985e-04, max_step=1.585e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-5.048e-09, signed_mean=-1.454e-07, mare=1.454e-07, max_step=3.430e-07 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.017e-01, mare_max=3.017e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.652e-03, mare_max=5.366e-03, max_step_max=2.408e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
