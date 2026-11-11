# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-e-bf16acc-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`、`L4.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | inconclusive | d0=-4.069e-07, signed_mean=5.152e-06, mare=1.011e-05, max_step=1.070e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | loss_series | L4 | inconclusive | d0=-4.069e-07, signed_mean=5.579e-06, mare=1.081e-05, max_step=1.070e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-1.018e-07, signed_mean=-1.078e-05, mare=3.327e-05, max_step=1.658e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-1.018e-07, signed_mean=-1.188e-05, mare=3.664e-05, max_step=1.658e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-1.739e-04, signed_mean=1.068e-04, mare=3.364e-04, max_step=1.617e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=6.350e-05, signed_mean=1.012e-03, mare=3.505e-03, max_step=7.392e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=3.539e-07, mare=3.153e-06, max_step=1.878e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.691e-01, mare_max=1.716e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=2.024e-03, mare_max=5.122e-03, max_step_max=2.513e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
