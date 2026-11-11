# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-e-idx32-1ks`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 8、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=-1.628e-07, signed_mean=4.143e-06, mare=8.105e-06, max_step=4.619e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=-1.628e-07, signed_mean=4.548e-06, mare=8.655e-06, max_step=4.619e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=1.018e-07, signed_mean=1.508e-05, mare=3.188e-05, max_step=1.765e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=1.018e-07, signed_mean=1.690e-05, mare=3.504e-05, max_step=1.765e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=9.010e-06, signed_mean=-1.391e-04, mare=3.171e-04, max_step=1.695e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-4.086e-05, signed_mean=4.126e-04, mare=3.097e-03, max_step=5.588e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.196e-06, mare=2.197e-06, max_step=1.596e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.047e-01, mare_max=4.133e-01, max_step_max=1.346e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=6.182e-04, mare_max=4.701e-03, max_step_max=2.663e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
