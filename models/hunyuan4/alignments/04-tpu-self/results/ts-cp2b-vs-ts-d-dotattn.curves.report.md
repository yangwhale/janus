# janus 报告：hunyuan4-mini / tpu-self-vs-ts-d-dotattn

- 被测：`ts-cp2b`；golden：`ts-d-dotattn`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-d-dotattn.yaml`（内容指纹 0a7b47c58e50…）
- 结果：pass 5、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`C.loss`、`C.loss.tail`、`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=-8.139e-08, signed_mean=3.836e-06, mare=6.608e-06, max_step=3.141e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | inconclusive | d0=-8.139e-08, signed_mean=4.612e-06, mare=8.020e-06, max_step=3.141e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=5.089e-07, signed_mean=-1.165e-05, mare=1.525e-05, max_step=8.151e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=5.256e-05, signed_mean=4.832e-05, mare=2.397e-04, max_step=1.011e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-5.735e-05, signed_mean=1.030e-04, mare=3.424e-04, max_step=1.780e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=7.924e-08, mare=9.111e-08, max_step=3.641e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.395e-01, mare_max=4.240e-01, max_step_max=7.005e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.048e-03, mare_max=5.058e-03, max_step_max=2.420e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
