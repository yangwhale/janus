# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-c-scag`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 6、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`C.grad_norm`、`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=6.788e-07, mare=6.461e-06, max_step=2.650e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=1.567e-06, mare=7.577e-06, max_step=2.650e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=1.018e-07, signed_mean=-3.258e-06, mare=1.217e-05, max_step=6.043e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-1.716e-06, signed_mean=1.506e-05, mare=2.082e-04, max_step=9.015e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=2.489e-05, signed_mean=-6.196e-05, mare=3.989e-04, max_step=2.143e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.016e-07, mare=1.121e-07, max_step=2.754e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.033e-01, mare_max=2.520e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.527e-03, mare_max=5.337e-03, max_step_max=2.513e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
