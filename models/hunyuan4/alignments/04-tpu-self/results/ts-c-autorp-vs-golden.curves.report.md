# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-c-autorp`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.689e-06, mare=6.463e-06, max_step=2.851e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.401e-06, mare=7.739e-06, max_step=2.851e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.011e-05, mare=1.537e-05, max_step=8.716e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.128e-05, mare=2.372e-04, max_step=1.240e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-1.049e-04, mare=4.037e-04, max_step=1.623e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-6.871e-08, mare=8.311e-08, max_step=5.233e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.173e-01, mare_max=2.593e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.466e-03, mare_max=5.144e-03, max_step_max=2.154e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
