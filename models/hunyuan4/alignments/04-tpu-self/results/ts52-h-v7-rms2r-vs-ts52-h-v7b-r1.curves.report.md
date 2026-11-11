# janus 报告：hunyuan4-mini / tpu-self-vs-ts52-h-v7b-r1

- 被测：`ts52-h-v7-rms2r`；golden：`ts52-h-v7b-r1`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts52-h-v7b-r1.yaml`（内容指纹 6ba43723648b…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.817e-06, mare=5.520e-06, max_step=2.693e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=3.194e-06, mare=6.537e-06, max_step=2.693e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-4.733e-06, mare=1.124e-05, max_step=6.072e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.073e-07, signed_mean=1.377e-05, mare=2.014e-04, max_step=9.766e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-5.998e-06, signed_mean=1.215e-04, mare=3.394e-04, max_step=1.748e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-5.918e-08, mare=6.908e-08, max_step=2.016e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=6.625e-01, mare_max=6.625e-01, max_step_max=4.791e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.480e-03, mare_max=4.923e-03, max_step_max=2.092e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
