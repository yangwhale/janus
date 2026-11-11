# janus 报告：hunyuan4-mini / tpu-self-vs-ts-f16

- 被测：`ts-f16-scar`；golden：`ts-f16`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-f16.yaml`（内容指纹 25ffe1fd0566…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.mtp_loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.628e-06, mare=5.952e-06, max_step=2.071e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.544e-06, mare=6.951e-06, max_step=2.071e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=1.304e-06, mare=1.267e-05, max_step=1.339e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-6.519e-06, mare=2.018e-04, max_step=1.003e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.177e-05, mare=3.254e-04, max_step=1.741e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=5.878e-08, mare=6.345e-08, max_step=2.903e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.288e-01, mare_max=2.984e-01, max_step_max=1.474e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.206e-03, mare_max=5.781e-03, max_step_max=2.265e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
