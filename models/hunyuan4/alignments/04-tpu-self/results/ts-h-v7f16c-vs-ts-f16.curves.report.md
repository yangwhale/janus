# janus 报告：hunyuan4-mini / tpu-self-vs-ts-f16

- 被测：`ts-h-v7f16c`；golden：`ts-f16`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-f16.yaml`（内容指纹 25ffe1fd0566…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.mtp_loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.790e-06, signed_mean=-2.233e-06, mare=6.204e-06, max_step=4.201e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.790e-06, signed_mean=-2.661e-06, mare=7.623e-06, max_step=4.201e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | inconclusive | d0=-3.053e-07, signed_mean=5.172e-06, mare=1.443e-05, max_step=1.341e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-2.467e-05, signed_mean=6.512e-06, mare=2.168e-04, max_step=8.819e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=7.753e-05, signed_mean=-9.017e-05, mare=3.415e-04, max_step=1.660e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-2.055e-08, signed_mean=-6.656e-08, mare=7.721e-08, max_step=4.772e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.272e-01, mare_max=2.289e-01, max_step_max=1.296e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.231e-03, mare_max=4.886e-03, max_step_max=2.265e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
