# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-e-rms2r-fsb`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.233e-06, mare=5.146e-06, max_step=2.833e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=3.150e-06, mare=6.269e-06, max_step=2.833e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-3.706e-06, mare=1.273e-05, max_step=8.257e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.947e-06, mare=1.953e-04, max_step=8.816e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-8.698e-06, signed_mean=6.278e-05, mare=3.111e-04, max_step=1.310e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.196e-08, mare=3.189e-08, max_step=1.286e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=6.038e-01, mare_max=6.038e-01, max_step_max=9.813e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.560e-03, mare_max=5.665e-03, max_step_max=2.645e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
