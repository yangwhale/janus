# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-e-ihcr`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.indexer_loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=1.872e-06, signed_mean=8.569e-07, mare=5.916e-06, max_step=2.012e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=1.872e-06, signed_mean=1.294e-06, mare=7.004e-06, max_step=2.012e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.120e-06, signed_mean=-5.015e-07, mare=1.384e-05, max_step=7.983e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=3.618e-04, signed_mean=-1.344e-04, mare=2.964e-04, max_step=1.177e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | pass | d0=3.010e-04, signed_mean=4.283e-05, mare=3.375e-04, max_step=1.359e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-4.301e-08, mare=5.027e-08, max_step=3.197e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.689e-01, mare_max=5.288e-01, max_step_max=8.411e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.218e-03, mare_max=5.140e-03, max_step_max=2.233e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
