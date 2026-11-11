# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-ebc-dp8`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 6、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`C.loss`、`C.loss.tail`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=-9.766e-07, signed_mean=3.603e-06, mare=5.427e-06, max_step=2.661e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | inconclusive | d0=-9.766e-07, signed_mean=4.979e-06, mare=6.735e-06, max_step=2.661e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=9.323e-07, mare=1.194e-05, max_step=9.909e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-5.985e-05, signed_mean=5.983e-05, mare=1.969e-04, max_step=1.142e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=7.138e-05, signed_mean=6.233e-05, mare=2.754e-04, max_step=1.617e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-4.508e-08, mare=7.553e-08, max_step=3.776e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.624e-01, mare_max=4.917e-01, max_step_max=8.411e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.268e-03, mare_max=5.474e-03, max_step_max=1.997e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
