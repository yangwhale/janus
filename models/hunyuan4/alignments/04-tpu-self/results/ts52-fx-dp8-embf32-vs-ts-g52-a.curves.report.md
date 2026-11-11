# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-fx-dp8-embf32`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 6、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`C.loss`、`C.loss.tail`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=-9.766e-07, signed_mean=4.678e-06, mare=6.241e-06, max_step=2.752e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | inconclusive | d0=-9.766e-07, signed_mean=6.848e-06, mare=7.683e-06, max_step=2.752e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=-1.772e-06, mare=1.097e-05, max_step=8.495e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-5.985e-05, signed_mean=9.079e-07, mare=1.934e-04, max_step=7.763e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=6.913e-05, signed_mean=8.672e-05, mare=3.271e-04, max_step=1.496e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-1.539e-07, mare=1.556e-07, max_step=5.492e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=6.384e-01, mare_max=7.326e-01, max_step_max=1.472e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=6.532e-04, mare_max=5.934e-03, max_step_max=2.321e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
