# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a-1000

- 被测：`ts52-fx-dp8-embf32-1k`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a-1000.yaml`（内容指纹 aa60356a0036…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=-9.766e-07, signed_mean=3.010e-06, mare=7.740e-06, max_step=4.349e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=-9.766e-07, signed_mean=3.309e-06, mare=8.225e-06, max_step=4.349e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=9.071e-07, mare=3.009e-05, max_step=1.512e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=9.120e-07, mare=3.304e-05, max_step=1.512e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-5.985e-05, signed_mean=-7.587e-05, mare=2.888e-04, max_step=1.393e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=6.913e-05, signed_mean=2.635e-04, mare=2.700e-03, max_step=6.278e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=1.884e-07, mare=1.609e-06, max_step=1.200e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.912e-01, mare_max=2.650e-01, max_step_max=1.472e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=5.500e-04, mare_max=5.050e-03, max_step_max=2.321e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
