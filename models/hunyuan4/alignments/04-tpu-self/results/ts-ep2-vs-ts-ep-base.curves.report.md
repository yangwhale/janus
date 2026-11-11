# janus 报告：hunyuan4-mini / tpu-self-vs-ts-ep-base

- 被测：`ts-ep2`；golden：`ts-ep-base`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-ep-base.yaml`（内容指纹 a041ba5e9666…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=3.255e-07, signed_mean=8.179e-07, mare=6.229e-06, max_step=2.902e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=3.255e-07, signed_mean=1.694e-06, mare=7.784e-06, max_step=2.902e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=2.036e-07, signed_mean=9.003e-07, mare=1.179e-05, max_step=5.867e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-4.076e-06, signed_mean=-6.664e-05, mare=2.049e-04, max_step=8.545e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=5.496e-05, signed_mean=4.952e-05, mare=3.433e-04, max_step=1.313e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.335e-12, signed_mean=2.272e-08, mare=2.728e-08, max_step=1.267e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.688e-01, mare_max=4.468e-01, max_step_max=2.791e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.260e-03, mare_max=4.295e-03, max_step_max=2.243e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
