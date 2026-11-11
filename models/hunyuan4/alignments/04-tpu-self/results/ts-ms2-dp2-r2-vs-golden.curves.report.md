# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-ms2-dp2-r2`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-8.952e-07, signed_mean=-6.441e-07, mare=5.213e-06, max_step=2.347e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-8.952e-07, signed_mean=1.346e-08, mare=5.905e-06, max_step=2.347e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-8.142e-07, signed_mean=1.693e-06, mare=1.171e-05, max_step=7.462e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-4.977e-05, signed_mean=-7.669e-06, mare=2.074e-04, max_step=8.949e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.198e-04, signed_mean=-9.042e-06, mare=3.521e-04, max_step=1.398e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-3.121e-08, mare=7.124e-08, max_step=4.118e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=7.667e-01, mare_max=7.673e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.479e-03, mare_max=6.141e-03, max_step_max=2.731e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
