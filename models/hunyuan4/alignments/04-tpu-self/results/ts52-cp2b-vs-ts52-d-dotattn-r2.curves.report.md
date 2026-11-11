# janus 报告：hunyuan4-mini / tpu-self-vs-ts52-d-dotattn-r2

- 被测：`ts52-cp2b`；golden：`ts52-d-dotattn-r2`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts52-d-dotattn-r2.yaml`（内容指纹 208d8c2c197d…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-7.520e-07, mare=4.926e-06, max_step=1.904e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-7.859e-07, mare=6.058e-06, max_step=1.904e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-3.053e-06, signed_mean=2.809e-06, mare=9.982e-06, max_step=6.612e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=2.145e-07, signed_mean=1.840e-05, mare=1.776e-04, max_step=7.630e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-6.642e-05, signed_mean=-1.718e-05, mare=3.274e-04, max_step=1.806e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=9.497e-08, mare=9.649e-08, max_step=2.800e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.253e-01, mare_max=3.264e-01, max_step_max=1.843e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.913e-03, mare_max=6.043e-03, max_step_max=2.606e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
