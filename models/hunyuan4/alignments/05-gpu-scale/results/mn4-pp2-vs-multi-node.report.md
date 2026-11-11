# janus 报告：hunyuan4-mini / gpu-scale-multi-node-self

- 被测：`mn4-pp2`；golden：`mn-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/multi-node-self.yaml`（内容指纹 634cd82d964b…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.628e-07, signed_mean=1.757e-06, mare=5.867e-06, max_step=2.665e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.628e-07, signed_mean=2.641e-06, mare=7.153e-06, max_step=2.665e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=5.089e-07, signed_mean=5.559e-06, mare=1.319e-05, max_step=8.236e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=2.488e-05, signed_mean=-2.973e-05, mare=2.274e-04, max_step=9.850e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-5.883e-05, signed_mean=8.809e-05, mare=3.547e-04, max_step=1.792e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=7.063e-09, signed_mean=-2.786e-08, mare=6.076e-08, max_step=1.569e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=7.606e-01, mare_max=7.606e-01, max_step_max=1.121e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.154e-03, mare_max=5.173e-03, max_step_max=2.251e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
