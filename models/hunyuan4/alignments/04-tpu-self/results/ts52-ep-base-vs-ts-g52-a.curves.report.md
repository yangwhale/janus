# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-ep-base`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-3.255e-07, signed_mean=-1.967e-07, mare=5.793e-06, max_step=2.513e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-3.255e-07, signed_mean=7.134e-07, mare=6.942e-06, max_step=2.513e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=5.089e-07, signed_mean=9.101e-07, mare=1.177e-05, max_step=7.792e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-8.173e-05, signed_mean=1.581e-06, mare=1.986e-04, max_step=9.627e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=1.138e-04, signed_mean=2.792e-05, mare=3.039e-04, max_step=1.551e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-9.651e-08, mare=1.046e-07, max_step=6.842e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.474e-01, mare_max=2.485e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.089e-03, mare_max=4.127e-03, max_step_max=2.779e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
