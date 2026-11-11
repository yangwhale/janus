# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-n2-dotf32w`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.628e-07, signed_mean=8.021e-07, mare=5.289e-06, max_step=2.347e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.628e-07, signed_mean=1.745e-06, mare=6.408e-06, max_step=2.347e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.018e-07, signed_mean=-5.278e-06, mare=1.438e-05, max_step=1.196e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=5.041e-06, signed_mean=-4.400e-05, mare=1.983e-04, max_step=9.807e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=4.731e-05, signed_mean=8.792e-05, mare=3.346e-04, max_step=1.374e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-6.999e-08, mare=1.151e-07, max_step=5.590e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.427e-01, mare_max=2.329e-01, max_step_max=2.804e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.706e-03, mare_max=4.140e-03, max_step_max=2.545e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
