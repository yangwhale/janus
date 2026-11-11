# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-c-nosrb`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=-8.691e-07, mare=4.779e-06, max_step=2.521e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=-7.560e-07, mare=5.818e-06, max_step=2.521e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-3.053e-07, signed_mean=-6.602e-06, mare=1.227e-05, max_step=6.368e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=3.218e-07, signed_mean=-6.578e-06, mare=1.969e-04, max_step=7.805e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=2.204e-05, signed_mean=9.960e-05, mare=3.243e-04, max_step=1.261e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=1.421e-07, mare=1.437e-07, max_step=3.747e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.618e-01, mare_max=2.915e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.275e-03, mare_max=4.224e-03, max_step_max=2.584e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
