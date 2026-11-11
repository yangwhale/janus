# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a-1000

- 被测：`ts52-e-ihcr-1k`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a-1000.yaml`（内容指纹 aa60356a0036…）
- 结果：pass 8、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.loss.tail`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=1.872e-06, signed_mean=4.596e-06, mare=8.730e-06, max_step=5.396e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | inconclusive | d0=1.872e-06, signed_mean=5.105e-06, mare=9.281e-06, max_step=5.396e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-1.120e-06, signed_mean=1.406e-05, mare=3.317e-05, max_step=2.159e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-1.120e-06, signed_mean=1.560e-05, mare=3.642e-05, max_step=2.159e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=3.618e-04, signed_mean=8.702e-05, mare=3.368e-04, max_step=1.405e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=3.010e-04, signed_mean=2.159e-04, mare=2.742e-03, max_step=5.508e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.550e-06, mare=2.552e-06, max_step=1.808e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.212e-01, mare_max=3.463e-01, max_step_max=8.411e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.123e-03, mare_max=4.696e-03, max_step_max=2.233e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
