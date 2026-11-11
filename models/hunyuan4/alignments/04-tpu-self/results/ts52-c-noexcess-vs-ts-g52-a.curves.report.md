# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-c-noexcess`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 3、fail 4、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.indexer_loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=-3.255e-07, signed_mean=3.336e-05, mare=3.503e-05, max_step=9.234e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=-3.255e-07, signed_mean=5.012e-05, mare=5.012e-05, max_step=9.234e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.018e-07, signed_mean=-8.217e-06, mare=1.571e-05, max_step=8.545e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=6.360e-05, signed_mean=-6.372e-05, mare=2.792e-04, max_step=1.181e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | fail | d0=-5.295e-03, signed_mean=-1.247e-03, mare=1.436e-03, max_step=6.577e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-6.766e-07, mare=6.773e-07, max_step=2.370e-06 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.608e-01, mare_max=1.800e-01, max_step_max=2.103e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=8.542e-04, mare_max=5.352e-03, max_step_max=2.229e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
