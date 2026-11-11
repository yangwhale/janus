# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-d-dotattn`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.302e-06, signed_mean=-1.125e-06, mare=5.205e-06, max_step=2.471e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.302e-06, signed_mean=-5.823e-07, mare=5.953e-06, max_step=2.471e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-9.160e-07, signed_mean=7.015e-06, mare=1.392e-05, max_step=9.524e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-5.191e-05, signed_mean=-4.533e-05, mare=2.306e-04, max_step=9.974e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=8.457e-05, signed_mean=3.036e-05, mare=3.444e-04, max_step=1.617e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-2.262e-07, mare=2.290e-07, max_step=9.776e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.582e-01, mare_max=3.013e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.373e-03, mare_max=6.454e-03, max_step_max=2.689e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
