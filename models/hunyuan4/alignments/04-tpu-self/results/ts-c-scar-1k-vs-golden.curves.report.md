# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-c-scar-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.298e-06, mare=5.528e-06, max_step=2.876e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.338e-06, mare=5.933e-06, max_step=2.876e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=2.809e-06, mare=1.345e-05, max_step=7.719e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=9.640e-07, mare=2.235e-04, max_step=1.059e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-6.986e-05, mare=3.957e-04, max_step=1.866e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-6.086e-08, mare=7.884e-08, max_step=3.397e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.835e-01, mare_max=3.870e-01, max_step_max=2.693e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.955e-03, mare_max=5.580e-03, max_step_max=2.655e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
