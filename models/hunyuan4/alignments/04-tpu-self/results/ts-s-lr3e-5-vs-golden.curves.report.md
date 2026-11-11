# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-s-lr3e-5`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 6、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`C.loss`、`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-3.177e-06, mare=6.309e-06, max_step=2.389e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-3.913e-06, mare=7.506e-06, max_step=2.389e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-7.756e-06, mare=1.465e-05, max_step=1.027e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=4.589e-05, mare=2.103e-04, max_step=1.304e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-4.313e-05, mare=3.890e-04, max_step=1.727e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=4.472e-08, mare=5.089e-08, max_step=1.777e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.895e-01, mare_max=2.895e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.697e-03, mare_max=5.954e-03, max_step_max=2.566e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
