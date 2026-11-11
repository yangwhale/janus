# janus 报告：hunyuan4-mini / tsnolhs-l4-n16

- 被测：`ts-c-nolhs-1k`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/gpu-noise/catalogs/tsnolhs-l4-n16.yaml`（内容指纹 75748d5ef2de…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=2.604e-06, signed_mean_max=4.532e-06, mare_max=1.109e-05, max_step_max=6.661e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=2.604e-06, signed_mean_max=4.796e-06, mare_max=1.183e-05, max_step_max=6.661e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.239e-06, signed_mean=-5.831e-06, mare=3.027e-05, max_step=1.822e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.239e-06, signed_mean=-6.454e-06, mare=3.326e-05, max_step=1.822e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-6.736e-05, signed_mean=-2.126e-04, mare=3.530e-04, max_step=1.720e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-5.225e-04, signed_mean=8.090e-05, mare=2.424e-03, max_step=5.076e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-3.538e-10, signed_mean=-1.589e-06, mare=2.104e-06, max_step=1.266e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.915e-01, mare_max=5.122e-01, max_step_max=4.206e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.307e-03, mare_max=4.515e-03, max_step_max=2.437e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
