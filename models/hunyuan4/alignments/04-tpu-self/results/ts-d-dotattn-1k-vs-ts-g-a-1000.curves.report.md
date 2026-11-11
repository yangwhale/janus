# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-d-dotattn-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | inconclusive | d0=-1.302e-06, signed_mean=4.966e-06, mare=9.316e-06, max_step=6.514e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | loss_series | L4 | inconclusive | d0=-1.302e-06, signed_mean=5.757e-06, mare=9.934e-06, max_step=6.514e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-9.160e-07, signed_mean=1.915e-05, mare=3.362e-05, max_step=2.062e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-9.160e-07, signed_mean=2.149e-05, mare=3.693e-05, max_step=2.062e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-5.191e-05, signed_mean=-1.622e-04, mare=3.331e-04, max_step=1.324e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=8.457e-05, signed_mean=6.057e-04, mare=3.153e-03, max_step=6.736e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-4.232e-06, mare=4.233e-06, max_step=2.300e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.739e-01, mare_max=2.798e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.129e-03, mare_max=5.712e-03, max_step_max=2.689e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
