# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-tp2-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.mtp_loss`、`L4.mtp_loss.tail`、`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=1.144e-06, mare=8.516e-06, max_step=4.412e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=1.345e-06, mare=8.633e-06, max_step=4.412e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=-2.545e-06, signed_mean=2.740e-05, mare=3.654e-05, max_step=1.779e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=-2.545e-06, signed_mean=3.057e-05, mare=4.021e-05, max_step=1.779e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.888e-05, signed_mean=-9.471e-05, mare=3.142e-04, max_step=1.589e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-1.319e-05, signed_mean=3.294e-04, mare=3.300e-03, max_step=6.153e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=-2.247e-09, signed_mean=-3.680e-06, mare=3.683e-06, max_step=2.456e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.011e-01, mare_max=2.560e-01, max_step_max=1.317e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.463e-03, mare_max=4.896e-03, max_step_max=2.408e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
