# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-e-swiglu-1ks`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 6、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | inconclusive | d0=-8.139e-08, signed_mean=2.723e-06, mare=9.046e-06, max_step=5.440e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | loss_series | L4 | inconclusive | d0=-8.139e-08, signed_mean=2.972e-06, mare=9.676e-06, max_step=5.440e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-5.089e-07, signed_mean=2.020e-05, mare=3.360e-05, max_step=1.839e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-5.089e-07, signed_mean=2.247e-05, mare=3.698e-05, max_step=1.839e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.770e-05, signed_mean=2.085e-05, mare=2.859e-04, max_step=1.317e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-1.374e-04, signed_mean=8.463e-04, mare=2.660e-03, max_step=7.146e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-2.087e-06, mare=2.109e-06, max_step=2.007e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.613e-01, mare_max=2.639e-01, max_step_max=1.077e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=7.893e-04, mare_max=5.191e-03, max_step_max=2.390e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
