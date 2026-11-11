# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-fx-dp16-embf32-1ks`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 5、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | inconclusive | d0=-8.139e-07, signed_mean=4.293e-06, mare=9.301e-06, max_step=7.101e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | loss_series | L4 | inconclusive | d0=-8.139e-07, signed_mean=4.891e-06, mare=9.886e-06, max_step=7.101e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=5.089e-07, signed_mean=1.425e-05, mare=3.327e-05, max_step=1.703e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=5.089e-07, signed_mean=1.610e-05, mare=3.650e-05, max_step=1.703e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-1.555e-05, signed_mean=-1.285e-04, mare=3.258e-04, max_step=1.541e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-1.247e-04, signed_mean=7.726e-04, mare=3.365e-03, max_step=8.993e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=-1.131e-07, signed_mean=-3.743e-06, mare=3.744e-06, max_step=2.793e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.435e-01, mare_max=2.481e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=8.601e-04, mare_max=5.232e-03, max_step_max=2.513e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
