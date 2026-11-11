# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-e-ihcr-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 6、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`C.indexer_loss`、`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=1.300e-06, mare=5.799e-06, max_step=2.448e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=8.139e-08, signed_mean=1.200e-06, mare=6.811e-06, max_step=2.448e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-3.461e-06, signed_mean=8.649e-08, mare=1.184e-05, max_step=6.698e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=3.944e-04, signed_mean=-1.026e-04, mare=2.733e-04, max_step=1.026e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=2.345e-04, signed_mean=-1.044e-04, mare=3.975e-04, max_step=1.693e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.029e-07, mare=1.060e-07, max_step=3.317e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.388e-01, mare_max=2.883e-01, max_step_max=1.077e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.954e-03, mare_max=5.286e-03, max_step_max=1.936e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
