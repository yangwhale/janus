# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a-1000

- 被测：`ts52-ebc-dp8-1k`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a-1000.yaml`（内容指纹 aa60356a0036…）
- 结果：pass 7、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | inconclusive | d0=-9.766e-07, signed_mean=5.865e-06, mare=8.534e-06, max_step=4.839e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | loss_series | L4 | inconclusive | d0=-9.766e-07, signed_mean=6.420e-06, mare=9.166e-06, max_step=4.839e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=2.825e-06, mare=3.181e-05, max_step=1.505e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=2.885e-06, mare=3.488e-05, max_step=1.505e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-5.985e-05, signed_mean=1.161e-05, mare=2.847e-04, max_step=1.316e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=7.138e-05, signed_mean=6.540e-05, mare=2.742e-03, max_step=5.941e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-5.540e-07, mare=7.508e-07, max_step=9.218e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.054e-01, mare_max=2.147e-01, max_step_max=8.411e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.581e-03, mare_max=5.019e-03, max_step_max=2.868e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
