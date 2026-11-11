# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-esd-dp8`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 5、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`C.loss`、`C.loss.tail`、`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=-9.766e-07, signed_mean=3.028e-06, mare=6.573e-06, max_step=2.540e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | inconclusive | d0=-9.766e-07, signed_mean=5.586e-06, mare=7.996e-06, max_step=2.540e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=4.671e-06, mare=1.332e-05, max_step=8.271e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-5.985e-05, signed_mean=-1.837e-05, mare=2.071e-04, max_step=1.018e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=7.131e-05, signed_mean=4.379e-05, mare=3.558e-04, max_step=1.801e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=-1.131e-07, signed_mean=-1.186e-07, mare=1.397e-07, max_step=7.740e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.575e-01, mare_max=4.575e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.120e-03, mare_max=5.820e-03, max_step_max=2.794e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
