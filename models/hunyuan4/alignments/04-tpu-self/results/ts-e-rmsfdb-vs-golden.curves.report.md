# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-e-rmsfdb`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 3、fail 2、inconclusive 3、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.indexer_loss`、`C.grad_norm`、`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=2.442e-07, signed_mean=2.056e-05, mare=2.075e-05, max_step=6.612e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=2.442e-07, signed_mean=2.792e-05, mare=2.793e-05, max_step=6.612e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-3.053e-07, signed_mean=8.906e-07, mare=1.284e-05, max_step=8.895e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=-7.294e-06, signed_mean=-1.819e-04, mare=2.633e-04, max_step=8.979e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-1.352e-04, signed_mean=-1.703e-04, mare=3.927e-04, max_step=1.671e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-2.374e-07, mare=2.415e-07, max_step=9.956e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.644e-01, mare_max=4.648e-01, max_step_max=9.077e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.404e-03, mare_max=5.257e-03, max_step_max=2.513e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
