# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-f16-dp16`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 3、fail 3、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.param_norm`、`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=-8.139e-07, signed_mean=7.692e-06, mare=9.644e-06, max_step=3.419e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=-8.139e-07, signed_mean=1.203e-05, mare=1.261e-05, max_step=3.419e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=5.089e-07, signed_mean=7.598e-07, mare=1.348e-05, max_step=7.294e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-1.555e-05, signed_mean=-9.206e-05, mare=2.466e-04, max_step=1.123e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | fail | d0=-6.035e-04, signed_mean=-2.122e-04, mare=5.003e-04, max_step=1.928e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=-1.131e-07, signed_mean=-2.849e-07, mare=2.871e-07, max_step=1.057e-06 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.211e-01, mare_max=3.599e-01, max_step_max=1.419e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.795e-03, mare_max=5.214e-03, max_step_max=2.798e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
