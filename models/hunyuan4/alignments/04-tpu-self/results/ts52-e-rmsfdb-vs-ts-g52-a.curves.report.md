# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-e-rmsfdb`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 3、fail 2、inconclusive 3、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.indexer_loss`、`C.grad_norm`、`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=3.255e-07, signed_mean=2.098e-05, mare=2.106e-05, max_step=4.997e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=3.255e-07, signed_mean=2.756e-05, mare=2.756e-05, max_step=4.997e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=3.053e-07, signed_mean=3.714e-06, mare=1.079e-05, max_step=8.005e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=-3.218e-06, signed_mean=-1.939e-04, mare=2.527e-04, max_step=8.411e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-1.213e-04, signed_mean=-1.256e-04, mare=3.525e-04, max_step=1.411e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-2.061e-07, mare=2.073e-07, max_step=6.490e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=7.616e-01, mare_max=7.616e-01, max_step_max=8.411e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.098e-03, mare_max=4.896e-03, max_step_max=2.105e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
