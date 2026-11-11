# janus 报告：hunyuan4-mini / tpu-self-vs-ts-dp8-ga1

- 被测：`ts-dp8-embf32-ga1`；golden：`ts-dp8-ga1`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-dp8-ga1.yaml`（内容指纹 1b0c60921915…）
- 结果：pass 3、fail 3、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.indexer_loss`、`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.185e-05, mare=1.271e-05, max_step=4.661e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.652e-05, mare=1.690e-05, max_step=4.661e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=7.976e-06, mare=1.413e-05, max_step=1.083e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=9.241e-05, mare=2.814e-04, max_step=1.184e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | fail | d0=1.139e-03, signed_mean=3.358e-04, mare=5.802e-04, max_step=2.244e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=3.038e-07, mare=3.051e-07, max_step=7.476e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.674e-01, mare_max=2.595e-01, max_step_max=6.051e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.103e-03, mare_max=5.843e-03, max_step_max=2.302e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
