# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-pc-logits`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 3、fail 2、inconclusive 3、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.loss.tail`、`C.indexer_loss`、`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=8.058e-06, mare=2.297e-05, max_step=2.692e-04 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=7.469e-06, mare=1.021e-05, max_step=3.563e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-7.193e-06, mare=1.567e-05, max_step=7.617e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-3.486e-05, mare=2.532e-04, max_step=1.134e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=2.369e-05, signed_mean=5.917e-05, mare=4.831e-04, max_step=2.781e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-4.875e-07, mare=4.891e-07, max_step=1.280e-06 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.925e-01, mare_max=4.520e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.864e-03, mare_max=5.368e-03, max_step_max=2.585e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
