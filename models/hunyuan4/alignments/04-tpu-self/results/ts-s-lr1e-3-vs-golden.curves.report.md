# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-s-lr1e-3`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 2、fail 3、inconclusive 3、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.mtp_loss`、`C.indexer_loss`、`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.392e-04, mare=1.392e-04, max_step=2.578e-04 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.823e-04, mare=1.823e-04, max_step=2.578e-04 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-2.970e-05, mare=2.997e-05, max_step=1.430e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-2.113e-04, mare=3.031e-04, max_step=1.150e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-1.397e-04, mare=4.082e-04, max_step=1.848e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=4.655e-06, mare=4.656e-06, max_step=1.377e-05 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.005e-01, mare_max=3.097e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.852e-03, mare_max=4.948e-03, max_step_max=2.513e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
