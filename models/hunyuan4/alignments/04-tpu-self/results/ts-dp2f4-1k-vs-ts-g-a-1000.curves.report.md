# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-dp2f4-1k`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 8、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=5.697e-07, signed_mean=4.277e-06, mare=8.577e-06, max_step=6.090e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=5.697e-07, signed_mean=4.823e-06, mare=9.130e-06, max_step=6.090e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=3.053e-07, signed_mean=-4.769e-06, mare=3.210e-05, max_step=1.839e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=3.053e-07, signed_mean=-4.955e-06, mare=3.517e-05, max_step=1.839e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.909e-05, signed_mean=-1.406e-04, mare=3.049e-04, max_step=1.376e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-1.185e-04, signed_mean=6.849e-04, mare=3.481e-03, max_step=8.211e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-3.178e-08, signed_mean=-2.818e-06, mare=2.824e-06, max_step=2.123e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.880e-01, mare_max=5.068e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.590e-03, mare_max=4.923e-03, max_step_max=2.513e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
