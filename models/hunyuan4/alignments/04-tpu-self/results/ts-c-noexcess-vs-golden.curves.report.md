# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-c-noexcess`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 3、fail 4、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.indexer_loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=6.511e-07, signed_mean=3.300e-05, mare=3.474e-05, max_step=9.469e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=6.511e-07, signed_mean=4.982e-05, mare=4.982e-05, max_step=9.469e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.832e-06, signed_mean=6.269e-06, mare=1.615e-05, max_step=1.126e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=8.431e-05, signed_mean=-1.662e-04, mare=3.204e-04, max_step=1.427e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | fail | d0=-5.452e-03, signed_mean=-1.318e-03, mare=1.498e-03, max_step=6.937e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-8.202e-07, mare=8.219e-07, max_step=2.824e-06 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=5.889e-01, mare_max=5.893e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.965e-03, mare_max=5.119e-03, max_step_max=1.967e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
