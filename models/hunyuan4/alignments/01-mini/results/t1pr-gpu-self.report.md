# janus 报告：hunyuan4-mini / t1pr-gpu-self

- 被测：`p1g-b`；golden：`p1g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1pr-gpu-self.yaml`（内容指纹 59bdee47690e…）
- 结果：pass 5、fail 3、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=3.255e-07, signed_mean_max=1.915e-05, mare_max=2.082e-05, max_step_max=8.130e-05 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | fail | d0_max=3.255e-07, signed_mean_max=2.118e-05, mare_max=2.280e-05, max_step_max=8.130e-05 | 超过噪声参照两倍 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=3.526e-06, mare=3.079e-05, max_step=1.570e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=3.939e-06, mare=3.395e-05, max_step=1.570e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-2.574e-06, signed_mean=1.045e-04, mare=3.281e-04, max_step=1.390e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | fail | d0=7.884e-05, signed_mean=-7.793e-05, mare=6.830e-03, max_step=1.005e-01 | 超过噪声参照两倍 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=-5.475e-06, mare=5.476e-06, max_step=1.434e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=6.499e-06, mare_max=6.543e-06, max_step_max=2.821e-05, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.863e-03, mare_max=5.277e-03, max_step_max=3.141e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
