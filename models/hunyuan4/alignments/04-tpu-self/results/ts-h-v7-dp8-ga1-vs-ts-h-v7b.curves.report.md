# janus 报告：hunyuan4-mini / tpu-self-vs-ts-h-v7b

- 被测：`ts-h-v7-dp8-ga1`；golden：`ts-h-v7b`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-h-v7b.yaml`（内容指纹 3e97e2842d7e…）
- 结果：pass 4、fail 3、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=-8.139e-07, signed_mean=1.223e-05, mare=1.302e-05, max_step=5.340e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=-8.139e-07, signed_mean=1.755e-05, mare=1.771e-05, max_step=5.340e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=1.018e-07, signed_mean=-1.032e-06, mare=1.202e-05, max_step=7.526e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-5.706e-05, signed_mean=-7.423e-05, mare=2.213e-04, max_step=7.744e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | fail | d0=-1.231e-03, signed_mean=-3.415e-04, mare=5.856e-04, max_step=2.106e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=-9.691e-08, signed_mean=-3.166e-07, mare=3.172e-07, max_step=9.216e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.183e-01, mare_max=2.252e-01, max_step_max=3.026e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.946e-03, mare_max=5.258e-03, max_step_max=2.180e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
