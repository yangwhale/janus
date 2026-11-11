# janus 报告：hunyuan4-mini / tpu-self-vs-ts-h-v7b

- 被测：`ts-h-v7-dp8b`；golden：`ts-h-v7b`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-h-v7b.yaml`（内容指纹 3e97e2842d7e…）
- 结果：pass 5、fail 3、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=-1.790e-06, signed_mean=6.634e-06, mare=8.210e-06, max_step=4.039e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=-1.790e-06, signed_mean=9.367e-06, mare=1.060e-05, max_step=4.039e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.425e-06, signed_mean=6.514e-07, mare=1.182e-05, max_step=8.452e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-9.857e-05, signed_mean=-4.833e-05, mare=2.292e-04, max_step=9.180e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | fail | d0=-5.133e-04, signed_mean=-2.193e-04, mare=4.408e-04, max_step=1.740e-03 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | pass | d0=-9.691e-08, signed_mean=-4.104e-08, mare=5.135e-08, max_step=1.681e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.422e-01, mare_max=2.520e-01, max_step_max=5.043e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.322e-03, mare_max=4.301e-03, max_step_max=2.306e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
