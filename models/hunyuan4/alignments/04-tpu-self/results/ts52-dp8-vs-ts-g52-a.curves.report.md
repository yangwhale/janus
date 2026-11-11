# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-dp8`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 5、fail 2、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=-9.766e-07, signed_mean=7.692e-06, mare=1.014e-05, max_step=4.135e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=-9.766e-07, signed_mean=1.278e-05, mare=1.335e-05, max_step=4.135e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=9.160e-07, signed_mean=-4.217e-07, mare=1.228e-05, max_step=8.715e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-5.985e-05, signed_mean=-4.407e-05, mare=1.864e-04, max_step=7.395e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-3.985e-04, signed_mean=-1.441e-04, mare=4.156e-04, max_step=1.857e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-8.801e-08, mare=1.269e-07, max_step=6.859e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.008e-01, mare_max=5.087e-01, max_step_max=7.009e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.711e-03, mare_max=5.587e-03, max_step_max=2.377e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
