# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-d-dotattn-r2`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.465e-06, signed_mean=2.286e-06, mare=4.875e-06, max_step=2.513e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.465e-06, signed_mean=3.422e-06, mare=6.080e-06, max_step=2.513e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=1.832e-06, signed_mean=-7.029e-06, mare=1.192e-05, max_step=9.032e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-1.341e-05, signed_mean=1.326e-05, mare=1.684e-04, max_step=7.900e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=1.904e-04, signed_mean=1.405e-04, mare=2.987e-04, max_step=2.044e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.330e-07, mare=1.362e-07, max_step=4.850e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.051e-01, mare_max=1.537e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=7.136e-04, mare_max=4.676e-03, max_step_max=2.321e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
