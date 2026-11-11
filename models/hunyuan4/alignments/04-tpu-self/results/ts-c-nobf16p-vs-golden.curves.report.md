# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-c-nobf16p`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 6、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`C.loss`、`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=-5.697e-07, signed_mean=-7.983e-07, mare=7.541e-06, max_step=2.487e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-5.697e-07, signed_mean=-6.862e-07, mare=7.451e-06, max_step=2.487e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.527e-06, signed_mean=5.721e-06, mare=1.348e-05, max_step=9.419e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=5.449e-05, signed_mean=3.548e-05, mare=2.232e-04, max_step=1.133e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-1.927e-05, signed_mean=-3.093e-05, mare=3.941e-04, max_step=1.476e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=8.022e-09, mare=2.929e-08, max_step=9.986e-08 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.053e-01, mare_max=3.155e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.761e-03, mare_max=5.251e-03, max_step_max=2.361e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
