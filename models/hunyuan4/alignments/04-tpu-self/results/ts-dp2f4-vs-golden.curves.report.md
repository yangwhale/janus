# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-dp2f4`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 7、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=5.697e-07, signed_mean=-4.506e-08, mare=5.069e-06, max_step=2.142e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=5.697e-07, signed_mean=2.710e-07, mare=5.817e-06, max_step=2.142e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=3.053e-07, signed_mean=-7.376e-06, mare=1.334e-05, max_step=7.501e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.909e-05, signed_mean=-5.574e-06, mare=2.059e-04, max_step=7.869e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.185e-04, signed_mean=-4.473e-05, mare=3.759e-04, max_step=1.675e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-3.178e-08, signed_mean=-6.811e-08, mare=8.616e-08, max_step=4.492e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.017e-01, mare_max=4.644e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=3.173e-03, mare_max=5.647e-03, max_step_max=2.513e-02, stats=7 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
