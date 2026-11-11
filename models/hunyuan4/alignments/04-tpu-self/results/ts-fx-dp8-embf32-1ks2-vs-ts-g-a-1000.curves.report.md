# janus 报告：hunyuan4-mini / tpu-self-vs-golden-1000

- 被测：`ts-fx-dp8-embf32-1ks2`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden-1000.yaml`（内容指纹 fe6301013c83…）
- 结果：pass 8、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=1.533e-06, mare=7.996e-06, max_step=5.375e-05 | 在噪声参照内 |
| `L4.loss.tail` | loss_series | L4 | pass | d0=7.325e-07, signed_mean=1.832e-06, mare=8.442e-06, max_step=5.375e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=3.241e-06, mare=3.061e-05, max_step=1.650e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=3.874e-06, mare=3.358e-05, max_step=1.650e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=1.738e-05, signed_mean=-4.699e-05, mare=3.033e-04, max_step=1.457e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-1.116e-04, signed_mean=6.695e-04, mare=2.971e-03, max_step=6.822e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.131e-07, signed_mean=-1.208e-06, mare=1.637e-06, max_step=1.959e-05 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.489e-01, mare_max=2.709e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.151e-03, mare_max=4.919e-03, max_step_max=2.466e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
