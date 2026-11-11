# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`mn-pp2`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 4、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`C.loss`、`C.loss.tail`、`C.grad_norm`、`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=-1.058e-06, signed_mean=-7.590e-07, mare=5.733e-06, max_step=3.541e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | inconclusive | d0=-1.058e-06, signed_mean=-7.048e-07, mare=7.026e-06, max_step=3.541e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=1.120e-06, signed_mean=5.418e-06, mare=1.199e-05, max_step=7.928e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=3.379e-06, signed_mean=2.426e-05, mare=2.241e-04, max_step=9.717e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=2.793e-05, signed_mean=5.756e-05, mare=3.195e-04, max_step=2.165e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=4.991e-09, signed_mean=1.218e-07, mare=1.218e-07, max_step=5.462e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=8.240e-01, mare_max=8.240e-01, max_step_max=9.813e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.305e-03, mare_max=5.337e-03, max_step_max=2.655e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
