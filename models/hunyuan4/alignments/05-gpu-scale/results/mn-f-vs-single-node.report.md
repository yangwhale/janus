# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`mn-f`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 4、fail 0、inconclusive 4、error 0

**证据不足（inconclusive）**：`C.loss`、`C.loss.tail`、`C.grad_norm`、`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=-1.872e-06, signed_mean=1.324e-06, mare=5.799e-06, max_step=3.234e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.loss.tail` | loss_series | L4 | inconclusive | d0=-1.872e-06, signed_mean=1.475e-06, mare=7.172e-06, max_step=3.234e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=5.089e-07, signed_mean=-2.389e-06, mare=1.055e-05, max_step=6.046e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-7.615e-06, signed_mean=7.264e-05, mare=1.930e-04, max_step=9.510e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=7.442e-06, signed_mean=8.593e-05, mare=3.369e-04, max_step=2.198e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=4.776e-10, signed_mean=4.984e-08, mare=6.139e-08, max_step=5.100e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.191e-01, mare_max=2.191e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.117e-03, mare_max=4.240e-03, max_step_max=2.440e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
