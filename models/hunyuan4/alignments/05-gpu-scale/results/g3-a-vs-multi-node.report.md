# janus 报告：hunyuan4-mini / gpu-scale-multi-node-self

- 被测：`g3-a`；golden：`mn-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/multi-node-self.yaml`（内容指纹 634cd82d964b…）
- 结果：pass 6、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`C.indexer_loss`、`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=1.302e-06, signed_mean=3.209e-06, mare=6.457e-06, max_step=3.547e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=1.302e-06, signed_mean=5.065e-06, mare=8.215e-06, max_step=3.547e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-5.089e-07, signed_mean=-2.037e-06, mare=1.402e-05, max_step=7.774e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=5.899e-06, signed_mean=-8.576e-05, mare=2.242e-04, max_step=9.046e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.319e-06, signed_mean=3.207e-05, mare=3.391e-04, max_step=1.425e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=3.013e-09, signed_mean=-1.728e-07, mare=1.770e-07, max_step=7.679e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.121e-01, mare_max=2.322e-01, max_step_max=6.619e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.269e-03, mare_max=6.204e-03, max_step_max=2.828e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
