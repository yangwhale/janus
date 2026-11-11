# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`mn-c`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 6、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`C.indexer_loss`、`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-2.197e-06, signed_mean=-1.335e-06, mare=5.716e-06, max_step=2.469e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-2.197e-06, signed_mean=-2.580e-06, mare=6.889e-06, max_step=2.469e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=8.274e-06, mare=1.551e-05, max_step=9.845e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | inconclusive | d0=1.351e-05, signed_mean=6.444e-05, mare=2.123e-04, max_step=1.282e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.grad_norm` | loss_series | L4 | pass | d0=1.295e-05, signed_mean=5.543e-05, mare=3.029e-04, max_step=1.583e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=5.287e-10, signed_mean=1.749e-07, mare=1.811e-07, max_step=6.943e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.107e-01, mare_max=4.432e-01, max_step_max=7.009e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.168e-03, mare_max=5.204e-03, max_step_max=2.016e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
