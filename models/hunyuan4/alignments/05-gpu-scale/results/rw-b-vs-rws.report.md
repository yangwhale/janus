# janus 报告：hunyuan4-rw3 / gpu-scale-rw-vs-single-node

- 被测：`rw-b`；golden：`rws-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/rw-vs-single-node.yaml`（内容指纹 cb56f7c2ce60…）
- 结果：pass 6、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | inconclusive | d0=-7.937e-07, signed_mean=4.158e-07, mare=8.668e-06, max_step=3.729e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-8.835e-05, signed_mean=9.094e-06, mare=6.271e-05, max_step=1.303e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-3.752e-06, signed_mean=3.777e-08, mare=3.865e-05, max_step=9.913e-05 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-2.067e-05, signed_mean=7.861e-06, mare=1.628e-04, max_step=4.811e-04 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.390e-12, signed_mean=-1.968e-11, mare=1.999e-11, max_step=4.998e-11 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.687e-02, mare_max=4.824e-02, max_step_max=1.395e-01, stats=3 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=5.063e-04, mare_max=1.121e-03, max_step_max=4.477e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
