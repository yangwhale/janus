# janus 报告：hunyuan4-rw3 / gpu-scale-rw-vs-single-node

- 被测：`rw-e`；golden：`rws-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/rw-vs-single-node.yaml`（内容指纹 cb56f7c2ce60…）
- 结果：pass 7、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=7.937e-08, signed_mean=1.244e-07, mare=8.981e-06, max_step=2.504e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=3.319e-05, signed_mean=-9.114e-05, mare=1.441e-04, max_step=4.209e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.433e-05, signed_mean=-1.923e-05, mare=3.828e-05, max_step=9.705e-05 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=1.121e-04, signed_mean=8.608e-05, mare=1.499e-04, max_step=5.419e-04 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-9.682e-12, signed_mean=-6.430e-11, mare=6.430e-11, max_step=1.413e-10 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=6.617e-02, mare_max=7.842e-02, max_step_max=1.468e-01, stats=3 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=3.890e-04, mare_max=7.763e-04, max_step_max=3.115e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
