# janus 报告：hunyuan4-rw3 / gpu-scale-rw-vs-single-node

- 被测：`rw-a`；golden：`rws-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/rw-vs-single-node.yaml`（内容指纹 cb56f7c2ce60…）
- 结果：pass 7、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=1.270e-06, signed_mean=-3.013e-06, mare=8.184e-06, max_step=2.540e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-1.136e-05, signed_mean=-5.332e-05, mare=1.352e-04, max_step=3.707e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-2.567e-05, signed_mean=-7.573e-06, mare=4.886e-05, max_step=1.388e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-1.164e-04, signed_mean=7.127e-05, mare=1.365e-04, max_step=5.545e-04 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.407e-11, signed_mean=-8.227e-12, mare=1.218e-11, max_step=2.741e-11 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.002e-02, mare_max=5.576e-02, max_step_max=1.531e-01, stats=3 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.300e-04, mare_max=9.315e-04, max_step_max=3.103e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
