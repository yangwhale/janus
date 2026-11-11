# janus 报告：hunyuan4-rw3 / gpu-scale-rw-vs-single-node

- 被测：`rw-d`；golden：`rws-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/rw-vs-single-node.yaml`（内容指纹 cb56f7c2ce60…）
- 结果：pass 7、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.587e-07, signed_mean=-1.325e-07, mare=7.978e-06, max_step=2.491e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=3.539e-05, signed_mean=-8.741e-05, mare=1.473e-04, max_step=4.143e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-1.902e-05, signed_mean=-2.325e-06, mare=3.450e-05, max_step=1.009e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-6.531e-05, signed_mean=1.675e-05, mare=1.306e-04, max_step=4.140e-04 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.948e-11, signed_mean=-8.775e-11, mare=8.775e-11, max_step=1.295e-10 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=3.681e-02, mare_max=4.463e-02, max_step_max=9.396e-02, stats=3 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=6.632e-04, mare_max=7.829e-04, max_step_max=3.649e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
