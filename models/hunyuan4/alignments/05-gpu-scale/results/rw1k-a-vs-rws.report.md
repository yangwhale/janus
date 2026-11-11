# janus 报告：hunyuan4-rw3 / gpu-scale-rw-vs-single-node

- 被测：`rw1k-a`；golden：`rws-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/rw-vs-single-node.yaml`（内容指纹 cb56f7c2ce60…）
- 结果：pass 5、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`C.grad_norm`、`C.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=6.350e-07, signed_mean=-3.635e-07, mare=6.677e-06, max_step=2.360e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=3.991e-05, signed_mean=-8.647e-05, mare=1.425e-04, max_step=4.829e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=1.484e-05, signed_mean=-2.306e-05, mare=4.077e-05, max_step=2.390e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-1.290e-04, signed_mean=1.253e-04, mare=1.728e-04, max_step=6.434e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | inconclusive | d0=-1.731e-11, signed_mean=-1.374e-10, mare=1.374e-10, max_step=1.972e-10 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=7.007e-02, mare_max=7.130e-02, max_step_max=1.651e-01, stats=3 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.157e-04, mare_max=9.812e-04, max_step_max=2.957e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
