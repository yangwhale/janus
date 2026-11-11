# janus 报告：hunyuan4-rw3 / gpu-scale-rw-vs-single-node

- 被测：`rw-c`；golden：`rws-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/rw-vs-single-node.yaml`（内容指纹 cb56f7c2ce60…）
- 结果：pass 6、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`C.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-4.286e-06, signed_mean=-1.279e-06, mare=7.743e-06, max_step=2.612e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-9.512e-05, signed_mean=2.611e-06, mare=6.778e-05, max_step=2.493e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=9.721e-06, signed_mean=-1.357e-05, mare=4.286e-05, max_step=1.112e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | inconclusive | d0=-3.179e-05, signed_mean=6.040e-05, mare=1.797e-04, max_step=6.651e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `C.param_norm` | loss_series | L4 | pass | d0=-1.208e-11, signed_mean=-3.708e-11, mare=3.708e-11, max_step=6.737e-11 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.715e-02, mare_max=6.287e-02, max_step_max=1.619e-01, stats=3 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=5.602e-04, mare_max=1.023e-03, max_step_max=3.970e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
