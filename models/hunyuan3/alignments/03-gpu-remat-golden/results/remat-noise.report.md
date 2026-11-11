# janus 报告：hunyuan3-4l / remat-noise

- 被测：`rc-a`；golden：`rc-b`, `rc-c`, `rc-d`, `rc-e`
- janus 0.0.1；目录 `/data/rc/cat/remat-noise.yaml`（sha256 db1bcc93a6b8…）
- 结果：pass 6、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.data.fingerprint` | data_fingerprint | L0 | pass | units_per_field={tokens:16000 labels:16000 loss_mask:16000 position_ids:16000}, first_loss={test:11.5 ref:11.5}, units=65000, differ=0 | token 与首步 loss 逐位相同 |
| `DET.replay.vs_first` | bitwise_equal | L2 | pass | tensors=57, differ=0 | 全部逐位相同 |
| `L3.step1.act.bitwise` | bitwise_equal | L3 | pass | tensors=46, differ=0 | 全部逐位相同 |
| `L4.loss` | multi_golden | L4 | pass | d0_max=0.000e+00, signed_mean_max=4.939e-05, mare_max=1.026e-04, max_step_max=4.876e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-6.091e-06, signed_mean=1.474e-03, mare=2.495e-02, max_step=1.548e-01 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=5.569e-08, mare=7.148e-08, max_step=2.161e-07 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=1.513e-03, mare_max=2.648e-03, max_step_max=1.386e-02, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
