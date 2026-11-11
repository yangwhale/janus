# janus 报告：hunyuan3-4l / remat-vs-g1k

- 被测：`rc-a`；golden：`g1k-a`, `g1k-b`, `g1k-c`
- janus 0.0.1；目录 `/data/rc/cat/remat-vs-g1k.yaml`（sha256 ceaa7e0d0423…）
- 结果：pass 3、fail 1、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.grad_norm`、`L4.param_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.data.fingerprint` | data_fingerprint | L0 | pass | units_per_field={tokens:16000 labels:16000 loss_mask:16000 position_ids:16000}, units=65000, differ=0 | token 与首步 loss 逐位相同 |
| `L3.step1.act.bitwise` | bitwise_equal | L3 | fail | tensors=46, differ=26 | 26 / 46 个张量不同 |
| `L4.loss` | multi_golden | L4 | pass | d0_max=7.486e-07, signed_mean_max=2.790e-05, mare_max=1.047e-04, max_step_max=5.262e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=5.656e-06, signed_mean=1.164e-04, mare=2.402e-02, max_step=5.662e-01 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=1.208e-07, mare=1.272e-07, max_step=3.241e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.475e-03, mare_max=2.702e-03, max_step_max=1.083e-02, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
