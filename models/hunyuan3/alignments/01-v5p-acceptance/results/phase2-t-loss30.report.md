# janus 报告：hunyuan3-4l / phase2-t-loss30

- 被测：`t-loss30`；golden：`g1k-a`, `g1k-b`, `g1k-c`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-t-loss30.yaml`（内容指纹 137ae5e007f4…）
- 结果：pass 3、fail 1、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.config.diff.dropout` | config_diff | L0 | pass | fields=6, must_equal_differ=0, other_differ=0, missing=0 | 必须相同的字段都相同；其他不同 0 项 |
| `L0.data.fingerprint` | data_fingerprint | L0 | pass | units_per_field={tokens:480 labels:480 loss_mask:480 position_ids:480}, units=1950, differ=0 | token 与首步 loss 逐位相同 |
| `L4.loss` | multi_golden | L4 | fail | d0_max=2.495e-06, signed_mean_max=1.011e-05, mare_max=2.188e-05, max_step_max=6.359e-05 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=3.094e-04, signed_mean=1.658e-04, mare=4.794e-04, max_step=2.047e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.lr` | scalar_series | L4 | pass | steps=30, bad_steps=0, first_bad=None | 逐步在容差内 |

数字全部来自 `results.json`，本报告不手写数字。
