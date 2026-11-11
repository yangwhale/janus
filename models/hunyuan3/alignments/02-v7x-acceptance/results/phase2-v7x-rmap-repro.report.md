# janus 报告：hunyuan3-4l / phase2-v7x-rmap-repro

- 被测：`t-v7x-rmap-1000-b`；golden：`t-v7x-rmap-1000-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-b200x8-to-v7x4-4layers/catalogs/phase2-v7x-rmap-repro.yaml`（sha256 300780fa815e…）
- 结果：pass 4、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.repro.lm_loss` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |
| `L4.repro.grad_norm` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |
| `L4.repro.param_norm` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |
| `L0.repro.data.fingerprint` | data_fingerprint | L0 | pass | units_per_field={tokens:16000 labels:16000 loss_mask:16000 position_ids:16000}, units=65000, differ=0 | token 与首步 loss 逐位相同 |

数字全部来自 `results.json`，本报告不手写数字。
