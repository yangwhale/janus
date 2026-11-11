# janus 报告：hunyuan4-mini / t1zg-l0-data

- 被测：`t1zg-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zg-l0-data.yaml`（内容指纹 6cce6eb46be1…）
- 结果：pass 1、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.data.fingerprint` | data_fingerprint | L0 | pass | units_per_field={tokens:16000 labels:16000 loss_mask:16000 position_ids:16000}, units=65000, differ=0 | token 与首步 loss 逐位相同 |

数字全部来自 `results.json`，本报告不手写数字。
