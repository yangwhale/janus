# janus 报告：hunyuan3-4l / phase1-obs

- 被测：`g61-obs`；golden：`d-a`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-obs.yaml`（sha256 ea01775758be…）
- 结果：pass 4、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `DET.obs.data` | data_fingerprint | L0 | pass | units_per_field={tokens:32 labels:32 loss_mask:32 position_ids:32}, first_loss={test:11.5 ref:11.5}, units=130, differ=0 | token 与首步 loss 逐位相同 |
| `DET.obs.metrics` | determinism | L4 | pass | fields=3, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
| `DET.obs.step1.bitwise` | bitwise_equal | L3 | pass | tensors=425, differ=0 | 全部逐位相同 |
| `DET.obs.step1.coverage` | coverage | L3 | pass |  | 全部比到 |

数字全部来自 `results.json`，本报告不手写数字。
