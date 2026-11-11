# janus 报告：hunyuan3-4l / phase2-resume

- 被测：`r-res20`；golden：`r-cont20`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-b200x8-to-v5p8-4layers/catalogs/phase2-resume.yaml`（sha256 521369b6ab29…）
- 结果：pass 5、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.resume.data` | data_fingerprint | L0 | pass | units_per_field={tokens:160 labels:160 loss_mask:160 position_ids:160}, units=650, differ=0 | token 与首步 loss 逐位相同 |
| `L4.resume.overlap` | overlap_replay | L4 | pass | overlap_steps=10, first_differ_step=None, first_differ_fields=[] | 重叠步全部逐位相同 |
| `L3.resume.state` | checkpoint_equal | L3 | pass | params=91, differ=0, only_one_side=0 | 全部参数逐位相同 |
| `L3.resume.step11` | bitwise_equal | L3 | pass | tensors=256, differ=0 | 全部逐位相同 |
| `L3.resume.coverage` | coverage | L3 | pass |  | 全部比到 |

数字全部来自 `results.json`，本报告不手写数字。
