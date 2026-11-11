# janus 报告：hunyuan3-4l / phase1-resume

- 被测：`r-resume`；golden：`r-cont`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-resume.yaml`（sha256 6b2775414f60…）
- 结果：pass 4、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.resume.data` | data_fingerprint | L0 | pass | units=650, differ=0 | token 与首步 loss 逐位相同 |
| `L4.resume.overlap` | overlap_replay | L4 | pass | overlap_steps=10, first_differ_step=None, first_differ_fields=[] | 重叠步全部逐位相同 |
| `L3.resume.step11` | bitwise_equal | L3 | pass | tensors=74, differ=0 | 全部逐位相同 |
| `L3.resume.coverage` | coverage | L3 | pass |  | 全部比到 |

数字全部来自 `results.json`，本报告不手写数字。
