# janus 报告：hunyuan3-4l / phase1-truncate

- 被测：`t-2l`；golden：`d-a`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-truncate.yaml`（sha256 b6199984e5c1…）
- 结果：pass 4、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.truncate.embed` | bitwise_equal | L2 | pass | tensors=1, differ=0 | 全部逐位相同 |
| `L2.truncate.layer0` | bitwise_equal | L2 | pass | tensors=10, differ=0 | 全部逐位相同 |
| `L2.truncate.layer1` | bitwise_equal | L2 | pass | tensors=10, differ=0 | 全部逐位相同 |
| `L2.truncate.router1` | bitwise_equal | L2 | pass | tensors=3, differ=0 | 全部逐位相同 |

数字全部来自 `results.json`，本报告不手写数字。
