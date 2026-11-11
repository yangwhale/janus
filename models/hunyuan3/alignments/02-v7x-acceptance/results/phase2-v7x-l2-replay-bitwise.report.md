# janus 报告：hunyuan3-4l / phase2-v7x-l2-replay-bitwise

- 被测：`r-v7x-l2`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-b200x8-to-v7x4-4layers/catalogs/phase2-v7x-l2-replay-bitwise.yaml`（sha256 711ca42cf51e…）
- 结果：pass 0、fail 1、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L3.step1.act.bitwise` | bitwise_equal | L3 | fail | tensors=41, differ=41 | 41 / 41 个张量不同 |

数字全部来自 `results.json`，本报告不手写数字。
