# janus 报告：hunyuan3-4l / phase1-inject

- 被测：`i-weight`；golden：`d-a`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-inject.yaml`（sha256 705b047e9600…）
- 结果：pass 3、fail 2、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.seam.inputs` | bitwise_equal | L2 | pass | tensors=4, differ=0 | 全部逐位相同 |
| `L2.inject.layer0` | bitwise_equal | L2 | pass | tensors=10, differ=0 | 全部逐位相同 |
| `L2.inject.layer1` | bitwise_equal | L2 | pass | tensors=10, differ=0 | 全部逐位相同 |
| `L2.inject.layer2` | bitwise_equal | L2 | fail | tensors=10, differ=6 | 6 / 10 个张量不同 |
| `L2.inject.layer3` | bitwise_equal | L2 | fail | tensors=10, differ=1 | 1 / 10 个张量不同 |

数字全部来自 `results.json`，本报告不手写数字。
