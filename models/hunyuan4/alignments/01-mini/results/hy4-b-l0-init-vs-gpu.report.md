# janus 报告：hunyuan4-trunc / hy4-b-l0-init-vs-gpu

- 被测：`hy4-b-maxtext-gpunames`；golden：`gb-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-v5p-acceptance/catalogs/l0-init-vs-gpu-b.yaml`（内容指纹 670da22f346c…）
- 结果：pass 1、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.init.tpu_vs_gpu.gb-a` | checkpoint_equal | L0 | pass | params=472, differ=0, only_one_side=0 | 全部参数逐位相同 |

数字全部来自 `results.json`，本报告不手写数字。
