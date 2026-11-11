# janus 报告：hunyuan4-trunc / hy4-a-l0-init-vs-gpu

- 被测：`hy4-a-maxtext-gpunames`；golden：`ga-b`, `ga1-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-v5p-acceptance/catalogs/l0-init-vs-gpu-a.yaml`（内容指纹 9475edcc6c8d…）
- 结果：pass 2、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.init.tpu_vs_gpu.ga-b` | checkpoint_equal | L0 | pass | params=248, differ=0, only_one_side=0 | 全部参数逐位相同 |
| `L0.init.tpu_vs_gpu.ga1-a` | checkpoint_equal | L0 | pass | params=248, differ=0, only_one_side=0 | 全部参数逐位相同 |

数字全部来自 `results.json`，本报告不手写数字。
