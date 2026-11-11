# janus 报告：hunyuan3-4l / phase2-v7x-l0-state

- 被测：`t-v7x-l0`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-b200x8-to-v7x4-4layers/catalogs/phase2-v7x-l0-state.yaml`（sha256 f24cabbb1770…）
- 结果：pass 7、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.weights.init` | checkpoint_equal | L0 | pass | params=1480, differ=0, only_one_side=0 | 全部参数逐位相同 |
| `L0.weights.init.prod` | checkpoint_equal | L0 | pass | params=1480, differ=0, only_one_side=0 | 全部参数逐位相同 |
| `L0.weights.loaded` | bitwise_equal | L0 | pass | tensors=50, differ=0 | 全部逐位相同 |
| `L0.main_params.loaded` | bitwise_equal | L0 | pass | tensors=50, differ=0 | 全部逐位相同 |
| `L0.buffers.loaded` | bitwise_equal | L0 | pass | tensors=3, differ=0 | 全部逐位相同 |
| `L0.dtype.manifest` | dtype_manifest | L0 | pass | entries=2960, wrong_dtype=0, unmatched=0, rule_hits={buffer.*expert_bias:48 * !buffer.*:2912} | 2960 项 dtype 全部符合规则 |
| `L0.model.structure` | artifact_check | L0 | pass | rules=15, bad=0 | 全部规则满足 |

数字全部来自 `results.json`，本报告不手写数字。
