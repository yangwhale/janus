# janus 报告：hunyuan3-4l / phase2-ckpt-convert

- 被测：`hy3-4l-maxtext-ckpt`；golden：`hy3-4l-hf`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase2-ckpt-convert.yaml`（sha256 359cd74f42b3…）
- 结果：pass 3、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.ckpt.convert.bitwise` | checkpoint_equal | L0 | pass | params=1781, differ=0, only_one_side=0 | 全部参数逐位相同 |
| `L0.ckpt.convert.dtype` | dtype_manifest | L0 | pass | entries=1781, wrong_dtype=0, unmatched=0, rule_hits={*.expert_bias:3 * !*.expert_bias:1778} | 1781 项 dtype 全部符合规则 |
| `L0.ckpt.source.dtype` | dtype_manifest | L0 | pass | entries=1781, wrong_dtype=0, unmatched=0, rule_hits={*.expert_bias:3 * !*.expert_bias:1778} | 1781 项 dtype 全部符合规则 |

数字全部来自 `results.json`，本报告不手写数字。
