# janus 报告：hunyuan4-mid / hy4-mid-l0-ckpt-convert

- 被测：`hy4-mid-maxtext-ckpt`；golden：`hy4-mid-hf`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/03-hy4-mid/catalogs/l0-ckpt-convert-mid.yaml`（内容指纹 f34681583dfe…）
- 结果：pass 3、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.ckpt.convert.bitwise` | checkpoint_equal | L0 | pass | params=212, differ=0, only_one_side=0 | 全部参数逐位相同 |
| `L0.ckpt.convert.dtype` | dtype_manifest | L0 | pass | entries=212, wrong_dtype=0, unmatched=0, rule_hits={model.hc_head.*:3 model.layers.*.hc_attn_layer.*:21 model.layers.*.hc_mlp_layer.*:21 model.layers.*.mlp.gate.e_score_correction_bias:6 model.layers.*.self_attn.learnable_sink_param:7 *:154} | 212 项 dtype 全部符合规则 |
| `L0.ckpt.source.dtype` | dtype_manifest | L0 | pass | entries=212, wrong_dtype=0, unmatched=0, rule_hits={model.hc_head.*:3 model.layers.*.hc_attn_layer.*:21 model.layers.*.hc_mlp_layer.*:21 model.layers.*.mlp.gate.e_score_correction_bias:6 model.layers.*.self_attn.learnable_sink_param:7 *:154} | 212 项 dtype 全部符合规则 |

数字全部来自 `results.json`，本报告不手写数字。
