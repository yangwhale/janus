# janus 报告：hunyuan4-trunc / hy4-d1-l0-ckpt-convert

- 被测：`hy4-d1-maxtext-ckpt`；golden：`hy4-d1-hf`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/02-t2-real-weights/catalogs/l0-ckpt-convert-d1.yaml`（内容指纹 bf86e5e98e1d…）
- 结果：pass 3、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.ckpt.convert.bitwise` | checkpoint_equal | L0 | pass | params=58, differ=0, only_one_side=0 | 全部参数逐位相同 |
| `L0.ckpt.convert.dtype` | dtype_manifest | L0 | pass | entries=58, wrong_dtype=0, unmatched=0, rule_hits={model.hc_head.*:3 model.layers.*.hc_attn_layer.*:3 model.layers.*.hc_mlp_layer.*:3 model.layers.*.mlp.gate.e_score_correction_bias:0 model.layers.*.self_attn.learnable_sink_param:1 *:48} | 58 项 dtype 全部符合规则 |
| `L0.ckpt.source.dtype` | dtype_manifest | L0 | pass | entries=58, wrong_dtype=0, unmatched=0, rule_hits={model.hc_head.*:3 model.layers.*.hc_attn_layer.*:3 model.layers.*.hc_mlp_layer.*:3 model.layers.*.mlp.gate.e_score_correction_bias:0 model.layers.*.self_attn.learnable_sink_param:1 *:48} | 58 项 dtype 全部符合规则 |

数字全部来自 `results.json`，本报告不手写数字。
