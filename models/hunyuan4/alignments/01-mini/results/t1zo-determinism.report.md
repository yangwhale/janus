# janus 报告：hunyuan4-mini / t1zo-determinism

- 被测：`tz-off-qkc`；golden：`tz-off-rep`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zo-determinism.yaml`（内容指纹 b507f4988843…）
- 结果：pass 1、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `DET.tpu.rerun.current` | determinism | L4 | pass | fields=47, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |

数字全部来自 `results.json`，本报告不手写数字。
