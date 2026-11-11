# janus 报告：hunyuan3-4l / phase1-hooks

- 被测：`h-light2`；golden：`d-a`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-hooks.yaml`（sha256 a3c1d0e2ecb9…）
- 结果：pass 2、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.hooks.data` | data_fingerprint | L0 | pass | first_loss={'test': 11.4657564163208, 'ref': 11.4657564163208}, units=1950, differ=0 | token 与首步 loss 逐位相同 |
| `DET.hooks.noop` | determinism | L4 | pass | fields=8, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |

数字全部来自 `results.json`，本报告不手写数字。
