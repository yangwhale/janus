# janus 报告：hunyuan3-4l / phase1-opt8

- 被测：`m-opt8`；golden：`d-a`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-opt8.yaml`（sha256 3aba42519088…）
- 结果：pass 1、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L1.optim.adamw.8steps` | optimizer_step | L1 | pass | rel_l2_max=3.372e-03, max_abs_max=1.122e-07, tensors=42, lr=1.000e-05 | 连续 8 步重放：全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
