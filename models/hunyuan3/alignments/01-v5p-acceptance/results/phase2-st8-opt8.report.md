# janus 报告：hunyuan3-4l / phase2-st8-opt8

- 被测：`t-st8-prod-gm`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-b200x8-to-v5p8-4layers/catalogs/phase2-st8-opt8.yaml`（sha256 13d8a74b45ae…）
- 结果：pass 1、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L1.optim.adamw.8steps` | optimizer_step | L1 | pass | rel_l2_max=4.796e-03, max_abs_max=1.180e-07, tensors=66, lr=1.000e-05 | 连续 8 步重放：全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
