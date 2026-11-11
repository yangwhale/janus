# janus 报告：hunyuan4-mini / t1zk-l0-settings

- 被测：`t1zk-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-v5p-acceptance/catalogs/t1zk-l0-settings.yaml`（内容指纹 e6fd09ed11aa…）
- 结果：pass 1、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.settings.lr` | scalar_series | L0 | pass | steps=1000, bad_steps=0, first_bad=None, equals=1.000e-05 | 两边每一步都等于登记值 1e-05 |

数字全部来自 `results.json`，本报告不手写数字。
