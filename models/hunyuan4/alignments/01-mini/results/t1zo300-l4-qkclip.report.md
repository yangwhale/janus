# janus 报告：hunyuan4-mini / t1zo300-l4-qkclip

- 被测：`tz-off-qkq`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1zo300-l4-qkclip.yaml`（内容指纹 b27827158141…）
- 结果：pass 0、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.qkclip.max_logit`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.qkclip.max_logit` | loss_series | L4 | inconclusive | d0=1.274e-03, signed_mean=-2.018e-04, mare=2.048e-03, max_step=4.540e-02 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
