# janus 报告：hunyuan4-mini / t1mz300-l4-qkclip

- 被测：`mz-g1q`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1mz300-l4-qkclip.yaml`（内容指纹 aaff965fe7dd…）
- 结果：pass 0、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.qkclip.max_logit`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.qkclip.max_logit` | loss_series | L4 | inconclusive | d0=1.662e-03, signed_mean=-5.060e-04, mare=1.907e-03, max_step=4.272e-02 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
