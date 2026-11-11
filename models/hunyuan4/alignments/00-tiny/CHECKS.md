[janus](../../../../README.md) › [混元 4](../../README.md) › [tiny](README.md) ｜ 检查清单

# 混元 4 · tiny（hy4-tiny，5.4M）· 检查清单

这一页回答：hy4-tiny 上做过哪些检查、结果如何。页面由 `models/hunyuan4/dashboard/build.py` 按 [`dashboard/checks.yaml`](dashboard/checks.yaml) 从结果文件生成。hy4-tiny 只做了语义检查（CPU），没有 GPU 对 TPU 的数值比较；做了什么、没做什么见 [README](README.md)。

## 规模

| 级 | 目的 | 检查项 | 检查条数 | 张量级比较 | 其他比较单元（hy4-tiny · CPU 冒烟） | hy4-tiny · CPU 冒烟 判定 |
|---|---|---|---|---|---|---|
| [L0](#l0--地基权重数据设置) | 转换与链路逐位 | 3 | 40 | — | 88 个对象、38 条规则 | ✅ 2；只报 1 |
| [L1](#l1l4--gpu-对-tpu) | 两边逐层与多步比较 | 2 | 0 | — | — | — |
| 合计 |  | 5 | 40 | 0 |  |  |

## L0 · 地基：权重、数据、设置

L0 共 3 项检查，检查条数 40。冒烟一行里一条检查是冒烟脚本的一步（`ok` 为真才过）。Muon 分区表（每个参数走 Muon 还是 Adam）是写出的表、不判，见 [README](README.md)。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | hy4-tiny · CPU 冒烟 |
|---|---|---|---|---|---|
| CPU 冒烟各步<br>`T0.smoke.make_tiny_hf`<br>`T0.smoke.make_fake_golden`<br>`T0.smoke.names_shapes`<br>`T0.smoke.A.convert`<br>… | hy4-tiny 维度、8 个假 CPU 设备，四个阶段（A–D）各走一遍：生成 HF 目录、转换、往返逐位、RoPE 去交错与负对照、训练、初始权重等于 HF、索引器冻结、数据指纹（8 个 rank 的 data/rank*.jsonl 与假 golden 逐字节相同；结果文件里 detail 为空，比法见 cpu_smoke.py）、学习率常数 | 冒烟脚本的每一步<br>比较单元：38 条规则<br>聚合：逐条规则核，数不满足的条数 | 逐位｜参考：无 | TPU 一侧 CPU 冒烟（`frameworks/maxtext/cpu_smoke.py`） | [38 / 38 条 ok ✅](results/hy4-cpu-smoke.json) |
| QK-Clip 在扫描层下按层缩放（修后，M1）<br>`T0.qkclip.scan_per_layer` | hy4-tiny 的 11 层变体 hy4-tiny11（1 稠密 + 10 MoE，扫描长度 2，另有 2 层余数块），τ 1e-3 下每层每头实测的 √scale 与按该层最大 logit 算的期望逐位比（`qk_clip_scan_check.py`，台账 J1；覆盖根源一览第 8 行） | 7 个 wq_b 叶子、11 层 × 4 头<br>比较单元：44 个对象<br>聚合：逐条规则核，数不满足的条数 | 逐位｜参考：无 | TPU 一侧 CPU（`frameworks/maxtext/qk_clip_scan_check.py`） | [0 / 44 不同 ✅](../../results/hy4-tiny11-qk-clip-scan-postfix.json) |
| QK-Clip 在扫描层下按层缩放（修前）（只报）<br>`T0.qkclip.scan_per_layer` | 同一脚本、同一参数在修复 M1 之前的读数：扫描块里两层共用一个缩放（等于两层期望的逐头最小值） | 7 个 wq_b 叶子、11 层 × 4 头<br>比较单元：44 个对象<br>聚合：逐条规则核，数不满足的条数 | 逐位｜参考：无 | TPU 一侧 CPU（`frameworks/maxtext/qk_clip_scan_check.py`） | [16 / 44 不同 ❌](../../results/hy4-tiny11-qk-clip-scan.json) |

非 ✅ 格子的原因：

- **QK-Clip 在扫描层下按层缩放（修前）**（hy4-tiny · CPU 冒烟 ❌）：修前的 bug 读数，留作对照；修后见上一行

## L1–L4 · GPU 对 TPU

hy4-tiny 上没有做 GPU 对 TPU 的比较。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | hy4-tiny · CPU 冒烟 |
|---|---|---|---|---|---|
| T0：两边逐层对 FP64<br> | hy4-tiny 两边逐层前向 / 反向、第 1 步更新对 FP64 的 R | 7 层 + MTP | — | — | ⬜ 未开始（待做） |
| 多步比较<br> | GPU 对 TPU 的逐步曲线 | — | — | — | ⬜ 未开始（没有做，L4 在 hy4-mini 上做） |

## 未列入本页的结果文件

| 结果 | 是什么、为什么不在清单里 | 判定 |
|---|---|---|
| [`hy4-cpu-smoke`](results/hy4-cpu-smoke.json) | 同一份冒烟结果里「不去交错的检查点必须判不一致」那一条负对照，列在 hy4-mini 检查清单 L0 的负对照一行的「真实权重」一列；那一格用的是 hy4-tiny 的合成权重（阶段 A 的结构），不是真实权重 |  |
