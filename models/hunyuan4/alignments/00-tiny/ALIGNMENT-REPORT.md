[janus](../../../../README.md) › [混元 4](../../README.md) › [tiny](README.md) ｜ 对齐报告

# 混元 4 · tiny（hy4-tiny）· 对齐报告

> **一句话**：TPU 一侧在 CPU 上跑（8 个假 JAX 设备），GPU 一侧只在 B200 × 2 上做过 20 步冒烟。hy4-tiny 只用来证明 TPU 一侧的链路通、语义对（CPU 冒烟全部通过、Muon 分区表写出、QK-Clip 扫描层各层共用缩放的 bug 已修），没有做 GPU 对 TPU 的数值比较，所以没有对齐结论。

## 判定

| 级 · 检查 | 判定 | 读数 | 出处 |
|---|---|---|---|
| L0 · CPU 冒烟各步（四个阶段的转换、往返、RoPE 去交错与负对照、训练、初始权重、数据指纹、学习率） | ✅ | 全部 `ok`。数据指纹四项（A–D）在结果文件里 detail 为空：比的是 8 个 rank 的 `data/rank*.jsonl` 与假 golden 逐字节相同（比法只在 [`cpu_smoke.py`](../../frameworks/maxtext/cpu_smoke.py) 里，结果文件没存证） | [`hy4-cpu-smoke.json`](results/hy4-cpu-smoke.json)、[检查清单](CHECKS.md) |
| 语义 · QK-Clip 在扫描层下按层缩放 | 已修（M1） | 修前扫描块里实测缩放等于两层期望的逐头最小值；修后同一脚本复跑（MaxText `b87a55aa`），11 层每层每头与期望逐位相同 | 修前 [`hy4-tiny11-qk-clip-scan.json`](../../results/hy4-tiny11-qk-clip-scan.json)、修后 [`hy4-tiny11-qk-clip-scan-postfix.json`](../../results/hy4-tiny11-qk-clip-scan-postfix.json)、[案例 07](../../cases/07-qk-clip-bugs.md) |
| L1–L3 · T0 两边逐层对 FP64 | ⬜ 没做 | — | 待做 |
| L4 · GPU 对 TPU 多步 | ⬜ 没做 | — | 在 [mini](../01-mini/ALIGNMENT-REPORT.md) 上做 |

## 本尺寸覆盖的根源

本尺寸的检查覆盖的根源（[`ROOT-CAUSES.md`](../../ROOT-CAUSES.md) 第 8 行）：QK-Clip 在扫描层下各层共用缩放（M1）。

## 不能下的结论

hy4-tiny 的宽度、头数、专家数都远小于生产，数值上的结论一律不从这里推。
