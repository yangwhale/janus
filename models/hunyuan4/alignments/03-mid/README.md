[janus](../../../../README.md) › [混元 4](../../README.md) ｜ 03 · mid（支线）

# 混元 4 · 03 · mid：hy4-mid（5.19B，随机初始化；v5p 2x2x2 ↔ B200 × 8）

支线：主线结论在 [mini](../01-mini/README.md)。mini（随机初始化，4.3 亿参数）到真实权重截层（约 121 亿参数）中间同时换了尺寸和权重；hy4-mid 只换尺寸，把 mini 的结构放到生产的宽度与注意力，设置、检查、判法与 mini 相同。

| 项 | hy4-mini | hy4-mid | 生产（hy4-preview） |
|---|---|---|---|
| 宽 / 头数 / q_lora | 1024 / 8 / 512 | 6144 / 64 / 2048 | 6144 / 64 / 2048 |
| 稠密 FFN | 3072 | 18432 | 18432 |
| 路由专家 | 16 个（top-4）宽 256 | 16 个（top-4）宽 512 | 256 个（top-8）宽 2048 |
| 层 | 1 稠密 + 6 MoE + 1 MTP | 同 hy4-mini | — |
| 参数量（含 MTP） | 427,783,489 | 5,189,678,849 | — |

## 现状

- 300 步现行 C 口径 `tmc-a`（按 PROTOCOL 2026-11-12 的前瞻登记判）没有不过（过 5、待定 4，都只略超 N：主 loss 全程 r 1.01、MTP loss r 1.25 / 1.15、索引器 KD r 1.003）。上一代 `tm-b`（base2 开、早于 M50）也是过 5、待定 4，超线的是主 loss（r 1.38）与 param_norm（r 1.79）。
- 第 1 步主 loss 对 CPU FP64 的相对差在 GPU 一侧（GPU +1.42e-5，TPU C 口径 −2.67e-6，R 0.19；上一代 `tm-b` +4.0e-6，[`mid-step1-loss-vs-fp64.json`](results/mid-step1-loss-vs-fp64.json)）。
- 第 1 步逐张量梯度（C 口径 `tmcs-c`，训练步梯度；GPU 参照只有 3 份，参照不足、待补到 6 份）：不过，5 个未定责（iHC 4 个，最大第 3 层 FFN 前 `hc_base` R 13.7；第 4 层 sink R 1.21），落盘核对全对（[`mid-c-step1-train-grad-vs-fp64.json`](results/mid-c-step1-train-grad-vs-fp64.json)）。上一代 `tm2diag` 是 7 个未定责、9 个文件损坏。
- 1000 步、L1 / L2 没做。对齐开关全开比全关贵 +46.8%，M13 约占 60%（运行没有登记、无结果文件，数字未核；见[对齐报告](ALIGNMENT-REPORT.md)）。

适用范围：只在 v5p 8 个设备上、只用 FSDP，随机初始化，300 步（[对齐报告第 5 节](ALIGNMENT-REPORT.md#5-适用范围)）。

判定：[对齐报告](ALIGNMENT-REPORT.md)；逐格：[支线总览](../../dashboard/other-sizes.md)、[检查清单](CHECKS.md)；设置与 GPU golden 的作业规格：[`setup.md`](setup.md)；与 mini 的配置差别：[`config-alignment.md`](config-alignment.md)；运行：[`goldens.md`](goldens.md)、[`runs.yaml`](runs.yaml)。
