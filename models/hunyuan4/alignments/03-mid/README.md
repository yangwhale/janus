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

- 300 步最近一代 `tm-b` 没有不过（过 5、待定 4，都只略超 N）。`tm-b` base2 开、早于 M50；2026-11-11 起的现行对齐配置（关 base2、含 M50）在 mid 上尚无运行。
- 第 1 步主 loss 对 CPU FP64 的相对差在 GPU 一侧（GPU +1.42e-5，TPU `tm-b` +4.0e-6，R 0.28；[`mid-step1-loss-vs-fp64.json`](results/mid-step1-loss-vs-fp64.json)）。
- 第 1 步逐张量梯度：7 个张量未定责（sink 4 个、iHC 3 个，最大第 3 层 FFN 前 `hc_base` R 15.3），9 个张量 TPU 梯度文件取回损坏（取回时断流留下空洞，见 [`tm2diag-grad-file-holes.json`](results/tm2diag-grad-file-holes.json)），要重跑诊断再比。
- 1000 步、L1 / L2 没做。对齐开关全开比全关贵 +46.8%，M13 约占 60%（运行没有登记、无结果文件，数字未核；见[对齐报告](ALIGNMENT-REPORT.md)）。

适用范围：只在 v5p 8 个设备上、只用 FSDP，随机初始化，300 步（[对齐报告第 5 节](ALIGNMENT-REPORT.md#5-适用范围)）。

判定：[对齐报告](ALIGNMENT-REPORT.md)；逐格：[支线总览](../../dashboard/other-sizes.md)、[检查清单](CHECKS.md)；设置与 GPU golden 的作业规格：[`setup.md`](setup.md)；与 mini 的配置差别：[`config-alignment.md`](config-alignment.md)；运行：[`goldens.md`](goldens.md)、[`runs.yaml`](runs.yaml)。
