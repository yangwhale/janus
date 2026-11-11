[janus](../../../../README.md) › [混元 4](../../README.md) › [mid](README.md) ｜ 对齐报告（支线）

# 混元 4 · mid（hy4-mid，5.19B，随机初始化）· 对齐报告

支线报告：只写与 [mini 主线报告](../01-mini/ALIGNMENT-REPORT.md)不同的地方；判法、开关与生产建议同 mini，N（噪声上限）、r、R、MARE 的读法见[总览细节「读法」](../../dashboard/detail.md#读法)。训练运行的代号（`tm-b`、`gm-a` 等）见 [`goldens.md`](goldens.md) 与[历代比较](../../dashboard/history.md)，第 1 步诊断运行（`tm2diag`）见[检查清单](CHECKS.md) L3 一行。hy4-mid 只换尺寸：mini 的结构放到生产的宽度与注意力（宽 6144、64 头、q_lora 2048、稠密 FFN 18432），路由专家仍 16 个 top-4；300 步，GPU B200 × 8 对 TPU v5p 2x2x2。

> **一句话**：300 步没有不过（过 5、待定 4，都只略超 N）；第 1 步主 loss 的差在 GPU 一侧；第 1 步逐张量梯度有 7 个张量未定责（sink 4 个、iHC 3 个）；1000 步、L1 / L2 没做。

## 1. 判定（一屏）

| 级 · 检查 | mid | 读数 |
|---|---|---|
| L0 · 学习率、权重转换往返（212 / 212）、RoPE（32 / 32） | ✅ | — |
| L3 · 第 1 步逐张量梯度对 CPU FP64 | ❌ 7 个未定责 | 最大第 3 层 FFN 前 `hc_base`：GPU 2.79e-3，TPU 4.27e-2，R 15.3；sink 4 个 R 1.23–1.47（[根源一览第 21 行](../../ROOT-CAUSES.md#r21)）；另 9 个张量 TPU 梯度文件落盘损坏、不可判 |
| L4 · 主 loss 全程 / 后段（[D27](#d27)） | 🟡 / 🟡 | 带符号均值：N 4.66e-6，TPU 6.41e-6，r 1.38；后段 r 1.01 |
| L4 · 索引器 KD | 🟡 | MARE：N 3.57e-4，TPU 3.99e-4，r 1.12 |
| L4 · param_norm | 🟡 | 带符号均值：N 8.94e-8，TPU −1.60e-7，r 1.79 |
| L4 · MTP loss、grad_norm、路由偏置 | ✅ | — |

逐格读数见[支线总览](../../dashboard/other-sizes.md)；出处 [`tmb-l4`](results/tmb-l4.report.md)、[`mid-tm2-step1-grad-vs-fp64.json`](results/mid-tm2-step1-grad-vs-fp64.json)；上一代 300 步见[历代比较](../../dashboard/history.md)。

### 第 1 步主 loss 对 CPU FP64

GPU 对 FP64 +1.68e-4，TPU +2.8e-6，R 0.02：差在 GPU 一侧，而且比 GPU 8 份第 1 步极差（1.1e-4）还大（推断：宽度 6144 以后 GPU 一侧 BF16 的系统偏差，真实权重截层 D1 上同样如此）。

## 2. 与 mini 不同的地方

- **sink 梯度**：mini 上只在单层探针里量过，mid 第 1 步整网上 4 个 sink 张量未定责；M42b（照 GPU 写法）的第 1 步还没在 mid 上跑。
- **主 loss、param_norm 略超**：mini 现行配置都过，mid 与真实权重截层上略超。
- **对齐开关的代价**：mid 上全开比全关贵 +46.8%（全开 2.8125 s/步、全关 1.9158 s/步），其中 M13 一项 +24.4%（只关 M13 2.2610 s/步；运行名见 goldens.md），约占总代价的 60%，与 mini 的 +23.7% 一致。这三份只量步时（40 步，第 11–40 步中位数，不含 M39–M41），不做判定。

## 3. 偏差台账

编号接 mini 与 real-weight 的 D 系列。

| # | 差别 | 机制 | 处理 |
|---|---|---|---|
| <a id="d27"></a>D27 | 300 步 L4 四项略超：主 loss 全程（带符号均值：N 4.66e-6，TPU 6.41e-6，r 1.38）与后段（r 1.01）、索引器 KD（MARE：N 3.57e-4，TPU 3.99e-4，r 1.12）、param_norm（带符号均值：N 8.94e-8，TPU −1.60e-7，r 1.79） | 未查；mini 现行配置这几项都过，推断与宽度有关；这一代运行早于 M50（KD 记录写法的病态见 mini [D1](../01-mini/DETAILS.md#d1)） | 待查（[根源一览「还没查明」](../../ROOT-CAUSES.md#u-mid-l4)） |
| <a id="d28"></a>D28 | 第 1 步逐张量梯度 7 个未定责（sink 4 个 R 1.23–1.47，iHC 3 个，最大第 3 层 FFN 前 `hc_base`：GPU 2.79e-3，TPU 4.27e-2，R 15.3）；第 1 步主 loss 对 FP64 GPU +1.68e-4，TPU +2.8e-6 | sink 同[根源一览第 21 行](../../ROOT-CAUSES.md#r21)；iHC 未查；主 loss 的差在 GPU 一侧（推断：宽度 6144 时 GPU 一侧 BF16 的系统偏差） | 待查（[根源一览「还没查明」](../../ROOT-CAUSES.md#u-mid-step1)） |

## 4. 覆盖缺口

- L3 的 9 个落盘损坏的张量要用重跑的第 1 步诊断重比；sink 要用带 M42b 的第 1 步再比一次。
- 只有 300 步；没有 L1 / L2；GPU 的第 1 步偏高从哪一层起没查。
