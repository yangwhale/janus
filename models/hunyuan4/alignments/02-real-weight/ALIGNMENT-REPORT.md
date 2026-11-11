[janus](../../../../README.md) › [混元 4](../../README.md) › [real-weight](README.md) ｜ 对齐报告（支线）

# 混元 4 · real-weight（真实权重截层 B 与 D1）· 对齐报告

支线：只写与 [mini](../01-mini/DETAILS.md) 不同的地方，判法、开关与生产建议同 mini；N（噪声上限）、r、R、MARE 的读法见[总览细节「读法」](../../dashboard/detail.md#读法)。代号见 [`runs.yaml`](runs.yaml)；B、D1 现行一代都没有 M50。B 测 MoE（第 0 层稠密 + 第 1 层 MoE），D1 测 MTP（第 0 层稠密 + 1 层 MTP），hy4-preview 真实权重，1000 步。

> **一句话**：B 没有不过；D1 索引器 KD 不过（[D21](#d21)）；第 1 步梯度两种截层都不过（[D22](#d22)）；截层 B、D1 的逐算子与逐层没做。不能签字。

## 1. 判定（一屏）

| 级 · 检查 | B | D1 |
|---|---|---|
| L0 · 学习率、权重转换往返、RoPE | ✅ | ✅ |
| L3 · 第 1 步逐张量梯度对 CPU FP64 | ❌ R > 1.2 的 16 个 | ❌ 11 个 |
| L4 · 主 loss 全程 | 🟡 r 1.15（最大单步：N 1.18e-4，TPU 1.36e-4） | 🟡 r 1.25（MARE：N 2.21e-5，TPU 2.76e-5） |
| L4 · MTP loss | — | 🟡 r 1.35（MARE：N 5.83e-5，TPU 7.89e-5） |
| L4 · 索引器 KD | ✅ | ❌ r 2.32（带符号均值：N 8.67e-4，TPU +2.01e-3） |
| L4 · grad_norm / param_norm / 路由偏置 | 🟡 / 🟡 / ✅（grad_norm MARE：N 8.54e-3，TPU 9.64e-3，r 1.13；param_norm 最大单步：N 1.87e-7，TPU 1.92e-7，r 1.03） | ✅ / 🟡 / 🟡（最差路由偏置均值最大单步：N 7.16e-3，TPU 1.20e-2，r 1.68） |

逐格读数见[支线总览](../../dashboard/other-sizes.md)；出处 [`t2b5b-l4`](results/t2b5b-l4.report.md)、[`t2d15b-l4`](results/t2d15b-l4.report.md)。

### 签字清单

| # | 条件 | 证据 | B | D1 |
|---|---|---|---|---|
| 1a | L0 全部逐位 | [检查清单](CHECKS.md) L0 | 学习率、转换、RoPE ✅；数据指纹、跨侧初始权重、配置 ⬜ | 同 B |
| 1b | TPU 重跑逐位 | — | ⬜ | ⬜ |
| 1c | 抓数不改数值 | — | ⬜ | ⬜ |
| 2a | L4 对 golden 全过 | 第 1 节 | 过 4、不过 0、待定 3 | 过 1、不过 1（索引器 KD，[D21](#d21)）、待定 7 |
| 2b | L2、L3 用噪声判的检查全过 | 第 1 节 | L2 没做；L3 不过（[D22](#d22)） | 同 B |
| 2c | 噪声参照够多 | 第 2 节 | L4 8 份 28 对 ✅；第 1 步 3 份 ❌ | 同 B |
| 3a | 超线处都已用 FP64 定责 | 第 3 节 | D20、D22 待查；grad_norm、param_norm 略超（第 1 节）未编号 | D19、D21、D22 待查；param_norm（MARE：N 9.30e-8，TPU 9.50e-8）、偏置均值（第 1 节）、偏置标准差（最大单步：N 1.69e-3，TPU 1.70e-3）略超未编号 |
| 3b | golden 对生产程序 | — | 没比 | 没比 |

## 2. 与 mini 不同的地方

- **L3 只看对 FP64 的 R**：GPU 三份第 1 步梯度对 FP64 的误差几乎同向（误差余弦中位数 B 0.997、D1 0.996），三份之间的差（中位数 B 2.0e-4）比离 FP64 的距离（1.41e-3）小一个量级，N 量不出 GPU 自己的系统偏差。
- **只在真实权重上显出来的两条**：RMSNorm 权重梯度（dγ 多舍一次 BF16，[根源一览第 23 行](../../ROOT-CAUSES.md#r23)）；q latent norm 偏小（splash 的 base2 指数，[第 24 行](../../ROOT-CAUSES.md#r24)）。

## 3. 偏差台账

编号接 mini 的 D 系列。

| # | 差别 | 机制 | 处理 |
|---|---|---|---|
| <a id="d19"></a>D19 | D1 第 1 步前向主 loss、MTP loss 就差；MTP loss 多步不过线 | 第 1 步对 CPU FP64：主 loss GPU +9.24e-4、TPU −2.23e-4，R 0.24；MTP loss GPU +1.22e-3、TPU −7.44e-4，R 0.61。主 loss 的差在 GPU 一侧，约 100 步衰减；MTP loss 不衰减，原因未查 | 待查 |
| <a id="d20"></a>D20 | B 第 1 步更新后出过 NaN；主 loss 前 100 步同号偏差 | NaN：QK-Clip 在最大 logit 不是正数时开负数的根号（M35 已修，[第 15 行](../../ROOT-CAUSES.md#r15)）；第 1 步主 loss 对 CPU FP64：GPU +7.58e-4、TPU −9.29e-4，R 1.23，两边方向相反。TPU 偏低来自 splash 的 base2 模式把注意力 logits 整体放大 1.0018 倍（[根源第 24 行](../../ROOT-CAUSES.md#r24)）：只关这个开关（`tnorm-sg` → `tqa-nb2`）TPU 的相对差 −8.06e-5 → +7.3e-6，R 1.24 → 0.11；CPU FP64 前向只加这一处放大，移动 −8.41e-5（[06 的定责](../06-rw3/README.md#第-1-步主-loss-为什么-tpu-偏低)） | NaN 已修；第 1 步主 loss 已定责（对齐配置仍开 base2）；前 100 步同号偏差是它的延续（推断） |
| <a id="d21"></a>D21 | D1 索引器 KD 不过，越往后越偏高 | 这一代运行早于 M50（KD 记录写法在 FP32 下病态，mini [D1](../01-mini/DETAILS.md#d1)）；对齐配置 `hy4-t2-d1.yml` 现已打开 M50 | 待查：用打开 M50 的配置重跑 1000 步 |
| <a id="d22"></a>D22 | 第 1 步逐张量梯度：B 16 个、D1 11 个 R > 1.2 | RMSNorm 权重（打开 `rmsnorm_scale_grad_fp32` 后 B 的 9 个 R 中位数 1.60 → 0.98）；q latent norm（关 base2 后第 0 层 `q_a_layernorm`：GPU 1.07e-3，TPU 3.09e-3 → 9.42e-4，R 2.89 → 0.88）；sink 同 mid；iHC、`hc_head_scale` 未查 | 待查 |

## 4. 覆盖缺口

- L0 的数据指纹、跨侧初始权重、配置对照没做；L1 只有阶段 A 第 0 层前向 10 个算子，截层 B、D1 的逐算子与逐层没做。
- L3 用的是上一代开关（没有 M42b）；D1 有 5 个张量的 TPU 梯度文件损坏，以重跑的诊断为准（[`t2d1-td7-step1-grad-vs-fp64.json`](results/t2d1-td7-step1-grad-vs-fp64.json)）。
