[janus](../../../../README.md) › [混元 4](../../README.md) › [real-weight](README.md) ｜ 对齐报告（支线）

# 混元 4 · real-weight（真实权重截层 B 与 D1）· 对齐报告

支线：只写与 [mini](../01-mini/DETAILS.md) 不同的地方，判法、开关与生产建议同 mini；N（噪声上限）、r、R、MARE 的读法见[总览细节「读法」](../../dashboard/detail.md#读法)。代号见 [`runs.yaml`](runs.yaml)。下面的判定来自最近一代 1000 步运行 `tb5b-a`、`td5b-b`（MaxText `2c28a4f4`、janus `9c0f19a`，含 M42b；`sa_use_base2_exp` 开，早于 M50，两份 `meta.json` 已核）。2026-11-11 起对齐配置全部关 base2，现行配置在 B、D1 上尚无运行。B 测 MoE（第 0 层稠密 + 第 1 层 MoE），D1 测 MTP（第 0 层稠密 + 1 层 MTP），hy4-preview 真实权重，1000 步。结论在 TPU 一侧 libtpu 0.0.46、jax 0.11.1 上测得（[版本](setup.md#软件版本)），换 libtpu 或 jax 要按 [`libtpu-052.md`](../04-tpu-self/libtpu-052.md) 的做法重测。

> **一句话**：B 没有不过；D1 索引器 KD 不过（[D21](#d21)）；第 1 步梯度两种截层都不过（[D22](#d22)）；截层 B、D1 的逐算子与逐层没做。不能签字。

## 1. 判定（一屏）

| 级 · 检查 | B | D1 |
|---|---|---|
| L0 · 学习率、权重转换往返、RoPE | ✅ | ✅ |
| L3 · 第 1 步逐张量梯度对 CPU FP64 | ❌ R > 1.2 的 16 个 | ❌ 11 个 |
| L4 · 主 loss 全程（`tb5b-a` / `td5b-b`） | 🟡 r 1.15（最大单步：N 1.18e-4，TPU 1.36e-4） | 🟡 r 1.25（MARE：N 2.21e-5，TPU 2.76e-5） |
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
| 3a | 超线处都已用 FP64 定责 | 第 3 节 | D20、D22 待查；grad_norm、param_norm 略超（第 1 节），未编号、待查 | D19、D21、D22 待查；param_norm（MARE：N 9.30e-8，TPU 9.50e-8）、偏置均值（第 1 节）、偏置标准差（最大单步：N 1.69e-3，TPU 1.70e-3）略超，未编号、待查 |
| 3b | golden 对生产程序 | — | 没比 | 没比 |

## 2. 与 mini 不同的地方

- **L3 只看对 FP64 的 R**：GPU 三份第 1 步梯度对 FP64 的误差几乎同向（误差余弦中位数 B 0.997、D1 0.996），三份之间的差（中位数 B 2.0e-4）比离 FP64 的距离（1.41e-3）小一个量级，N 量不出 GPU 自己的系统偏差。
- **只在真实权重上显出来的两条**：RMSNorm 权重梯度（dγ 多舍一次 BF16，[根源一览第 23 行](../../ROOT-CAUSES.md#r23)）；q latent norm 偏小（splash 的 base2 指数，[第 24 行](../../ROOT-CAUSES.md#r24)）。

## 3. 偏差台账

编号接 mini 的 D 系列。

| # | 差别 | 机制 | 处理 |
|---|---|---|---|
| <a id="d19"></a>D19 | D1 第 1 步前向主 loss、MTP loss 就差；MTP loss 多步不过线 | 第 1 步对 CPU FP64 的相对差（(x − FP64) ÷ FP64）：主 loss GPU +6.81e-5、TPU −1.64e-5，R 0.24；MTP loss GPU +1.12e-4、TPU −6.82e-5，R 0.61。主 loss 的差在 GPU 一侧，约 100 步衰减；MTP loss 不衰减，原因未查。06 在 D3 上查到 MTP 偏低主要是轨迹差、base2 占一部分（[06 MTP 一节](../06-rw3/README.md#mtp-loss-全程偏低主要是轨迹同一份权重上的前向差不显著)），D1 上没核是否同源 | 待查 |
| <a id="d20"></a>D20 | B 第 1 步更新后出过 NaN；主 loss 前 100 步同号偏差 | NaN：QK-Clip 在最大 logit 不是正数时开负数的根号（M35 已修，[第 15 行](../../ROOT-CAUSES.md#r15)）；第 1 步主 loss 对 CPU FP64 的相对差：GPU +6.48e-5、TPU −7.94e-5（`tb6diag`），R 1.23，两边方向相反。TPU 偏低来自 splash 的 base2 模式把注意力 logits 整体放大 1.0018 倍（[根源第 24 行](../../ROOT-CAUSES.md#r24)）：只关这个开关（`tnorm-sg` → `tqa-nb2`）TPU 的相对差 −8.06e-5 → +7.3e-6，R 1.24 → 0.11；CPU FP64 前向只加这一处放大，移动 −8.41e-5（[06 的定责](../06-rw3/README.md#第-1-步主-loss-为什么-tpu-偏低)） | NaN 已修；第 1 步主 loss 已定责。对齐配置 2026-11-11 起关 base2，B 上还没用现行配置重跑。前 100 步同号偏差：06 在 D3 上的 250 步两臂里，关 base2 后前段偏低变小但没消失，其余是轨迹差（[06](../06-rw3/README.md#base2-开--关的-250-步回放)）；B 上没核 |
| <a id="d21"></a>D21 | D1 索引器 KD 不过，越往后越偏高 | 这一代运行早于 M50（KD 记录写法在 FP32 下病态，mini [D1](../01-mini/DETAILS.md#d1)）；对齐配置 `hy4-t2-d1.yml` 现已打开 M50。另一个候选：06 在 D3 上查到 GPU 一侧索引器 RoPE 在 BF16 里算，让 KD 偏高约 5e-4（[根源一览第 27 行](../../ROOT-CAUSES.md#r27)），D1 上没核 | 待查：用现行配置（含 M50、关 base2）重跑 1000 步 |
| <a id="d22"></a>D22 | 第 1 步逐张量梯度：B 16 个、D1 11 个 R > 1.2 | RMSNorm 权重（打开 `rmsnorm_scale_grad_fp32` 后 B 的 9 个 R 中位数 1.60 → 0.98）；q latent norm（关 base2 后第 0 层 `q_a_layernorm`：GPU 1.07e-3，TPU 3.09e-3 → 9.42e-4，R 2.89 → 0.88）；sink 同 mid；iHC、`hc_head_scale` 未查 | 待查 |

## 4. 覆盖缺口

- L0 的数据指纹、跨侧初始权重、配置对照没做；L1 只有阶段 A 第 0 层前向 10 个算子，截层 B、D1 的逐算子与逐层没做。
- L3 的证据运行是 `tb6diag`、`td6diag`（MaxText `3741e818`，含 M39–M41，没有 M42b、M50，base2 开），比 L4 的 `tb5b-a`、`td5b-b` 早一代；D1 有 5 个张量的 TPU 梯度文件损坏，以重跑的诊断为准（[`t2d1-td7-step1-grad-vs-fp64.json`](results/t2d1-td7-step1-grad-vs-fp64.json)）。
- 显存装不装得下的判据见 [`setup.md`](setup.md)「显存」一节。

## 5. 适用范围

- 只在 v5p 2x2x2（8 个设备）上跑过，TPU 一侧只用 FSDP 8；EP、TP、多切片没测。
- 只有两种截层：第 0 层稠密加一层 MoE（B），或加一层 MTP（D1）；最多 2 层，最长 1000 步。完整模型没测。
- L4 判定用的 TPU 运行 base2 开、早于 M50；2026-11-11 起的现行对齐配置在这一尺寸上没有运行。
- 尺子是 GPU 单机 8 份两两 28 对；mini 上 16 份的上限是 8 份的 1.0–2.1 倍（[知识库](../../knowledge.md#gpu-1000-步尺子扩到-16-份要点与出处)），这里的「待定」「不过」可能偏严。
- TPU 程序版本：1000 步四份运行（`td5-a`、`tb5-b`、`td5b-b`、`tb5b-a`）的 `meta.json` 记 jax 0.11.1，libtpu 构建串与 06 的 `rfx-on` 相同（那份 `meta.json` 记 libtpu 0.0.46）；更早的运行没核。
