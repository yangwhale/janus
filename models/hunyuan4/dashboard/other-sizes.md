# 混元 4 · 其他尺寸（支线） · 对齐总览

由 `models/hunyuan4/dashboard/build.py` 从 `alignments/*/results/` 生成，数字和判定都不手写。

**本页结构**：怎么读 → 这几个尺寸各测什么 → 各列是什么 → L0–L4（只列有非 ✅ 格子的行）→ 汇总。主线（hy4-mini）在[总览](dashboard.md)。

## 怎么读这张表

- 读法同[总览](dashboard.md)；只列有非 ✅ 格子的行（全过的行照样计入汇总）。跨尺寸的对照见 mini 对齐细节「跨尺寸看」。

### 这几个尺寸各测什么

- **mid**（hy4-mid，5.19B，随机初始化）：只换尺寸，看 mini 的结论放大宽度后还在不在（[报告](../alignments/03-mid/ALIGNMENT-REPORT.md)）。
- **B、D1**（真实权重截层）：B 测 MoE（第 0 层稠密 + 第 1 层 MoE），D1 测 MTP（第 0 层稠密 + 1 层 MTP）（[报告](../alignments/02-real-weight/ALIGNMENT-REPORT.md)）。
- **tiny**（hy4-tiny，5.4M）：只做 TPU 一侧的 CPU 冒烟（[报告](../alignments/00-tiny/ALIGNMENT-REPORT.md)）。
- 判法同 mini：每个尺寸用 GPU 同设置多份两两之差量噪声上限 N（这里各尺寸都是 8 份 28 对；mini 现行 1000 步是 16 份 120 对），TPU 只和指定的一份比；L3 对 CPU FP64 看 R。

## 各列是什么

| 列 | 含义 |
|---|---|
| mid | 03 · hy4-mid 最近一代（base2 开、早于 M50；现行对齐配置在 mid 上尚无运行）：L4 是 TPU tm-b 300 步对 GPU gm-a（噪声 gm 两两 28 对）；L3 是 tm2diag；L0 学习率核对用 tm-b，权重转换取 tm-a 一代（同一份检查点） |
| B | 02 · real-weight 截层 B：L4 是 TPU tb5b-a 1000 步（base2 开、早于 M50；现行对齐配置在 B 上尚无运行）对 GPU gb3-a（噪声 gb3 两两 28 对）；L3 是 tb6diag；L0 学习率核对也用 tb5b-a |
| D1 | 02 · real-weight 截层 D1：L4 是 TPU td5b-b 1000 步（base2 开、早于 M50；现行对齐配置在 D1 上尚无运行）对 GPU gd3-a（噪声 gd3 两两 28 对）；L3 是 td6diag（取回时断流损坏的 5 个 TPU 梯度文件不判）；L0 学习率核对也用 td5b-b |
| tiny | 00 · hy4-tiny：只有 TPU 一侧的 CPU 冒烟（8 个假 CPU 设备，四个阶段整条链路），没有 GPU 对 TPU 的比较 |

## L0 地基（设置 / 数据 / 权重 / 配置，必须逐位）

| 检查项 | 指标 | 门槛（严格｜参考） | mid | B | D1 | tiny |
|---|---|---|---|---|---|---|
| 跨侧初始权重逐位（TPU 载入后对 GPU 初始参数清单） | — | — | ⬜ | ⬜ | ⬜ | — |
| 每 rank 每步输入指纹（TPU 回放 GPU 的 batch） | — | — | ⬜ | ⬜ | ⬜ | — |
| 配置逐项对照 | — | — | ⬜ | ⬜ | ⬜ | — |

<details><summary>说明（非 ✅ 格子，以及参考档不过或无法判的格子）</summary>

- **跨侧初始权重逐位（TPU 载入后对 GPU 初始参数清单） · mid**（未开始）：没做（与 B、D1 同样是缺口）
- **跨侧初始权重逐位（TPU 载入后对 GPU 初始参数清单） · B**（未开始）：登记了，还没做
- **跨侧初始权重逐位（TPU 载入后对 GPU 初始参数清单） · D1**（未开始）：登记了，还没做
- **每 rank 每步输入指纹（TPU 回放 GPU 的 batch） · mid**（未开始）：没做（与 B、D1 同样是缺口）
- **每 rank 每步输入指纹（TPU 回放 GPU 的 batch） · B**（未开始）：登记了，还没做
- **每 rank 每步输入指纹（TPU 回放 GPU 的 batch） · D1**（未开始）：登记了，还没做
- **配置逐项对照 · mid**（未开始）：没做（与 B、D1 同样是缺口）
- **配置逐项对照 · B**（未开始）：登记了，还没做
- **配置逐项对照 · D1**（未开始）：登记了，还没做

</details>

## L1 单算子（对 FP64；逐元素函数按 ULP）

| 检查项 | 指标 | 门槛（严格｜参考） | mid | B | D1 | tiny |
|---|---|---|---|---|---|---|
| T0：hy4-tiny 逐层前向 / 反向对 FP64 | — | — | — | — | — | ⬜ |

<details><summary>说明（非 ✅ 格子，以及参考档不过或无法判的格子）</summary>

- **T0：hy4-tiny 逐层前向 / 反向对 FP64 · tiny**（未开始）：待做

</details>

## L3 第 1 步整网

| 检查项 | 指标 | 门槛（严格｜参考） | mid | B | D1 | tiny |
|---|---|---|---|---|---|---|
| 全部参数梯度：对 GPU 第 1 步噪声，超出的对 CPU FP64 定责 | 超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数 | 严格：超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数 ≤ 0｜参考：无 | 7 ❌ | 16 ❌ | 11 ❌ | — |

<details><summary>说明（非 ✅ 格子，以及参考档不过或无法判的格子）</summary>

- **全部参数梯度：对 GPU 第 1 步噪声，超出的对 CPU FP64 定责 · mid**：实测 7（超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数）｜严格档 ≤ 0 → 不过｜参考档 无对应门槛｜待查：sink 一组是 TPU 一侧 sink 梯度的系统差（[根源一览第 21 行](../ROOT-CAUSES.md#r21)）；iHC 同 hy4-mini（[mini 对齐细节 D11](../alignments/01-mini/DETAILS.md#d11)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-mid-step1)）；梯度文件取回时断流损坏的张量不进判定
- **全部参数梯度：对 GPU 第 1 步噪声，超出的对 CPU FP64 定责 · B**：实测 16（超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数）｜严格档 ≤ 0 → 不过｜参考档 无对应门槛｜待查；real-weight 的 L3 只看对 CPU FP64 的 R（报告第 2 节）。RMSNorm 权重大部分是 dγ 多舍一次 BF16（[根源一览第 23 行](../ROOT-CAUSES.md#r23)），sink 见第 21 行，q latent norm 偏小已定位到 splash 的 base2 指数（[根源一览第 24 行](../ROOT-CAUSES.md#r24)），iHC、`hc_head_scale` 待查（[real-weight 报告 D22](../alignments/02-real-weight/ALIGNMENT-REPORT.md#d22)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）
- **全部参数梯度：对 GPU 第 1 步噪声，超出的对 CPU FP64 定责 · D1**：实测 11（超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数）｜严格档 ≤ 0 → 不过｜参考档 无对应门槛｜待查；real-weight 的 L3 只看对 CPU FP64 的 R（报告第 2 节）。sink、RMSNorm 权重、q 的上下投影、hc_head TPU 更远（[根源一览第 21、23 行](../ROOT-CAUSES.md#r21)；[real-weight 报告 D22](../alignments/02-real-weight/ALIGNMENT-REPORT.md#d22)）；判定以 td6diag 的 52 个张量为准；取回时断流损坏的 5 个另用 td7diag 看过，R 都不超 1.2。td7diag 不是同配置重跑（MaxText `2306c7a5`，含已撤下的 M42），sink 不能拿它换

</details>

## L4 多步（B、D1 1000 步；mid 300 步）

| 检查项 | 指标 | 门槛（严格｜参考） | mid | B | D1 | tiny |
|---|---|---|---|---|---|---|
| loss（全程） | 带符号均值 | 严格：≤ N=4.66e-6 过，> 2N 不过｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | 6.41e-6（1.38×N） 🟡｜参✅ | 1.36e-4（最大单步，1.15×N，N=1.18e-4） 🟡｜参✅ | 2.76e-5（MARE，1.25×N，N=2.21e-5） 🟡｜参✅ | — |
| loss 后段（第 101 步起） | 带符号均值 | 严格：≤ N=4.91e-6 过，> 2N 不过｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | 4.95e-6（1.007×N） 🟡｜参✅ | 9.74e-5（最大单步，0.824×N，N=1.18e-4） ✅｜参✅ | 2.52e-5（MARE，1.055×N，N=2.38e-5） 🟡｜参✅ | — |
| MTP loss（全程与后段；B 没有 MTP 层） | MARE | 严格：≤ N=2.22e-5 过，> 2N 不过｜参考：无 | 2.12e-5（0.956×N） ✅ | — | 7.89e-5（1.35×N，N=5.83e-5） 🟡 | — |
| 索引器 KD loss | MARE | 严格：≤ N=3.57e-4 过，> 2N 不过｜参考：无 | 3.99e-4（1.12×N） 🟡 | 9.25e-4（0.9692×N，N=9.55e-4） ✅ | 2.01e-3（带符号均值，2.32×N，N=8.67e-4） ❌ | — |
| grad_norm（裁剪前） | MARE | 严格：≤ N=7.28e-4 过，> 2N 不过｜参考：无 | 6.77e-4（0.9307×N） ✅ | 9.65e-3（1.13×N，N=8.55e-3） 🟡 | 1.14e-3（带符号均值，0.9805×N，N=1.17e-3） ✅ | — |
| param_norm（更新后参数范数，两边比精确值） | 带符号均值 | 严格：≤ N=8.94e-8 过，> 2N 不过｜参考：带符号均值 ≤ 5e-6、最大单步 ≤ 1.25e-5 | −1.60e-7（1.79×N） 🟡｜参✅ | 1.92e-7（最大单步，1.028×N，N=1.87e-7） 🟡｜参✅ | 9.50e-8（MARE，1.021×N，N=9.30e-8） 🟡｜参✅ | — |
| 路由偏置曲线（每层均值、标准差） | MARE | 严格：≤ N=2.98e-3 过，> 2N 不过｜参考：无 | 2.03e-3（0.681×N） ✅ | 6.05e-3（最大单步，0.831×N，N=7.28e-3） ✅ | 1.20e-2（最大单步，1.68×N，N=7.16e-3） 🟡 | — |

<details><summary>说明（非 ✅ 格子，以及参考档不过或无法判的格子）</summary>

- **loss（全程） · mid**：实测 6.41e-6（带符号均值）｜严格档 N=4.66e-6（28 对）→ 1.38×N，待定｜参考档 首步差 ≤ 2e-4：1.52e-5（0.0758×）；带符号均值 ≤ 2e-4：6.41e-6（0.032×）；无 NaN / Inf，过｜待查（[mid 报告 D27](../alignments/03-mid/ALIGNMENT-REPORT.md#d27)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-mid-l4)）
- **loss（全程） · B**：实测 1.36e-4（最大单步）｜严格档 N=1.18e-4（28 对）→ 1.15×N，待定｜参考档 首步差 ≤ 2e-4：1.45e-4（0.725×）；带符号均值 ≤ 2e-4：8.02e-6（0.0401×）；无 NaN / Inf，过｜待查：前 100 步同号衰减，第 1 步前向就差，后段在噪声内（[real-weight 报告 D20](../alignments/02-real-weight/ALIGNMENT-REPORT.md#d20)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）
- **loss（全程） · D1**：实测 2.76e-5（MARE）｜严格档 N=2.21e-5（28 对）→ 1.25×N，待定｜参考档 首步差 ≤ 2e-4：8.45e-5（0.422×）；带符号均值 ≤ 2e-4：9.22e-6（0.0461×）；无 NaN / Inf，过｜待查（[real-weight 报告 D19](../alignments/02-real-weight/ALIGNMENT-REPORT.md#d19)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）
- **loss 后段（第 101 步起） · mid**：实测 4.95e-6（带符号均值）｜严格档 N=4.91e-6（28 对）→ 1.007×N，待定｜参考档 首步差 ≤ 2e-4：1.52e-5（0.0758×）；带符号均值 ≤ 2e-4：4.95e-6（0.0247×）；无 NaN / Inf，过｜待查（[mid 报告 D27](../alignments/03-mid/ALIGNMENT-REPORT.md#d27)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-mid-l4)）
- **loss 后段（第 101 步起） · D1**：实测 2.52e-5（MARE）｜严格档 N=2.38e-5（28 对）→ 1.055×N，待定｜参考档 首步差 ≤ 2e-4：8.45e-5（0.422×）；带符号均值 ≤ 2e-4：1.57e-5（0.0784×）；无 NaN / Inf，过｜待查（[real-weight 报告 D19](../alignments/02-real-weight/ALIGNMENT-REPORT.md#d19)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）
- **MTP loss（全程与后段；B 没有 MTP 层） · D1**：2 条取最差 `L4.mtp_loss`：实测 7.89e-5（MARE）｜严格档 N=5.83e-5（28 对）→ 1.35×N，待定｜参考档 无对应门槛｜待查（[real-weight 报告 D19](../alignments/02-real-weight/ALIGNMENT-REPORT.md#d19)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）
- **索引器 KD loss · mid**：实测 3.99e-4（MARE）｜严格档 N=3.57e-4（28 对）→ 1.12×N，待定｜参考档 无对应门槛｜待查（[mid 报告 D27](../alignments/03-mid/ALIGNMENT-REPORT.md#d27)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-mid-l4)）
- **索引器 KD loss · D1**：实测 2.01e-3（带符号均值）｜严格档 N=8.67e-4（28 对）→ 2.32×N，不过｜参考档 无对应门槛｜待查（[real-weight 报告 D21](../alignments/02-real-weight/ALIGNMENT-REPORT.md#d21)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）
- **grad_norm（裁剪前） · B**：实测 9.65e-3（MARE）｜严格档 N=8.55e-3（28 对）→ 1.13×N，待定｜参考档 无对应门槛｜待查（[real-weight 报告第 1 节](../alignments/02-real-weight/ALIGNMENT-REPORT.md#1-判定一屏)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）
- **param_norm（更新后参数范数，两边比精确值） · mid**：实测 −1.60e-7（带符号均值）｜严格档 N=8.94e-8（28 对）→ 1.79×N，待定｜参考档 带符号均值 ≤ 5e-6：1.60e-7（0.0321×）；最大单步 ≤ 1.25e-5：3.27e-7（0.0262×），过｜待查（[mid 报告 D27](../alignments/03-mid/ALIGNMENT-REPORT.md#d27)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-mid-l4)）
- **param_norm（更新后参数范数，两边比精确值） · B**：实测 1.92e-7（最大单步）｜严格档 N=1.87e-7（28 对）→ 1.028×N，待定｜参考档 带符号均值 ≤ 5e-6：6.00e-8（0.012×）；最大单步 ≤ 1.25e-5：1.92e-7（0.0154×），过｜待查（[real-weight 报告第 1 节](../alignments/02-real-weight/ALIGNMENT-REPORT.md#1-判定一屏)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）
- **param_norm（更新后参数范数，两边比精确值） · D1**：实测 9.50e-8（MARE）｜严格档 N=9.30e-8（28 对）→ 1.021×N，待定｜参考档 带符号均值 ≤ 5e-6：9.14e-8（0.0183×）；最大单步 ≤ 1.25e-5：2.71e-7（0.0217×），过｜待查（[real-weight 报告第 1 节](../alignments/02-real-weight/ALIGNMENT-REPORT.md#1-判定一屏)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）
- **路由偏置曲线（每层均值、标准差） · D1**：2 条取最差 `L4.router.bias_mean`：实测 1.20e-2（最大单步）｜严格档 N=7.16e-3（28 对）→ 1.68×N，待定｜参考档 无对应门槛｜待查（[real-weight 报告第 1 节](../alignments/02-real-weight/ALIGNMENT-REPORT.md#1-判定一屏)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）

</details>

## 汇总

严格档数全部格子；参考档只数有对应门槛的格子（演练不适用）。

| 列 | 严格档 ✅ | 🟡 | ❌ | ⬜ | 参考档 ✅ | ☑️ | ❌ | ❔ |
|---|---|---|---|---|---|---|---|---|
| mid | 6 | 4 | 1 | 3 | 3 | 0 | 0 | 0 |
| B | 6 | 3 | 1 | 3 | 3 | 0 | 0 | 0 |
| D1 | 4 | 5 | 2 | 3 | 3 | 0 | 0 | 0 |
| tiny | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
