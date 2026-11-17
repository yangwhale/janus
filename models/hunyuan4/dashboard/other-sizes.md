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
| mid | 03 · hy4-mid 现行 C 口径（PROTOCOL 2026-11-12 前瞻登记）：L4 是 TPU tmc-a 300 步对 GPU gm-a（噪声 gm 两两 28 对）；L3 是 tmcs-c（训练步梯度，GPU 第 1 步参照只有 3 份，参照不足）；L0 学习率核对用 tmc-a，权重转换取 tm-a 一代（同一份检查点）；上一代 tm-b / tm2diag 见历代比较 |
| B | 02 · real-weight 截层 B：L4 是现行 C 口径的 TPU tbc-a 1000 步（PROTOCOL 2026-11-12 前瞻登记）对 GPU gb3-a（噪声 gb3 两两 28 对）；L3 是现行 C 口径的 tb8s-c-r1（训练步梯度，GPU 第 1 步参照 6 份，PROTOCOL 2026-11-12 前瞻登记）；L0 学习率核对也用 tbc-a |
| D1 | 02 · real-weight 截层 D1：L4 是现行 C 口径的 TPU tdc-a 1000 步（PROTOCOL 2026-11-12 前瞻登记）对 GPU gd3-a（噪声 gd3 两两 28 对）；L3 是现行 C 口径的 td8s-c-r2（训练步梯度，GPU 第 1 步参照 6 份，PROTOCOL 2026-11-12 前瞻登记）；L0 学习率核对也用 tdc-a |
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
| 全部参数梯度：对 GPU 第 1 步噪声，超出的对 CPU FP64 定责 | 超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数 | 严格：超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数 ≤ 0｜参考：无 | 5 ❌ | 13 ❌ | 9 ❌ | — |

<details><summary>说明（非 ✅ 格子，以及参考档不过或无法判的格子）</summary>

- **全部参数梯度：对 GPU 第 1 步噪声，超出的对 CPU FP64 定责 · mid**：实测 5（超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数）｜严格档 ≤ 0 → 不过｜参考档 无对应门槛｜参照不足（GPU 第 1 步参照只有 gms-a … c 三份，2026-11-12 起要 6 份，待补）；现行 C 口径 tmcs-c 不过，未定责 5 个：iHC 4 个（第 3 层 FFN 前 `hc_base` R 13.7 最大；同 hy4-mini 的 [D11](../alignments/01-mini/DETAILS.md#d11)）与第 4 层 sink（R 1.21，[根源一览第 21 行](../ROOT-CAUSES.md#r21)）；[根源一览「还没查明」](../ROOT-CAUSES.md#u-mid-step1)
- **全部参数梯度：对 GPU 第 1 步噪声，超出的对 CPU FP64 定责 · B**：实测 13（超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数）｜严格档 ≤ 0 → 不过｜参考档 无对应门槛｜前瞻登记（PROTOCOL 2026-11-12，判法看结果之前固定、不用豁免）的判定：不过。未定责的 13 个：RMSNorm 权重 7 个（dγ 多舍一次 BF16，[根源一览第 23 行](../ROOT-CAUSES.md#r23)；开关 `rmsnorm_scale_grad_fp32` 不在对齐配置里）、第 1 层注意力前 iHC 3 个与第 0 层 FFN 前 `hc_scale`、第 1 层 sink（第 21 行）、N = 0 的索引器 `k_norm.bias`；GPU 3 份对照判定相同（[real-weight 报告 D22](../alignments/02-real-weight/ALIGNMENT-REPORT.md#d22)）
- **全部参数梯度：对 GPU 第 1 步噪声，超出的对 CPU FP64 定责 · D1**：实测 9（超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数）｜严格档 ≤ 0 → 不过｜参考档 无对应门槛｜前瞻登记（PROTOCOL 2026-11-12，判法看结果之前固定、不用豁免）的判定：不过。未定责的 9 个：RMSNorm 权重 4 个（[根源一览第 23 行](../ROOT-CAUSES.md#r23)）、第 0 层与 MTP 层 sink（第 21 行）、`hc_head_scale`、第 0 层 `q_b_proj`、N = 0 的索引器 `k_norm.bias`；落盘核对 57 个张量全部完好；GPU 3 份对照判定相同（[real-weight 报告 D22](../alignments/02-real-weight/ALIGNMENT-REPORT.md#d22)）

</details>

## L4 多步（B、D1 1000 步；mid 300 步；都是现行 C 口径）

| 检查项 | 指标 | 门槛（严格｜参考） | mid | B | D1 | tiny |
|---|---|---|---|---|---|---|
| loss（全程） | MARE | 严格：≤ N=1.62e-5 过，> 2N 不过｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | 1.64e-5（1.01×N） 🟡｜参✅ | 2.85e-5（0.9216×N，N=3.09e-5） ✅｜参✅ | 1.78e-5（0.807×N，N=2.21e-5） ✅｜参✅ | — |
| MTP loss（全程与后段；B 没有 MTP 层） | 带符号均值 | 严格：≤ N=1.82e-5 过，> 2N 不过｜参考：无 | 2.28e-5（1.25×N） 🟡 | — | 5.58e-5（MARE，0.9574×N，N=5.83e-5） ✅ | — |
| 索引器 KD loss | 最大单步 | 严格：≤ N=1.27e-3 过，> 2N 不过｜参考：无 | 1.27e-3（1.003×N） 🟡 | −7.97e-4（带符号均值，1.4×N，N=5.69e-4） 🟡 | 1.66e-2（2.3×N，N=7.24e-3） ❌ | — |
| 路由偏置曲线（每层均值、标准差） | MARE | 严格：≤ N=2.98e-3 过，> 2N 不过｜参考：无 | 2.02e-3（0.677×N） ✅ | 0.153（最大单步，0.709×N，N=0.215） ✅ | 9.01e-3（最大单步，1.26×N，N=7.16e-3） 🟡 | — |

<details><summary>说明（非 ✅ 格子，以及参考档不过或无法判的格子）</summary>

- **loss（全程） · mid**：实测 1.64e-5（MARE）｜严格档 N=1.62e-5（28 对）→ 1.01×N，待定｜参考档 首步差 ≤ 2e-4：2.19e-5（0.109×）；带符号均值 ≤ 2e-4：1.24e-6（0.0062×）；无 NaN / Inf，过｜待查：MARE 刚超线（[mid 报告 D27](../alignments/03-mid/ALIGNMENT-REPORT.md#d27)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-mid-l4)）
- **MTP loss（全程与后段；B 没有 MTP 层） · mid**：2 条取最差 `L4.mtp_loss`：实测 2.28e-5（带符号均值）｜严格档 N=1.82e-5（28 对）→ 1.25×N，待定｜参考档 无对应门槛｜待查：TPU 偏高，带符号均值略超（[mid 报告 D27](../alignments/03-mid/ALIGNMENT-REPORT.md#d27)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-mid-l4)）
- **索引器 KD loss · mid**：实测 1.27e-3（最大单步）｜严格档 N=1.27e-3（28 对）→ 1.003×N，待定｜参考档 无对应门槛｜待查：最大单步刚超线（[mid 报告 D27](../alignments/03-mid/ALIGNMENT-REPORT.md#d27)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-mid-l4)）
- **索引器 KD loss · B**：实测 −7.97e-4（带符号均值）｜严格档 N=5.69e-4（28 对）→ 1.4×N，待定｜参考档 无对应门槛｜待查：TPU 偏低，带符号均值略超（[real-weight 报告 D21](../alignments/02-real-weight/ALIGNMENT-REPORT.md#d21)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）
- **索引器 KD loss · D1**：实测 1.66e-2（最大单步）｜严格档 N=7.24e-3（28 对）→ 2.3×N，不过｜参考档 无对应门槛｜待查：TPU 偏高、越往后越高，M50 已开仍不过（[real-weight 报告 D21](../alignments/02-real-weight/ALIGNMENT-REPORT.md#d21)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）
- **路由偏置曲线（每层均值、标准差） · D1**：2 条取最差 `L4.router.bias_mean`：实测 9.01e-3（最大单步）｜严格档 N=7.16e-3（28 对）→ 1.26×N，待定｜参考档 无对应门槛｜待查：均值最大单步略超（[real-weight 报告第 1 节](../alignments/02-real-weight/ALIGNMENT-REPORT.md#1-判定一屏)；[根源一览「还没查明」](../ROOT-CAUSES.md#u-real-weight)）

</details>

## 汇总

严格档数全部格子；参考档只数有对应门槛的格子（演练不适用）。

| 列 | 严格档 ✅ | 🟡 | ❌ | ⬜ | 参考档 ✅ | ☑️ | ❌ | ❔ |
|---|---|---|---|---|---|---|---|---|
| mid | 7 | 3 | 1 | 3 | 3 | 0 | 0 | 0 |
| B | 8 | 1 | 1 | 3 | 3 | 0 | 0 | 0 |
| D1 | 8 | 1 | 2 | 3 | 3 | 0 | 0 | 0 |
| tiny | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
