[janus](../../../../../README.md) › [混元 4](../../../README.md) › [TPU 自比](../README.md) › [影响逐件](README.md) ｜ 输出头反向省掉一次舍入

# 输出头反向省掉一次舍入（hc_head 梯度偏）

## 改了什么

不是哪一个开关，是好几种普通改动会碰上的同一处编译结果：换 v7x（H2，libtpu 0.0.46）、einsum 改广播乘（D3）、两个 v5p 切片走 DCN（B15）、v7x 上 0.0.52 纯数据并行不累积（V17），还有关小矩阵乘改写（C15，归第 03 件）。这些程序里，编译器把末层 RMSNorm 反向的每 token 行和 S 并进了输出头反向（dgrad）的卷积 kernel。

## 影响多大

曲线看不出：这几次 300 步曲线最差 r 0.89–1.10，1000 步 0.91–1.33，都在 GPU 重跑的范围里。

第 1 步看得出：`hc_head` 三个参数的梯度离 FP64 是对照的 7.5–46 倍（R），`hc_head_base` 四个元素对 FP64 差同一个常数，0.0.46 上大到把前三个或四个元素推过零。

| 运行 | `hc_head_base` 离 FP64 | 对照 | `hc_head` 三个张量的 R |
|---|---|---|---|
| D3 einsum 改广播乘（v5p 0.0.46） | 1.26 | golden 0.061 | 20.5 / 18.3 / 46.2 |
| H2 换 v7x（0.0.46） | 1.27 | golden 0.061 | 20.7 / 18.4 / 46.2 |
| B15 两个 v5p 切片走 DCN（0.0.46） | 0.51 | golden 0.061 | 8.4 / 7.5 / 21.1 |
| V17 v7x 纯数据并行不累积（0.0.52） | 0.56 | V16 0.014 | 17.3 / 8.8 / 25.7 |

（R 的顺序 base / fn / scale；GPU 三份离 FP64 的中位数：base 0.032、fn 0.066、scale 0.013。）

2026-11-12 前，iHC 豁免没有倍数上限，这三个张量被单列，这几次判成略超或在噪声内；加上限（5.22）后都判「超」，和 GPU 重跑分得开（PROTOCOL 变更记录 2026-11-12，[MATRIX「iHC 豁免」](../MATRIX.md#ihc-豁免定义与受影响的运行)）。

## 问题出在哪

输出头 dgrad 卷积算出 FP32 的 dy，HLO 里写着先舍成 BF16 再给末层 RMSNorm 反向用。S 被并进卷积 kernel 后，excess precision 下生成机器码时这次舍入被省掉：S 用 kernel 里没舍的 FP32 dy，主项读写进显存的 BF16 dy，同一个 dy 两处精度不一致。

`hc_head` 的梯度大半是末层 RMSNorm 的 ε 项，各 token 之间又大量相消，S 上的一点系统误差就放大成一个常数偏差。机理与证据见 [MATRIX「hc_head」](../MATRIX.md#hc_head被-ihc-豁免盖住的三个张量)、差异清单 [A16](../../../../../docs/reference/tpu-vs-gpu-differences.md#a16-hc_head输出头反向-kernel-合并末层-rmsnorm-反向行和时省掉一次-bf16-舍入)。

## 为什么

会不会被切成「S 在卷积 kernel 里」，随芯片、libtpu 版本和切分变：v5p FSDP 8 没有（golden），v7x FSDP 8 在 0.0.46 有、0.0.52 没有；v7x 纯数据并行不累积反过来；v5p 单切片 16 片没有，同一切法走 DCN 就有。判据：优化后 HLO 里主头卷积 kernel 有 S、且 excess precision 开着，17 次存了 HLO 的运行里满足的 6 次都偏，其余 11 次都不偏（[`hc-head-dgrad-fusion.json`](../results/hc-head-dgrad-fusion.json)）。

## 换 libtpu 版本

0.0.52 没修好，只是换了哪些程序被切成这样：v7x FSDP 8（V16）、关小矩阵乘改写（V6）不再偏，v7x 纯数据并行不累积（V17）开始偏。

## 能不能忍

第 1 步 `hc_head` 梯度符号都能反，属于真实的数值缺陷，不是换了求和顺序。曲线 300 步、1000 步都看不出，单看曲线会放过它。

## 怎么修

MaxText 开关 `hc_head_dgrad_force_bf16_rounding`（默认关，台账 M65）：只在末层 norm 上，前向不变，反向时把它输出的余切先舍成 BF16（`reduce_precision`，编译器不会删），S 和主项用同一份 BF16 dy。诊断参数 `hc_head_dgrad_force_bf16_rounding_mode=barrier` 改用 `optimization_barrier`，把 dy 先落显存。

上卡核过（第 1 步对 CPU FP64，[`hc-head-dgrad-fix.json`](../results/hc-head-dgrad-fix.json)）：

| 程序 | 开关关 | 开（reduce_precision） | 开（barrier） | 对照 |
|---|---|---|---|---|
| v7x 纯数据并行不累积，0.0.52（V17 那一型） | 0.561（与 V17 逐位相同） | 0.059 | — | V16 0.014，GPU 中位数 0.032 |
| v7x FSDP 8，0.0.46（H2 那一型） | 1.27（与 H2 逐位相同） | 0.041 | — | golden 0.061；同一程序关 excess precision（H4b）0.041 |
| v5p 关小矩阵乘改写，0.0.46（C15） | 1.26（与 C15 逐位相同） | 0.060 | 0.060 | golden 0.061 |

表里是 `hc_head_base` 离 FP64。开着时第 1 步 loss 逐位不变，步时不变；V17 那一型对 V16、H2 那一型对 golden，第 1 步都没有未定责张量（V17 那一型 `hc_head` 三个 R 1.84 / 1.17 / 0.74，H2 那一型 0.67 / 0.75 / 1.11）；300 步曲线最差 r：V17 那一型对 V16 0.94，H2 那一型对 golden 0.93。在本来不偏的 golden 上开它，第 1 步 205 个梯度都在 GPU 重跑尺子内、不逐位，300 步曲线最差 r 1.03，只换了求和顺序。

## 代价

步时不变（v5p 0.546 → 0.545 s；v7x 0.337 → 0.337 s）。

## 坑

- 现行对齐配置（C 口径 `tc-a`，v5p 0.0.46）离线重编没有 S、按判据不偏，C 口径第 1 步全量抓数 `s1f-c` 的 `hc_head` 也在噪声内；M52（`hc_head_force_bf16_rounding`）名字相近，但不是这处的修法：它舍的是末层 RMSNorm 的输入与余切，这处在末层 RMSNorm 输出的余切上。
- 开关的验证都在 golden 那一代配置上（base2 开、C 口径之前）；C 口径、生产形状、多切片没上卡测。
- MTP 头的 dgrad kernel 里同样并进了行和、同样省了那次舍入，但 MTP 三个 norm 梯度没放大；开关只管主头。

## 一句话

几种普通改动会让编译器把末层 RMSNorm 反向的行和并进输出头反向 kernel、省掉一次 BF16 舍入，`hc_head` 第 1 步梯度偏 7.5–46 倍、曲线看不出；`hc_head_dgrad_force_bf16_rounding` 在 v5p、v7x 上都把它拉回对照水平，前向与步时不变。

## 出处

- 结果文件：[`ts-h-v7b-vs-golden.json`](../results/ts-h-v7b-vs-golden.json)、[`ts-d-einsum-vs-golden.json`](../results/ts-d-einsum-vs-golden.json)、[`ts-ms2-dp2-r2-vs-golden.json`](../results/ts-ms2-dp2-r2-vs-golden.json)、[`ts52-h-v7-dp8-ga1-r1-vs-ts52-h-v7b-r1.json`](../results/ts52-h-v7-dp8-ga1-r1-vs-ts52-h-v7b-r1.json)
- 判据与离线核：[`hc-head-dgrad-fusion.json`](../results/hc-head-dgrad-fusion.json)；修法上卡：[`hc-head-dgrad-fix.json`](../results/hc-head-dgrad-fix.json)
