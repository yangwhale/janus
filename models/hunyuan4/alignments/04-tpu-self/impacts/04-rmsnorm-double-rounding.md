[janus](../../../../../README.md) › [混元 4](../../../README.md) › [TPU 自比](../README.md) › [影响逐件](README.md) ｜ RMSNorm 多舍一次

# RMSNorm 多舍一次

## 改了什么

对齐开关 `rmsnorm_single_rounding=false`：RMSNorm 回到多舍一次 BF16 的写法。

## 影响多大

libtpu 0.0.46 上，曲线最差 r 1.06（路由偏置标准差的带符号均值），略超，和 GPU 重跑分不开。libtpu 0.0.52 上大得多，见下面「换 libtpu 版本」。

## 问题出在哪

第 1 步只有第 0 层 `input_layernorm` 的权重梯度变远：离 FP64 4.41e-3，golden 3.54e-3，GPU 3.67e-3，1.20 倍，正好压在定责线上。

## 为什么

RMSNorm 多一次 BF16 舍入，舍入位置和 GPU 不同（[根源一览第 2 行](../../../ROOT-CAUSES.md#r2)）。

## 换 libtpu 版本

在 libtpu 0.0.52 上重测（V7，v5p，对 0.0.52 的 golden `ts-g52-a` 比；[`libtpu-052.md`](../libtpu-052.md)），结果**变了**：

| | 0.0.46 | 0.0.52 |
|---|---|---|
| 300 步曲线最差 r | 1.06（路由偏置标准差） | **5.50**（主 loss·带符号均值） |
| 1000 步曲线最差 r | 没跑 | **3.68**（主 loss 后段·带符号均值） |
| 前 300 步主 loss 对 golden 的带符号均值 | 1.3e-6 | 1.65e-5 |
| 第 1 步第 0 层 `input_layernorm` 的 R | 1.20 | 1.17 |

- 0.0.52 上主 loss 系统性偏高，判「超」，和 GPU 重跑分得开；换一组机器重跑逐位相同（V7b），不是偶然。
- 第 1 步看不出来（没有定不了责的张量），差别是跑起来以后累积出来的。
- 两个版本里，这一改动在优化后 HLO 上的差别是同一类（reduce-precision 少 88 个、convert 多几百个），所以变的是后端怎么生成 kernel；具体哪一步没有查。
- golden 本身（开着单次舍入）换 0.0.52 在噪声内，所以开着开关就没有这件事。

## 能不能忍

libtpu 0.0.46 上能忍：曲线和 GPU 重跑分不开，只有这一个张量压线。libtpu 0.0.52 上不能忍：主 loss 系统性偏高，r 5.50。

## 怎么修

开着 `rmsnorm_single_rounding`。对齐配置开着；舍一次不比两次差，生产也建议开（[对齐报告「生产配置建议」](../../01-mini/ALIGNMENT-REPORT.md#生产配置建议)）。

## 代价

关掉本身多花 +0.6%。

## 坑

待补。

## 一句话

RMSNorm 多舍一次在 libtpu 0.0.46 上只让一个张量压线，在 0.0.52 上让主 loss 系统性偏高；开着单次舍入的开关就没有这件事，对齐、生产都要开。

## 出处

- [实验矩阵「E 真实改动」](../MATRIX.md#e-真实改动曲线抓不到时第-1-步定责能不能抓到)
- 结果文件：[`ts-e-rms2r-vs-golden.json`](../results/ts-e-rms2r-vs-golden.json)
