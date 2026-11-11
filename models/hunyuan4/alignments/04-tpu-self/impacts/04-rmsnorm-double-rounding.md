[janus](../../../../../README.md) › [混元 4](../../../README.md) › [TPU 自比](../README.md) › [影响逐件](README.md) ｜ RMSNorm 多舍一次

# RMSNorm 多舍一次

## 改了什么

对齐开关 `rmsnorm_single_rounding=false`：RMSNorm 回到多舍一次 BF16 的写法。

## 影响多大

曲线最差 r 1.06（路由偏置标准差的带符号均值），略超，和 GPU 重跑分不开。

## 问题出在哪

第 1 步只有第 0 层 `input_layernorm` 的权重梯度变远：离 FP64 4.41e-3，golden 3.54e-3，GPU 3.67e-3，1.20 倍，正好压在定责线上。

## 为什么

RMSNorm 多一次 BF16 舍入，舍入位置和 GPU 不同（[根源一览第 2 行](../../../ROOT-CAUSES.md#r2)）。

## 能不能忍

能忍：曲线和 GPU 重跑分不开，只有这一个张量压线。

## 怎么修

开着 `rmsnorm_single_rounding`。对齐配置开着；舍一次不比两次差，生产也建议开（[对齐报告「生产配置建议」](../../01-mini/ALIGNMENT-REPORT.md#生产配置建议)）。

## 代价

关掉本身多花 +0.6%。

## 坑

待补。

## 一句话

RMSNorm 多舍一次只让一个张量压线；开着单次舍入的开关就没有这件事。

## 出处

- [实验矩阵「E 真实改动」](../MATRIX.md#e-真实改动曲线抓不到时第-1-步定责能不能抓到)
- 结果文件：[`ts-e-rms2r-vs-golden.json`](../results/ts-e-rms2r-vs-golden.json)
