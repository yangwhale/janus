[janus](../../../README.md) › [混元 4](../README.md) › [总览](dashboard.md) ｜ 历代比较

# 混元 4 · 历代比较

由 `models/hunyuan4/dashboard/build.py` 从结果文件生成，数字和判定都不手写。[总览](dashboard.md)只放各尺寸的现行比较；这里放历代与诊断代，便于看每一代开关的效果。

- L4 每格：判定与 r = TPU 对 GPU 基线的差 ÷ GPU 自己重跑的噪声上限 N，带符号均值、MARE、最大单步三个判定量里取 r 最大的；r ≤ 1 过（✅），1–2 待定（🟡），> 2 不过（❌）；— 是这次没有这一项。每格的 N 与 TPU 实测见各结果文件的 report.md。
- 300 步与 1000 步的 r 不能直接比大小：GPU 自己重跑的散布随步数变宽。
- 每一代改了什么：开关台账 [`CHANGES-LEDGER.md`](../CHANGES-LEDGER.md)；判法的改动：[`PROTOCOL.md`](../PROTOCOL.md) 变更记录。

## mini · 1000 步（对 GPU g3-a；噪声 g3 两两：现行 tc-a、上一代 tz-off-rep 与 x5-prod 16 份 120 对，其余各代 8 份 28 对，两种 N 下的 r 不能直接比）

M 编号是开关台账的行；「诊断」是只为拆开某个开关效果的运行，不当现行。

| 被测 | 设置 | 主 loss | 主 loss 后段 | MTP loss | MTP 后段 | 索引器 KD | grad_norm | param_norm | 偏置均值 | 偏置标准差 |
|---|---|---|---|---|---|---|---|---|---|---|
| [`t1u-a`](../alignments/01-mini/results/t1u-l4.report.md) | M28 之前 | ✅ 0.92 | ✅ 0.92 | ✅ 0.85 | ✅ 0.85 | 🟡 1.93 | ✅ 0.90 | 🟡 1.06 | ✅ 0.75 | ✅ 1.00 |
| [`t1w`](../alignments/01-mini/results/t1w-l4.report.md) | +M28（t1v 的设置） | 🟡 1.88 | 🟡 1.93 | ✅ 0.95 | ✅ 0.95 | ❌ 2.37 | ✅ 0.88 | ✅ 0.43 | ✅ 0.50 | ✅ 0.83 |
| [`t1zk-a`](../alignments/01-mini/results/t1zk-l4.report.md) | +M30、M32（t1z 的设置） | ❌ 2.12 | ❌ 2.16 | 🟡 1.01 | 🟡 1.00 | 🟡 1.52 | ✅ 0.86 | ✅ 0.39 | ✅ 0.87 | ✅ 0.96 |
| [`t1zr-a`](../alignments/01-mini/results/t1zr-l4.report.md) | +M36 | 🟡 1.05 | 🟡 1.04 | ✅ 0.86 | ✅ 0.86 | ✅ 0.89 | ✅ 0.83 | ✅ 0.38 | 🟡 1.18 | ✅ 0.91 |
| [`t1zs-a`](../alignments/01-mini/results/t1zs-l4.report.md) | 诊断：t1zr + M37 | 🟡 1.20 | 🟡 1.18 | 🟡 1.33 | 🟡 1.34 | 🟡 1.64 | ✅ 0.90 | ✅ 0.34 | 🟡 1.86 | ✅ 0.91 |
| [`t1zt-a`](../alignments/01-mini/results/t1zt-l4.report.md) | 诊断：t1zs + M34 | 🟡 1.40 | 🟡 1.41 | ✅ 0.86 | ✅ 0.86 | ✅ 0.93 | 🟡 1.21 | ✅ 0.67 | ❌ 2.54 | ✅ 0.96 |
| [`t1zu-b`](../alignments/01-mini/results/t1zub-l4.report.md) | 诊断：t1zt + M39 | 🟡 1.23 | 🟡 1.23 | ✅ 0.92 | ✅ 0.92 | 🟡 1.13 | 🟡 1.13 | 🟡 1.54 | ✅ 0.89 | ✅ 0.98 |
| [`t1zu-c`](../alignments/01-mini/results/t1zuc-l4.report.md) | 诊断：t1zt + M39、M41 | 🟡 1.91 | 🟡 1.91 | ✅ 0.90 | ✅ 0.90 | 🟡 1.37 | ✅ 0.89 | ✅ 0.59 | 🟡 1.57 | 🟡 1.09 |
| [`t1zu-d`](../alignments/01-mini/results/t1zud-l4.report.md) | 诊断：t1zt + M39、M40 | 🟡 1.09 | 🟡 1.10 | ✅ 0.88 | ✅ 0.88 | ✅ 0.94 | ✅ 0.89 | ✅ 0.74 | 🟡 1.18 | ✅ 0.89 |
| [`t1zu-a`](../alignments/01-mini/results/t1zu-l4.report.md) | t1zt + M39、M40、M41 | 🟡 1.07 | 🟡 1.06 | ✅ 0.88 | ✅ 0.88 | ✅ 0.96 | ✅ 0.97 | ✅ 0.51 | ✅ 0.77 | ✅ 0.90 |
| [`t1zv-b`](../alignments/01-mini/results/t1zv-l4.report.md) | 诊断：t1zu + M42 | 🟡 1.32 | 🟡 1.32 | ✅ 0.87 | ✅ 0.87 | ✅ 0.96 | ✅ 0.81 | ✅ 0.60 | 🟡 1.61 | ✅ 0.81 |
| [`t1zz-a`](../alignments/01-mini/results/t1zz-l4.report.md) | 验证：t1zw 再关 splash 的 base2 指数 | 🟡 1.91 | 🟡 1.94 | ✅ 1.00 | ✅ 1.00 | 🟡 1.24 | 🟡 1.01 | ✅ 0.56 | ✅ 0.41 | ✅ 0.81 |
| [`t1zw-a`](../alignments/01-mini/results/t1zw-l4.report.md) | 上一代：t1zu + M42b | ✅ 0.98 | ✅ 0.95 | ✅ 0.85 | ✅ 0.85 | 🟡 1.06 | 🟡 1.23 | ✅ 0.44 | ✅ 0.81 | 🟡 1.14 |
| [`t1zg-a`](../alignments/01-mini/results/t1zga-l4.report.md) | 上一代：t1zw + M50（KD 广义 KL，只改记录值） | ✅ 0.98 | ✅ 0.95 | ✅ 0.85 | ✅ 0.85 | ✅ 0.85 | 🟡 1.23 | ✅ 0.44 | ✅ 0.81 | 🟡 1.14 |
| [`t1zl-a`](../alignments/01-mini/results/t1zla-l4.report.md) | 验证：t1zg + M51（精确 log），主 loss 变差，没采用 | 🟡 1.44 | 🟡 1.45 | 🟡 1.27 | 🟡 1.27 | ✅ 0.91 | ✅ 0.88 | ✅ 0.38 | ✅ 0.18 | 🟡 1.01 |
| [`t1zg-b`](../alignments/01-mini/results/t1zgb-l4.report.md) | t1zg-a 的开关、M50 修订版代码（MaxText fd30fbfb）：KD 记录值逐位相同，其余最多差 2 ulp | ✅ 0.98 | ✅ 0.95 | ✅ 0.85 | ✅ 0.85 | ✅ 0.85 | 🟡 1.23 | ✅ 0.44 | ✅ 0.81 | 🟡 1.14 |
| [`tz-off-rep`](../alignments/01-mini/results/t1zo-l4.report.md) | 上一代：t1zg 再关 base2（噪声换 g3 16 份 120 对） | 🟡 1.55 | 🟡 1.52 | ✅ 1.00 | ✅ 1.00 | ✅ 0.77 | ✅ 0.81 | ✅ 0.55 | ✅ 0.20 | ✅ 0.77 |
| [`tc-a`](../alignments/01-mini/results/t1zc-l4.report.md) | **现行**：C 口径，tz-off-rep 再开 M52、M58（2026-11-11 起；6 份取第一份；tc-b … tc-d、tc-f 只改执行方式，tc-e 改了重算策略） | ✅ 0.98 | ✅ 0.97 | ✅ 0.81 | ✅ 0.80 | ✅ 0.75 | ✅ 0.79 | ✅ 0.14 | 🟡 1.31 | ✅ 0.85 |
| [`x5-prod`](../alignments/01-mini/results/t1zp-l4.report.md) | 验证：生产建议组合（tz-off-rep（C 口径之前）再开 M42、嵌入与 RMSNorm scale 梯度 FP32，关 M8、M10、M27、M30、M36、M42b） | ✅ 0.83 | ✅ 0.80 | ✅ 0.98 | ✅ 0.98 | 🟡 1.26 | ✅ 0.86 | ✅ 0.54 | ✅ 0.54 | ✅ 0.86 |

## mini · 300 步（t1n、t1s 对 GPU g1-a，其余对 g2-a；噪声各自 8 份两两 28 对）

t1n、t1s 两代按当时的角色（GPU 当被测）记差；g1 → g2 是 GPU 参照的一处修正（台账 G17）。

| 被测 | 设置 | 主 loss | 主 loss 后段 | MTP loss | MTP 后段 | 索引器 KD | grad_norm | param_norm | 偏置均值 | 偏置标准差 |
|---|---|---|---|---|---|---|---|---|---|---|
| [`t1n-a`](../alignments/01-mini/results/t1-l4-nozm.report.md) | 第一代（路由偏置零均值关） | ❌ 7.17 | 🟡 1.47 | 🟡 1.17 | 🟡 1.17 | ❌ 8.74 | ❌ 2.68 | 🟡 1.58 | ✅ 0.31 | ✅ 0.88 |
| [`t1s-a`](../alignments/01-mini/results/t1s-l4.report.md) | 第二代（再开四个舍入开关与索引器真 FP32） | ❌ 7.00 | 🟡 1.09 | ✅ 0.85 | ✅ 0.85 | ❌ 16.00 | ❌ 2.75 | 🟡 1.45 | ✅ 0.25 | ✅ 0.85 |
| [`t1t-c`](../alignments/01-mini/results/t1t-l4.report.md) | M28 之前 | 🟡 1.02 | 🟡 1.03 | ✅ 0.93 | ✅ 0.93 | ❌ 3.10 | 🟡 1.34 | 🟡 1.60 | ✅ 0.80 | ✅ 0.83 |
| [`t1v`](../alignments/01-mini/results/t1v-l4.report.md) | +M28 | ✅ 0.96 | ✅ 0.84 | ✅ 0.56 | ✅ 0.55 | ❌ 3.67 | 🟡 1.39 | ✅ 0.88 | ✅ 0.96 | ✅ 0.88 |
| [`t1z-a`](../alignments/01-mini/results/t1z-l4.report.md) | +M30、M32 | ✅ 0.91 | ✅ 0.75 | ✅ 0.61 | ✅ 0.61 | 🟡 1.04 | 🟡 1.31 | ✅ 0.53 | ✅ 0.88 | 🟡 1.01 |
| [`t1zr-a`](../alignments/01-mini/results/t1zr300-l4.report.md) | +M36（前 300 步） | ✅ 0.88 | ✅ 0.76 | ✅ 0.67 | ✅ 0.67 | 🟡 1.11 | 🟡 1.26 | ✅ 0.39 | ✅ 0.15 | ✅ 0.83 |
| [`t1zs-a`](../alignments/01-mini/results/t1zs300-l4.report.md) | 诊断：+M37（前 300 步） | 🟡 1.79 | 🟡 1.32 | ✅ 0.78 | ✅ 0.78 | 🟡 1.30 | 🟡 1.30 | 🟡 1.16 | ✅ 0.37 | ✅ 0.77 |
| [`t1zt-a`](../alignments/01-mini/results/t1zt300-l4.report.md) | 诊断：+M34（前 300 步） | ✅ 0.96 | ✅ 0.82 | ✅ 0.49 | ✅ 0.50 | 🟡 1.34 | 🟡 1.27 | ✅ 0.56 | ✅ 0.64 | ✅ 0.78 |
| [`t1zu-b`](../alignments/01-mini/results/t1zub300-l4.report.md) | 诊断：t1zt + M39（前 300 步） | ✅ 0.96 | ✅ 0.87 | ✅ 0.65 | ✅ 0.66 | 🟡 1.42 | 🟡 1.36 | ✅ 0.55 | ✅ 0.15 | ✅ 0.83 |
| [`t1zu-c`](../alignments/01-mini/results/t1zuc300-l4.report.md) | 诊断：t1zt + M39、M41（前 300 步） | ❌ 2.09 | 🟡 1.48 | ✅ 0.53 | ✅ 0.52 | 🟡 1.15 | 🟡 1.32 | ✅ 0.84 | ✅ 0.88 | ✅ 0.92 |
| [`t1zu-d`](../alignments/01-mini/results/t1zud300-l4.report.md) | 诊断：t1zt + M39、M40（前 300 步） | ✅ 0.89 | ✅ 0.84 | ✅ 0.58 | ✅ 0.58 | 🟡 1.09 | 🟡 1.33 | ✅ 0.45 | ✅ 0.15 | ✅ 0.83 |
| [`t1zu-a`](../alignments/01-mini/results/t1zu300-l4.report.md) | t1zt + M39、M40、M41（前 300 步） | ✅ 0.89 | ✅ 0.82 | ✅ 0.58 | ✅ 0.58 | 🟡 1.06 | 🟡 1.32 | ✅ 0.57 | ✅ 0.48 | ✅ 0.76 |
| [`t1zv-b`](../alignments/01-mini/results/t1zv300-l4.report.md) | 诊断：t1zu + M42（前 300 步） | ✅ 0.92 | ✅ 0.77 | ✅ 0.96 | ✅ 0.96 | 🟡 1.56 | 🟡 1.32 | ✅ 0.99 | ✅ 0.32 | ✅ 0.67 |
| [`t1zz-a`](../alignments/01-mini/results/t1zz300-l4.report.md) | 验证：再关 base2（前 300 步） | 🟡 1.10 | ✅ 0.78 | ✅ 0.80 | ✅ 0.80 | 🟡 1.09 | 🟡 1.23 | ✅ 0.57 | ✅ 0.92 | ✅ 0.67 |
| [`t1zw-a`](../alignments/01-mini/results/t1zw300-l4.report.md) | 上一代：t1zu + M42b（前 300 步） | ✅ 0.94 | ✅ 0.78 | ✅ 0.58 | ✅ 0.57 | 🟡 1.20 | 🟡 1.32 | ✅ 0.48 | ✅ 0.92 | ✅ 0.97 |
| [`t1zg-a`](../alignments/01-mini/results/t1zga300-l4.report.md) | 上一代（前 300 步） | ✅ 0.94 | ✅ 0.78 | ✅ 0.58 | ✅ 0.57 | ✅ 0.90 | 🟡 1.32 | ✅ 0.48 | ✅ 0.92 | ✅ 0.97 |
| [`t1zl-a`](../alignments/01-mini/results/t1zla300-l4.report.md) | 验证：+M51（前 300 步） | 🟡 1.21 | ✅ 0.93 | ✅ 0.68 | ✅ 0.70 | ✅ 0.86 | 🟡 1.42 | ✅ 0.68 | ✅ 0.48 | ✅ 0.77 |
| [`tz-off-rep`](../alignments/01-mini/results/t1zo300-l4.report.md) | 上一代：再关 base2（前 300 步） | 🟡 1.10 | ✅ 0.78 | ✅ 0.80 | ✅ 0.80 | ✅ 0.94 | 🟡 1.23 | ✅ 0.57 | ✅ 0.92 | ✅ 0.67 |
| [`tc-a`](../alignments/01-mini/results/t1zc300-l4.report.md) | **现行**：C 口径，再开 M52、M58（前 300 步） | ✅ 0.87 | ✅ 0.81 | ✅ 0.62 | ✅ 0.61 | ✅ 0.94 | 🟡 1.01 | ✅ 0.18 | ✅ 0.88 | ✅ 0.83 |
| [`x5-prod`](../alignments/01-mini/results/t1zp300-l4.report.md) | 验证：生产建议组合（前 300 步） | 🟡 1.01 | ✅ 0.90 | ✅ 0.61 | ✅ 0.61 | ❌ 4.91 | 🟡 1.27 | ✅ 0.41 | ✅ 0.80 | ✅ 0.80 |

## real-weight · D1（300 步对 gd2-a、1000 步对 gd3-a；噪声各自 28 对）

| 被测 | 设置 | 主 loss | 主 loss 后段 | MTP loss | MTP 后段 | 索引器 KD | grad_norm | param_norm | 偏置均值 | 偏置标准差 |
|---|---|---|---|---|---|---|---|---|---|---|
| [`td2-a`](../alignments/02-real-weight/results/t2d1-l4.report.md) | 第一代，300 步（t1v 等价设置） | ❌ 3.12 | 🟡 1.19 | ❌ 4.69 | ❌ 2.94 | 🟡 1.07 | 🟡 1.46 | 🟡 1.25 | 🟡 1.04 | ✅ 0.68 |
| [`td3-a`](../alignments/02-real-weight/results/t2d1z-l4.report.md) | 第二代，300 步（t1z 的开关、M25 暂关） | ❌ 2.97 | 🟡 1.20 | ❌ 4.28 | ❌ 2.70 | 🟡 1.13 | 🟡 1.69 | 🟡 1.24 | ✅ 0.90 | ✅ 0.81 |
| [`td4-a`](../alignments/02-real-weight/results/t2d14-l4.report.md) | 第三代，300 步（t1zr 再加 M34、M37，M25 重新打开） | ❌ 2.64 | 🟡 1.13 | ❌ 4.09 | ❌ 2.54 | ❌ 2.20 | 🟡 1.44 | 🟡 1.42 | 🟡 1.36 | ✅ 0.85 |
| [`td5-a`](../alignments/02-real-weight/results/t2d15-l4.report.md) | 1000 步（当时 T2 配置的全部开关，含 M39–M41，不含 M42b） | ✅ 0.94 | ✅ 0.73 | 🟡 1.72 | 🟡 1.43 | ❌ 3.46 | ✅ 0.80 | ✅ 0.38 | 🟡 1.68 | ✅ 0.92 |
| [`td5b-b`](../alignments/02-real-weight/results/t2d15b-l4.report.md) | 最近一代，1000 步（含 M42b；base2 开、早于 M50，现行对齐配置在 D1 上尚无运行） | 🟡 1.25 | 🟡 1.06 | 🟡 1.35 | 🟡 1.27 | ❌ 2.32 | ✅ 0.98 | 🟡 1.02 | 🟡 1.68 | 🟡 1.01 |

## real-weight · B（300 步对 gb2-a、1000 步对 gb3-a；噪声各自 28 对；第一代 tb2-a 第 1 步更新后出 NaN，没有 L4）

| 被测 | 设置 | 主 loss | 主 loss 后段 | MTP loss | MTP 后段 | 索引器 KD | grad_norm | param_norm | 偏置均值 | 偏置标准差 |
|---|---|---|---|---|---|---|---|---|---|---|
| [`tb3-b`](../alignments/02-real-weight/results/t2bz-l4.report.md) | 第二代，300 步（同 td3-a 的设置） | ❌ 2.34 | ✅ 0.98 | — | — | 🟡 1.58 | ✅ 0.96 | 🟡 1.88 | 🟡 1.10 | 🟡 1.14 |
| [`tb4-a`](../alignments/02-real-weight/results/t2b4-l4.report.md) | 第三代，300 步（同 td4-a 的设置） | ❌ 2.29 | ✅ 0.83 | — | — | 🟡 1.61 | 🟡 1.61 | 🟡 1.60 | ✅ 0.58 | ✅ 0.56 |
| [`tb5-b`](../alignments/02-real-weight/results/t2b5-l4.report.md) | 1000 步（同 td5-a 的设置） | 🟡 1.19 | ✅ 0.83 | — | — | 🟡 1.12 | 🟡 1.13 | 🟡 1.31 | ✅ 0.66 | ✅ 0.83 |
| [`tb5b-a`](../alignments/02-real-weight/results/t2b5b-l4.report.md) | 最近一代，1000 步（含 M42b；base2 开、早于 M50，现行对齐配置在 B 上尚无运行） | 🟡 1.15 | ✅ 0.82 | — | — | ✅ 0.97 | 🟡 1.13 | 🟡 1.03 | ✅ 0.66 | ✅ 0.83 |

## mid · 300 步（对 GPU gm-a，噪声 gm 两两 28 对）

tm-a 与 tm-b 差不止一个开关，不能当 A/B。

| 被测 | 设置 | 主 loss | 主 loss 后段 | MTP loss | MTP 后段 | 索引器 KD | grad_norm | param_norm | 偏置均值 | 偏置标准差 |
|---|---|---|---|---|---|---|---|---|---|---|
| [`tm-a`](../alignments/03-mid/results/tm-l4.report.md) | fd796f6 时的配置（没有 M39–M41、M42b） | 🟡 1.01 | ✅ 0.94 | 🟡 1.04 | 🟡 1.04 | ✅ 0.94 | ✅ 0.97 | 🟡 1.02 | ✅ 0.53 | ✅ 0.86 |
| [`tm-b`](../alignments/03-mid/results/tmb-l4.report.md) | 最近一代（含 M39–M41 与 M42b；base2 开、早于 M50，现行对齐配置在 mid 上尚无运行） | 🟡 1.38 | 🟡 1.01 | ✅ 0.96 | ✅ 0.91 | 🟡 1.12 | ✅ 0.93 | 🟡 1.79 | ✅ 0.61 | ✅ 0.68 |

## mini · 第 1 步整网（L3）

全部参数梯度对 GPU 第 1 步参照的噪声（现行 6 份 15 对，之前各代 3 份 3 对），超出的对 CPU FP64 整网重算定责（规则见总览 L3 一组）。

| 被测 | 设置 | 判定 | 张量 | 超出 N 而未定责 | 其中超出 2N |
|---|---|---|---|---|---|
| `t1diag15` | t1s 设置，对 g1s 三份 | ❌ | 205 | 7 | 3 |
| `t1diag17` | t1t 设置，对 g2s 三份 | ❌ | 205 | 7 | 2 |
| `t1vdiag` | t1v 设置（+M28），对 g2s 三份 | ❌ | 205 | 5 | 2 |
| `t1zdiag2` | 上一代：t1z 设置（+M30、M32），对 g2s 三份 | ❌ | 205 | 4 | 2 |
| `mz-diag1` | 旁证：base2 开那一代配置的诊断程序梯度，对 g2s 三份 | ❌ | 205 | 6 | 2 |
| `mz-opt2` | 上一代：base2 开的配置训练步实际用的梯度，对 g2s 三份 | ❌ | 205 | 6 | 2 |
| `tz-off-rep` | C 口径之前的配置（base2 关）训练步实际用的梯度，对 g2s 三份（2026-11-11 前的尺子） | ❌ | 205 | 5 | 1 |
| `tz-off-rep` | 上一代（C 口径之前）：同上，尺子换成 g2s 6 份 15 对（2026-11-11 补 g2s-d … f） | 🟡 | 205 | 1 | 0 |
| `s1f-c` | **现行**：C 口径（M36、M52、M58 都开）训练步实际用的梯度，对 g2s 6 份 15 对 | 🟡 | 205 | 1 | 0 |
