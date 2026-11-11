[janus](../../../../../README.md) › [混元 4](../../../README.md) › [TPU 自比](../README.md) › [影响逐件](README.md) ｜ 关小矩阵乘改写

# 关小矩阵乘改写

## 改了什么

编译开关 `xla_tpu_enable_dot_strength_reduction=false`：不让编译器改写小的矩阵乘。默认开着。

## 影响多大

libtpu 0.0.46 上，曲线最差 r 0.94（梯度范数的 MARE），曲线在噪声内。

## 问题出在哪

第 1 步有两处梯度变远：

| 张量 | 离 FP64：关掉后 | 默认（golden） | GPU |
|---|---|---|---|
| MTP 层 `enorm` 权重 | 2.60e-3 | 1.85e-3 | 1.57e-3 |
| 最后收拢超连接的 `hc_head` 三个参数 | 1.25–1.29 | 2.7e-2–7.1e-2 | 1.3e-2–6.6e-2 |

`hc_head` 那三个张量梯度很小（1e-5 量级），关掉后整体偏了一个常数：`hc_head_base` 第 1 步 FP64 是 [6.9, 8.1, 9.0, 11.2]e-6，默认 [6.0, 7.8, 9.6, 11.2]e-6，关掉后 [−5.1, −3.3, −1.5, 0.2]e-6，前三个符号反了。判档时它们归在 iHC 小参数里单列（01-mini D11），所以只有 `enorm` 被记成「离精确值更远」；`hc_head` 实际偏得更多。

## 为什么

RMSNorm 乘 scale 的那一步，MaxText 写成 `einsum("...k,k->...k", normed, scale)`，没给精度，用默认精度。这在 XLA 里是一个矩阵乘（`dot_general`）：

- 默认开着改写：编译器把这种输出只有一个向量的矩阵乘改成逐元素乘再求和，在 FP32 里算。golden 训练步 HLO 里 scale 梯度是 `reduce(multiply(convert(dy), x·rsqrt))`，归一化后的值留在 FP32。
- 关掉改写：它留成 MXU 上的卷积，默认精度只做一遍 BF16，两个操作数先舍成 BF16。关掉后的 HLO 里新增 55 个卷积，其中 37 个是这一类（RMSNorm 的前向乘法、scale 梯度、对归一化值的梯度），操作数全是 `bf16`。

于是归一化后的值在乘 scale 之前、在算 scale 梯度之前都多舍了一次 BF16。`enorm` 与 `hc_head` 的梯度是大量项的相互抵消，多一次舍入就显出来（`hc_head` 经过最后那层 RMSNorm 的反向）。

验证：关掉改写，同时把这 4 处 einsum 的精度改成 HIGHEST（测试分支 `test/68-n2-norm-highest`，MaxText `880e3cd5`，只作消融），第 1 步 `enorm` 回到 1.85e-3、`hc_head_base` 回到 [6.1, 7.8, 9.6, 11.2]e-6，与 golden 相同（`ts-n2-nosrb-hi`）。

## 换 libtpu 版本

在 libtpu 0.0.52 上重测（V6，v5p，对 `ts-g52-a` 比；[`libtpu-052.md`](../libtpu-052.md)）：曲线最差 r 1.00（0.0.46 是 0.94）；第 1 步变远的仍是 MTP `enorm`，R 1.35（0.0.46 是 1.40）。这两个版本、这个配置下结论相同。`hc_head` 在 0.0.52 上不再偏：`hc_head_base` 离 FP64 5.2e-2，0.0.52 的 golden 5.0e-2。

## 能不能忍

能忍：曲线在噪声内。但关了没有好处。

## 怎么修

别关。真要关，把 RMSNorm 里这几处 einsum 的精度设成 HIGHEST（消融里已验证能恢复）。

## 代价

关掉本身多花 +10.4%。

## 坑

- 同样的 `hc_head` 偏差在 D3（einsum 改广播乘，含只改末层、只改末层以外两种拆法）和 v7x 默认配置里也出现（都是 libtpu 0.0.46），对 FP64 的差几乎是同一个常数（四个元素都约 −11e-6）；关掉省舍入后都回到 FP64 附近。这个梯度的大部分是末层 RMSNorm 的 ε 项（FP64 的 6.87–11.18e-6 里 8.8–9.2e-6），偏差约等于 ε 项的 1.26 倍，所以四个元素一起被推过零。根因定到输出头反向那颗卷积 kernel：偏的程序里末层 RMSNorm 反向的每 token 行和 S 用的是 kernel 里没舍的 dy，主项用的是写进显存的 BF16 dy；在 TPU 上给这份余切加 `optimization_barrier` 就回到 golden（`ts-n3-d3final-barrier`）。这条因果只在 v5p、只改末层的程序上上卡验证过。另外两个程序离线核过，都没加 barrier 上卡，同一机制仍是推断：
  - v7x 默认（0.0.46）：编了 LLO，主头卷积 kernel 里有 S，`vpack.i.bf16` 0 条，关 excess precision 后 256 条，与 v5p 偏的程序同型；
  - 关小矩阵乘改写：只核了 HLO，S 也并进了卷积 kernel。

  写进显存那次是「截断式」舍入、所以幅度对得上，也是推断。
- libtpu 0.0.52 上，v7x 默认（V16，`hc_head_base` 1.267 → 0.014）和关小矩阵乘改写（V6）都不再偏；v7x 纯数据并行不累积（V17）却偏到 0.56。V17 离线核过，是同一型（[`hc-head-dgrad-fusion.json`](../results/hc-head-dgrad-fusion.json)）：
  - 主头卷积 kernel 里并进了 S；
  - LLO 里 `vpack.i.f32.bf16` 0 条，关 excess precision 后 768 条，其余指令条数不变；
  - 逐元素差 −4.26、−5.51、−5.06、−5.13 e-6，同号、接近常数，约为 0.0.46 那几次的一半。

  0.0.52 没修好这类偏差，只是 v7x FSDP 8、V6 这些程序没被切成「S 在卷积 kernel 里」。会不会切成这样，随版本和切分变，0.0.46 的纯数据并行不累积（H4d）就没有。

  判据：优化后 HLO 里主头卷积 kernel 有 S、且 excess precision 开着。15 次存了 HLO 的运行里，满足的 5 次都偏（0.56–1.27），其余 10 次都在 FP64 附近。离线可查：`analysis/hc_head_dgrad_fusion.py`。V17 的幅度为什么是一半、要不要上卡加 barrier 验证，没做。详见[实验矩阵「hc_head：被 iHC 豁免盖住的三个张量」](../MATRIX.md#hc_head被-ihc-豁免盖住的三个张量)。
- 工具对 iHC 小参数单列、不挡判档（`ihc_accepted`，定义见 [MATRIX「iHC 豁免」](../MATRIX.md#ihc-豁免定义与受影响的运行)），原意是放过 1.5–2 倍的小偏差；这里偏到 18–46 倍、符号反了也被单列。收窄豁免的提议已登记在 PROTOCOL 变更记录（2026-11-11），豁免是否收窄待定；看判档时要看 `ihc_accepted` 里的倍数。

## 一句话

关小矩阵乘改写让 RMSNorm 的乘法改走 MXU 单遍 BF16，归一化值多舍一次，`enorm` 和 `hc_head` 的梯度变远；曲线在噪声内，但更慢，别关。

## 出处

- 结果文件：[`ts-c-nosrb-vs-golden.json`](../results/ts-c-nosrb-vs-golden.json)
- 消融：`ts-n2-nosrb-hi`（关改写 + HIGHEST），第 1 步对 FP64 的数字见 [MATRIX「C15 关小矩阵乘改写」](../MATRIX.md#c15-关小矩阵乘改写改了哪一处)
