[janus](../../../../../README.md) › [混元 4](../../../README.md) › [TPU 自比](../README.md) › [影响逐件](README.md) ｜ 关掉编译器的省舍入

# 关掉编译器的省舍入（excess precision）

## 改了什么

XLA 开关 `xla_allow_excess_precision=false`：不许编译器把本该舍成 BF16 的中间结果留在 FP32。默认是开着的。

## 影响多大

libtpu 0.0.46 上，v5p 曲线最差 r 13.28（梯度范数的带符号均值），判「超」；v7x 上 r 3.43。

## 问题出在哪

v5p 第 1 步词嵌入梯度离 FP64 1.09e-1，是 golden / GPU 里较远那个的 26.13 倍；另有第 4 层 MoE 路由门的权重梯度 1.20 倍。

v7x 第 1 步词嵌入梯度没有变远（1.11 倍，在定责线以内），曲线照样超。

## 为什么

v5p：词嵌入的反向是对 BF16 表的 scatter-add，关掉 excess precision 后每加一次就舍一次 BF16（[根源一览第 22 行](../../../ROOT-CAUSES.md#r22)；单设备 scatter 测试逐位对上，见[改动台账](../../../CHANGES-LEDGER.md) J40）。

v7x 有两件事，和 v5p 不同：

1. **词嵌入不受影响，因为 v7x 上这次 scatter 交给了 SparseCore。** 两代芯片训练步 HLO 里词嵌入梯度的 scatter 写法相同（`bf16[120832,128]`、按 token 号排好序），但 v5p 编成 TensorCore 上的 `scatter_custom_fusion`，v7x 编成 `scatter_offload_custom_fusion`（SparseCore 卸载，带 scratchpad）。v7x 上关掉省舍入，词嵌入梯度离 FP64 4.25e-3 → 4.71e-3；v5p 上 4.0e-3 → 1.09e-1。
2. **曲线超的是梯度范数的带符号均值（r 3.43），来自主干全部权重梯度一起变小。** 第 1 步梯度范数：FP64 6.35944，v7x 默认 6.36051（+1.7e-4），关掉后 6.35669（−4.3e-4），GPU 6.36361（+6.5e-4）；300 步里关掉的梯度范数平均比默认小 3.4e-4，7.4e-1 的步数都是小的。按参数类别看（平方和对 FP64），关掉后各层注意力投影从 +6e-4 变成 −5e-4、稠密 MLP 从 −8e-4 变成 −2.7e-3、专家权重也变小，而输出头、MTP 层、最后一层 RMSNorm 不变（`hc_head` 反而回到 FP64 的值：v7x 默认时它是偏的，见[关小矩阵乘改写](03-small-matmul-rewrite.md)）；第 6 层已经变小，到第 0 层略多。v5p 上关掉也有同样的缩小（被词嵌入梯度的 R 26.1 盖住了）。所以范数离 FP64 的偏差从正变负，大小与 GPU 的偏差同一量级；带符号均值的尺子是 GPU 重跑两两之差，而 GPU 几份重跑的偏差同向（第 1 步 `g3-a` 6.36361、`g3-b` 6.36347，都是 +6.5e-4 左右），两两相减就抵消了，系统性的 3e-4 于是显成 3.4 倍。

   **定位到超连接（iHC）这一段。** 下面这些消融（Pallas 那组、`ts-n2-xfg-noexc`、`ts-n2-k1` / `k2`、`ts-n3-dlo-prodround`）都在 v5p 2x2x2、libtpu 0.0.46 上跑，只抓第 1 步（结果文件不记芯片与版本，按各运行目录的 meta.json 核过）。运行登记见 [`runs.yaml`](../runs.yaml)。所以机理是在 v5p 上定位的，v7x 上只看到同样的现象（主干梯度整体变小），没在 v7x 上重做这些消融。让 iHC 改走 Pallas 内核（`use_ihc_pallas_kernel=true`，全程 FP32、手写反向，不受编译器省舍入影响）后，关掉省舍入的主干缩小基本消失：注意力 `o_proj` 平方和对 FP64 +5.4e-4 → +4.0e-4、稠密 MLP `down_proj` −1.09e-3 → −1.06e-3（同为 Pallas 内核时开着 / 关掉省舍入）；而走 XLA 的 iHC 是 +6.0e-4 → −2.7e-4、−9.8e-4 → −2.2e-3（v5p）。训练步 HLO 里，关掉省舍入后 iHC 反向多了一处显式舍入：4 路流的梯度在每层由门控与流之和两路贡献相加（`add_any`，`f32[1,4096,4,1024]`），默认时这个和留在 FP32，关掉后先舍成 BF16。但它不是缩小的来源：让门控也吃同一份 FP32 的流（测试分支 `486ac74e`，这处舍入在 HLO 里消失了），关掉省舍入照样缩小（`o_proj` −1.9e-4、`down_proj` −2.2e-3，`ts-n2-xfg-noexc`）。

   已排除：梯度累积（累积 1 照样缩小）、重算（不重算照样缩小）、RMSNorm 乘法的精度（那 4 处 einsum 改 HIGHEST 后与不改逐值相同）、超越函数近似（E4 回到 TPU 默认近似不缩小）、带 FP32 累加的 dense 输出（开着省舍入、只把这些输出强制舍成 BF16，不缩小）。关掉省舍入后编译器一共多插了 584 个 reduce-precision（v7x；v5p 673 个，默认时一个都没有），多数落在带 FP32 累加的 dense 层，但它们不是缩小的来源。所以缩小在 iHC 这一段、但不在编译器插的那处 reduce-precision 上。再拆开 iHC 的前后两半（都是测试分支，只作消融）：前半（门控 + 加权流之和）走 Pallas、后半（残差更新 `out = x + post·layer_out` 及其反向）走 XLA 时，关掉省舍入照样缩小（`o_proj` +5.3e-4 → −1.9e-4、`down_proj` −1.13e-3 → −2.18e-3，`ts-n2-k1`）；前半走 XLA、后半走 Pallas 时不缩小，反而略大（+4.3e-4 → +7.9e-4、−1.29e-3 → −2.5e-4，`ts-n2-k2`）。所以缩小来自 XLA 编出来的 iHC 残差更新那一段。这一段开着与关掉省舍入时训练步 HLO 里源码在 `mhc.py` 的指令逐项计数相同，差别在 TPU 后端。

   **后端里多出的那次舍入（LLO）**：把这一段写成最小复现（batch 1、seq 4096、4 路流、d 1024，前向 + 反向），在 CPU 上对 v5p 做 AOT 编译，打开后端 dump（`LIBTPU_INIT_ARGS="--xla_jf_dump_to=… --xla_jf_dump_llo_text=true"`），开、关省舍入两份优化后 HLO 逐字相同。最终指令（`final_bundles`）只有一颗 kernel 不同：反向里算分支输出梯度 d layer_out 的那颗。它在 HLO 里是「FP32 乘积 post_k·g_k → 按 k 求和 → 输出 BF16」一颗 reduce。开着省舍入：乘积不舍，FP32 累加，最后舍一次。关掉：每条 `vmul.f32` 的结果先过一条 `vpack.i.bf16` 舍到 BF16 精度，再进 `vadd.f32` 累加（静态 256 条，开着时 0 条；`vadd.f32`、`vmul.f32` 等其余条数相同）。对 x、post 的梯度那两颗逐字相同。这也解释了 Pallas 消融：Pallas 的残差更新反向在核里用 FP32 算 d layer_out，不经过这颗 reduce。

   **TPU 上验证**：开着省舍入，只把这一处照后端的做法改写（测试分支 `f4966773`：自定义反向，d layer_out 的每个乘积先 `reduce_precision` 成 BF16 再累加），第 1 步主干梯度的缩小大部分复现：稠密 MLP `down_proj` 平方和对 FP64 −9.8e-4 → −2.05e-3（关掉省舍入是 −2.22e-3）、`up_proj` −8.2e-4 → −1.88e-3（−2.22e-3）、注意力 `o_proj` +6.0e-4 → +0.8e-4（−2.7e-4）、`kv_b_proj` +5.3e-4 → −0.0e-4（−3.7e-4），梯度范数 6.36028 → 6.35851（`ts-n3-dlo-prodround`）。MLP 几乎全部复现，注意力复现了约 2/3；剩下的那部分没定位（最小复现里关掉省舍入后，两颗余切求和的 kernel 也多出成批的 `vadd → vpack.i`，可能是它们）。

   为什么「每个乘积先舍再加」会让梯度系统性变小，没有查清：单算子的 numpy 仿真里这一处只让 d layer_out 离 FP64 远约 1.4 倍（1.66e-3 → 2.36e-3），对 FP64 的投影没有系统性缩小，所以缩小是这次舍入经分支反向往下传以后才显出来的（推断）。

   另外排除了：分支输出、分支输入梯度的显式舍入（开着省舍入时强制舍，与 golden 逐值相同），门控的 einsum 精度改 default（照样缩小）。

## 换 libtpu 版本

在 libtpu 0.0.52 上重测（V5，v5p，对 0.0.52 的 golden `ts-g52-a` 比；[`libtpu-052.md`](../libtpu-052.md)）：

| | 0.0.46 | 0.0.52 |
|---|---|---|
| 曲线最差 r | 13.28 | 12.56 |
| 第 1 步词嵌入梯度 R | 26.13 | 26.11 |

这两个版本、v5p 这个配置下结论相同：关掉就超，变远的是同一个张量，倍数几乎一样。离线编译看到 FSDP 那次 scatter 两个版本都排序、都标 `indices_are_sorted=true`。v7x 上关 excess precision 没在 0.0.52 上重测：v7x 的机理在后端 kernel 里，正是会随 libtpu 变的那一类，换版本要重测。

## 能不能忍

不能。

## 怎么修

别关。

就算要关，v5p 上打开 `embedding_grad_fp32` 能把词嵌入那一项救回来：第 1 步词嵌入梯度 R 26.1 → 1.06（根源一览第 22 行，在 mini 对齐配置上量的；曲线没测）。v7x 上关掉后超的是 iHC 造成的主干梯度变小，`embedding_grad_fp32` 管不到。

## 代价

关掉本身多花 +1.4%（v7x −0.3%），保持默认反而省。

## 坑

- v7x 上第 1 步每个张量都没变远，只有范数整体小了 4e-4：逐张量定责看不出这种同向的小偏差，要另看梯度范数与按类别的平方和。
- 只在 FSDP 8（v5p、v7x）上量过关 excess precision。纯数据并行 + 关 excess precision 没有单独测；[案例 14](../../../cases/14-ddp-fsdp-embedding.md) 里拿来对照的纯数据并行（词嵌入梯度离 FP64 1.51e-2）是默认开着的。

## 一句话

关掉省舍入：v5p 上词嵌入梯度每加一次就舍一次，R 26.1（离 FP64 是 GPU 的 26 倍、golden 的 27 倍）；v7x 上词嵌入走 SparseCore 不受影响，但主干梯度整体小 4e-4，梯度范数曲线超出。主干缩小的机理在 v5p 上定位到：iHC 残差更新反向里 d layer_out 的乘积被后端逐个舍成 BF16；v7x 上现象相同，同一机制是推断。别关。

## 出处

- [实验矩阵「C 编译开关：在哪一处开始不同」](../MATRIX.md#c-编译开关在哪一处开始不同)
- 结果文件：[`ts-c-noexcess-vs-golden.json`](../results/ts-c-noexcess-vs-golden.json)、[`ts-h-v7-noexcess3-vs-ts-h-v7b.json`](../results/ts-h-v7-noexcess3-vs-ts-h-v7b.json)
- 第 1 步梯度平方和按类别、按层：[`night2-noexcess-grad-sq.json`](../results/night2-noexcess-grad-sq.json)（`analysis/step1_grad_sq_by_class.py`）
- 最小复现的 LLO 指令计数：[`night3-llo-ihc-counts.json`](../results/night3-llo-ihc-counts.json)（`analysis/ihc_llo_probe.py`，jax 0.11.1、libtpu 0.0.46，v5p:2x2x1 AOT）
