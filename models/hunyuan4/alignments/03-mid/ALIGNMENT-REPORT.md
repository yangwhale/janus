[janus](../../../../README.md) › [混元 4](../../README.md) › [mid](README.md) ｜ 对齐报告（支线）

# 混元 4 · mid（hy4-mid，5.19B，随机初始化）· 对齐报告

支线报告：只写与 [mini 主线报告](../01-mini/ALIGNMENT-REPORT.md)不同的地方；判法、开关与生产建议同 mini，N（噪声上限）、r、R、MARE 的读法见[总览细节「读法」](../../dashboard/detail.md#读法)。训练运行的代号（`tm-b`、`gm-a` 等）见 [`goldens.md`](goldens.md) 与[历代比较](../../dashboard/history.md)，第 1 步诊断运行（`tm2diag`）见[检查清单](CHECKS.md) L3 一行。hy4-mid 只换尺寸：mini 的结构放到生产的宽度与注意力（宽 6144、64 头、q_lora 2048、稠密 FFN 18432），路由专家仍 16 个 top-4；300 步，GPU B200 × 8 对 TPU v5p 2x2x2。判定来自 `tm-b`（MaxText `2c28a4f4`、janus `9c0f19a`，含 M39–M41 与 M42b；`sa_use_base2_exp` 开，早于 M50，`meta.json` 已核）；2026-11-11 起对齐配置全部关 base2，现行配置在 mid 上尚无运行。结论在 TPU 一侧 libtpu 0.0.46、jax 0.11.1 上测得（[版本](setup.md#软件版本)），换 libtpu 或 jax 要按 [`libtpu-052.md`](../04-tpu-self/libtpu-052.md) 的做法重测。

> **一句话**：300 步没有不过（过 5、待定 4，都只略超 N）；第 1 步主 loss 的差在 GPU 一侧；第 1 步逐张量梯度有 7 个张量未定责（sink 4 个、iHC 3 个）；1000 步、L1 / L2 没做。

## 1. 判定（一屏）

| 级 · 检查 | mid | 读数 |
|---|---|---|
| L0 · 学习率、权重转换往返（212 / 212）、RoPE（32 / 32） | ✅ | — |
| L3 · 第 1 步逐张量梯度对 CPU FP64（`tm2diag`） | ❌ 7 个未定责 | 最大第 3 层 FFN 前 `hc_base`：GPU 2.79e-3，TPU 4.27e-2，R 15.3；sink 4 个 R 1.23–1.47（[根源一览第 21 行](../../ROOT-CAUSES.md#r21)）；另 9 个张量 TPU 梯度文件落盘损坏、不可判 |
| L4 · 主 loss 全程 / 后段（`tm-b`，[D27](#d27)） | 🟡 / 🟡 | 带符号均值：N 4.66e-6，TPU 6.41e-6，r 1.38；后段 r 1.01 |
| L4 · 索引器 KD | 🟡 | MARE：N 3.57e-4，TPU 3.99e-4，r 1.12 |
| L4 · param_norm | 🟡 | 带符号均值：N 8.94e-8，TPU −1.60e-7，r 1.79 |
| L4 · MTP loss、grad_norm、路由偏置 | ✅ | — |

逐格读数见[支线总览](../../dashboard/other-sizes.md)；出处 [`tmb-l4`](results/tmb-l4.report.md)、[`mid-tm2-step1-grad-vs-fp64.json`](results/mid-tm2-step1-grad-vs-fp64.json)；上一代 300 步见[历代比较](../../dashboard/history.md)。

### 第 1 步主 loss 对 CPU FP64

相对差（(x − FP64) ÷ FP64，同 [`metrics.md`](../../../../docs/reference/metrics.md#逐步统计量) 的 dₜ）：GPU +1.42e-5，TPU +2.4e-7，R 0.02。差在 GPU 一侧，而且比 GPU 8 份第 1 步的极差（相对 9.4e-6）还大（推断：宽度 6144 以后 GPU 一侧 BF16 的系统偏差，真实权重截层 D1 上同样如此）。06 在 D3 上把 GPU 的第 1 步偏高拆出约 55%，没有单一来源（[06](../06-rw3/README.md#后段主-loss-偏低不是前向的差是轨迹的差) 第 2 条）；mid 上没拆。

## 2. 与 mini 不同的地方

- **sink 梯度**：mini 上只在单层探针里量过，mid 第 1 步整网上 4 个 sink 张量未定责；M42b（照 GPU 写法）的第 1 步还没在 mid 上跑。
- **主 loss、param_norm 略超**：以现行比较 `tz-off-rep` 为准，mini 的 param_norm 过、主 loss 两个窗口待定（r 1.55、1.52，见 mini 对齐报告）；mid 与真实权重截层上两项都略超。
- **对齐开关的代价**：mid 上全开比全关贵 +46.8%（全开 2.8125 s/步、全关 1.9158 s/步），其中 M13 一项 +24.4%（只关 M13 2.2610 s/步；运行名见 goldens.md），约占总代价的 60%，与 mini 的 +23.7% 一致。这三份只量步时（40 步，第 11–40 步中位数，不含 M39–M41），不做判定；运行没有登记，结果文件没有入库，数字未核。

## 3. 偏差台账

编号接 mini 与 real-weight 的 D 系列。

| # | 差别 | 机制 | 处理 |
|---|---|---|---|
| <a id="d27"></a>D27 | 300 步 L4 四项略超：主 loss 全程（带符号均值：N 4.66e-6，TPU 6.41e-6，r 1.38）与后段（r 1.01）、索引器 KD（MARE：N 3.57e-4，TPU 3.99e-4，r 1.12）、param_norm（带符号均值：N 8.94e-8，TPU −1.60e-7，r 1.79） | 未查；以现行比较 `tz-off-rep` 为准，mini 的索引器 KD、param_norm 过，主 loss 也待定；mid 的 KD、param_norm 超线推断与宽度有关；这一代运行 base2 开、早于 M50（KD 记录写法的病态见 mini [D1](../01-mini/DETAILS.md#d1)）。主 loss 对 8 份 GPU 各比一次（[`mid-l4-vs-8gpu.json`](results/mid-l4-vs-8gpu.json)）：带符号均值的 r 中位数 1.30，8 份里 7 份 > 1，是一致的 TPU 偏高，不是只对 `gm-a` 偶然超线 | 待查（[根源一览「还没查明」](../../ROOT-CAUSES.md#u-mid-l4)） |
| <a id="d28"></a>D28 | 第 1 步逐张量梯度 7 个未定责（sink 4 个 R 1.23–1.47，iHC 3 个，最大第 3 层 FFN 前 `hc_base`：GPU 2.79e-3，TPU 4.27e-2，R 15.3）；第 1 步主 loss 对 FP64 的相对差 GPU +1.42e-5，TPU +2.4e-7 | sink 同[根源一览第 21 行](../../ROOT-CAUSES.md#r21)；iHC 3 个未定责：mini 的 [D11](../01-mini/DETAILS.md#d11) 是 6 个 iHC 张量 R 1.57–5.22、已接受，接受理由没在 mid 上核，mid 最大 R 15.3，不能直接套用；主 loss 的差在 GPU 一侧（推断：宽度 6144 时 GPU 一侧 BF16 的系统偏差） | 待查（[根源一览「还没查明」](../../ROOT-CAUSES.md#u-mid-step1)） |

## 4. 覆盖缺口

- L3 的 9 个落盘损坏的张量要用重跑的第 1 步诊断重比；sink 要用带 M42b 的第 1 步再比一次。
- 只有 300 步；没有 L1 / L2；GPU 的第 1 步偏高从哪一层起没查。

## 5. 适用范围

- 只在 v5p 2x2x2（8 个设备）上跑过，TPU 一侧只用 FSDP 8；EP、TP、多切片没测。
- 随机初始化，7 层 + 1 层 MTP，路由专家 16 个；只有 300 步。
- L4 判定用的 `tm-b` base2 开、早于 M50；现行对齐配置在 mid 上没有运行。
- 尺子是 GPU 单机 8 份两两 28 对。
- TPU 程序：`tm-b` 的 `meta.json` 记 jax 0.11.1，libtpu 构建串与 06 的 `rfx-on` 相同（那份记 libtpu 0.0.46）；`tm-a` 没核。
