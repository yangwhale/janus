[janus](../../../../README.md) › [混元 4](../../README.md) › [mid](README.md) ｜ 对齐报告（支线）

# 混元 4 · mid（hy4-mid，5.19B，随机初始化）· 对齐报告

支线报告：只写与 [mini 主线报告](../01-mini/ALIGNMENT-REPORT.md)不同的地方；判法、开关与生产建议同 mini，N（噪声上限）、r、R、MARE 的读法见[总览细节「读法」](../../dashboard/detail.md#读法)。训练运行的代号（`tmc-a`、`gm-a` 等）见 [`goldens.md`](goldens.md) 与[历代比较](../../dashboard/history.md)，第 1 步运行（`tmcs-c`）见[检查清单](CHECKS.md) L3 一行。hy4-mid 只换尺寸：mini 的结构放到生产的宽度与注意力（宽 6144、64 头、q_lora 2048、稠密 FFN 18432），路由专家仍 16 个 top-4；300 步，GPU B200 × 8 对 TPU v5p 2x2x2。判定来自现行 C 口径的 `tmc-a`（L4）与 `tmcs-c`（L3 第 1 步全量抓数），MaxText `8d3d7561`、janus `7314865d`，M36、M52、M58 都开，base2 关，含 M50（`meta.json` 的 `maxtext_config_full` 对上一代 `tm-b` 只差这几项与默认关的诊断键），按 PROTOCOL 2026-11-12「C 口径在 mini 以外各尺寸」的前瞻登记判。上一代 `tm-b`（MaxText `2c28a4f4`，base2 开、早于 M50）与 `tm2diag` 的读数留在[历代比较](../../dashboard/history.md)与下面的偏差台账。结论在 TPU 一侧 libtpu 0.0.46、jax 0.11.1 上测得（[版本](setup.md#软件版本)），换 libtpu 或 jax 要按 [`libtpu-052.md`](../04-tpu-self/libtpu-052.md) 的做法重测。

> **一句话**：C 口径 300 步没有不过（过 5、待定 4，都只略超 N；主 loss 全程 r 1.01、MTP r 1.25 最大）；第 1 步主 loss 的差在 GPU 一侧；第 1 步逐张量梯度对 GPU 3 份不过（参照不足，6 份起步待补），未定责 5 个（iHC 4 个、第 4 层 sink）；1000 步、L1 / L2 没做。

## 1. 判定（一屏）

| 级 · 检查 | mid | 读数 |
|---|---|---|
| L0 · 学习率、权重转换往返（212 / 212）、RoPE（32 / 32） | ✅ | — |
| L3 · 第 1 步逐张量梯度对 CPU FP64（C 口径 `tmcs-c`，训练步梯度；GPU `gms-a … c` 3 份，参照不足） | ❌ 5 个未定责（[D28](#d28)） | 205 个张量里 174 个超 N、122 个超 2N，R > 1.2 的 13 个；超 2N 且 R > 1.2：第 3 层 FFN 前 `hc_base`（GPU 2.79e-3，TPU 3.82e-2，R 13.7，7.12 N）、第 5 层 FFN 前 `hc_scale`（R 6.16）、第 6 层 FFN 前 `hc_base`（R 1.71）、第 3 层注意力前 `hc_scale`（R 1.42）、第 4 层 sink（GPU 3.69e-2，TPU 4.44e-2，R 1.21）；落盘核对 728 个张量 sha256 全对 |
| L4 · 主 loss 全程 / 后段（C 口径 `tmc-a`，[D27](#d27)） | 🟡 / ✅ | 全程 MARE：N 1.62e-5，TPU 1.64e-5，r 1.01；后段 r 0.96 |
| L4 · MTP loss 全程 / 后段 | 🟡 / 🟡 | 带符号均值：N 1.82e-5，TPU +2.28e-5，r 1.25；后段 N 2.71e-5，TPU +3.11e-5，r 1.15 |
| L4 · 索引器 KD | 🟡 | 最大单步：N 1.269e-3，TPU 1.272e-3，r 1.003 |
| L4 · grad_norm、param_norm、路由偏置 | ✅ | r 0.85、0.73、0.37 / 0.68 |

逐格读数见[支线总览](../../dashboard/other-sizes.md)；出处 [`tmc-l4`](results/tmc-l4.report.md)、[`tmc-l0-settings`](results/tmc-l0-settings.report.md)（学习率逐步核对，过）、[`mid-c-step1-train-grad-vs-fp64.json`](results/mid-c-step1-train-grad-vs-fp64.json)（落盘核对 [`mid-c-step1-train-grad-file-check.json`](results/mid-c-step1-train-grad-file-check.json)）；上一代 `tm-b` 300 步过 5、待定 4（主 loss 全程 r 1.38、param_norm r 1.79 最大），见[历代比较](../../dashboard/history.md)。

### 第 1 步主 loss 对 CPU FP64

相对差（(x − FP64) ÷ FP64，同 [`metrics.md`](../../../../docs/reference/metrics.md#逐步统计量) 的 dₜ）：GPU +1.42e-5（第 1 步全量抓数参照，[`mid-tm2-step1-grad-vs-fp64.json`](results/mid-tm2-step1-grad-vs-fp64.json) 的 `loss_vs_gpu`），TPU C 口径 −2.67e-6（`tmc-a` 与 `tmcs-c` 第 1 步主 loss 逐位相同；对同一份 FP64 参照的主 loss 11.81886298 算，参照值记在 `mid-c-step1-train-grad-vs-fp64.json` 的 `reference`），R 0.19；上一代 `tm-b` +4.0e-6（R 0.28），再上一代 `tm-a` +2.4e-7。出处 [`mid-step1-loss-vs-fp64.json`](results/mid-step1-loss-vs-fp64.json)。差在 GPU 一侧，而且比 GPU 8 份训练运行第 1 步的极差（相对 9.4e-6，8 份各 +9.8e-6 – +1.92e-5）还大（推断：宽度 6144 以后 GPU 一侧 BF16 的系统偏差，真实权重截层 D1 上同样如此）。06 在 D3 上把 GPU 的第 1 步偏高拆出约 57%（六类简单相加，各类不一定可加），没有单一来源（[06](../06-rw3/README.md#后段主-loss-偏低不是前向的差是轨迹的差) 第 2 条）；mid 上没拆。

## 2. 与 mini 不同的地方

- **sink 梯度**：mini 上只在单层探针里量过。mid 第 1 步整网上，C 口径 `tmcs-c`（对齐 yml 含 M42b，照 GPU 写法）8 个 sink 张量都超 2N（2.04–2.63 N），R 0.97–1.21，只有第 4 层（R 1.21）一个按 R 计入未定责、MTP 层 R 1.20 刚在线内；上一代 `tm2diag`（没有 M42b）是 R 1.09–1.47、4 个未定责。
- **主 loss、param_norm 略超**：mid 只有 300 步（GPU 8 份 28 对尺子，`tm-b` base2 开），两项都略超（r 1.38、1.79）。同窗口、同一种尺子对照 mini：以现行比较 `tc-a`（C 口径）为准，它的前 300 步（`t1zc300-l4`）主 loss 全程 0.87、后段 0.81、param_norm 0.18 都过，只有 grad_norm 1.01 待定。C 口径之前的 `tz-off-rep`（`t1zo300-l4`）是主 loss 全程 1.10 待定、后段 0.78 过、param_norm 0.57 过。所以「mini 同窗口主 loss 也略超」只在 C 口径之前成立，现行配置上 mini 前 300 步主 loss 已过。`tc-a` 与 mid 这一代差五处：base2（mini 关、mid 开）、M50（只改 KD 记录值，mini 有、mid 没有）、C 口径的 M52 与 M58（mini 开；mid 用的 MaxText 里还没有这两个开关）、MaxText 提交（`8d3d7561` 对 `2c28a4f4`）、打进 Pod 的 janus 程序（`3ca7af66` 对 `9c0f19af`，见两边 setup）。真实权重截层上两项也略超（base2 开、C 口径之前的运行）。
- **对齐开关的代价**：mid 上全开比全关贵 +46.8%（全开 2.8125 s/步、全关 1.9158 s/步），其中 M13 一项 +24.4%（只关 M13 2.2610 s/步），约占总代价的 60%，与 mini 的 +23.7% 一致。这三份只量步时（40 步，第 11–40 步中位数，不含 M39–M41），不做判定；运行没有登记，结果文件没有入库，数字未核。

## 3. 偏差台账

编号接 mini 与 real-weight 的 D 系列。

| # | 差别 | 机制 | 处理 |
|---|---|---|---|
| <a id="d27"></a>D27 | 300 步 L4 四项略超。现行 C 口径 `tmc-a`：主 loss 全程（MARE：N 1.62e-5，TPU 1.64e-5，r 1.01）、MTP loss 全程与后段（带符号均值：N 1.82e-5，TPU +2.28e-5，r 1.25；后段 r 1.15）、索引器 KD（最大单步 r 1.003）；主 loss 后段、param_norm 过。上一代 `tm-b`（base2 开、早于 M50）：主 loss 全程（带符号均值：N 4.66e-6，TPU 6.41e-6，r 1.38）与后段（r 1.01）、索引器 KD（MARE r 1.12）、param_norm（带符号均值：N 8.94e-8，TPU −1.60e-7，r 1.79） | 未查；以现行比较 `tc-a` 为准，mini 同窗口（前 300 步，`t1zc300-l4`，同样是 8 份 28 对尺子；C 口径、base2 关）主 loss 全程 0.87、后段 0.81、索引器 KD 0.94、param_norm 0.18 都过，只有 grad_norm 1.01 待定（C 口径之前的 `tz-off-rep` 同窗口主 loss 全程 1.10 待定）；mini 现行与 mid 这一代还差 base2、M50、C 口径的 M52 与 M58、MaxText 提交几处，其中 M50 本身就改 KD 记录值，所以 KD 这一项两边口径不同，不能拿来比；mid 的 param_norm 超线推断与宽度有关；这一代运行 base2 开、早于 M50（KD 记录写法的病态见 mini [D1](../01-mini/DETAILS.md#d1)）。主 loss 对 8 份 GPU 各比一次（[`mid-l4-vs-8gpu.json`](results/mid-l4-vs-8gpu.json)）：带符号均值的 r 中位数 1.30，8 份里 7 份 > 1，是一致的 TPU 偏高，不是只对 `gm-a` 偶然超线 | 待查（[根源一览「还没查明」](../../ROOT-CAUSES.md#u-mid-l4)） |
| <a id="d28"></a>D28 | 第 1 步逐张量梯度：现行 C 口径 `tmcs-c`（训练步梯度，对 GPU `gms-a … c` 3 份，参照不足）5 个未定责：iHC 4 个（第 3 层 FFN 前 `hc_base`：GPU 2.79e-3，TPU 3.82e-2，R 13.7，7.12 N；第 5 层 FFN 前 `hc_scale` R 6.16；第 6 层 FFN 前 `hc_base` R 1.71；第 3 层注意力前 `hc_scale` R 1.42）与第 4 层 sink（R 1.21）。上一代 `tm2diag` 7 个（sink 4 个 R 1.23–1.47，iHC 3 个，最大同一个第 3 层 FFN 前 `hc_base` R 15.3）、9 个文件损坏不可判。第 1 步主 loss 对 FP64 的相对差 GPU +1.42e-5，TPU C 口径 −2.67e-6（R 0.19） | sink 同[根源一览第 21 行](../../ROOT-CAUSES.md#r21)，M42b 进了对齐配置后只剩第 4 层略超；iHC 同 mini 的 [D11](../01-mini/DETAILS.md#d11)：mini 的 C 口径第 3 层 FFN 前 `hc_fn`（R 1.43）已查到大部分来自 TPU 自己的离散选择（路由与 DSA top-k），其余连续误差与 CPU 上 GPU 写法同量级（[mini D11 hc_fn 一段](../01-mini/DETAILS.md#d11-hcfn-c)）；mid 的 R 13.7 远大于 mini，同一机制能不能解释没核；主 loss 的差在 GPU 一侧（推断：宽度 6144 时 GPU 一侧 BF16 的系统偏差） | 待查（[根源一览「还没查明」](../../ROOT-CAUSES.md#u-mid-step1)）；GPU 第 1 步参照补到 6 份后重判 |

## 4. 覆盖缺口

- L3 的 GPU 第 1 步参照只有 `gms-a … c` 3 份（2026-11-12 起 6 份起步），现行 C 口径的判档写作参照不足、待补。上一代 `tm2diag` 的 9 个取回损坏张量不再补比：现行 `tmcs-c` 的落盘核对 728 个张量 sha256 全对。
- L0 只有学习率、权重转换往返、RoPE；跨侧初始权重、每 rank 输入指纹、配置逐项对照没做（与 B、D1 一样）。
- 只有 300 步；没有 L1 / L2；GPU 的第 1 步偏高从哪一层起没查。

## 5. 适用范围

- 只在 v5p 2x2x2（8 个设备）上跑过，TPU 一侧只用 FSDP 8；EP、TP、多切片没测。
- 随机初始化，7 层 + 1 层 MTP，路由专家 16 个；只有 300 步。
- L4 判定用的 TPU 运行是现行 C 口径的 `tmc-a`，1 份；TPU 一侧同配置重跑没做。
- 尺子是 GPU 单机 8 份两两 28 对。
- TPU 程序：`tmc-a`、`tmcs-c`、`tm-b` 的 `meta.json` 记 jax 0.11.1，libtpu 构建号 `cl/963683785`，即镜像自带的 0.0.46（[`jit-miscompile-probe-v5p.json`](../../results/jit-miscompile-probe-v5p.json) 读到这一构建的 pip 版本）；`tm-a` 没核。
