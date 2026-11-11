[janus](../../../../README.md) › [混元 4](../../README.md) › [mid](README.md) ｜ 检查清单

# 混元 4 · mid（hy4-mid，5.19B，随机初始化）· 检查清单

这一页回答：hy4-mid 比了哪些量、用什么门槛判、现在各是什么结果。页面由 `models/hunyuan4/dashboard/build.py` 按 [`dashboard/checks.yaml`](dashboard/checks.yaml) 从结果文件生成，数字和判定都不手写。

- 判定同 T1（[mini 的检查清单](../01-mini/CHECKS.md)「怎么读」）：L4 每条序列算带符号均值、MARE、最大单步，各自对 GPU 8 份两两 28 对的噪声上限 N 判，≤ N 过、> 2N 不过；格子写最差那个判定量的 TPU 实测与倍数。r 只量两边差多大，不定责（[总览细节「读法」](../../dashboard/detail.md#读法)）。
- TPU 配置与 hy4-mini 的 T1 配置除 `model_name` 外逐键相同（[`config-alignment.md`](config-alignment.md)）。L4 的 `tm-b` 是 base2 开、早于 M50 的运行；2026-11-11 起的现行对齐配置在 mid 上尚无运行。

## 规模

| 级 | 目的 | 检查项 | 检查条数 | 张量级比较 | 其他比较单元（hy4-mid · tm-b（300 步，base2 开、早于 M50）） | hy4-mid · tm-b（300 步，base2 开、早于 M50） 判定 | hy4-mid · L0 取 tm-a 一代（同一份检查点）；L3 tm2diag 判定 |
|---|---|---|---|---|---|---|---|
| [L0](#l0--地基设置与权重) | 设置与权重逐位 | 3 | 1 / 5 | 0 / 244 | 300 步 | ✅ 1 | ✅ 3 |
| [L3](#l3--整网第-1-步) | 第 1 步逐张量梯度对 GPU 噪声，超出的对 CPU FP64 参照定责 | 1 | 0 / 1 | 0 / 196 | — | — | ❌ 1 |
| [L4](#l4--多步训练) | 300 步逐步曲线在 GPU 重跑的噪声内 | 7 | 9 / 0 | — | 1,900 步 | ✅ 3 🟡 4 | — |
| 合计 |  | 11 | 10 / 6 | 0 / 440 |  |  |  |

## L0 · 地基：设置与权重

L0 共 3 项检查，检查条数 1 / 5。GPU 一侧 8 份运行的载入自检与清单 sha256 见 [`hy4-mid-gpu-load-check.json`](results/hy4-mid-gpu-load-check.json)（不是 janus 结果格式，不进本表）。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | hy4-mid · tm-b（300 步，base2 开、早于 M50） | hy4-mid · L0 取 tm-a 一代（同一份检查点）；L3 tm2diag |
|---|---|---|---|---|---|---|
| 设置核对：逐步学习率<br>`L0.settings.lr` | 两边每一步记下的学习率都等于登记的常数 1e-5 | 300 步<br>比较单元：300 步<br>聚合：逐步比，任一步超出容差就不过 | rtol ≤ 0、abs_ulp ≤ 0｜参考：无 | 生产程序（逐步记录） | [两边每一步都等于登记值 1e-05 ✅](results/tmb-l0-settings.report.md) | [两边每一步都等于登记值 1e-05 ✅](results/tm-l0-settings.report.md) |
| 权重转换往返与 dtype 清单<br>`L0.ckpt.convert.bitwise`<br>`L0.ckpt.convert.dtype`<br>`L0.ckpt.source.dtype` | TPU 检查点反向取回逐张量对 HF 清单逐位，dtype 清单全符合 | 212 个张量<br>比较单元：212 个参数张量、424 项 dtype<br>聚合：逐张量比 sha256，数不同的个数；逐项核 dtype，数不合规的个数 | 逐位｜参考：无 | 离线核对（CPU，检查点文件，MaxText 469dbb85） | — | [0 / 212 不同 ✅](results/hy4-mid-l0-ckpt-convert.report.md) |
| RoPE 列去交错<br>`L0.ckpt.rope_deinterleave.bitwise` | 带 RoPE 的列正向去交错、逆置换逐位回到原件 | 32 个张量<br>比较单元：32 个张量<br>聚合：逐条规则核，数不满足的条数 | 只报，不判｜参考：无 | 离线核对（CPU，检查点文件，MaxText 469dbb85） | — | [32/32 个带 RoPE 列的张量：正向 = 偶数位接奇数位、逆置换逐位回到原件 ✅](results/hy4-mid-l0-rope.results.json) |

## L3 · 整网第 1 步

L3 共 1 项检查。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | hy4-mid · tm-b（300 步，base2 开、早于 M50） | hy4-mid · L0 取 tm-a 一代（同一份检查点）；L3 tm2diag |
|---|---|---|---|---|---|---|
| 第 1 步逐张量梯度<br>`L3.step1.grads.vs_gpu` | TPU `tm2diag`（MaxText `3741e818`，当时的开关，含 M39–M41，没有 M42b、M50，base2 开）对 GPU 第 1 步参照 `gms-*` 三份；超出 GPU 噪声的张量对 CPU FP64 整网重算定责（规则同 T1，PROTOCOL 第 5 节） | 205 个张量（HF 名字；路由专家只比 GPU 全量抓数里有的 rank 0 本地 0–3 号：全量抓数只由 rank 0 写盘，专家并行 4 时它本地是 16 个里的 0–3 号；`--dump-all-experts` 只去掉「本地专家只留前两个（weight0、weight1）」的筛选，要收齐其余 12 个得加 `--gather-experts`，`gms-*` 没加（读 `train_hy4mg.py` 与 `janus/dump/megatron.py`，没核抓数文件清单））<br>比较单元：196 个张量<br>聚合：逐张量算 r 与 R；有超 2N 而 R > 1.2 的判不过，只有 1–2N 的判待定 | 超出 N（或 N = 0 且不低于分辨率下限）而未定责的张量数 ≤ 0｜参考：无 | 第 1 步诊断（TPU tm2diag 对 GPU 三份第 1 步参照 gms）＋CPU FP64 整网重算 | — | [7 ❌](results/mid-tm2-step1-grad-vs-fp64.json) |

非 ✅ 格子的原因：

- **第 1 步逐张量梯度**（hy4-mid · L0 取 tm-a 一代（同一份检查点）；L3 tm2diag ❌）：TPU 梯度文件 9 个张量取回损坏（文件平方和对不上设备上算的 per_tensor.sq。当时按 256 MiB 分块并行取回、只核总字节数，有的块中途断流，本机文件留下没写入的空洞，读出来是 0；9 个空洞都止于块边界，见 `tm2diag-grad-file-holes.json`；mid 没有用同一份代码重跑诊断，这 9 个仍不可判（之后的 `tm3diag` 文件完好，但不是同配置重跑：MaxText `2306c7a5`，含 M42；比它要重算 CPU FP64 参照梯度、重新取回 GPU 三份全量抓数，还没做）；第 0 层 gate_proj、第 4 层 o_proj、第 5 / 6 层与 MTP 层输出门、第 5 层与 MTP 层专家 gate_up、输出头、词嵌入），不可判。其余 196 个的 rel_L2 对 FP64 中位数 GPU 2.48e-2、TPU 2.41e-2，R 0.98，误差余弦中位数 0.54。未定责 7 个：第 3 层 FFN 前 `hc_base`（7.98 N）GPU 2.79e-3、TPU 4.27e-2，R 15.3；第 3 层注意力前 `hc_base`（3.79 N）GPU 1.06e-2、TPU 5.81e-2，R 5.46；第 2 层 FFN 前 `hc_scale`（2.34 N）GPU 4.67e-2、TPU 1.18e-1，R 2.53；第 0 / 2 / 4 层与 MTP 层注意力的 sink（1.73–2.22 N）GPU 3.47e-2 / 3.70e-2 / 3.69e-2 / 4.65e-2、TPU 4.27e-2 / 4.59e-2 / 4.78e-2 / 6.83e-2，R 1.23 / 1.24 / 1.30 / 1.47。sink 一组 8 个 R 1.09–1.47、余弦 0.49–0.75，与上一代 `tmdiag`（没有 M39–M41）持平，是 TPU 一侧 sink 梯度的系统差（推断），待查；iHC 3 个未定责（mini 的 D11 是 6 个 iHC 张量 R 1.57–5.22、已接受，接受理由没在 mid 上核），待查。上一代 `tmdiag` 的判定（未定责 8 个）在 `mid-step1-grad-vs-fp64.json`。

## L4 · 多步训练

L4 共 7 项检查，检查条数 9 / 0。窗口与口径同 T1（主 loss 第 1–300 步与第 101–300 步，grad_norm 以参照作分母（2026-11-10 起，原为两边均值），param_norm 比精确值）。

| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | hy4-mid · tm-b（300 步，base2 开、早于 M50） | hy4-mid · L0 取 tm-a 一代（同一份检查点）；L3 tm2diag |
|---|---|---|---|---|---|---|
| 主 loss<br>`L4.loss` | 全程逐步主 loss 的相对差 | `lm_loss`，第 1–300 步<br>比较单元：300 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（带符号均值 的 N = 4.66e-6）｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | 生产程序（逐步记录） | [6.41e-6（带符号均值，1.38×N） 🟡｜参✅](results/tmb-l4.report.md) | — |
| 主 loss 后段<br>`L4.loss.tail` | 第 101 步起 | `lm_loss`，第 101–300 步<br>比较单元：200 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（带符号均值 的 N = 4.91e-6）｜参考：首步差 ≤ 2e-4、带符号均值 ≤ 2e-4、无 NaN / Inf | 生产程序（逐步记录） | [4.95e-6（带符号均值，1.007×N） 🟡｜参✅](results/tmb-l4.report.md) | — |
| MTP loss<br>`L4.mtp_loss`<br>`L4.mtp_loss.tail` | 逐步 MTP loss（MaxText 口径 `mtp_loss_mt`），全程与后段 | `mtp_loss_mt`，全程与第 101 步起<br>比较单元：500 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 2.22e-5）｜参考：无 | 生产程序（逐步记录） | [2.12e-5（MARE，0.956×N） ✅](results/tmb-l4.report.md) | — |
| 索引器 KD loss<br>`L4.indexer_loss` | 逐步索引器 KD loss | `indexer_loss`，全程<br>比较单元：300 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 3.57e-4）｜参考：无 | 生产程序（逐步记录） | [3.99e-4（MARE，1.12×N） 🟡](results/tmb-l4.report.md) | — |
| grad_norm<br>`L4.grad_norm` | 逐步裁剪前全局梯度范数；分母用参照（2026-11-10 起，原为两边均值） | `grad_norm`，全程<br>比较单元：300 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（MARE 的 N = 7.28e-4）｜参考：无 | 生产程序（逐步记录） | [6.77e-4（MARE，0.9307×N） ✅](results/tmb-l4.report.md) | — |
| param_norm<br>`L4.param_norm` | 逐步更新后参数范数，两边比精确值 | `param_norm_fp64` 对 `param_norm_exact`，全程<br>比较单元：300 步<br>聚合：每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判 | ≤ N 过，> 2N 不过（带符号均值 的 N = 8.94e-8）｜参考：带符号均值 ≤ 5e-6、最大单步 ≤ 1.25e-5 | 生产程序（逐步记录） | [−1.60e-7（带符号均值，1.79×N） 🟡｜参✅](results/tmb-l4.report.md) | — |
| 路由偏置曲线<br>`L4.router.bias_mean`<br>`L4.router.bias_std` | 逐步每层路由偏置的均值、标准差 | 第 1–6 层与 MTP 层<br>聚合：每步每层一个统计量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步 | ≤ N 过，> 2N 不过（MARE 的 N = 2.98e-3）｜参考：无 | 生产程序（逐步记录） | [2.03e-3（MARE，0.681×N） ✅](results/tmb-l4.report.md) | — |

非 ✅ 格子的原因：

- **主 loss**（hy4-mid · tm-b（300 步，base2 开、早于 M50） 🟡）：带符号均值刚超线（TPU 偏高），待查；上一代 tm-a 见[历代比较](../../dashboard/history.md)。
- **主 loss 后段**（hy4-mid · tm-b（300 步，base2 开、早于 M50） 🟡）：带符号均值刚超线，待查。
- **索引器 KD loss**（hy4-mid · tm-b（300 步，base2 开、早于 M50） 🟡）：MARE 略超线，待查。
- **param_norm**（hy4-mid · tm-b（300 步，base2 开、早于 M50） 🟡）：带符号均值超线（TPU 偏低），待查。

| 结果 | 是什么、为什么不在清单里 | 判定 |
|---|---|---|
| [`tm-l4`](results/tm-l4.report.md) | 上一代 tm-a 的 300 步 L4（fd796f6 时的配置，没有 M39–M41、M42b）；最近一代是 tmb-l4，历代见 models/hunyuan4/dashboard/history.md | pass 5、inconclusive 4 |
