[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 06

# 案例 06：GPU 参照路径把 q·k 先舍成 BF16 再 softmax——错在 GPU 一侧，开 `hy4_attn_scores_fp32` 修正

> **状态：已修（GPU 开关 G17，默认开；GPU 参照由 `g1-*` 换成 `g2-*`）。** 台账 [G17](../CHANGES-LEDGER.md)；[根源一览第 4 行](../ROOT-CAUSES.md#r4)；对齐报告 [D9](../alignments/01-mini/DETAILS.md#d9)。

## 现象

GPU 的 `sparse_sink_attention` 是 janus 的 eager 实现。在 `hy4_attn_scores_fp32: False` 时，它把 q·k 算成 BF16，再送进 FP32 softmax（提交 `bc08bf1` 第（c）条）。生产融合内核和 TPU 内核（tokamax splash）都把 q·k 留在 FP32。

## 机制

这是 GPU 参照路径上的一个 16 位边界：矩阵乘在 FP32 里累加，结果先存成 BF16 再给 softmax（[精度边界](../../../docs/reference/precision-boundaries.md)第 1 类）。这是 eager 实现自己的写法，不是生产语义。

## 证据

- 单测 `test_attn_scores_fp32.py`：开关打开时离 FP64 参照更近，说明开关确实改变结果（提交 `37ccee8`）。同一组输入在 CPU 上重算，对 FP64 的相对 L2 误差：舍 BF16 8.18e-3，留 FP32 1.62e-3；种子 0–9 十组都是留 FP32 更近，比值 0.20–0.40（[`attn-scores-fp32-unit.json`](../alignments/01-mini/results/attn-scores-fp32-unit.json)）。
- 第 1 步只改这一处（`g2-a` 对 `g1-a`，提交 `2d5dcf6`）：lm_loss 变 +8.1e-7，grad_norm 变 −3.0e-5，indexer_loss 变 −1.2e-6。同窗口的 GPU 自己的散布（同设置 8 份第 1 步的相对极差，[`gpu-load-check-and-step1-spread.json`](../alignments/01-mini/gpu-noise/results/gpu-load-check-and-step1-spread.json)）：`g2-*` 的 lm_loss 1.7e-6、grad_norm 9.3e-5；`g1-*` 的 lm_loss 2.1e-6、grad_norm 7.3e-5。单对 `g2-a` − `g1-a` 只是一个样本；两组各 8 份直接比均值（相对值，同一结果文件的 16 个数）：lm_loss 均值差 +1.5e-6 ± 3.5e-7（标准误），约 4.3 个标准误，是 `g2-*` 极差的 0.88 倍；grad_norm 均值差 +1.6e-5 ± 1.4e-5，约 1.2 个标准误，和单对的符号相反。所以第 1 步 lm_loss 上能分辨出这一处的系统移动，量级和 8 份的散布相当；grad_norm 上分辨不出。散布本身几乎没变。300 步上它单独的效果仓库里没有记录。
- 300 步 L4，修前 `t1s-l4`（GPU `g1-a` 对 TPU `t1s-a`，尺子 `g1`；GPU 当时是被测，写「差」）、修后 `t1t-l4`（TPU `t1t-c` 对 GPU `g2-a`，尺子 `g2`）：

| 检查 | 修前：N、差、r | 修后：N、TPU、r |
|---|---|---|
| 主 loss | 不过：最大单步 3.74e-5、2.62e-4、7.00 | 待定：MARE 6.72e-6、6.88e-6、1.02（最大单步 4.53e-5、3.10e-5、0.69） |
| grad_norm | 不过：带符号均值 8.29e-5、−2.28e-4、2.75 | 待定：MARE 3.85e-4、5.16e-4、1.34（带符号均值 9.92e-5、+5.79e-5、0.58） |

出处：[`t1s-l4.results.json`](../alignments/01-mini/results/t1s-l4.results.json)、[`t1t-l4.results.json`](../alignments/01-mini/results/t1t-l4.results.json)；报告 [`t1s-l4.report.md`](../alignments/01-mini/results/t1s-l4.report.md)、[`t1t-l4.report.md`](../alignments/01-mini/results/t1t-l4.report.md)。

这张表的变化不能只归给 G17。从 `t1s-l4` 到 `t1t-l4` 还同时改了：TPU 的 M13（[案例 04](04-lm-head-single-bf16-pass.md)）、M25（[案例 07](07-qk-clip-bugs.md)）、M26（[案例 05](05-full-remat-routing-reselect.md)）、M27；噪声尺子从 `g1` 换成 `g2`（各 8 份）；被测与基线的角色互换；param_norm 改读精确值。这几处在 `t1s-l4` → `t1t-l4` 里没有拆开；之后有只关单个开关的 300 步对照（`tq-m13`、`tq-m25`、`tq-m26`、`tq-m27`，见[根源一览](../ROOT-CAUSES.md#r5)第 5、12 行与 mini 对齐报告「生产配置建议」）。

## 结论

- **修法**：GPU 训练入口 `--attn-scores-fp32`，默认开（`g1-*` 是关），`meta.json` 记 `config.attn_scores_fp32`；提交 `81f9ebf`、`37ccee8`。此后的比较以 `g2-*` 作 GPU 参照（PROTOCOL 变更记录）。
- **定责依据：读代码，对照写明的语义标准**。标准是「生产融合内核与 TPU 内核都把 q·k 留在 FP32」；janus 的 eager 稀疏注意力把 FP32 累加完的 q·k 先存成 BF16，错在 GPU 一侧。
- **收拢不能单独作依据**：L4 主 loss 从 r 7.00（N 3.74e-5，差 2.62e-4）到 r 1.02（N 6.72e-6，TPU 6.88e-6），但同一代里 TPU 改了四处、尺子也换了，效果拆不开。
- **没有对 FP64 的 R**：仓库里没有注意力分数第 1 步对 FP64 的 R。
