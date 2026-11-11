[janus](../../../README.md) › [混元 4](../README.md) ｜ 梯度范数分母统一：重判前后对照

# 混元 4 · 梯度范数曲线检查改用参照作分母：重判前后对照（2026-11-10）

混元 4 的曲线检查一律以参照作分母，梯度范数也不例外，不再用两边均值 |a + b| ÷ 2（[PROTOCOL](../PROTOCOL.md) 变更记录 2026-11-10）。本页列出受影响的全部结果文件在改前、改后的 r 与判定。混元 3 已签字，不动。

## 怎么重判的

- 范围：混元 4 下所有写了 `denominator: mean` 的检查项目录（87 个），以及用这些目录产出的全部结果文件（262 份，每份一条梯度范数检查）。
- 做法：目录去掉 `params: {denominator: mean}`，用结果文件里登记的同一批运行目录（逐个核对 `meta.json` 的 sha256）重跑 `janus run`，结果原名覆盖；改前的值以本表与 git 历史为准。
- 先用改前的目录重跑一遍：262 份结果的全部检查与已提交的逐位相同（判定、各统计量、N），说明重跑本身可复现。
- 改后：梯度范数以外的检查一项都没变；变的只有梯度范数的三个统计量与 N。

## 要点

- **判定变化 1 项**：`t1zz-l4.results.json` 的 `L4.grad_norm` 过 → 待定（最差 r 0.917 → 1.006，带符号均值领头）。这是一次历史比较（`t1zz-a`，诊断运行），不是现行比较。
- 最差 r 在 5 以内的 252 份：最差 r 改前改后之差的中位数 1.1e-04、最大 0.10；带符号均值的 r 之差中位数 1.3e-03、最大 0.43。MARE 与最大单步几乎不动，变的主要是带符号均值。
- 变化最大的几份（最差 r 在 5 以内）：`t2d15b-l4` 0.88 → 0.98；`t1zz-l4` 0.92 → 1.01；`ts-d-einsum-1k-vs-ts-g-a-1000.curves` 1.25 → 1.33；`t2b4-l4` 1.52 → 1.61；`t2d1z-l4` 1.62 → 1.69；`t1zu-l4` 0.90 → 0.97；`ts-dense-moe-1k-vs-ts-g-a-1000.curves` 1.09 → 1.16；`t1zla-l4` 0.92 → 0.88。
- 故意埋大 bug 的灵敏度臂 r 变化大（loss 多乘 2：6528 → 9677），因为差到了 1e-1 量级，两种分母本来就差一截；判定都还是不过。
- 混元 4 mini 现行比较 `t1zga-l4` 的梯度范数：1000 步带符号均值 r 1.25 → 1.23（N 3.52e-4 → 3.36e-4，TPU −4.39e-4 → −4.14e-4），仍待定、已接受；前 300 步 `t1zga300-l4` 1.32 不变。
- TPU 自比（`04-tpu-self`）：171 个比较的判档（逐位相同 / 在刻度内 / 略超 / 超）一个没变；「和 GPU 重跑比」一列也没变。
- 留一法（`analysis/gpu_loo_r.py`）按新目录重算：300 步最大值 2.47（不变）、1000 步 1.82（不变）；1000 步最差 r > 1 的比例 2.2e-1 → 2.2e-01。`analysis/noise_leave_one_run.py` 16 份梯度范数的最大 r 0.899 → 0.898，janus 的 `noise_loo`（`g3x16-noise`）梯度范数 1.057 → 1.026。
- 8 份换 16 份时梯度范数带符号均值上限的放大倍数 1.70 → 1.84（[`g3x16-noise-vs-8.json`](../alignments/01-mini/gpu-noise/results/g3x16-noise-vs-8.json) 的 `rejudge_2026_11_10`）。

## TPU 自比判档对照

只列曲线最差 r 两位小数有变化的比较；判档全部不变。

| 比较 | 判档 | 曲线最差 r 改前 | 改后 |
|---|---|---|---|
| `ts-c-nodd-1k-vs-ts-g-a-1000` | 略超 → 略超 | 1.47 | 1.49 |
| `ts-d-einsum-1k-vs-ts-g-a-1000` | 略超 → 略超 | 1.26 | 1.33 |
| `ts-d-metrics-1k-vs-ts-g-a-1000` | 略超 → 略超 | 1.37 | 1.40 |
| `ts-dense-moe-1k-vs-ts-g-a-1000` | 略超 → 略超 | 1.09 | 1.16 |
| `ts-e-approx-1k-vs-ts-g-a-1000` | 略超 → 略超 | 1.05 | 1.07 |
| `ts-e-bf16acc-1k-vs-golden` | 在刻度内 → 在刻度内 | 0.96 | 0.97 |
| `ts-e-bf16acc-1k-vs-ts-g-a-1000` | 略超 → 略超 | 1.62 | 1.64 |
| `ts-e-bf16acc-vs-golden` | 在刻度内 → 在刻度内 | 0.96 | 0.97 |
| `ts-e-swiglu-1ks-vs-ts-g-a-1000` | 略超 → 略超 | 1.38 | 1.37 |
| `ts-f-ltpu45-1k-vs-ts-g-a-1000` | 略超 → 略超 | 1.18 | 1.19 |
| `ts-fx-dp8-embf32-1ks2-vs-ts-g-a-1000` | 略超 → 略超 | 1.07 | 1.09 |
| `ts-remat-all-1k-vs-golden` | 在刻度内 → 在刻度内 | 0.94 | 0.95 |
| `ts-remat-all-vs-golden` | 在刻度内 → 在刻度内 | 0.94 | 0.95 |
| `ts-remat-full-1k-vs-ts-g-a-1000` | 在刻度内 → 在刻度内 | 0.91 | 0.92 |
| `ts-s-loss101-vs-golden` | 超 → 超 | 93.93 | 94.50 |
| `ts-s-loss2-vs-golden` | 超 → 超 | 6528.04 | 9676.75 |

## 全部结果文件

r = 被测的统计量 ÷ 噪声上限 N；「改前 → 改后」。

| 实践 | 结果文件 | 检查 | 判定 | 带符号均值 r | MARE r | 最大单步 r | 最差 r |
|---|---|---|---|---|---|---|---|
| 01-mini | [`t1-l4-nozm`](../alignments/01-mini/results/t1-l4-nozm.results.json) | `L4.grad_norm` | 不过 → 不过 | 2.69 → 2.68 | 1.75 → 1.75 | 1.14 → 1.14 | 2.69 → 2.68 |
| 01-mini | [`t1-l4`](../alignments/01-mini/results/t1-l4.results.json) | `L4.grad_norm` | 不过 → 不过 | 2.08 → 2.07 | 1.71 → 1.71 | 1.13 → 1.13 | 2.08 → 2.07 |
| 01-mini | [`t1s-l4`](../alignments/01-mini/results/t1s-l4.results.json) | `L4.grad_norm` | 不过 → 不过 | 2.75 → 2.75 | 1.72 → 1.72 | 1.07 → 1.07 | 2.75 → 2.75 |
| 01-mini | [`t1t-l4`](../alignments/01-mini/results/t1t-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.58 → 0.583 | 1.34 → 1.34 | 0.991 → 0.993 | 1.34 → 1.34 |
| 01-mini | [`t1u-l4`](../alignments/01-mini/results/t1u-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.596 → 0.567 | 0.905 → 0.904 | 0.836 → 0.788 | 0.905 → 0.904 |
| 01-mini | [`t1v-l4`](../alignments/01-mini/results/t1v-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.201 → 0.199 | 1.39 → 1.39 | 0.78 → 0.78 | 1.39 → 1.39 |
| 01-mini | [`t1w-l4`](../alignments/01-mini/results/t1w-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.0675 → 0.122 | 0.882 → 0.882 | 0.646 → 0.639 | 0.882 → 0.882 |
| 01-mini | [`t1z-l4`](../alignments/01-mini/results/t1z-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.321 → 0.323 | 1.31 → 1.31 | 0.662 → 0.663 | 1.31 → 1.31 |
| 01-mini | [`t1zga-l4`](../alignments/01-mini/results/t1zga-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.25 → 1.23 | 1.02 → 1.01 | 0.939 → 0.882 | 1.25 → 1.23 |
| 01-mini | [`t1zga300-l4`](../alignments/01-mini/results/t1zga300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.13 → 1.13 | 1.32 → 1.32 | 0.806 → 0.806 | 1.32 → 1.32 |
| 01-mini | [`t1zgb-l4`](../alignments/01-mini/results/t1zgb-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.25 → 1.23 | 1.02 → 1.01 | 0.939 → 0.882 | 1.25 → 1.23 |
| 01-mini | [`t1zk-l4`](../alignments/01-mini/results/t1zk-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.177 → 0.139 | 0.86 → 0.86 | 0.635 → 0.603 | 0.86 → 0.86 |
| 01-mini | [`t1zla-l4`](../alignments/01-mini/results/t1zla-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.101 → 0.0481 | 0.881 → 0.88 | 0.921 → 0.866 | 0.921 → 0.88 |
| 01-mini | [`t1zla300-l4`](../alignments/01-mini/results/t1zla300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.556 → 0.559 | 1.42 → 1.42 | 1 → 1 | 1.42 → 1.42 |
| 01-mini | [`t1zr-l4`](../alignments/01-mini/results/t1zr-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.255 → 0.314 | 0.824 → 0.825 | 0.667 → 0.637 | 0.824 → 0.825 |
| 01-mini | [`t1zr300-l4`](../alignments/01-mini/results/t1zr300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.675 → 0.678 | 1.26 → 1.26 | 0.649 → 0.65 | 1.26 → 1.26 |
| 01-mini | [`t1zs-l4`](../alignments/01-mini/results/t1zs-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.474 → 0.447 | 0.901 → 0.9 | 0.746 → 0.705 | 0.901 → 0.9 |
| 01-mini | [`t1zs300-l4`](../alignments/01-mini/results/t1zs300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.634 → 0.637 | 1.3 → 1.3 | 0.695 → 0.695 | 1.3 → 1.3 |
| 01-mini | [`t1zt-l4`](../alignments/01-mini/results/t1zt-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.224 → 0.179 | 0.873 → 0.874 | 1.2 → 1.21 | 1.2 → 1.21 |
| 01-mini | [`t1zt300-l4`](../alignments/01-mini/results/t1zt300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.615 → 0.618 | 1.27 → 1.27 | 0.64 → 0.641 | 1.27 → 1.27 |
| 01-mini | [`t1zu-l4`](../alignments/01-mini/results/t1zu-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.878 → 0.971 | 0.899 → 0.899 | 0.613 → 0.582 | 0.899 → 0.971 |
| 01-mini | [`t1zu300-l4`](../alignments/01-mini/results/t1zu300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.05 → 1.06 | 1.31 → 1.32 | 0.661 → 0.662 | 1.31 → 1.32 |
| 01-mini | [`t1zub-l4`](../alignments/01-mini/results/t1zub-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.713 → 0.693 | 0.909 → 0.91 | 1.13 → 1.13 | 1.13 → 1.13 |
| 01-mini | [`t1zub300-l4`](../alignments/01-mini/results/t1zub300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.933 → 0.936 | 1.36 → 1.36 | 0.772 → 0.774 | 1.36 → 1.36 |
| 01-mini | [`t1zuc-l4`](../alignments/01-mini/results/t1zuc-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.783 → 0.874 | 0.884 → 0.885 | 0.745 → 0.738 | 0.884 → 0.885 |
| 01-mini | [`t1zuc300-l4`](../alignments/01-mini/results/t1zuc300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.32 → 1.32 | 1.28 → 1.28 | 0.837 → 0.837 | 1.32 → 1.32 |
| 01-mini | [`t1zud-l4`](../alignments/01-mini/results/t1zud-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.199 → 0.26 | 0.892 → 0.893 | 0.683 → 0.676 | 0.892 → 0.893 |
| 01-mini | [`t1zud300-l4`](../alignments/01-mini/results/t1zud300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.434 → 0.436 | 1.33 → 1.33 | 0.731 → 0.732 | 1.33 → 1.33 |
| 01-mini | [`t1zv-l4`](../alignments/01-mini/results/t1zv-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.216 → 0.179 | 0.808 → 0.808 | 0.722 → 0.683 | 0.808 → 0.808 |
| 01-mini | [`t1zv300-l4`](../alignments/01-mini/results/t1zv300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1 → 1 | 1.32 → 1.32 | 0.832 → 0.834 | 1.32 → 1.32 |
| 01-mini | [`t1zw-l4`](../alignments/01-mini/results/t1zw-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.25 → 1.23 | 1.02 → 1.01 | 0.939 → 0.882 | 1.25 → 1.23 |
| 01-mini | [`t1zw300-l4`](../alignments/01-mini/results/t1zw300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.13 → 1.13 | 1.32 → 1.32 | 0.806 → 0.806 | 1.32 → 1.32 |
| 01-mini | [`t1zz-l4`](../alignments/01-mini/results/t1zz-l4.results.json) | `L4.grad_norm` | 过 → 待定 | 0.917 → 1.01 | 0.826 → 0.827 | 0.594 → 0.564 | 0.917 → 1.01 |
| 01-mini | [`t1zz300-l4`](../alignments/01-mini/results/t1zz300-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.588 → 0.591 | 1.23 → 1.23 | 0.675 → 0.676 | 1.23 → 1.23 |
| 01-mini | [`tq-alloff-l4`](../alignments/01-mini/results/tq-alloff-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.21 → 1.21 | 1.62 → 1.62 | 1.17 → 1.18 | 1.62 → 1.62 |
| 01-mini | [`tq-base-l4`](../alignments/01-mini/results/tq-base-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.05 → 1.06 | 1.31 → 1.32 | 0.661 → 0.662 | 1.31 → 1.32 |
| 01-mini | [`tq-base2-l4`](../alignments/01-mini/results/tq-base2-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.14 → 1.14 | 1.36 → 1.36 | 0.827 → 0.829 | 1.36 → 1.36 |
| 01-mini | [`tq-m10-l4`](../alignments/01-mini/results/tq-m10-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.174 → 0.176 | 1.26 → 1.26 | 0.675 → 0.675 | 1.26 → 1.26 |
| 01-mini | [`tq-m12-l4`](../alignments/01-mini/results/tq-m12-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.404 → 0.406 | 1.34 → 1.34 | 0.736 → 0.737 | 1.34 → 1.34 |
| 01-mini | [`tq-m13-l4`](../alignments/01-mini/results/tq-m13-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.23 → 1.23 | 1.63 → 1.63 | 1.13 → 1.13 | 1.63 → 1.63 |
| 01-mini | [`tq-m23-l4`](../alignments/01-mini/results/tq-m23-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.517 → 0.52 | 1.26 → 1.26 | 0.665 → 0.666 | 1.26 → 1.26 |
| 01-mini | [`tq-m25-l4`](../alignments/01-mini/results/tq-m25-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.445 → 0.448 | 1.25 → 1.25 | 0.648 → 0.649 | 1.25 → 1.25 |
| 01-mini | [`tq-m26-l4`](../alignments/01-mini/results/tq-m26-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.358 → 0.36 | 1.25 → 1.25 | 0.7 → 0.701 | 1.25 → 1.25 |
| 01-mini | [`tq-m27-l4`](../alignments/01-mini/results/tq-m27-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.477 → 0.479 | 1.3 → 1.3 | 0.747 → 0.749 | 1.3 → 1.3 |
| 01-mini | [`tq-m28-l4`](../alignments/01-mini/results/tq-m28-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.773 → 0.776 | 1.31 → 1.31 | 0.665 → 0.666 | 1.31 → 1.31 |
| 01-mini | [`tq-m31-l4`](../alignments/01-mini/results/tq-m31-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.156 → 0.154 | 1.34 → 1.34 | 0.669 → 0.67 | 1.34 → 1.34 |
| 01-mini | [`tq-m33-l4`](../alignments/01-mini/results/tq-m33-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.75 → 1.76 | 1.37 → 1.37 | 0.892 → 0.894 | 1.75 → 1.76 |
| 01-mini | [`tq-m34-l4`](../alignments/01-mini/results/tq-m34-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.372 → 0.375 | 1.29 → 1.29 | 0.706 → 0.707 | 1.29 → 1.29 |
| 01-mini | [`tq-m36-l4`](../alignments/01-mini/results/tq-m36-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.54 → 0.543 | 1.28 → 1.28 | 0.712 → 0.714 | 1.28 → 1.28 |
| 01-mini | [`tq-m40-l4`](../alignments/01-mini/results/tq-m40-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.32 → 1.32 | 1.28 → 1.28 | 0.837 → 0.837 | 1.32 → 1.32 |
| 01-mini | [`tq-m41-l4`](../alignments/01-mini/results/tq-m41-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.434 → 0.436 | 1.33 → 1.33 | 0.731 → 0.732 | 1.33 → 1.33 |
| 01-mini | [`tq-m8-l4`](../alignments/01-mini/results/tq-m8-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.51 → 1.51 | 1.38 → 1.38 | 0.771 → 0.772 | 1.51 → 1.51 |
| 01-mini | [`tq-m9-l4`](../alignments/01-mini/results/tq-m9-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.23 → 1.23 | 1.31 → 1.31 | 0.786 → 0.788 | 1.31 → 1.31 |
| 01-mini/gpu-noise | [`g3x16-noise`](../alignments/01-mini/gpu-noise/results/g3x16-noise.results.json) | `L4.grad_norm` | 过 → 过 | 0.154 → 0.127 | 0.789 → 0.788 | 0.494 → 0.513 | 0.789 → 0.788 |
| 01-mini/gpu-noise | [`t1zga-l4-n16`](../alignments/01-mini/gpu-noise/results/t1zga-l4-n16.results.json) | `L4.grad_norm` | 过 → 过 | 0.734 → 0.672 | 0.999 → 0.995 | 0.773 → 0.764 | 0.999 → 0.995 |
| 02-real-weight | [`t2b4-l4`](../alignments/02-real-weight/results/t2b4-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.52 → 1.61 | 1.04 → 1.04 | 0.372 → 0.36 | 1.52 → 1.61 |
| 02-real-weight | [`t2b5-l4`](../alignments/02-real-weight/results/t2b5-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.672 → 0.722 | 1.12 → 1.13 | 0.785 → 0.825 | 1.12 → 1.13 |
| 02-real-weight | [`t2b5b-l4`](../alignments/02-real-weight/results/t2b5b-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.503 → 0.567 | 1.13 → 1.13 | 0.767 → 0.805 | 1.13 → 1.13 |
| 02-real-weight | [`t2bz-l4`](../alignments/02-real-weight/results/t2bz-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.0502 → 0.0165 | 0.965 → 0.962 | 0.291 → 0.273 | 0.965 → 0.962 |
| 02-real-weight | [`t2d1-l4`](../alignments/02-real-weight/results/t2d1-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.397 → 0.331 | 1.46 → 1.46 | 1.07 → 1.11 | 1.46 → 1.46 |
| 02-real-weight | [`t2d14-l4`](../alignments/02-real-weight/results/t2d14-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.26 → 1.22 | 1.44 → 1.44 | 1.34 → 1.4 | 1.44 → 1.44 |
| 02-real-weight | [`t2d15-l4`](../alignments/02-real-weight/results/t2d15-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.122 → 0.413 | 0.807 → 0.802 | 0.391 → 0.264 | 0.807 → 0.802 |
| 02-real-weight | [`t2d15b-l4`](../alignments/02-real-weight/results/t2d15b-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.553 → 0.98 | 0.88 → 0.876 | 0.322 → 0.221 | 0.88 → 0.98 |
| 02-real-weight | [`t2d1z-l4`](../alignments/02-real-weight/results/t2d1z-l4.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.35 → 1.3 | 1.43 → 1.43 | 1.62 → 1.69 | 1.62 → 1.69 |
| 03-mid | [`tm-l4`](../alignments/03-mid/results/tm-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.0364 → 0.0342 | 0.969 → 0.969 | 0.782 → 0.778 | 0.969 → 0.969 |
| 03-mid | [`tmb-l4`](../alignments/03-mid/results/tmb-l4.results.json) | `L4.grad_norm` | 过 → 过 | 0.342 → 0.344 | 0.931 → 0.931 | 0.668 → 0.665 | 0.931 → 0.931 |
| 04-tpu-self | [`mz-g1-vs-golden.curves`](../alignments/04-tpu-self/results/mz-g1-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`t1zg-b-vs-golden.curves`](../alignments/04-tpu-self/results/t1zg-b-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-a-8c-vs-golden.curves`](../alignments/04-tpu-self/results/ts-a-8c-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-a-8f-vs-golden.curves`](../alignments/04-tpu-self/results/ts-a-8f-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-a-shared-vs-golden.curves`](../alignments/04-tpu-self/results/ts-a-shared-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-c-autorp-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-autorp-1k-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 1.06 → 1.06 | 1.05 → 1.05 | 0.723 → 0.723 | 1.06 → 1.06 |
| 04-tpu-self | [`ts-c-autorp-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-c-autorp-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.14 → 1.14 | 0.94 → 0.943 | 0.946 → 1 | 1.14 → 1.14 |
| 04-tpu-self | [`ts-c-autorp-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-autorp-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 1.06 → 1.06 | 1.05 → 1.05 | 0.723 → 0.723 | 1.06 → 1.06 |
| 04-tpu-self | [`ts-c-crsbf16b-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-crsbf16b-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-c-mfags-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-mfags-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-c-mmhigh-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-mmhigh-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-c-noacf-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-noacf-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-c-nobf16p-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-nobf16p-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.313 → 0.312 | 1.02 → 1.02 | 0.657 → 0.658 | 1.02 → 1.02 |
| 04-tpu-self | [`ts-c-nocopy-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-nocopy-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.468 → 0.467 | 0.97 → 0.97 | 0.705 → 0.705 | 0.97 → 0.97 |
| 04-tpu-self | [`ts-c-nodd-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-nodd-1k-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.0241 → 0.0228 | 1.02 → 1.02 | 0.693 → 0.694 | 1.02 → 1.02 |
| 04-tpu-self | [`ts-c-nodd-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-c-nodd-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.47 → 1.49 | 1.07 → 1.07 | 0.935 → 0.988 | 1.47 → 1.49 |
| 04-tpu-self | [`ts-c-nodd-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-nodd-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.0241 → 0.0228 | 1.02 → 1.02 | 0.693 → 0.694 | 1.02 → 1.02 |
| 04-tpu-self | [`ts-c-nodparb-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-nodparb-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-c-noexcess-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-noexcess-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 13.3 → 13.3 | 3.9 → 3.89 | 3.1 → 3.09 | 13.3 → 13.3 |
| 04-tpu-self | [`ts-c-nolhs-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-nolhs-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.688 → 0.688 | 0.91 → 0.91 | 0.691 → 0.691 | 0.91 → 0.91 |
| 04-tpu-self | [`ts-c-nolhs-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-c-nolhs-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.841 → 0.85 | 0.912 → 0.913 | 0.616 → 0.643 | 0.912 → 0.913 |
| 04-tpu-self | [`ts-c-nolhs-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-nolhs-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.688 → 0.688 | 0.91 → 0.91 | 0.691 → 0.691 | 0.91 → 0.91 |
| 04-tpu-self | [`ts-c-noovl-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-noovl-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-c-norwb-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-norwb-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-c-nosrb-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-nosrb-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.39 → 0.39 | 0.943 → 0.943 | 0.713 → 0.713 | 0.943 → 0.943 |
| 04-tpu-self | [`ts-c-nowe-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-nowe-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.999 → 0.999 | 0.968 → 0.968 | 0.685 → 0.685 | 0.999 → 0.999 |
| 04-tpu-self | [`ts-c-nowe-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-c-nowe-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.059 → 0.0973 | 0.962 → 0.961 | 0.699 → 0.698 | 0.962 → 0.961 |
| 04-tpu-self | [`ts-c-nowe-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-nowe-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.999 → 0.999 | 0.968 → 0.968 | 0.685 → 0.685 | 0.999 → 0.999 |
| 04-tpu-self | [`ts-c-scag-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-scag-1k-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.625 → 0.624 | 1.04 → 1.04 | 0.955 → 0.955 | 1.04 → 1.04 |
| 04-tpu-self | [`ts-c-scag-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-c-scag-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.187 → 0.236 | 1.08 → 1.08 | 0.762 → 0.793 | 1.08 → 1.08 |
| 04-tpu-self | [`ts-c-scag-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-scag-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.625 → 0.624 | 1.04 → 1.04 | 0.955 → 0.955 | 1.04 → 1.04 |
| 04-tpu-self | [`ts-c-scar-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-scar-1k-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.704 → 0.704 | 1.03 → 1.03 | 0.832 → 0.832 | 1.03 → 1.03 |
| 04-tpu-self | [`ts-c-scar-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-c-scar-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.383 → 0.408 | 0.847 → 0.848 | 0.785 → 0.825 | 0.847 → 0.848 |
| 04-tpu-self | [`ts-c-scar-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-scar-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.704 → 0.704 | 1.03 → 1.03 | 0.832 → 0.832 | 1.03 → 1.03 |
| 04-tpu-self | [`ts-c-scf32-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-scf32-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-c-scrs-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-scrs-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-c-vmem-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c-vmem-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.776 → 0.776 | 0.887 → 0.887 | 0.76 → 0.762 | 0.887 → 0.887 |
| 04-tpu-self | [`ts-c300-vs-golden.curves`](../alignments/04-tpu-self/results/ts-c300-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-cp2b-vs-ts-d-dotattn.curves`](../alignments/04-tpu-self/results/ts-cp2b-vs-ts-d-dotattn.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 1.04 → 1.04 | 0.889 → 0.889 | 0.792 → 0.794 | 1.04 → 1.04 |
| 04-tpu-self | [`ts-cp2c-vs-ts-cp2b.curves`](../alignments/04-tpu-self/results/ts-cp2c-vs-ts-cp2b.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-cp2c-vs-ts-d-dotattn.curves`](../alignments/04-tpu-self/results/ts-cp2c-vs-ts-d-dotattn.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 1.04 → 1.04 | 0.889 → 0.889 | 0.792 → 0.794 | 1.04 → 1.04 |
| 04-tpu-self | [`ts-d-ckname-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-ckname-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-d-clip-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-clip-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-d-commute-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-commute-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-d-dotattn-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-dotattn-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.304 → 0.306 | 0.894 → 0.894 | 0.72 → 0.721 | 0.894 → 0.894 |
| 04-tpu-self | [`ts-d-dotattn-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-d-dotattn-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.955 → 0.983 | 1.08 → 1.08 | 0.868 → 0.876 | 1.08 → 1.08 |
| 04-tpu-self | [`ts-d-dotattn-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-dotattn-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.304 → 0.306 | 0.894 → 0.894 | 0.72 → 0.721 | 0.894 → 0.894 |
| 04-tpu-self | [`ts-d-einsum-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-einsum-1k-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.316 → 0.315 | 1.1 → 1.1 | 0.779 → 0.779 | 1.1 → 1.1 |
| 04-tpu-self | [`ts-d-einsum-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-d-einsum-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.851 → 0.904 | 1.09 → 1.09 | 1.25 → 1.33 | 1.25 → 1.33 |
| 04-tpu-self | [`ts-d-einsum-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-einsum-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.316 → 0.315 | 1.1 → 1.1 | 0.779 → 0.779 | 1.1 → 1.1 |
| 04-tpu-self | [`ts-d-helper-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-helper-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-d-lossdiv-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-lossdiv-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-d-meansum-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-meansum-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-d-metrics-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-metrics-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.547 → 0.547 | 0.894 → 0.894 | 0.552 → 0.553 | 0.894 → 0.894 |
| 04-tpu-self | [`ts-d-metrics-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-d-metrics-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.37 → 1.4 | 1.07 → 1.08 | 1.17 → 1.25 | 1.37 → 1.4 |
| 04-tpu-self | [`ts-d-metrics-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-metrics-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.547 → 0.547 | 0.894 → 0.894 | 0.552 → 0.553 | 0.894 → 0.894 |
| 04-tpu-self | [`ts-d-qklog-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-qklog-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-d-rsqrt-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-rsqrt-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.889 → 0.889 | 0.979 → 0.979 | 0.872 → 0.872 | 0.979 → 0.979 |
| 04-tpu-self | [`ts-d-rsqrt-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-d-rsqrt-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.737 → 0.744 | 0.846 → 0.846 | 0.544 → 0.554 | 0.846 → 0.846 |
| 04-tpu-self | [`ts-d-rsqrt-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-rsqrt-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.889 → 0.889 | 0.979 → 0.979 | 0.872 → 0.872 | 0.979 → 0.979 |
| 04-tpu-self | [`ts-d-square-1ks-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-square-1ks-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.597 → 0.596 | 0.981 → 0.981 | 0.703 → 0.703 | 0.981 → 0.981 |
| 04-tpu-self | [`ts-d-square-1ks-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-d-square-1ks-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.488 → 0.512 | 0.958 → 0.96 | 0.738 → 0.774 | 0.958 → 0.96 |
| 04-tpu-self | [`ts-d-square-vs-golden.curves`](../alignments/04-tpu-self/results/ts-d-square-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.597 → 0.596 | 0.981 → 0.981 | 0.703 → 0.703 | 0.981 → 0.981 |
| 04-tpu-self | [`ts-dense-moe-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-dense-moe-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.0514 → 0.0524 | 0.822 → 0.822 | 0.72 → 0.721 | 0.822 → 0.822 |
| 04-tpu-self | [`ts-dense-moe-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-dense-moe-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.877 → 0.9 | 0.965 → 0.969 | 1.09 → 1.16 | 1.09 → 1.16 |
| 04-tpu-self | [`ts-dense-moe-vs-golden.curves`](../alignments/04-tpu-self/results/ts-dense-moe-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.0514 → 0.0524 | 0.822 → 0.822 | 0.72 → 0.721 | 0.822 → 0.822 |
| 04-tpu-self | [`ts-dp2f4-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-dp2f4-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.451 → 0.451 | 0.976 → 0.976 | 0.746 → 0.747 | 0.976 → 0.976 |
| 04-tpu-self | [`ts-dp2f4-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-dp2f4-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.06 → 1.11 | 1.19 → 1.19 | 1.01 → 1.07 | 1.19 → 1.19 |
| 04-tpu-self | [`ts-dp2f4-vs-golden.curves`](../alignments/04-tpu-self/results/ts-dp2f4-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.451 → 0.451 | 0.976 → 0.976 | 0.746 → 0.747 | 0.976 → 0.976 |
| 04-tpu-self | [`ts-dp8-embf32-ga1-vs-golden.curves`](../alignments/04-tpu-self/results/ts-dp8-embf32-ga1-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.223 → 0.225 | 0.919 → 0.919 | 0.619 → 0.619 | 0.919 → 0.919 |
| 04-tpu-self | [`ts-dp8-embf32-ga1-vs-ts-dp8-ga1.curves`](../alignments/04-tpu-self/results/ts-dp8-embf32-ga1-vs-ts-dp8-ga1.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 3.38 → 3.38 | 1.51 → 1.51 | 0.998 → 1 | 3.38 → 3.38 |
| 04-tpu-self | [`ts-dp8-embf32-vs-golden.curves`](../alignments/04-tpu-self/results/ts-dp8-embf32-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 180 → 183 | 53.3 → 53.8 | 14.8 → 15.1 | 180 → 183 |
| 04-tpu-self | [`ts-dp8-ga1-vs-golden.curves`](../alignments/04-tpu-self/results/ts-dp8-ga1-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 3.15 → 3.16 | 1.54 → 1.54 | 1.07 → 1.07 | 3.15 → 3.16 |
| 04-tpu-self | [`ts-dp8-vs-golden.curves`](../alignments/04-tpu-self/results/ts-dp8-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 2.62 → 2.62 | 1.36 → 1.36 | 0.818 → 0.818 | 2.62 → 2.62 |
| 04-tpu-self | [`ts-e-adameps-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-adameps-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 47.2 → 47.1 | 12.2 → 12.1 | 3.44 → 3.43 | 47.2 → 47.1 |
| 04-tpu-self | [`ts-e-approx-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-approx-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.384 → 0.383 | 0.898 → 0.898 | 0.674 → 0.674 | 0.898 → 0.898 |
| 04-tpu-self | [`ts-e-approx-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-e-approx-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.05 → 1.07 | 0.93 → 0.933 | 0.828 → 0.871 | 1.05 → 1.07 |
| 04-tpu-self | [`ts-e-approx-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-approx-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.384 → 0.383 | 0.898 → 0.898 | 0.674 → 0.674 | 0.898 → 0.898 |
| 04-tpu-self | [`ts-e-bf16acc-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-bf16acc-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.111 → 0.11 | 0.965 → 0.965 | 0.72 → 0.72 | 0.965 → 0.965 |
| 04-tpu-self | [`ts-e-bf16acc-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-e-bf16acc-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.62 → 1.64 | 1.19 → 1.2 | 0.911 → 0.961 | 1.62 → 1.64 |
| 04-tpu-self | [`ts-e-bf16acc-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-bf16acc-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.111 → 0.11 | 0.965 → 0.965 | 0.72 → 0.72 | 0.965 → 0.965 |
| 04-tpu-self | [`ts-e-embf32-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-embf32-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.482 → 0.482 | 1.02 → 1.02 | 0.799 → 0.801 | 1.02 → 1.02 |
| 04-tpu-self | [`ts-e-eps-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-eps-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 26.6 → 26.6 | 6.86 → 6.85 | 2.44 → 2.44 | 26.6 → 26.6 |
| 04-tpu-self | [`ts-e-idx32-1ks-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-idx32-1ks-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.547 → 0.546 | 0.949 → 0.949 | 0.621 → 0.622 | 0.949 → 0.949 |
| 04-tpu-self | [`ts-e-idx32-1ks-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-e-idx32-1ks-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.638 → 0.67 | 1.06 → 1.06 | 0.734 → 0.727 | 1.06 → 1.06 |
| 04-tpu-self | [`ts-e-idx32-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-idx32-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.547 → 0.546 | 0.949 → 0.949 | 0.621 → 0.622 | 0.949 → 0.949 |
| 04-tpu-self | [`ts-e-ihcr-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-ihcr-1k-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 1.05 → 1.05 | 1.03 → 1.03 | 0.755 → 0.755 | 1.05 → 1.05 |
| 04-tpu-self | [`ts-e-ihcr-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-e-ihcr-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.691 → 0.712 | 1.03 → 1.03 | 0.631 → 0.627 | 1.03 → 1.03 |
| 04-tpu-self | [`ts-e-ihcr-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-ihcr-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 1.05 → 1.05 | 1.03 → 1.03 | 0.755 → 0.755 | 1.05 → 1.05 |
| 04-tpu-self | [`ts-e-lossorder-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-lossorder-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-e-nsfp32-1ks-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-nsfp32-1ks-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.894 → 0.894 | 0.98 → 0.98 | 0.891 → 0.891 | 0.98 → 0.98 |
| 04-tpu-self | [`ts-e-nsfp32-1ks-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-e-nsfp32-1ks-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.546 → 0.572 | 0.973 → 0.973 | 0.826 → 0.869 | 0.973 → 0.973 |
| 04-tpu-self | [`ts-e-nsfp32-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-nsfp32-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.894 → 0.894 | 0.98 → 0.98 | 0.891 → 0.891 | 0.98 → 0.98 |
| 04-tpu-self | [`ts-e-rms2r-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-rms2r-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.0448 → 0.046 | 0.909 → 0.909 | 0.825 → 0.827 | 0.909 → 0.909 |
| 04-tpu-self | [`ts-e-swiglu-1ks-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-swiglu-1ks-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.443 → 0.442 | 0.985 → 0.985 | 0.84 → 0.842 | 0.985 → 0.985 |
| 04-tpu-self | [`ts-e-swiglu-1ks-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-e-swiglu-1ks-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.38 → 1.37 | 0.906 → 0.909 | 0.881 → 0.929 | 1.38 → 1.37 |
| 04-tpu-self | [`ts-e-swiglu-vs-golden.curves`](../alignments/04-tpu-self/results/ts-e-swiglu-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.443 → 0.442 | 0.985 → 0.985 | 0.84 → 0.842 | 0.985 → 0.985 |
| 04-tpu-self | [`ts-ep-base-vs-golden.curves`](../alignments/04-tpu-self/results/ts-ep-base-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.317 → 0.316 | 0.959 → 0.959 | 0.751 → 0.752 | 0.959 → 0.959 |
| 04-tpu-self | [`ts-ep2-vs-ts-ep-base.curves`](../alignments/04-tpu-self/results/ts-ep2-vs-ts-ep-base.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.497 → 0.499 | 0.891 → 0.892 | 0.585 → 0.585 | 0.891 → 0.892 |
| 04-tpu-self | [`ts-ep4-vs-ts-ep-base.curves`](../alignments/04-tpu-self/results/ts-ep4-vs-ts-ep-base.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.514 → 0.515 | 0.814 → 0.814 | 0.74 → 0.741 | 0.814 → 0.814 |
| 04-tpu-self | [`ts-ep8-vs-ts-ep-base.curves`](../alignments/04-tpu-self/results/ts-ep8-vs-ts-ep-base.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.411 → 0.411 | 0.759 → 0.759 | 0.837 → 0.837 | 0.837 → 0.837 |
| 04-tpu-self | [`ts-f-ltpu45-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f-ltpu45-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.429 → 0.428 | 0.986 → 0.986 | 0.782 → 0.782 | 0.986 → 0.986 |
| 04-tpu-self | [`ts-f-ltpu45-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-f-ltpu45-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.18 → 1.19 | 1.01 → 1.01 | 0.76 → 0.798 | 1.18 → 1.19 |
| 04-tpu-self | [`ts-f-ltpu45-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f-ltpu45-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.429 → 0.428 | 0.986 → 0.986 | 0.782 → 0.782 | 0.986 → 0.986 |
| 04-tpu-self | [`ts-f-ltpu47-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f-ltpu47-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.266 → 0.265 | 0.966 → 0.966 | 0.8 → 0.8 | 0.966 → 0.966 |
| 04-tpu-self | [`ts-f-ltpu47-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-f-ltpu47-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.517 → 0.539 | 0.918 → 0.919 | 0.705 → 0.738 | 0.918 → 0.919 |
| 04-tpu-self | [`ts-f-ltpu47-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f-ltpu47-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.266 → 0.265 | 0.966 → 0.966 | 0.8 → 0.8 | 0.966 → 0.966 |
| 04-tpu-self | [`ts-f-ltpu48-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f-ltpu48-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.266 → 0.265 | 0.966 → 0.966 | 0.8 → 0.8 | 0.966 → 0.966 |
| 04-tpu-self | [`ts-f-ltpu48-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-f-ltpu48-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.517 → 0.539 | 0.918 → 0.919 | 0.705 → 0.738 | 0.918 → 0.919 |
| 04-tpu-self | [`ts-f-ltpu48-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f-ltpu48-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.266 → 0.265 | 0.966 → 0.966 | 0.8 → 0.8 | 0.966 → 0.966 |
| 04-tpu-self | [`ts-f-ltpu48-vs-ts-f-ltpu47.curves`](../alignments/04-tpu-self/results/ts-f-ltpu48-vs-ts-f-ltpu47.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-f-sc45-1ks-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f-sc45-1ks-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.145 → 0.144 | 0.923 → 0.923 | 0.574 → 0.574 | 0.923 → 0.923 |
| 04-tpu-self | [`ts-f-sc45-1ks-vs-ts-c-scar-1k-1000.curves`](../alignments/04-tpu-self/results/ts-f-sc45-1ks-vs-ts-c-scar-1k-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.505 → 0.523 | 0.86 → 0.861 | 0.845 → 0.89 | 0.86 → 0.89 |
| 04-tpu-self | [`ts-f-sc45-1ks-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-f-sc45-1ks-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.888 → 0.912 | 1.02 → 1.02 | 0.915 → 0.966 | 1.02 → 1.02 |
| 04-tpu-self | [`ts-f-sc45b-vs-ts-c-scar.curves`](../alignments/04-tpu-self/results/ts-f-sc45b-vs-ts-c-scar.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.56 → 0.562 | 0.9 → 0.9 | 0.989 → 0.992 | 0.989 → 0.992 |
| 04-tpu-self | [`ts-f-sc45s2-vs-ts-c-scar.curves`](../alignments/04-tpu-self/results/ts-f-sc45s2-vs-ts-c-scar.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.56 → 0.562 | 0.9 → 0.9 | 0.989 → 0.992 | 0.989 → 0.992 |
| 04-tpu-self | [`ts-f16-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f16-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.0988 → 0.0979 | 0.886 → 0.886 | 0.788 → 0.789 | 0.886 → 0.886 |
| 04-tpu-self | [`ts-f16-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-f16-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.888 → 0.917 | 1.09 → 1.1 | 0.929 → 0.981 | 1.09 → 1.1 |
| 04-tpu-self | [`ts-f16-dp16-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f16-dp16-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 2.14 → 2.14 | 1.3 → 1.3 | 0.859 → 0.86 | 2.14 → 2.14 |
| 04-tpu-self | [`ts-f16-dp2-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f16-dp2-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.712 → 0.712 | 0.953 → 0.953 | 0.753 → 0.753 | 0.953 → 0.953 |
| 04-tpu-self | [`ts-f16-dp2-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-f16-dp2-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.504 → 0.537 | 0.996 → 0.996 | 0.836 → 0.88 | 0.996 → 0.996 |
| 04-tpu-self | [`ts-f16-dp2-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f16-dp2-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.712 → 0.712 | 0.953 → 0.953 | 0.753 → 0.753 | 0.953 → 0.953 |
| 04-tpu-self | [`ts-f16-scar-vs-ts-f16.curves`](../alignments/04-tpu-self/results/ts-f16-scar-vs-ts-f16.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.218 → 0.219 | 0.845 → 0.845 | 0.775 → 0.776 | 0.845 → 0.845 |
| 04-tpu-self | [`ts-f16-tp2-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f16-tp2-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.704 → 0.704 | 0.994 → 0.994 | 0.632 → 0.633 | 0.994 → 0.994 |
| 04-tpu-self | [`ts-f16-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f16-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.0988 → 0.0979 | 0.886 → 0.886 | 0.788 → 0.789 | 0.886 → 0.886 |
| 04-tpu-self | [`ts-f16b-vs-golden.curves`](../alignments/04-tpu-self/results/ts-f16b-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.0988 → 0.0979 | 0.886 → 0.886 | 0.788 → 0.789 | 0.886 → 0.886 |
| 04-tpu-self | [`ts-f16b-vs-ts-f16.curves`](../alignments/04-tpu-self/results/ts-f16b-vs-ts-f16.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-fx-dp16-embf32-1ks-vs-golden.curves`](../alignments/04-tpu-self/results/ts-fx-dp16-embf32-1ks-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.0743 → 0.0733 | 0.941 → 0.941 | 0.65 → 0.65 | 0.941 → 0.941 |
| 04-tpu-self | [`ts-fx-dp16-embf32-1ks-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-fx-dp16-embf32-1ks-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.22 → 1.25 | 1.15 → 1.15 | 1.1 → 1.17 | 1.22 → 1.25 |
| 04-tpu-self | [`ts-fx-dp16-embf32-vs-golden.curves`](../alignments/04-tpu-self/results/ts-fx-dp16-embf32-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.0743 → 0.0733 | 0.941 → 0.941 | 0.65 → 0.65 | 0.941 → 0.941 |
| 04-tpu-self | [`ts-fx-dp8-embf32-1ks2-vs-golden.curves`](../alignments/04-tpu-self/results/ts-fx-dp8-embf32-1ks2-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.934 → 0.934 | 0.984 → 0.984 | 0.893 → 0.893 | 0.984 → 0.984 |
| 04-tpu-self | [`ts-fx-dp8-embf32-1ks2-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-fx-dp8-embf32-1ks2-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.07 → 1.09 | 1.01 → 1.02 | 0.843 → 0.887 | 1.07 → 1.09 |
| 04-tpu-self | [`ts-fx-dp8-embf32-vs-golden.curves`](../alignments/04-tpu-self/results/ts-fx-dp8-embf32-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.934 → 0.934 | 0.984 → 0.984 | 0.893 → 0.893 | 0.984 → 0.984 |
| 04-tpu-self | [`ts-fx-g-vs-golden.curves`](../alignments/04-tpu-self/results/ts-fx-g-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-g-a-vs-gpu`](../alignments/04-tpu-self/results/ts-g-a-vs-gpu.results.json) | `L4.grad_norm` | 待定 → 待定 | 1.25 → 1.23 | 1.02 → 1.01 | 0.939 → 0.882 | 1.25 → 1.23 |
| 04-tpu-self | [`ts-g-b-vs-golden.curves`](../alignments/04-tpu-self/results/ts-g-b-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-h-v7-1k3-vs-golden.curves`](../alignments/04-tpu-self/results/ts-h-v7-1k3-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.0573 → 0.0587 | 1.02 → 1.02 | 0.759 → 0.761 | 1.02 → 1.02 |
| 04-tpu-self | [`ts-h-v7-1k3-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-h-v7-1k3-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.265 → 0.298 | 0.91 → 0.909 | 0.734 → 0.769 | 0.91 → 0.909 |
| 04-tpu-self | [`ts-h-v7-dp8b-vs-ts-h-v7b.curves`](../alignments/04-tpu-self/results/ts-h-v7-dp8b-vs-ts-h-v7b.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 2.21 → 2.21 | 1.14 → 1.14 | 0.775 → 0.776 | 2.21 → 2.21 |
| 04-tpu-self | [`ts-h-v7-noexcess3-vs-ts-h-v7b.curves`](../alignments/04-tpu-self/results/ts-h-v7-noexcess3-vs-ts-h-v7b.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 3.43 → 3.43 | 1.29 → 1.29 | 0.953 → 0.953 | 3.43 → 3.43 |
| 04-tpu-self | [`ts-h-v7-scar2-vs-ts-h-v7b.curves`](../alignments/04-tpu-self/results/ts-h-v7-scar2-vs-ts-h-v7b.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-h-v7b-vs-golden.curves`](../alignments/04-tpu-self/results/ts-h-v7b-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.0573 → 0.0587 | 1.02 → 1.02 | 0.759 → 0.761 | 1.02 → 1.02 |
| 04-tpu-self | [`ts-h-v7c-vs-ts-h-v7b.curves`](../alignments/04-tpu-self/results/ts-h-v7c-vs-ts-h-v7b.curves.results.json) | `C.grad_norm` | 过 → 过 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| 04-tpu-self | [`ts-h-v7f16c-vs-golden.curves`](../alignments/04-tpu-self/results/ts-h-v7f16c-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 1.01 → 1.01 | 0.926 → 0.926 | 0.593 → 0.594 | 1.01 → 1.01 |
| 04-tpu-self | [`ts-h-v7f16c-vs-ts-f16.curves`](../alignments/04-tpu-self/results/ts-h-v7f16c-vs-ts-f16.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.909 → 0.909 | 0.887 → 0.887 | 0.738 → 0.74 | 0.909 → 0.909 |
| 04-tpu-self | [`ts-mb2-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-mb2-1k-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.145 → 0.144 | 1.03 → 1.03 | 0.717 → 0.717 | 1.03 → 1.03 |
| 04-tpu-self | [`ts-mb2-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-mb2-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.547 → 0.567 | 0.926 → 0.927 | 0.82 → 0.863 | 0.926 → 0.927 |
| 04-tpu-self | [`ts-mb2-vs-golden.curves`](../alignments/04-tpu-self/results/ts-mb2-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.145 → 0.144 | 1.03 → 1.03 | 0.717 → 0.717 | 1.03 → 1.03 |
| 04-tpu-self | [`ts-pc-logits-vs-golden.curves`](../alignments/04-tpu-self/results/ts-pc-logits-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.593 → 0.596 | 1.25 → 1.25 | 1.24 → 1.24 | 1.25 → 1.25 |
| 04-tpu-self | [`ts-pc-zm-vs-golden.curves`](../alignments/04-tpu-self/results/ts-pc-zm-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.322 → 0.321 | 0.905 → 0.905 | 0.93 → 0.93 | 0.93 → 0.93 |
| 04-tpu-self | [`ts-remat-all-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-remat-all-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.295 → 0.297 | 0.945 → 0.945 | 0.728 → 0.729 | 0.945 → 0.945 |
| 04-tpu-self | [`ts-remat-all-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-remat-all-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.105 → 0.15 | 1.04 → 1.04 | 0.702 → 0.735 | 1.04 → 1.04 |
| 04-tpu-self | [`ts-remat-all-vs-golden.curves`](../alignments/04-tpu-self/results/ts-remat-all-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.295 → 0.297 | 0.945 → 0.945 | 0.728 → 0.729 | 0.945 → 0.945 |
| 04-tpu-self | [`ts-remat-full-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-remat-full-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.596 → 0.595 | 0.993 → 0.993 | 0.745 → 0.746 | 0.993 → 0.993 |
| 04-tpu-self | [`ts-remat-full-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-remat-full-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.735 → 0.749 | 0.915 → 0.916 | 0.604 → 0.63 | 0.915 → 0.916 |
| 04-tpu-self | [`ts-remat-full-vs-golden.curves`](../alignments/04-tpu-self/results/ts-remat-full-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.596 → 0.595 | 0.993 → 0.993 | 0.745 → 0.746 | 0.993 → 0.993 |
| 04-tpu-self | [`ts-remat-min-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-remat-min-1k-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 1.04 → 1.04 | 0.985 → 0.985 | 0.776 → 0.776 | 1.04 → 1.04 |
| 04-tpu-self | [`ts-remat-min-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-remat-min-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 过 → 过 | 0.343 → 0.369 | 0.958 → 0.957 | 0.624 → 0.62 | 0.958 → 0.957 |
| 04-tpu-self | [`ts-remat-min-vs-golden.curves`](../alignments/04-tpu-self/results/ts-remat-min-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 1.04 → 1.04 | 0.985 → 0.985 | 0.776 → 0.776 | 1.04 → 1.04 |
| 04-tpu-self | [`ts-s-loss101-vs-golden.curves`](../alignments/04-tpu-self/results/ts-s-loss101-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 93.9 → 94.5 | 24.2 → 24.4 | 4.83 → 4.86 | 93.9 → 94.5 |
| 04-tpu-self | [`ts-s-loss2-vs-golden.curves`](../alignments/04-tpu-self/results/ts-s-loss2-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 6528 → 9677 | 1684 → 2494 | 295 → 441 | 6528 → 9677 |
| 04-tpu-self | [`ts-s-lr1e-1-vs-golden.curves`](../alignments/04-tpu-self/results/ts-s-lr1e-1-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 119 → 116 | 39.7 → 39 | 26.1 → 25.4 | 119 → 116 |
| 04-tpu-self | [`ts-s-lr1e-2-vs-golden.curves`](../alignments/04-tpu-self/results/ts-s-lr1e-2-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 10.8 → 10.8 | 3.9 → 3.89 | 3.38 → 3.37 | 10.8 → 10.8 |
| 04-tpu-self | [`ts-s-lr1e-3-vs-golden.curves`](../alignments/04-tpu-self/results/ts-s-lr1e-3-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 1.41 → 1.41 | 1.06 → 1.06 | 0.823 → 0.824 | 1.41 → 1.41 |
| 04-tpu-self | [`ts-s-lr1e-4-vs-golden.curves`](../alignments/04-tpu-self/results/ts-s-lr1e-4-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.662 → 0.662 | 0.976 → 0.976 | 0.581 → 0.582 | 0.976 → 0.976 |
| 04-tpu-self | [`ts-s-lr1e-5-vs-golden.curves`](../alignments/04-tpu-self/results/ts-s-lr1e-5-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.223 → 0.224 | 0.938 → 0.939 | 0.633 → 0.634 | 0.938 → 0.939 |
| 04-tpu-self | [`ts-s-lr3e-5-vs-golden.curves`](../alignments/04-tpu-self/results/ts-s-lr3e-5-vs-golden.curves.results.json) | `C.grad_norm` | 待定 → 待定 | 0.435 → 0.435 | 1.01 → 1.01 | 0.768 → 0.77 | 1.01 → 1.01 |
| 04-tpu-self | [`ts-s-mtp1e-1-vs-golden.curves`](../alignments/04-tpu-self/results/ts-s-mtp1e-1-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 56.6 → 56.9 | 14.6 → 14.7 | 6.75 → 6.81 | 56.6 → 56.9 |
| 04-tpu-self | [`ts-s-mtp1e-2-vs-golden.curves`](../alignments/04-tpu-self/results/ts-s-mtp1e-2-vs-golden.curves.results.json) | `C.grad_norm` | 不过 → 不过 | 4.82 → 4.83 | 1.56 → 1.56 | 1.03 → 1.04 | 4.82 → 4.83 |
| 04-tpu-self | [`ts-s-mtp1e-3-vs-golden.curves`](../alignments/04-tpu-self/results/ts-s-mtp1e-3-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.0754 → 0.0765 | 0.896 → 0.896 | 0.61 → 0.61 | 0.896 → 0.896 |
| 04-tpu-self | [`ts-tp2-1k-vs-golden.curves`](../alignments/04-tpu-self/results/ts-tp2-1k-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.209 → 0.208 | 0.913 → 0.913 | 0.706 → 0.707 | 0.913 → 0.913 |
| 04-tpu-self | [`ts-tp2-1k-vs-ts-g-a-1000.curves`](../alignments/04-tpu-self/results/ts-tp2-1k-vs-ts-g-a-1000.curves.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.488 → 0.535 | 1.13 → 1.13 | 0.763 → 0.8 | 1.13 → 1.13 |
| 04-tpu-self | [`ts-tp2-vs-golden.curves`](../alignments/04-tpu-self/results/ts-tp2-vs-golden.curves.results.json) | `C.grad_norm` | 过 → 过 | 0.209 → 0.208 | 0.913 → 0.913 | 0.706 → 0.707 | 0.913 → 0.913 |
| 05-gpu-scale | [`g3-a-vs-multi-node`](../alignments/05-gpu-scale/results/g3-a-vs-multi-node.results.json) | `C.grad_norm` | 过 → 过 | 0.22 → 0.221 | 0.842 → 0.842 | 0.541 → 0.541 | 0.842 → 0.842 |
| 05-gpu-scale | [`mn-a-vs-single-node`](../alignments/05-gpu-scale/results/mn-a-vs-single-node.results.json) | `C.grad_norm` | 过 → 过 | 0.262 → 0.261 | 0.867 → 0.867 | 0.676 → 0.675 | 0.867 → 0.867 |
| 05-gpu-scale | [`mn-b-vs-single-node`](../alignments/05-gpu-scale/results/mn-b-vs-single-node.results.json) | `C.grad_norm` | 过 → 过 | 0.352 → 0.353 | 0.799 → 0.799 | 0.69 → 0.69 | 0.799 → 0.799 |
| 05-gpu-scale | [`mn-c-vs-single-node`](../alignments/05-gpu-scale/results/mn-c-vs-single-node.results.json) | `C.grad_norm` | 过 → 过 | 0.453 → 0.454 | 0.775 → 0.775 | 0.749 → 0.749 | 0.775 → 0.775 |
| 05-gpu-scale | [`mn-d-vs-single-node`](../alignments/05-gpu-scale/results/mn-d-vs-single-node.results.json) | `C.grad_norm` | 过 → 过 | 0.292 → 0.293 | 0.777 → 0.777 | 0.764 → 0.764 | 0.777 → 0.777 |
| 05-gpu-scale | [`mn-e-vs-single-node`](../alignments/05-gpu-scale/results/mn-e-vs-single-node.results.json) | `C.grad_norm` | 过 → 过 | 0.49 → 0.491 | 0.799 → 0.799 | 0.822 → 0.822 | 0.822 → 0.822 |
| 05-gpu-scale | [`mn-ep16-vs-single-node`](../alignments/05-gpu-scale/results/mn-ep16-vs-single-node.results.json) | `C.grad_norm` | 过 → 过 | 0.161 → 0.161 | 0.788 → 0.788 | 0.746 → 0.744 | 0.788 → 0.788 |
| 05-gpu-scale | [`mn-f-vs-single-node`](../alignments/05-gpu-scale/results/mn-f-vs-single-node.results.json) | `C.grad_norm` | 待定 → 待定 | 0.703 → 0.704 | 0.861 → 0.862 | 1.04 → 1.04 | 1.04 → 1.04 |
| 05-gpu-scale | [`mn-g-vs-single-node`](../alignments/05-gpu-scale/results/mn-g-vs-single-node.results.json) | `C.grad_norm` | 过 → 过 | 0.929 → 0.93 | 0.885 → 0.886 | 0.912 → 0.912 | 0.929 → 0.93 |
| 05-gpu-scale | [`mn-h-vs-single-node`](../alignments/05-gpu-scale/results/mn-h-vs-single-node.results.json) | `C.grad_norm` | 过 → 过 | 0.589 → 0.59 | 0.805 → 0.805 | 0.671 → 0.671 | 0.805 → 0.805 |
| 05-gpu-scale | [`mn-lr-vs-single-node`](../alignments/05-gpu-scale/results/mn-lr-vs-single-node.results.json) | `C.grad_norm` | 不过 → 不过 | 95.5 → 93.5 | 38.9 → 38.2 | 27.1 → 26.3 | 95.5 → 93.5 |
| 05-gpu-scale | [`mn-pp2-vs-single-node`](../alignments/05-gpu-scale/results/mn-pp2-vs-single-node.results.json) | `C.grad_norm` | 待定 → 待定 | 0.47 → 0.472 | 0.817 → 0.817 | 1.02 → 1.02 | 1.02 → 1.02 |
| 05-gpu-scale | [`mn4-pp2-vs-multi-node`](../alignments/05-gpu-scale/results/mn4-pp2-vs-multi-node.results.json) | `C.grad_norm` | 过 → 过 | 0.605 → 0.606 | 0.88 → 0.88 | 0.679 → 0.681 | 0.88 → 0.88 |
| 05-gpu-scale | [`mn4-pp2-vs-single-node`](../alignments/05-gpu-scale/results/mn4-pp2-vs-single-node.results.json) | `C.grad_norm` | 过 → 过 | 0.458 → 0.46 | 0.833 → 0.833 | 0.721 → 0.721 | 0.833 → 0.833 |
| 05-gpu-scale | [`rw-a-vs-rws`](../alignments/05-gpu-scale/results/rw-a-vs-rws.results.json) | `C.grad_norm` | 过 → 过 | 0.757 → 0.757 | 0.642 → 0.642 | 0.862 → 0.862 | 0.862 → 0.862 |
| 05-gpu-scale | [`rw-b-vs-rws`](../alignments/05-gpu-scale/results/rw-b-vs-rws.results.json) | `C.grad_norm` | 过 → 过 | 0.0833 → 0.0835 | 0.766 → 0.766 | 0.748 → 0.748 | 0.766 → 0.766 |
| 05-gpu-scale | [`rw-c-vs-rws`](../alignments/05-gpu-scale/results/rw-c-vs-rws.results.json) | `C.grad_norm` | 待定 → 待定 | 0.641 → 0.642 | 0.846 → 0.846 | 1.03 → 1.03 | 1.03 → 1.03 |
| 05-gpu-scale | [`rw-d-vs-rws`](../alignments/05-gpu-scale/results/rw-d-vs-rws.results.json) | `C.grad_norm` | 过 → 过 | 0.178 → 0.178 | 0.615 → 0.615 | 0.644 → 0.644 | 0.644 → 0.644 |
| 05-gpu-scale | [`rw-e-vs-rws`](../alignments/05-gpu-scale/results/rw-e-vs-rws.results.json) | `C.grad_norm` | 过 → 过 | 0.914 → 0.915 | 0.705 → 0.705 | 0.842 → 0.843 | 0.914 → 0.915 |
| 05-gpu-scale | [`rw1k-a-vs-rws`](../alignments/05-gpu-scale/results/rw1k-a-vs-rws.results.json) | `C.grad_norm` | 待定 → 待定 | 1.33 → 1.33 | 0.813 → 0.813 | 1 → 1 | 1.33 → 1.33 |
| 05-gpu-scale | [`sn-a-vs-single-node`](../alignments/05-gpu-scale/results/sn-a-vs-single-node.results.json) | `C.grad_norm` | 过 → 过 | 0.641 → 0.643 | 0.883 → 0.883 | 0.78 → 0.779 | 0.883 → 0.883 |
| 05-gpu-scale | [`sn-ep8-vs-single-node`](../alignments/05-gpu-scale/results/sn-ep8-vs-single-node.results.json) | `C.grad_norm` | 过 → 过 | 0.603 → 0.604 | 0.757 → 0.757 | 0.67 → 0.67 | 0.757 → 0.757 |
| 06-rw3 | [`rw3-l4-first164`](../alignments/06-rw3/results/rw3-l4-first164.results.json) | `L4.grad_norm` | 待定 → 待定 | 0.896 → 0.895 | 1.2 → 1.2 | 0.575 → 0.577 | 1.2 → 1.2 |

数据（含各统计量与 N 的原值）：[`grad-norm-denominator-rejudge.json`](grad-norm-denominator-rejudge.json)。
