[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 11

# 案例 11：iHC 四路流加权和，TPU 默认配对相加、GPU 依次相加——M28 让顺序一致

> **状态：已修（`t1v-*` / `t1w-*` 起）。对 CPU FP64 的改善在后一代未复现（差异清单 U11）。现行 C 口径的 L3 第 1 步已重判（`tc-a` 同配置的第 1 步全量抓数 `s1f-c`，GPU 第 1 步参照 6 份）：还有 1 个 iHC 张量未定责，是第 3 层 FFN 前 `hc_fn`（1.14 N，R 1.43），按 D11 豁免规则单列、计入未定责；C 口径之前 `tz-off-rep` 未定责的那个（第 3 层注意力前 `hc_scale`，1.05 N；3 份时 5 个，3 个是关 base2 后新出现的）在 C 口径下 0.66 N，回到尺子内。base2 开那一代的 6 个已定责、已接受（[D11](../alignments/01-mini/DETAILS.md#d11)）。读数见 [PROTOCOL](../PROTOCOL.md) 第 0 节 L3 一行。** 层号按 HF 检查点从 0 数。

## 现象

第 1 步参数梯度对 CPU FP64 整网重算比 R。M28 关时（`t1t`），42 个 iHC `hc_pre` 张量（`hc_fn`、`hc_base`、`hc_scale`）里 11 个 R > 1.2，如第 2 层注意力前 `hc_scale`：GPU 2.15e-3，TPU 9.01e-3，R 4.18；第 3 层 FFN 前 `hc_base`：GPU 3.04e-2，TPU 6.74e-2，R 2.22（[`t1t-step1-grad-vs-fp64.json`](../alignments/01-mini/results/t1t-step1-grad-vs-fp64.json)）。300 步 L4 param_norm（精确值）MARE：N 1.90e-7，TPU 3.04e-7，r 1.60，待定（[案例 12](12-param-norm-fp32-sum-and-drift.md)）。

## 机制

记 tₖ = preₖ·xₖ（FP32 乘，各舍一次），四项相加后才舍成 BF16：

| 写法 | 优化后 HLO | 相加顺序 |
|---|---|---|
| TPU 默认：`einsum('bsk,bskd->bsd', precision=HIGHEST)` | algsimp 改写成 FP32 逐元素乘接沿 k 的 FP32 reduce（kLoop），不上矩阵单元 | (t₀ + t₂) + (t₁ + t₃) |
| TPU 开 M28：逐路乘加 | kLoop：FP32 乘与 add 链 | ((t₀ + t₁) + t₂) + t₃ |
| GPU `Hy4HyperConnection`：`torch.sum(pre * x)`，autocast 关 | — | ((t₀ + t₁) + t₂) + t₃ |

两种写法精度相同，只差加法树，差在 FP32 1 ulp 量级。iHC 参数梯度是对全部 token 的长求和，条件数 κ = Σ\|项\| ÷ \|Σ项\| 第 1 层 710–1359、第 2 层 25、第 3 层 93–96（GPU `g2s-a`，一条 4096 token 序列，[`t1-ihc-post-grad-conditioning.json`](../alignments/01-mini/results/t1-ihc-post-grad-conditioning.json)），把这一差放大到第 1 步梯度和 param_norm 上。κ 只解释放大，不定哪边错。

## 证据

- **HLO 实证**：CPU 上 AOT 编译 v5p 训练步（MaxText `3741e818`），只改开关。M28 关时 StableHLO 是 f32 的 `dot_general`（HIGHEST），`algsimp` 把 54 个 dot 全部改写成 broadcast → multiply → 沿 k 的 reduce；关掉 algsimp 与融合则留成 54 个 convolution（[片段](../alignments/01-mini/results/m28-ihc-stream-sum-hlo-snippets.txt)、[汇总](../alignments/01-mini/results/m28-ihc-stream-sum-hlo.json)）。
- **TPU 相加顺序（逐位）**：v5p 上单独运行各写法（`hlo_probe.py`，输入取 GPU 第 1 步第 1 层），与 numpy 按不同顺序重算比：HIGHEST einsum、`jnp.sum` 100% 等于配对相加（对 FP64 rel_L2 3.79e-8），逐路乘加 100% 等于依次相加（4.01e-8）（[`t1-hlo-probe-tpu.json`](../alignments/01-mini/results/t1-hlo-probe-tpu.json)）。
- **GPU 相加顺序**：B200 上 `torch.sum(pre * x)` 419 万个输出 100% 等于依次相加（配对相加 60.9%）（台账 G29，`analysis/ihc_sum_order_probe.py`）。
- **只开 M28 后**（`t1t` → `t1v`）：param_norm MARE TPU 3.04e-7 → 1.68e-7（N 1.90e-7），r 1.60 → 0.88；1000 步 r 1.06 → 0.43（8 份尺子，N 4.05e-6，`t1w-l4` MARE TPU 1.74e-6）。两组都是单份运行：300 步的变化 0.72 在只改执行方式的波动线 1.07 以内（[`exec-spread-300.json`](../alignments/01-mini/gpu-noise/results/exec-spread-300.json)）；1000 步只改执行方式的重复，主 loss r 就能在 0.90–1.69 之间摆（16 份尺子，[知识库](../knowledge.md#mini-上关-base2-为什么离-gpu-更远要点与出处)）。单凭曲线不能归到开关。第 1 步对 CPU FP64：`hc_pre` 42 个里 R > 1.2 的 11 → 7（含 `hc_head` 的 iHC 全组 45 个是 11 → 8），第 3 层 FFN 前 `hc_base` TPU 6.74e-2 → 3.14e-3（GPU 3.04e-2）（[`t1v-step1-grad-vs-fp64.json`](../alignments/01-mini/results/t1v-step1-grad-vs-fp64.json)）。
- **不是 excess precision**：`t1zu-a` 一代配置关掉 excess precision，iHC R > 1.2 的 11 → 14，iHC rel_L2 中位数 8.07e-3 → 9.47e-3（GPU 1.00e-2）；只关融合是 16 个（[`t1-step1-compiler-pass-variants.json`](../alignments/01-mini/results/t1-step1-compiler-pass-variants.json)）。同一组里 `embed_tokens` 梯度离 FP64 变远（[r22](../ROOT-CAUSES.md#r22)）。

## 结论

- **修法**：台账 M28（[CHANGES-LEDGER](../CHANGES-LEDGER.md)），MaxText 开关 `ihc_stream_sum_unrolled`（默认关，`2aa332cb`），T1 打开，`t1v-*` 起生效。代价：单开关 A/B +0.2%。根因一览 [r6](../ROOT-CAUSES.md#r6)。
- **定责**：读代码对照 + 两边逐位实测，M28 让 TPU 这一处的相加顺序与 GPU 相同。只改 TPU 一边后 param_norm 的差收拢（r 1.60 → 0.88），但这是单份运行、在只改执行方式的波动内，不作定责依据。M28 不是「更精确」：单算子上配对相加对 FP64 还略好（3.79e-8 对 4.01e-8）。
- **副作用**：`t1w-l4` 1000 步主 loss 带符号均值 N 3.78e-6，TPU +7.08e-6，r 1.88；由 M36 修掉大部分，`t1zr-l4` MARE N 8.24e-6，TPU 8.67e-6，r 1.05（[D26](../alignments/01-mini/DETAILS.md#d26)、[r16](../ROOT-CAUSES.md#r16)）。
- **编译选项的影响**：只换编译选项，iHC 里 R > 1.2 的张量就在 11–16 个之间摆动，按 R 给 iHC 张量定责要把这个摆动算进去。
- 待查：对 FP64 的改善要用只差 M28 的一对运行补 R（U11）；只差 M28 的第 1 步 iHC 梯度对 GPU 的 rel_L2 中位数开 8.31e-3、关 8.18e-3（GPU 三份之间 3.58e-3），对 GPU 没变近（[`t1-switch-isolation-tpu.json`](../alignments/01-mini/results/t1-switch-isolation-tpu.json)）。
