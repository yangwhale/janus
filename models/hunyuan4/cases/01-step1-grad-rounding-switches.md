[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 01

# 案例 01：第 1 步逐张量梯度普遍是 GPU 散布的 2–3 倍——TPU 的舍入位置与 Megatron-Core + TE 不同，四个开关对齐

> **状态：已修（台账 M8–M10、M12，T1 开，`t1s-*` 起）。** 剩下超过 2N 的张量对 CPU FP64 定责，见[检查清单](../alignments/01-mini/CHECKS.md) L3。

## 现象

第 1 步逐张量梯度，TPU 对最近一份 GPU 的 rel_L2 ÷ GPU 三份（`g1s-a/b/c`）两两的最大 rel_L2：中位数 2.67，195 个张量里 139 个超过 2（`t1diag4`，[`t1-step1-precision-switches.json`](../alignments/01-mini/results/t1-step1-precision-switches.json)，只存中位数与计数）。从输出头（`lm_head.weight` 3.41）到第 0 层（约 2）普遍 2–3，没有哪一处突出（[`t1-step1-grad-per-tensor.json`](../alignments/01-mini/results/t1-step1-grad-per-tensor.json)）。

## 机制

GPU 一侧（Megatron-Core 0.20 + TE 2.20）的舍入位置（[`PROTOCOL.md`](../PROTOCOL.md) 第 1.2 节「GPU 一侧的舍入位置」行），MaxText 默认写法四处都不同：

- TE GEMM 跨分片的部分和在 FP32 里累加、只舍一次（M12）；
- TE RMSNorm 只舍一次（M9）；
- 路由 GEMM 用先舍成 BF16 的权重在 FP32 里乘（M8）；
- 不融合的 SwiGLU 按算子舍入（M10）。

GPU 同设置重跑之间只差原子加的顺序，舍入位置相同，所以 GPU 噪声这把尺子窄；舍入位置不同造成的差在全网整体超出它。

## 证据

| 运行 | 设置 | 比值中位数 | 超过 2 的张量 / 195 |
|---|---|---|---|
| `t1diag4` | 开关全关 | 2.67 | 139 |
| `t1diag7` | + M12、M9、M8 | 1.65 | 52 |
| `t1diag8` | + 以上 + M10 | 1.72 | 62 |

出处：[`t1-step1-precision-switches.json`](../alignments/01-mini/results/t1-step1-precision-switches.json)。这一比值是两边的差，不定责。

- 只把输出头设成 `logits_matmul_precision: highest`，逐张量结论不变（提交 `0a39dd5`，没有单独的结果文件；见[案例 04](04-lm-head-single-bf16-pass.md)）。
- 修后对 CPU FP64（`t1diag15`，五个开关含 M23）：205 个参数梯度的 R 中位数 0.95（rel_L2 中位数 GPU 6.60e-3、TPU 6.18e-3）；R > 1.2 的 13 个全是 iHC 的 `hc_pre` 参数与 MTP `enorm`；超 2N 的 58 个里未定责的 3 个都是 iHC（[案例 11](11-ihc-stream-sum-highest-einsum.md)）（[`t1s-step1-grad-vs-fp64.json`](../alignments/01-mini/results/t1s-step1-grad-vs-fp64.json)）。
- M12 的旁证（混元 3 `phase2-l1-ops-own`，v5p 单算子对 FP64，开关关）：qkv GPU 1.89e-3、TPU 2.33e-3，R 1.24；稠密 fc1 GPU 1.99e-3、TPU 2.58e-3，R 1.29。
- 300 步 L4（`t1-l4-nozm` → `t1s-l4`）主 loss、grad_norm、索引器 KD 修后仍不过，这一代同时开了 M23，是两边的差、不定责（[`t1-l4-nozm.results.json`](../alignments/01-mini/results/t1-l4-nozm.results.json)、[`t1s-l4.results.json`](../alignments/01-mini/results/t1s-l4.results.json)）；后续见[案例 06](06-gpu-attn-scores-bf16.md)、[案例 02](02-indexer-kd-f32-lse.md)。

## 结论

- **修法**：TPU 打开四个 MaxText 开关（默认关，[`CHANGES-LEDGER.md`](../CHANGES-LEDGER.md)）：M8 `routed_gate_bf16_rounding`、M9 `rmsnorm_single_rounding`、M10 `mlp_activation_round_per_op`、M12 `dense_fp32_accumulation`，T1 都开。同批打开的 M23 `indexer_fp32_scores` 与本案统计无关，见[案例 02](02-indexer-kd-f32-lse.md)。
- **定责依据**：读代码，对照 TE 的舍入位置（上面四处）；修后对 CPU FP64 的 R 中位数 0.95（GPU 6.60e-3、TPU 6.18e-3）。修前没有对 FP64 的 R。M10 让统计略变差（1.65 → 1.72），仍按 GPU 实际写法（不融合、按算子舍入）打开（提交 `534fa84`）。
- **生产建议**：见[根源一览第 2 行](../ROOT-CAUSES.md#r2)；对齐报告 [D3](../alignments/01-mini/DETAILS.md#d3)。
