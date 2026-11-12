[janus](../../../../README.md) › [混元 4](../../README.md) ｜ 01 · mini（hy4-mini，T1）

# 混元 4 · 01 · mini：hy4-mini（427,783,489 参数），TPU v5p 8 个设备 ↔ B200 × 8

hy4-mini 保留混元 4 的全部结构与每头维度，只缩宽度与个数（7 层 + 1 层 MTP，随机初始化），在上面做生产式训练（Muon + QK-Clip + 索引器 KD + MTP）的完整 L0–L4 对齐。根源一览见 [`ROOT-CAUSES.md`](../../ROOT-CAUSES.md)。GPU（Megatron-Core）是 golden，TPU（MaxText）是被测，噪声尺子只用 GPU 重跑。

- **给人读的**：[总览](../../dashboard/dashboard.md)、[对齐报告](ALIGNMENT-REPORT.md)；依据在[对齐细节](DETAILS.md)（知识库，偏差台账、开关表、签字清单）。
- 逐项比什么：[检查清单](CHECKS.md)；历代比较：[历代比较](../../dashboard/history.md)。

## 这个文件夹

| 文件 / 文件夹 | 放什么 |
|---|---|
| [`ALIGNMENT-REPORT.md`](ALIGNMENT-REPORT.md) | 给人读的结论：现状、生产建议、挡签字、下一步 |
| [`DETAILS.md`](DETAILS.md) | 对齐细节（知识库）：偏差台账、开关表、签字清单、证据索引 |
| [`CHECKS.md`](CHECKS.md) | 检查清单（由 `models/hunyuan4/dashboard/build.py` 按 `dashboard/checks.yaml` 生成） |
| [`config-alignment.md`](config-alignment.md) | 两边每一项配置怎么对上 |
| [`goldens.md`](goldens.md) | 每一份运行的来历与代号 |
| [`setup.md`](setup.md)、[`runs.yaml`](runs.yaml) | 设置与运行登记 |
| `catalogs/`、`results/` | 检查项目录；结果文件。目录文件注释里的「现行」指成文那一天的现行比较，现在以 [`DETAILS.md`「代号」](DETAILS.md#代号)为准 |
| [`gpu-noise/`](gpu-noise/README.md) | GPU 一侧自比 |

本尺寸 TPU 配置是 [`hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml)。

## 现状

现行配置（2026-11-11 起 C 口径：M36、M52、M58 都开，base2 关）跑 1000 步，对 GPU 基线逐步比，噪声取同一代 GPU 16 份两两 120 对：

| 比较 | 步数 | 被测 / 基线 | 结果 |
|---|---|---|---|
| [`t1zc-l4`](results/t1zc-l4.report.md) | 1000 | TPU `tc-a`（C 口径 6 份的第一份；`tc-e` 改了重算策略，其余只改执行方式）/ GPU `g3-a` | 过 8、不过 0、待定 1（路由偏置均值 r 1.31，C 口径另 5 份过）；主 loss 过（r 0.98）。C 口径 6 份都没有不过（`tc-b` 主 loss 1.14、MTP 1.06–1.07，`tc-f` 梯度范数最大单步 1.04 待定（MARE 0.95 过）；[`t1zc-repeats-6.json`](results/t1zc-repeats-6.json)（5 份时 [`t1zc-repeats.json`](results/t1zc-repeats.json)））。后段偏高的主因已定位到 iHC 流梯度的舍入写法（只开 M36 + M58 那一格第 551–1000 步均值降到 +1.9e-6，p 0.016，5 份里 2 份仍超尾部阈值）；C 口径 6 份第 551–1000 步还高 +6.9e-6，和只开 M36 的对照分不开（p 0.32，[`mini-factorial-tpu-6.json`](results/mini-factorial-tpu-6.json)），C 口径的后段没修掉：M52 开着时补 M58 全套或任一半都测不出作用（对只开 M52 p 0.14 / 0.14 / 0.54，[`mini-factorial-c-parts-vs-m52.json`](results/mini-factorial-c-parts-vs-m52.json)），机制待查（[D30](DETAILS.md#d30)） |
| [`t1zo-l4`](results/t1zo-l4.report.md) | 1000 | 上一代：TPU `tz-off-rep`（C 口径之前，M52、M58 不开）/ GPU `g3-a` | 过 7、不过 0、待定 2（主 loss 全程与后段，r 1.55 / 1.52）；更早的 `t1zg-a`（base2 开）见[历代比较](../../dashboard/history.md) |

| mini 挡签字的项 | 状态 |
|---|---|
| 第 1 步 163 个超出噪声 | C 口径（2026-11-11 重判）对 GPU 6 份 15 对：超出噪声且比 GPU 远的是第 3 层 FFN 前 `hc_fn`（1.14 N）与 MTP enorm，按 [D11 豁免规则](DETAILS.md#d11-rule) 单列：`hc_fn` 计入未定责，`enorm` 不计入（挡不挡签字待拍板）；C 口径之前是 169 个超出、第 3 层注意力前 `hc_scale` 计入未定责；仍缺 2b |
| 训练曲线待定（2a） | 以现行比较 tc-a 为准：1000 步路由偏置均值待定（r 1.31，C 口径另 5 份过）；前 300 步 grad_norm 待定（r 1.01，C 口径另 4 份 `tc-b … tc-e` 0.81–0.92（`tc-f` 前 300 步 0.90，[`mini-config-c-vs-m58.json`](results/mini-config-c-vs-m58.json)）、都过）；主 loss 现已刚好过线（MARE 0.98；C 口径另 `tc-b` 主 loss 1.14 待定；后段没修掉，见 D30）；1000 步 grad_norm `tc-a` 过，C 口径 `tc-f` 最大单步 1.04 待定（MARE 0.95 过）。QK-Clip 全局最大 logit 1000 步与前 300 步待定（1.09、1.05）是 C 口径之前另一份运行（`tz-off-qkq`，配置与 `tc-a` 不同）的读数，C 口径上没重判；它与 grad_norm 在 base2 开时已接受，是否沿用尚未决定（[D2](DETAILS.md#d2)、[D29](DETAILS.md#d29)） |
| 低阈值 QK-Clip 第 1 步第 4 层（3a） | C 口径之前、关 base2 后 R 1.55，未定责；C 口径上要重测（[D31](DETAILS.md#d31)） |
| golden 对 GPU 生产程序 | 没比 |
| 三条事后判法的前瞻验证 | 已在 PROTOCOL 事先登记，用真实权重截层下一批运行验证 |

逐元素 log（[D14](DETAILS.md#d14)）已定责、已接受；param_norm 漂移（[D12](DETAILS.md#d12)）已过线，不挡签字。


## 真实权重截层的 L0 与 L1（不依赖优化器）

阶段 A–D 是混元 4 真实权重的截层（A＝第 0 层，B＝第 0–1 层，C＝第 0–2 层，D＝第 0–1 层加 1 层 MTP；`model/hy4_map.py` 的 `STAGES`）。L0 转换与阶段 A 的 L1 单算子不依赖优化器，仍有效；T2 的端到端训练在 [real-weight](../02-real-weight/README.md)。

| 检查 | 阶段 | 结果 | 结果文件 |
|---|---|---|---|
| HF 目录：135 个张量的名字、形状、dtype；参数量 42,070,313,621 | 全部 | pass | [`hy4-l0-names`](results/hy4-l0-names.results.json) |
| 转换后逐张量还原回 HF 对 `tensor_sha256.json` 逐位，含 dtype 清单 | A / B / C / D | pass（31 / 60 / 84 / 87 个张量） | `hy4-{a,b,c,d}-l0-ckpt-convert`，例 [`A`](results/hy4-a-l0-ckpt-convert.report.md) |
| RoPE 列去交错与逆置换 | A / B / C / D | pass（6 / 12 / 14 / 18 个张量） | `hy4-{a,b,c,d}-l0-rope` |
| 跨侧：TPU 初始权重对 GPU golden 初始参数清单 | A、B | pass（248 / 248、472 / 472，对 AdamW 时期的 GPU golden：A 是 `ga-b`、`ga1-a`，B 是 `gb-a`；现行 T2 的 Megatron-Core golden 上这一项没做，见 [real-weight 签字清单](../02-real-weight/ALIGNMENT-REPORT.md#签字清单) 1a） | [`A`](results/hy4-a-l0-init-vs-gpu.report.md)、[`B`](results/hy4-b-l0-init-vs-gpu.report.md) |
| 数据回放：300 步 × 8 rank × 2 microbatch 的输入指纹对 GPU 运行 | A | 4800 个单元 0 处不同 | [`ga-a`](results/hy4-a-replay-check-ga-a.txt) |
| 阶段 A 第 0 层 10 个算子隔离回放，对 FP64（GPU 一侧是 HF 参照模型） | A | 全过，R 0.18–1.00 | [`hy4-a-l1-op-tpu-fix.json`](results/hy4-a-l1-op-tpu-fix.json) |

- 单算子回放把每个算子的输入换成 GPU 第 1 步抓到的同一个输入，GPU 一侧的 FP64 结果在 [`ga1-a.fp64_ops.json`](../../results/ga1-a.fp64_ops.json)。这里 GPU 是 HF 写法（RMSNorm 两次舍入），RMSNorm 的 R 0.71 偏向 TPU（推断）。
- 接线核对（诊断）：CPU 上用阶段 A 检查点跑一次前向、按模块对 GPU 抓的张量，正常转换各模块 rel_L2 在 BF16 量级（logits 9.43e-3）；不去交错的负对照在注意力处大 20 倍以上（logits 1.67e-1）（[`hy4-a-probe-forward-cpu.json`](results/hy4-a-probe-forward-cpu.json)）。

## 其他

- **显存**：编译期估算 [`hy4-mini-aot-memory.json`](results/hy4-mini-aot-memory.json)，实测峰值在 [`hy4-mini-tpu-determinism.json`](results/hy4-mini-tpu-determinism.json) 的 `peak_bytes_in_use_max_local_devices`。
