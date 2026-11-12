[janus](../../README.md) › 混元 4

# 混元 4

混元 4 从 GPU（B200 × 8，Megatron-Core）迁到 TPU（v5p，MaxText）训练，这里检查两边训出来的数值是否一致。

主线是 **hy4-mini**：混元 4 的全部结构、缩小宽度和层数的随机初始化模型。

**现状：还不能签字**（签字＝正式宣布两边对齐了）。以现行比较 tc-a 为准：2026-11-11 起对齐配置改成 C 口径（TPU 在超连接与 hc_head 一带照 GPU 的写法「32 位算、16 位存」，GPU 不改），现行比较是 C 口径的第一份运行 `tc-a`，1000 步噪声用 GPU 16 份。hy4-mini 训练 1000 步的 9 项曲线没有一项不过：8 项对上，路由偏置均值 1 项略超出 GPU 自己重跑的波动（带符号均值 1.31 倍，未定责；C 口径另 5 份 0.36–0.80 倍，其中只改执行方式的 4 份 0.39–0.80，0.36 是改了重算策略、反向会重选路由的 `tc-e`）；主 loss 对上（三个量里最差的平均绝对相对误差 0.98 倍，贴线）。C 口径 6 份都判时没有不过，但不是 6 份都过：`tc-b`（只改 scoped VMEM 64 MiB）主 loss 全程 1.14 倍、MTP 1.06 倍待定，`tc-f`（只改 scoped VMEM 48 MiB）1000 步梯度范数最大单步 1.04 倍待定（平均绝对相对误差 0.95 倍，过；GPU 留一法 15 份最大 0.90；[`t1zc-repeats-6.json`](alignments/01-mini/results/t1zc-repeats-6.json)）。同一运行前 300 步另判也是 8 项对上，梯度范数 1 项略超（1.01 倍；另 5 份 `tc-b … tc-f` 0.81–0.92 倍、都过，[`t1zc300-repeats.json`](alignments/01-mini/results/t1zc300-repeats.json)；`tc-f` 0.90 见 [`mini-config-c-vs-m58.json`](alignments/01-mini/results/mini-config-c-vs-m58.json)），状态待定。C 口径之前的上一代（`tz-off-rep`）主 loss 全程与第 101 步起略超（1.55、1.52 倍），后段偏高的主因是超连接里流的梯度怎么舍：在只开 M36 的对照上只加开这一处舍法（M58），第 551–1000 步主 loss 对 GPU 16 份均值从 +9.4e-6 降到 +1.9e-6（p 0.016），但 5 份里 2 份仍超尾部阈值（第 751–1000 步均值超 3e-6）。但现行 C 口径（M36、M52、M58 都开）6 份同一窗口平均仍高 +6.9e-6，6 份里 5 份尾部还在，和只开 M36 的对照分不开（p 0.32；[`mini-factorial-tpu-6.json`](alignments/01-mini/results/mini-factorial-tpu-6.json)），也就是 C 口径的后段没修掉：M52 开着时补 M58 全套或只补任一半，对只开 M52 的 4 份都测不出作用（p 0.14 / 0.14 / 0.54，[`mini-factorial-c-parts-vs-m52.json`](alignments/01-mini/results/mini-factorial-c-parts-vs-m52.json)），机制待查（见[对齐报告](alignments/01-mini/ALIGNMENT-REPORT.md)）。下面两项判档不是过，且是 C 口径之前的读数，C 口径上没重判：另一份运行 `tz-off-qkc`（只多记逐头最大 logit）派生的每步全局最大注意力打分（`tz-off-qkq`）1000 步略超（1.09 倍；base2 开时已接受，关 base2 后没有重做同一状态的证据，是否沿用尚未决定）；低阈值 QK-Clip 第 1 步第 4 层逐头最大 logit 离精确值是 GPU 的 1.55 倍（不过，未定责）。另有这些检查也停在 C 口径之前或 base2 开的读数、C 口径上没重判（判档照旧沿用，清单见[协议](PROTOCOL.md)第 7 节 2026-11-11「补记 C 口径上没重做的检查」一行）：第 1 步优化器一步、路由一致率、L1 单算子回放、L2 逐层、健康信号、同设置重跑逐位、存档续跑逐位、抓数不改数值。第 1 步整网梯度已在 C 口径上重判（[`t1zc-step1-train-grad-vs-fp64-n6.json`](alignments/01-mini/results/t1zc-step1-train-grad-vs-fp64-n6.json)）：1 个超连接小参数（第 3 层 FFN 前 `hc_fn`）超出 GPU 6 份之间的波动 1.14 倍、离精确值是 GPU 的 1.43 倍，计入未定责（待定；MTP enorm 另列；两个都按 [D11 豁免规则](alignments/01-mini/DETAILS.md#d11-rule) 单列：`hc_fn` 计入未定责（base2 开那一代登记接受的 6 个之一，接受理由没有数值门槛、复核不了；是否沿用待拍板），`enorm` 不计入（D13 的理由「两边都小于 u」复核成立）；单列的挡不挡签字：待拍板；GPU 第 1 步参照 2026-11-11 从 3 份补到 6 份，C 口径之前那一代 3 份时是 5 个未定责、不过；6 份是看过 3 份的判定之后补跑的，能否作正式判定待拍板）。挡签字的事见[对齐报告](alignments/01-mini/ALIGNMENT-REPORT.md#挡签字)。项目的验收对象是真实权重截层（只取混元 4 真实权重的前几层组成的小模型），那边的签字条件一条都还没满足。

## 按这个顺序读

| # | 读什么 | 回答什么 |
|---|---|---|
| 1 | [总览](dashboard/dashboard.md) | 每一项检查对上没有，一行一句话 |
| 2 | [mini 对齐报告](alignments/01-mini/ALIGNMENT-REPORT.md) | 为什么、生产怎么配、还差什么 |

## 知识库（查询用）

台账、开关表、根源、协议、运行登记、其他尺寸等都在知识库，入口是[知识库索引](knowledge.md)：每份文件一行，写它回答什么问题、什么时候查；另有[术语对照表](knowledge.md#术语对照表)，知识库里的术语都能在那里查到白话。
