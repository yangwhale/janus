[janus](../../../../README.md) › [混元 4](../../README.md) ｜ 02 · real-weight（支线）

# 混元 4 · 02 · real-weight：真实权重截层 B、D1（v5p 2x2x2 ↔ B200 × 8）

支线：主线结论在 [mini](../01-mini/README.md)。这里用 hy4-preview 的真实权重截两种层，看换成真实权重后 mini 的结论还在不在：

| 截层 | 结构 | 参数量 | 测什么 |
|---|---|---|---|
| B | 第 0 层稠密 + 第 1 层 MoE | 12,078,550,957 | MoE（256 个专家、top-8 的真实路由） |
| D1 | 第 0 层稠密 + 1 层 MTP | 12,153,673,625 | MTP |

两种截层共用一份事先登记、一条 GPU golden 流水线、同一种 TPU 切片；两份 TPU 配置 [`hy4-t2-b.yml`](../../frameworks/maxtext/hy4-t2-b.yml)、[`hy4-t2-d1.yml`](../../frameworks/maxtext/hy4-t2-d1.yml) 除 `model_name` 外与 mini 的配置逐键相同。判法同 mini：GPU 是 golden，噪声只用 GPU 8 份重跑；L3 只看对 CPU FP64 的 R（原因见报告第 2 节）。

## 现状

- 1000 步现行一代：B 没有不过（过 4、待定 3）；D1 索引器 KD 不过，其余过 1、待定 7。
- 第 1 步逐张量梯度对 CPU FP64：两种截层都不过（R > 1.2 的 B 16 个、D1 11 个）；RMSNorm 权重梯度与 q latent norm 两类机制已查明（[根源一览第 23、24 行](../../ROOT-CAUSES.md#r23)），但对应开关在对齐配置里没开，所以照结果仍记未定责。
- L1 只有阶段 A 第 0 层前向 10 个算子；截层 B、D1 的逐算子与逐层没做。

判定与偏差：[对齐报告](ALIGNMENT-REPORT.md)；逐格：[支线总览](../../dashboard/other-sizes.md)、[检查清单](CHECKS.md)；历代：[历代比较](../../dashboard/history.md)。

## 这个文件夹

| 文件 | 放什么 |
|---|---|
| [`ALIGNMENT-REPORT.md`](ALIGNMENT-REPORT.md)、[`CHECKS.md`](CHECKS.md) | 判定与偏差台账（D19–D22）；检查清单（生成） |
| [`setup.md`](setup.md) | 模型与初始权重、程序与配置、显存、GPU golden 的作业规格（300 步与 1000 步） |
| [`config-alignment.md`](config-alignment.md)、[`goldens.md`](goldens.md)、[`runs.yaml`](runs.yaml) | 与 mini 的配置差别；运行登记 |
| `catalogs/`、`results/` | 检查项目录；结果文件 |

M35（QK-Clip 对最大 logit 不是正数的头缩放取 1）的触发条件在截层 B 上出现（[根源一览第 15 行](../../ROOT-CAUSES.md#r15)）。
