[janus](../../README.md) › 混元 3

# 混元 3

混元 3 截成 4 层，从 GPU（B200 × 8，Megatron）迁到 TPU（v5p-8、v7x-4，MaxText）。用户指南的例子全部取自这里。

## 按这个顺序读

| # | 读什么 | 回答什么 |
|---|---|---|
| 1 | [对齐报告 `ALIGNMENT-REPORT.md`](ALIGNMENT-REPORT.md) | 两边对上没有；全部偏差台账；想完全对上要动哪些开关。**先读这一份** |
| 2 | [检查清单 `CHECKS.md`](CHECKS.md) | 每一级比了哪些模块和张量、用什么指标和门槛判、v5p / v7x 各是什么结果 |
| 3 | [`goldens.md`](goldens.md) | 参照（GPU golden）每一份是怎么来的、比较协议、第一层验收的全部结果 |
| 4 | [`config-alignment.md`](config-alignment.md) | 两边每一项配置怎么对上、证据在哪 |
| 5 | [`alignments/00-gpu-noise-and-drills/`](alignments/00-gpu-noise-and-drills/README.md) | GPU 自比：工具自证、GPU 噪声尺子、故障演练 |
| 6 | [`alignments/03-gpu-remat-golden/`](alignments/03-gpu-remat-golden/README.md) | GPU 生产 golden `rc`（整层重算 + 路由回放）；最终 golden `rcc` 在它上面只改一处 |
| 7 | [`alignments/01-v5p-acceptance/`](alignments/01-v5p-acceptance/README.md) | TPU v5p 迁移验收，L0 到 L4；[案例 01](alignments/01-v5p-acceptance/cases/01-loss-offset-param-norm.md) 是「GPU 参照自己有偏」的完整证据 |
| 8 | [`alignments/02-v7x-acceptance/`](alignments/02-v7x-acceptance/README.md) | 同一套验收在 TPU v7x 上 |
| 9 | [`regression/`](regression/README.md) | 验收通过以后的日常回归：TPU 对上一次验证过的 TPU 运行（LKG） |

全部检查一张表：[总览 `dashboard/dashboard.md`](dashboard/dashboard.md)（怎么读见指南[第 5 章](../../docs/guide/05-reading-results.md)）。离签字还差什么：[签字清单](../../docs/guide/05-reading-results.md#56-签字清单现在离签字还差什么)。

## 文件夹地图

| 路径 | 放什么 |
|---|---|
| `ALIGNMENT-REPORT.md`、`CHECKS.md`、`goldens.md`、`config-alignment.md` | 结论、检查清单与两份主要依据（见上表）；`CHECKS.md` 由 `dashboard/build.py` 生成 |
| `alignments/` | 四个对齐实践。每个文件夹：`README.md`（逐级大表 + 结论）、`setup.md`、`runs.yaml`、`catalogs/`（检查项目录）、`results/`（结果文件）、`figures/` |
| `dashboard/` | `build.py` 从各实践的 `results/` 生成跨实践总览 `dashboard.md` 与检查清单 `../CHECKS.md`（骨架在 `checks.yaml`） |
| `frameworks/megatron/`、`frameworks/maxtext/` | GPU 侧 / TPU 侧操作手册与脚本：训练、抓数、回放、提交作业（[GPU](frameworks/megatron/README.md)、[TPU](frameworks/maxtext/README.md)） |
| `model/` | 两边共用的代码：名字与布局映射 `name_map.yaml`、截层、HF 与 Megatron / MaxText 互转、逐张量核对 |
| [`data/`](data/README.md) | TPU 原生数据加载器对拍 GPU |
| `analysis/` | 混元 3 自己的诊断脚本（多数为案例 01 写的，用法写在各自开头）；通用的分析在 `janus curves` / `offsets` / `fault-matrix` / `drift` |
| `regression/` | 日常回归：平台档案、档位、LKG 登记、回归目录、手动试跑记录 |

## 代码基准

- TPU：`maxtext` 的 `hunyuan` 分支（具体 commit 每次报告写明）。
- GPU：Megatron-Bridge 的混元 3 接入 + Megatron-Core，镜像 `nvcr.io/nvidia/nemo:26.10`（digest 每次报告写明）。

## 运行名前缀

各实践的 `runs.yaml` 里每次运行记一条，前缀的意思统一在这里说明。命名规矩见 [`../../docs/dev/conventions.md`](../../docs/dev/conventions.md#运行名)「运行名」，每个 golden 代号的定义见[术语表](../../docs/glossary.md#golden-codenames)。

| 前缀 | 意思 | 例 |
|---|---|---|
| `g1k-` | GPU golden，1000 步，全量抓数 | `g1k-a`、`g1k-det` |
| `s-` / `d-` | 第一阶段全覆盖：默认设置 / 确定性，100 步 | `s-a`…`s-i`、`d-a` |
| `f-` / `i-` | 故障演练：埋错 / 逐层注入 | `f-weight`、`i-weight` |
| `p-` / `r-` / `h-` | 只换执行方式：并行切分 / 重算与存档续训 / 抓数档位 | `p-ep4`、`r-remat`、`h-light2` |
| `m-` | 多步优化器重放 | `m-opt8` |
| `rc-` / `rv-` | GPU golden，1000 步，整层重算 + 路由回放（GPU 生产程序）/ 路由回放的逐位验证 | `rc-a`…`rc-e`、`rv-det` |
| `rcc-` | 最终 golden：`rc` 加输出头 dgrad 分段 FP32 累加（见表下） | `rcc-a`…`rcc-j` |
| `rce-` / `rcec-` / `rcx-` | 参考与历史：精确轨迹 / 真精确轨迹 / 生产 golden 输出头 dgrad 换 FP32 一次累加（见表下） | `rce-a`…`rce-e`、`rcec-a`、`rcx-a`…`rcx-e` |
| `g-` / `p2-` | 案例 01 的 GPU 诊断臂 | `g-bw-1k`、`p2-dgradx-a` |
| `t-` | TPU 被测 | `t-a`、`t-loss30`、`t-ga2-30` |
| `rd-` / `g61-` | 早期噪声底 / GPU 侧剩余检查的补充运行 | `rd-golden-1`、`g61-obs` |

`rcc-` 这一组里，L4 第一层的基线是 [`rcc-a`](../../docs/glossary.md#rcc-a)（最终 golden 的第 1 份），噪声取 `rcc-a..j` 10 份重跑两两配对得到的 45 对比较。

`rce-` / `rcec-` / `rcx-` 三组的区别：

- `rce`：`rc` 配置全程 FP32 的精确轨迹，输出头 dgrad 仍有偏；
- `rcec`：`rce` 再加分段累加，真精确；
- `rcx`：`rc` 加输出头 dgrad 换成 cuBLAS FP32 一次乘完，作归因证据。

现有两处前缀共用：`t-2l`（第一阶段 GPU 截 2 层的运行）和 TPU 被测共用 `t-`；`r-l2`、`r-l2-*`（TPU 逐层隔离回放）和只换执行方式的 `r-` 共用前缀。新的 GPU 运行不要用 `t-`。
