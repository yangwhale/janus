[janus](../../README.md) › GLM-5.2

# GLM-5.2

GLM-5.2（Hugging Face `zai-org/GLM-5.2`，架构不改）截层后从 GPU（B200 × 8，Megatron-Bridge）迁到 TPU（v5p 8 片，MaxText）的端到端数值对齐。方法、数据回放与检查清单照[混元 3](../hunyuan3/README.md)，文件夹结构与混元 3 一一对应；判法用 janus 现行判法（与混元 4 相同，见[比较协议](PROTOCOL.md)）。

为什么是 GLM-5.2：它的骨架和混元 4 相同（MLA、DSA 索引器与跨层共享、256 专家 sigmoid 路由、1 层 MTP），又有 Megatron-Bridge 现成的桥与训练配方。这里做完端到端对齐后，整理成逐组件的对齐指南，给混元 4 这类同骨架模型用。混元 4 的实践在 [`../hunyuan4/`](../hunyuan4/README.md)。

**现状：暂停，不排期**（2026-11-11 项目负责人定：混元 4 做完以后 GLM-5.2 先不做）。计划与比较协议已登记；GPU 参照程序停在 G0–G3，还没有 golden；TPU 一侧只做过编译期显存分析，没有训练。

## 重新启动时

- **可以直接复用混元 4 的方法和工具**：[尺子](../../docs/guide/ruler.md)（GPU 同设置重跑两两、只用 GPU）；第 1 步逐张量对 CPU FP64 定责（[混元 4 比较协议](../hunyuan4/PROTOCOL.md)第 5 节）；跑 TPU 之前在 CPU 上编训练步、比优化后 HLO 的[离线筛选](../hunyuan4/alignments/04-tpu-self/setup.md#跑前离线筛)；TPU 自比的[脾气表](../hunyuan4/alignments/04-tpu-self/MATRIX.md#tpu-脾气表)（哪类改动在哪条曲线上先露头）。
- **不能直接搬的**：GLM-5.2 走上游 MaxText 的 DeepSeek 解码器接入代码，不是 maxtext 的混元 4 集成分支。混元 4 [根源一览](../hunyuan4/ROOT-CAUSES.md)里的模型特有问题和对应开关，在这条代码路径上有没有、有没有开关，都要重新查。
- **第一步**：把混元 4 根源一览逐行对到当时的上游 MaxText 代码路径上（读代码，加[离线 HLO 筛选](../hunyuan4/alignments/04-tpu-self/setup.md#跑前离线筛)），列出哪些问题在 GLM-5.2 上也存在、上游有没有对应开关，结果写进 [`config-alignment.md`](config-alignment.md)；GPU 一侧照 [`PLAN.md`](PLAN.md) 从 G0 做起。

## 按这个顺序读

| # | 读什么 | 回答什么 |
|---|---|---|
| 1 | [实践计划 `PLAN.md`](PLAN.md) | 截哪几层、GPU 参照程序与「生产配置」、和配方不同的地方及理由、里程碑 |
| 2 | [比较协议 `PROTOCOL.md`](PROTOCOL.md) | 事先登记的基线、份数、噪声规则、指标与窗口、停止规则；看到结果之前冻结 |
| 3 | [`config-alignment.md`](config-alignment.md) | 两边每一项配置怎么对上、证据在哪 |
| 4 | [`goldens.md`](goldens.md) | 每一份 GPU golden 怎么来的 |
| 5 | [检查清单 `CHECKS.md`](CHECKS.md) | 每一级比哪些量、怎么判、结果（有结果后由生成器生成） |

## 文件夹地图

| 路径 | 放什么 |
|---|---|
| `PLAN.md`、`PROTOCOL.md`、`config-alignment.md`、`goldens.md`、`CHECKS.md` | 计划、协议、配置对照、golden 登记、检查清单 |
| `alignments/` | 对齐实践：`00-gpu-noise-and-drills/`（GPU 自比与故障演练）、`01-v5p-acceptance/`（TPU v5p 迁移验收）。每个文件夹：`README.md`、`setup.md`、`runs.yaml`、`catalogs/`、`results/` |
| `dashboard/` | 总览与检查清单的骨架（`items.yaml`、`checks.yaml`），有结果后用混元 3 的 `build.py` 生成 |
| `frameworks/megatron/`、`frameworks/maxtext/` | GPU 侧（Megatron-Bridge）/ TPU 侧（MaxText）的程序与操作手册 |
| `model/` | 截层、名字映射、HF ↔ Megatron ↔ MaxText 转换与逐张量核对 |
| `analysis/`、`results/` | 本模型的分析脚本与模型一层的结果（如[参数量](results/param-counts.json)） |
