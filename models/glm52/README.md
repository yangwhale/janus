[janus](../../README.md) › GLM-5.2

# GLM-5.2

GLM-5.2（Hugging Face `zai-org/GLM-5.2`，架构不改）截层后从 GPU（B200 × 8，Megatron-Bridge）迁到 TPU（v5p 8 片，MaxText）的端到端数值对齐。方法、判法、数据回放与检查清单全部照搬[混元 3](../hunyuan3/README.md)，文件夹结构与混元 3 一一对应。

为什么是 GLM-5.2：它的骨架和混元 4 相同（MLA、DSA 索引器与跨层共享、256 专家 sigmoid 路由、1 层 MTP），又有 Megatron-Bridge 现成的桥与训练配方。这里做完端到端对齐后，整理成逐组件的对齐指南，给混元 4 这类同骨架模型用。混元 4 的实践在 [`../hunyuan4/`](../hunyuan4/README.md)（2026-11-07 起改为反向方案、进行中）；GLM-5.2 暂停。

**现状（2026-11-07）**：计划与比较协议已登记；GPU 参照程序在搭（G0–G3），还没有 golden；TPU 一侧只做编译期显存分析，不训练。

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
