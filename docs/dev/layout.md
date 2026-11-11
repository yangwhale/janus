# 仓库布局：通用、模型、对齐实践三层

仓库按「换了什么就要动什么」分三层。模型层里另有一个日常回归目录，见下面「日常回归目录」。

| 层 | 放什么 | 什么时候动 |
|---|---|---|
| **通用** | 工具、方法文档、规格、模板（见注 1） | 换模型、换平台都不动；只在改方法、加检查类型时动 |
| **模型** | `models/<模型>/`（见注 2） | 接一个新模型时整层新建；同一模型的所有对齐实践共用 |
| **对齐实践** | `models/<模型>/alignments/<实践>/`（见注 3） | 每开一组新的对比就新建一个文件夹；旧实践的结论不被覆盖 |

注 1（通用层）：
- 工具代码 `janus/`；
- 方法文档 `docs/`；
- 每级检查的定义、判法与接口规范 `docs/reference/`；
- 模板 `templates/`。

注 2（模型层）：名字映射、截层、权重转换、各框架的训练与抓数脚本、数据加载对拍、配置对照表、跨实践总览。

注 3（对齐实践）：
- 一个「源平台 → 目标平台」一个文件夹；
- 放检查项目录、运行清单、结果、图、逐级大表。

## 目录树

```
janus/                          工具代码（Python 包）
docs/                           文档（索引 docs/README.md）
  guide/                        用户指南（主线）：按上手顺序分 8 章，另有专题 ruler.md（指标与尺子）
  glossary.md                   术语表（术语只在这里定义）
  reference/                    参考：原则、分级流程（每级一节）、指标与判定、环境、抓数、日常回归标准；
                                指标定义、checks/（每种检查类型一份规格，与实现同 PR 改）、命令行、dump / 目录 / 结果格式
  dev/                          给改 janus 本身的人：架构、约定、本页、设计文档 design/、路径变更记录
templates/                      报告、配置对照表、实践文件夹、排查案例的模板
models/<模型>/
  README.md                     模型目录的地图（按阅读顺序）
  ALIGNMENT-REPORT.md           对齐报告：结论、偏差台账、开关表
  goldens.md                    GPU golden 出处与比较协议
  config-alignment.md           两边配置逐项对照
  model/                        跨平台共用的代码：name_map.yaml、截层、HF ⇄ 各框架转换与逐张量核对
  frameworks/
    megatron/                   GPU 侧：训练、抓数、L1 回放、Pod
    maxtext/                    TPU 侧：训练、抓数、数据回放、JobSet
  data/                         原生数据加载器对拍
  analysis/                     分析工具（janus curves / offsets / fault-matrix）的模型配置：运行分组、故障期望；另有转发到 janus 的旧入口
  dashboard/                    跨实践总览（items.yaml + build.py）：每行一项检查，每列一个实践
  regression/                   日常回归（TPU 对 TPU 的 LKG）：实例参数、LKG 登记、回归目录、手动入口
    README.md                   平台档案、档位、各行的具体名字与个数、噪声标定计划、怎么手动跑
    lkg.yaml                    LKG 登记：每个「平台 × 档位 × 并行变体」一条，含换代记录
    tiers/                      档位配置 <档>.yaml：步数、训练参数、命令行覆盖（sha256 登记在 lkg.yaml）
    catalogs/                   回归目录 regression-<档>.yaml，各档共用的检查放 regression-common.yaml（include 引入）
    regress.py                  按 PR 声明改判、算覆盖与步时、写报告首屏、打印 LKG 条目
    run_regression.sh           手动跑一档：提交训练 → 取回 → janus 对 LKG → 报告 → 上传
    results/                    手动试跑阶段每次运行的报告（小文件）；排程上线后不再放
  alignments/
    NN-<做什么>/
      README.md                 逐级大表（build.py 生成）+ 结论
      setup.md                  两边硬件、并行切分、镜像与提交号；配置只写本实践特有的差异，其余引用 config-alignment.md
      runs.yaml                 运行清单：名字、步数、设置、提交号、数据位置（不写桶名）
      cases/                    （可选）排查案例：一次问题从发现到结论的完整记录，编号递增（写法见 guide/06-debugging.md）
      catalogs/                 检查项目录（YAML）
      results/                  <目录名>.results.json + .report.md（平铺）；series/ 放逐步原值
      figures/                  配图（build.py 生成，另有手画的分析图）
      faults/                   目标平台侧的故障演练
      impacts/                  （可选）逐件详解：一件影响一个文件（NN-<英文短横线>.md，标题中文），README.md 是目录（每件一句话结论 + 链接）；如混元 4 的 04-tpu-self/impacts/
tests/                          单元测试（含 Markdown 断链检查）
tools/                          一次性维护脚本（如结果格式迁移），连同运行日志一起提交
```

混元 3 的实际内容见 [`models/hunyuan3/README.md`](../../models/hunyuan3/README.md)。进行中的混元 4 在 [`models/hunyuan4/`](../../models/hunyuan4/README.md)（按尺寸分成几次独立的对齐：`alignments/NN-<尺寸>/` 各自带对齐报告、配置对照与检查清单；总览只有一份、每列一个尺寸，协议、开关台账、根源一览、排查案例也在模型层共用；TPU 自比 `04-tpu-self/` 的 README 是一页报告，真有影响的每一件在 `impacts/` 下各一个详解文件），暂停的 GLM-5.2 在 [`models/glm52/`](../../models/glm52/README.md)；归档的实践移到 `models/archived/<模型>/`（[`models/archived/README.md`](../../models/archived/README.md)），文件夹内部结构不变。

## 每层放什么、不放什么

### 三层各自的边界

- **通用层的方法与规格不依赖模型**。
  - 检查类型、指标、判法、dump 格式都按规范名写，换模型不用改。
  - 举例与实测出处可以引用混元 3（写全路径，如 `models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-full.results.json`）。
  - 但只能当例子和证据，不能成为规则的一部分。
- **模型层不放结论**。
  - `model/`、`frameworks/` 里是脚本和对照表；
  - 某次运行跑出来的数字只进对齐实践的 `results/`。
- **对齐实践不放脚本**。
  - 实践文件夹只放「这一组对比」的输入（目录、运行清单、设置）和输出（结果、图、表）；
  - 要改脚本，回模型层改。

### 结果与数据

- **结果文件名全模型唯一**。
  - 总览按文件名找结果（`dashboard/build.py` 遇到重名会报错）；
  - 所以文件名带上目录名，如 `phase2-t-loss30.results.json`。
- **结果平铺，不按级分子目录**。
  - 一个 `results.json` 对应一个检查项目录，一个目录常常同时覆盖 L0–L4。
  - 按级拆就要拆结果文件，这会破坏「`report.md` 只从 `results.json` 渲染」的规矩。
  - 按级看，用 README 里生成的逐级大表。
- **不放桶名、内部链接、原始抓数**。
  - 数据位置写成 `gs://<POC 桶>/…` 占位；
  - 只提交 `results.json`、`report.md` 和小的逐步原值（`results/series/<运行>.metrics.jsonl`）。

## 日常回归目录

`models/<模型>/regression/` 管的是另一件事：模型已在 TPU 上投产后，每天的提交有没有改变数值。

标准见 [`reference/regression.md`](../reference/regression.md)。它不按「源 → 目标」分文件夹：

- **一个模型一个**，所有平台共用。
  - 标准不分平台，平台之间只有参数不同，写在 README 的「平台档案」里；
  - LKG 按平台各登记各的。
- **只放输入和登记，不放每次运行的结果**。
  - 回归一天跑很多次，结果 JSON 与曲线存 POC 项目的存储桶（`<模型>/regression/<平台>/<档>/<日期>/<运行名>/`）；
  - 仓库里只留 `lkg.yaml` 的换代记录。
  - 例外：手动试跑阶段（还没排程、几次手动运行来确认流程能用）把每次运行的 `report.md` 与 `regression.json` 放进 `results/<运行名>/`，排程上线后停止。
- **回归逼出来的迁移验收重跑**（数值变了以后回去重跑 L0–L4 的某几级），结果进对应对齐实践的 `results/`，不进这里。

## 新对齐实践的命名

`NN-<做什么>`，用几个英文词说清这个实践在故事里的作用，如 `01-v5p-acceptance`、`00-gpu-noise-and-drills`：

- `NN`：两位序号，按开工顺序递增，不复用。
- `<做什么>`：小写、短横线连接，写目的（验收、标噪声、做 golden），不写硬件编码；
  硬件、规模、截几层写进实践的 `setup.md` 与 README 首行。
- 2026-11-06 以前的旧名（如 `01-b200x8-to-v5p8-4layers`）见 [`CHANGES-PATHS.md`](CHANGES-PATHS.md)。
- 例外：一个模型按尺寸分成几次独立的对齐时（混元 4，2026-11-08 起），`<做什么>` 写尺寸（`00-tiny`、`01-mini`、`02-real-weight`、`03-mid`）：每个文件夹就是一个尺寸的完整对齐，硬件与切分仍写进 `setup.md`。另有三个不按尺寸的专题实践：`04-tpu-self`（TPU 自比）、`05-gpu-scale`（GPU 换机器数与并行方式）、`06-rw3`（真实宽度三层）。

### 什么时候另开文件夹

**会改数值的变化另开文件夹，不在旧文件夹里改**：

- 截层数变了、全层、FP8；
- 换了 golden 的设置（确定性开关、分布式优化器开关、并行切分）。

这样旧实践的表和结论保持原样，可以和新的并排比。

同一实践里只追加运行（如 30 步之后再跑 1000 步）时，不用新开文件夹：

- 在 `runs.yaml` 里加一条；
- 结果文件用新名字。

### 混元 3 现有的实践

| 文件夹 | 是什么 |
|---|---|
| `00-gpu-noise-and-drills` | 第一阶段：B200×8 GPU 自比，工具自证、标定噪声底、故障演练 |
| `01-v5p-acceptance` | 第二阶段：B200×8 → TPU v5p-8 迁移验收（含排查案例 01） |
| `02-v7x-acceptance` | 第二阶段：B200×8 → TPU v7x-4 迁移验收 |
| `03-gpu-remat-golden` | B200×8 GPU 自比，整层重算 + 路由回放：生产近似 golden `rc-*`（换了 golden 的设置，所以另开文件夹） |

## 加一个新实践要做的事

1. 建文件夹，照 `templates/` 写 `setup.md`、`runs.yaml`，把目录放进 `catalogs/`。
2. 在 `models/<模型>/dashboard/items.yaml` 的 `columns` 加一列、`alignments` 加一行（文件夹名 → 列名），把每项检查对应的结果文件写进那一列（按尺寸分对齐的模型，如混元 4，也只有一份总览：列是各尺寸的现行比较，换了现行比较就改那一列，旧的进 `build.py` 的 `HISTORY`）。
3. 跑完把 `results.json` / `report.md` 放进 `results/`，逐步原值放进 `results/series/`。
   - 要画「被测对 golden」的逐步图，在 `dashboard/build.py` 的 `SERIES_COMPARE` 加一行。
4. `python models/<模型>/dashboard/build.py`（混元 3、混元 4 可以 `make dashboard`；生成逻辑在 `janus/report/overview.py`，各模型的 build.py 只放说明文字与配置）：
   - 重画总览；
   - 给每个实践的 README 重写逐级大表、重画配图。
5. 在 README 表格下面写结论。数字只引用表里的行或结果文件，不另抄。
