# 路径变更记录

文件搬家后旧链接、旧命令、并行分支里的引用会断。

- 每次搬文件都在这里追加一节：日期、PR、旧路径 → 新路径、旧路径还能不能用。
- 最新的在最上面。
- 已提交的 `results.json` 不随搬家改：里面的 `catalog.path` 是跑的时候的路径。结果记的是目录的内容指纹（`catalog.content_sha256`，见 [`../reference/result-schema.md`](../reference/result-schema.md)），搬家和改注释都不影响；按文件名到同一实践的 `catalogs/` 下找（`tests/test_committed_results.py` 就这样找）。

搬文件的规矩（[`design/0001-repo-standards.md`](design/0001-repo-standards.md)「风险」）：

- 搬家单独一个 PR、马上合；
- 旧路径能留转发的留到下一个次版本；
- 这里记一笔。

## 2026-11-09 · 混元 4 合成一份总览（分支 `feat/68-hy4-reverse`；未单独开 PR）

照混元 3 的总览模板：一份总览，每列是一个尺寸的现行比较；历代与诊断代移到生成的历代比较。各尺寸的总览删掉，检查清单不变。

| 旧 | 新 | 旧路径还能用吗 |
|---|---|---|
| `models/hunyuan4/alignments/<尺寸>/dashboard/dashboard.{md,html}` | `models/hunyuan4/dashboard/dashboard.{md,html}`（对应的列） | 不能 |
| `models/hunyuan4/alignments/<尺寸>/dashboard/items.yaml` | `models/hunyuan4/dashboard/items.yaml` | 不能 |
| mini 总览开头的「L4 一张表」与历代列 | `models/hunyuan4/dashboard/history.md` | 不能 |

## 2026-11-08 · 混元 4 按尺寸分成独立的对齐（分支 `feat/68-hy4-reverse`；未单独开 PR）

每个尺寸是一次独立的对齐、各有一套自成一体的文档。`models/hunyuan4/alignments/` 改成一个尺寸一个文件夹（`00-tiny`、`01-mini`、`02-real-weight`、`03-mid`，今后更大的模型照此新建 `04-*`），各自有过程 README、对齐报告、配置对照、检查清单、总览、运行登记；协议、开关台账、根源一览、排查案例留在 `models/hunyuan4/` 顶层共用。仓库内引用已改好（链接按目标重算），已提交的结果文件不改（`catalog.path` 是跑的时候的路径；`tests/test_committed_results.py` 先在同一实践的 `catalogs/` 下按文件名找，找不到再到同一模型各实践的 `catalogs/` 下找唯一同名的）。旧路径不留转发：下表旧路径都不能再用。

| 旧 | 新 |
|---|---|
| `models/hunyuan4/alignments/01-v5p-acceptance/` | `models/hunyuan4/alignments/01-mini/` |
| `models/hunyuan4/alignments/01-v5p-acceptance/cases/` | `models/hunyuan4/cases/` |
| `models/hunyuan4/alignments/00-gpu-noise-and-drills/` | `models/hunyuan4/alignments/01-mini/gpu-noise/` |
| `models/hunyuan4/alignments/02-t2-real-weights/` | `models/hunyuan4/alignments/02-real-weight/` |
| `models/hunyuan4/alignments/03-hy4-mid/` | `models/hunyuan4/alignments/03-mid/` |
| `models/hunyuan4/ALIGNMENT-REPORT.md`、`config-alignment.md`、`goldens.md`、`CHECKS.md` | `models/hunyuan4/alignments/01-mini/` 下同名文件（报告开头的根源一览表搬到新文件 `models/hunyuan4/ROOT-CAUSES.md`） |
| `models/hunyuan4/dashboard/dashboard.{md,html}`、`items.yaml`、`checks.yaml` | `models/hunyuan4/alignments/01-mini/dashboard/` 下同名文件（`dashboard/build.py` 不动，为每个尺寸各生成一份） |
| `models/hunyuan4/alignments/01-v5p-acceptance/results/hy4-cpu-smoke.json` | `models/hunyuan4/alignments/00-tiny/results/hy4-cpu-smoke.json` |
| — | 新建：`models/hunyuan4/ROOT-CAUSES.md`、`alignments/00-tiny/`，以及 `02-real-weight`、`03-mid` 的 `ALIGNMENT-REPORT.md`、`config-alignment.md`、`goldens.md`、`CHECKS.md`、`dashboard/` |

## 2026-11-07 · 混元 4 恢复到 `models/hunyuan4/`

改为反向方案：以 TPU 一侧 MaxText 的混元 4 实现为准，GPU 一侧补齐，混元 4 恢复为进行中。文件夹从 `models/archived/` 移回原位，内部结构不变，仓库内引用已改回，已提交的结果文件不改。GLM-5.2 原地不动（暂停，未归档）。

| 旧 | 新 |
|---|---|
| `models/archived/hunyuan4/` | `models/hunyuan4/` |
| `tests/models/archived/hunyuan4/` | `tests/models/hunyuan4/` |

## 2026-11-07 · 混元 4 归档，新建 GLM-5.2

混元 4 的实践暂停（GLM-5.2 完成后继续），整个文件夹移进 `models/archived/`；文件夹内部结构不变，仓库内引用已改好，已提交的结果文件不改（见上）。旧路径不留转发。同一个 PR 新建 `models/glm52/`（这次搬家与新建放在一个 PR 里）。

| 旧 | 新 |
|---|---|
| `models/hunyuan4/` | `models/archived/hunyuan4/` |
| `tests/models/hunyuan4/` | `tests/models/archived/hunyuan4/` |

## 2026-11-06 · 仓库重整：一套文档、实践文件夹改成说目的的名字

文档只留一套：用户指南是主线，查用的进 `docs/reference/`，给工具开发者的进 `docs/dev/`。全仓库的引用已一并改好；旧路径不留转发。

| 旧 | 新 |
|---|---|
| `docs/01-principles.md` | `docs/reference/principles.md` |
| `docs/02-funnel.md`、`docs/levels/L0-foundation.md` … `L4-steps.md` | `docs/reference/levels.md`（合并：每级一节） |
| `docs/03-metrics.md` | `docs/reference/judging.md` |
| `docs/04-environment.md` | `docs/reference/environment.md` |
| `docs/05-dump.md` | `docs/reference/dump.md` |
| `docs/levels/checks/` | `docs/reference/checks/` |
| `docs/architecture.md`、`docs/conventions.md` | `docs/dev/architecture.md`、`docs/dev/conventions.md` |
| `docs/06-layout.md` | `docs/dev/layout.md` |
| `docs/design/` | `docs/dev/design/` |
| `docs/CHANGES-PATHS.md` | `docs/dev/CHANGES-PATHS.md`（本页） |
| `docs/07-new-model.md`、`docs/process.md`（转发页） | 删除；内容在 `docs/guide/08-new-model.md`、`docs/guide/06-debugging.md` |
| `models/hunyuan3/model/goldens.md`、`config-alignment.md`、`issue-coverage.md`（公开版未收） | `models/hunyuan3/` 下同名文件（和对齐报告并列；`model/` 只留代码） |
| `models/hunyuan3/alignments/00-b200x8-self-4layers/` | `models/hunyuan3/alignments/00-gpu-noise-and-drills/` |
| `models/hunyuan3/alignments/01-b200x8-to-v5p8-4layers/` | `models/hunyuan3/alignments/01-v5p-acceptance/` |
| `models/hunyuan3/alignments/02-b200x8-to-v7x4-4layers/` | `models/hunyuan3/alignments/02-v7x-acceptance/` |
| `models/hunyuan3/alignments/03-b200x8-self-4layers-remat/` | `models/hunyuan3/alignments/03-gpu-remat-golden/` |
| `models/hunyuan3/alignments/01-…/plan.md` | 删除（第二阶段工作计划，已被对齐报告与签字清单取代） |

- 2026-11-04 #35 三层重排时的旧路径（`catalogs/hunyuan3-4l/<名>.yaml`、`models/hunyuan3/results/<名>.results.json`）对应今天的 `alignments/<实践>/catalogs/<名>.yaml`、`alignments/<实践>/results/<名>.results.json`。

## 2026-11-05 · 两篇操作文档搬进用户指南

用户指南（`docs/guide/`）按上手顺序分章，已经是「一步一步做」写法的两篇直接成为其中两章。

全仓库的引用已一并改好；旧路径留一页转发，到下一个次版本删除。

| 旧 | 新 | 旧路径还能用吗 |
|---|---|---|
| `docs/process.md` | `docs/guide/06-debugging.md`（第 6 章 发现不对齐时怎么排查） | 能：转发页 |
| `docs/07-new-model.md` | `docs/guide/08-new-model.md`（第 8 章 接入新模型） | 能：转发页 |

## 2026-11-05 · 日常回归测试里的通用部分按模块归位

`tests/models/hunyuan3/test_regression.py` 只留混元 3 的部分（回归目录、`regress.py`、HLO 收集）；测 janus 本身的用例搬到对应模块的测试：

| 用例 | 搬到 |
|---|---|
| FP32 状态清单、`sha256`、`write_state_sha` | `tests/dump/test_maxtext.py` |
| `run_complete`、`artifact_check`、`config_diff` 字典比 | `tests/checks/test_extra.py` |
| `first_differ_step` 与 `loss_series` 的 exact | `tests/checks/test_series.py` |
| 待标定门槛（`pending`） | `tests/test_run.py` |
| 目录 `include` / `vars` | `tests/test_catalog.py`（新） |

共用的造数据函数进了 `tests/helpers.py`：`tpu_state()`（原 `_flat`）、`tpu_meta()`（原 `_meta`）、`HLO_SAMPLE`（原 `HLO`）。

## 2026-11-05 · 设计文档搬到 docs/design/

`docs/reference/` 只留接口规范；三份设计文档搬到 `docs/design/` 并按设计说明的写法编号。

全仓库的引用（文档、GPU 侧脚本的说明）已一并改好。

| 旧 | 新 | 旧路径还能用吗 |
|---|---|---|
| `docs/reference/l0-data-weights.md` | `docs/design/0002-l0-data-weights.md` | 不能：文件已搬 |
| `docs/reference/l1-ops.md` | `docs/design/0003-l1-op-replay.md` | 不能 |
| `docs/reference/l1-kernels.md` | `docs/design/0004-l1-gpu-kernels.md` | 不能 |

## 2026-11-04 · 漂移分析进包

`models/hunyuan3/analysis/` 里四个与模型无关的漂移分析搬进 `janus/analysis/drift.py`，挂成 `janus drift`。

旧脚本留作薄包装，命令与输出逐字节不变：

- `03` 的 `remat-pair-summary.json`、`01` 的 `phase2-t-a-l4.summary.json`（之后已删除）重算核对过；
- 另两个在合成数据上新旧对拍。

| 旧 | 新 | 旧路径还能用吗 |
|---|---|---|
| `models/hunyuan3/analysis/pair_summary.py` | `janus drift pairs`（混元 3 的路由偏置字段由旧脚本作为 `--fields` 默认值传入） | 能：旧脚本转发 |
| `models/hunyuan3/analysis/l4_drift.py` | `janus drift trend` | 能 |
| `models/hunyuan3/analysis/group_norm_drift.py` | `janus drift groups` | 能 |
| `models/hunyuan3/analysis/output_row_drift.py` | `janus drift rows` | 能 |

`remat_router_check.py`、`adamw_step_check.py` 留在模型层：一个要 GPU 与 Megatron 的路由输出，一个按 Megatron 的参数命名判权重衰减。

## 2026-11-04 · 指标速查并入指标规格

| 旧 | 新 | 旧路径还能用吗 |
|---|---|---|
| `docs/reference/metric-cheatsheet.md` | `docs/reference/metrics.md`（怎么算、边界、实测量级合成一张表；为什么要它、盲点只留在 `docs/03-metrics.md`） | 不能：文件已删 |

## 2026-11-04 · 测试目录照包的结构

| 旧 | 新 |
|---|---|
| `tests/test_checks.py` | 拆成 `tests/checks/test_base.py`、`test_tensor.py`、`test_discrete.py`、`test_meta.py` |
| `tests/test_series.py`、`test_extra.py` | `tests/checks/` 下同名 |
| `tests/test_ops.py` | 拆成 `tests/analysis/test_ops.py`（CPU 参照）与 `tests/checks/test_ops.py`（`op_replay`） |
| `tests/test_analysis.py` | 拆成 `tests/analysis/test_curves.py`、`test_offsets.py`、`test_fault_matrix.py` |
| `tests/test_dump_names.py`、`test_name_map.py`、`test_maxtext_dump.py`、`test_dump_observe.py` | `tests/dump/test_names.py`、`test_name_map.py`、`test_maxtext.py`、`test_megatron.py` |
| `tests/test_alignment.py` | `tests/report/test_tables.py` |
| `tests/test_regression.py`、`test_native_loader_parity.py` | `tests/models/hunyuan3/` 下同名 |
| `tests/test_links.py`、`test_repo_hygiene.py` | `tests/docs/` 下同名 |
| `from test_run import make, catalog, BASE_META`、`from test_run_full import make_full, checks` | `from helpers import make, catalog, BASE_META, make_full, full_checks`（`tests/helpers.py`） |

- 测试文件里取仓库根一律用 `from helpers import ROOT`（目录深度变了，`Path(__file__).parents[1]` 不再是仓库根）。
- 并行分支改过旧路径上的测试：rebase 时 git 会按改名跟过去。
- 新写的测试放到对应子目录。

## 2026-11-04 · 渲染模块合进 `janus/report/`

| 旧 | 新 | 旧路径还能用吗 |
|---|---|---|
| `janus/report_html.py`（`janus.report_html`） | `janus/report/html.py`（`janus.report.html`） | 能：导入时给 `DeprecationWarning`，0.2 删除 |
| `janus/alignment.py`（`janus.alignment`） | `janus/report/tables.py`（`janus.report.tables`） | 同上 |
| `janus/figures.py`（`janus.figures`） | `janus/report/figures.py`（`janus.report.figures`） | 同上 |

各实践 README 里生成的大表说明行随之改成「调 `janus/report/tables.py`」（`dashboard/build.py` 重画，只有这一行变）。

## 2026-11-04 · 分析脚本进包

`models/hunyuan3/analysis/` 里的通用算法搬进 `janus/analysis/`，挂成子命令；模型特有的分组与故障期望改成 YAML。

旧脚本留作薄包装，命令与输出逐字节不变。以下都重算核对过：

- 第一阶段的 `curves-summary.json`、`offsets.json`、全部 PNG；
- 案例 01 的 `case01-offset-attribution.json`。

| 旧 | 新 | 旧路径还能用吗 |
|---|---|---|
| `models/hunyuan3/analysis/curves.py` 的算法 | `janus/analysis/curves.py`（`janus curves compare` / `groups`） | 能：旧脚本转发，命令不变 |
| `models/hunyuan3/analysis/curves.py` 里写死的 `GROUPS`、`SETTINGS` | `models/hunyuan3/analysis/phase1-curves.yaml` | — |
| `models/hunyuan3/analysis/offset_attribution.py` | `janus/analysis/offsets.py`（`janus offsets`） | 能：旧脚本转发 |
| `models/hunyuan3/analysis/fault_matrix.py` 的算法 | `janus/analysis/fault_matrix.py`（`janus fault-matrix`） | 能：旧脚本转发 |
| `fault_matrix.py` 里写死的 `FAULTS`、`EXPECT` | `models/hunyuan3/analysis/phase1-faults.yaml` | — |
| `import curves`（`dashboard/build.py` 曾经把 `analysis/` 加进 `sys.path`） | `from janus.analysis import curves` | 旧脚本仍导出 `compare`、`load_file`、`rel`、`rolling` |

<!-- 新条目加在最上面（紧跟开头说明之后），格式：
## YYYY-MM-DD · 一句话
| 旧 | 新 | 旧路径还能用吗 |
|---|---|---|
-->
