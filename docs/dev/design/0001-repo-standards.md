---
title: "设计说明 0001：仓库的工程标准（目录、包结构、工具链、文档、命名）"
status: implemented
author: janus 维护者
date: 2026-11-04
---

# 设计说明 0001：仓库的工程标准

## 概述

janus 从「几个人边做边写」走到「别的工程师要能自己接着用」。这份说明定下仓库的工程标准：

- 顶层文件、包结构、命令行、版本、lint 与测试、文档的组织、命名。
- 每条写明选了什么、为什么、没选什么。

它只管本仓库内部的组织，不改方法论。

## 背景与动机

2026-11-04 时的状况（`3fff02d`）：

- **包结构**：分析、报告、核心模块平铺在 `janus/` 下。
  - 通用的分析脚本（逐步曲线、偏移归因、故障矩阵）放在 `models/hunyuan3/analysis/`。
  - 换一个模型就得复制一份，再改写死的运行名。
- **命令行**：只有 `python -m janus run`，参数手工解析。
  - 未知参数被静默忽略。
  - `--out` 后面漏写值直接抛 `IndexError`。
  - 目录文件不存在时抛栈。
- **工具链**：
  - 没有 lint。
  - 版本号写在两处（`pyproject.toml` 与 `janus/__init__.py`）。
  - 没有 CHANGELOG。
- **测试**：全平铺在 `tests/`，靠 `from test_run import …` 互相借用造数据的函数。
- **文档**：质量高，但索引只按「读的顺序」排。
  - 概念、操作、规格、设计、模型指南混在一张表里。
  - 术语表埋在 `principles.md` 中间。

### 目标

1. 新人 30 秒知道仓库干什么、2 条命令跑通测试（`pip install -e '.[dev]'`、`make check`）。
2. 包结构按职责分层，通用代码不出现模型名；模型层只放数据与薄包装。
3. 一个 `janus` 命令，子命令有 `--help`、出错有退出码。
4. CI 跑 lint 与测试，全部只要 CPU，Python 3.10 与 3.12 都过。
5. 文档有分类索引、术语表、架构图、迭代流程、命名规矩；所有相对链接有效（CI 检查）。

### 非目标（必填）

- **不改任何检查、指标、判法的行为**，不改 `results.json` 格式。工程整理不能顺手改数值语义。
- **不改、不删任何已提交的结果**（`results/`、`figures/`、`faults/`、`cases/`），不重跑实验。
- 不批量重排老代码的格式（理由见「lint 与格式」）。
- 不改 `AGENTS.md`。

## 方案

### 顶层文件

| 文件 | 决定 | 理由 |
|---|---|---|
| `README.md` | 重写成：30 秒概述 → 快速开始 → 目录地图 → 去哪读 | 首页只回答「这是什么、怎么跑、东西在哪」；细节都在 `docs/` |
| `CONTRIBUTING.md` | 保留本仓库特有规矩，补开发流程（见注 1） | 规矩已经有了，缺的是「怎么做」 |
| `CHANGELOG.md` | 新建，keep-a-changelog 格式，从 git 历史回填 | 复现写 tag，读者要知道每个版本改了什么 |
| `LICENSE` | 专有声明 | 私有仓库，代码归属与使用范围要写明；不用开源许可证 |
| `SECURITY.md` | 新建：数据与机密的处理、发现泄露怎么办、去哪提问题 | 仓库内容对外可见，「什么不能进仓库」需要一个固定入口 |
| `CODEOWNERS` | **不加** | 只有一位人类维护者（见注 2） |
| `Makefile` | 新建 | 见「工具链」 |

- 注 1：补的开发流程包括环境、`make check`、PR 要求、加检查类型 / 子命令 / 模型的清单。
- 注 2：目前只有一位人类维护者，分支保护也没开（需 Team 档）。
  - CODEOWNERS 只会自动请同一个人评审，没有信息量。
  - 有第二位维护者、或按目录分工时再加。

### 包结构

```
janus/
  cli.py  __main__.py         命令行入口（子命令按需导入）
  run.py                      引擎：读目录 → 取数 → 跑检查 → 写结果
  catalog.py                  检查项目录的读入与校验
  store.py  dtypes.py         运行目录读写（io 层）
  metrics.py                  指标（FP64）
  replay.py                   FP64 优化器重放（optimizer_step 用）
  data_layout.py              两边数据加载器的样本布局与换序
  checks/                     检查类型，按族分模块（张量、序列、离散、元检查、单算子；其余在 extra.py），注册到 REGISTRY
  dump/                       两侧抓数：megatron.py、maxtext.py；名字映射 names.py、name_map.py
  analysis/                   不判 pass / fail 的分析：CPU 参照、逐步曲线、偏移归因、漂移拆解、故障矩阵
  report/                     渲染：report.md、HTML 报告、逐级大表、配图
  demo.py  validate.py        合成演示；只校验检查项目录
```

- **analysis 搬进包**：`curves.py`、`offset_attribution.py`、`fault_matrix.py` 的算法与模型无关。
  - 只有运行分组、故障说明是混元 3 的。
  - 算法进 `janus/analysis/`，挂成 `janus curves` / `janus offsets` / `janus fault-matrix`。
  - 模型特有的部分写成数据（YAML）。
  - `models/hunyuan3/analysis/` 只留薄包装，旧命令照样能跑。
- **report 成子包**：`alignment.py`、`figures.py`、`report_html.py` 都是「从 `results.json` 渲染，不重算」，合在 `janus/report/`。
  - 旧模块名留一层转发（导入时给 `DeprecationWarning`），0.2 删除。
  - 并行开发中的脚本不会因此断。
- **store 不搬进 `janus/io/`**：它是被导入最多的模块（训练脚本、抓数、检查、测试都用）。
  - 搬了只换来一层目录，却要改并行开发中的训练脚本。
  - 在架构文档里把 `store.py`、`dtypes.py` 标为 io 层：概念上分层、物理上不动。
- **dump 不动**：抓数代码随训练镜像跑，路径写在各框架的启动脚本里。

### 命令行

- 一个总命令 `janus`（`[project.scripts]`），`python -m janus` 等价。
- 子命令登记在 `janus/cli.py` 的 `COMMANDS`。
  - 每个是一个模块的 `main(argv) -> int`，按需导入。
  - 只跑 `janus run` 不会导入 matplotlib。
- `janus run` 改用 argparse：参数与原来一一对应。
  - 未知参数、缺值、目录文件不存在都按约定返回 2，不再抛栈。
- 旧写法 `python -m janus <目录文件>` 继续认。

### 版本

- 版本号只写在 `janus/__init__.py`，`pyproject.toml` 用 `dynamic = ["version"]` 读它。
  - `results.json` 记的也是它。
  - 另记 janus 的 git 提交号（有未提交改动时带 `+dirty`）。
- 两次发版之间版本号带 `.devN`（如 `0.1.0.dev0`）：没发版的代码跑出的结果不会冒充某个已发的版本。
- 0.x 期间的升版规则：
  - 改了检查判法、结果格式或命令行参数的不兼容变更，升次版本号；
  - 其余升修订号；
  - 每次升版在 CHANGELOG 加一节。
- 打 tag 由维护者做（`v<版本>`），复现一律写 tag。

### 工具链

- **Makefile**：`make check`（= `lint` + `test`，CI 跑的就是它）、`make fmt`、`make docs-check`、`make help`。
  - 选 make 不选 just：make 在 Linux 开发机与 CI runner 上默认就有，just 要另装。
- **依赖分组**：
  - 核心只要 numpy、ml_dtypes、pyyaml；
  - `[report]` 加 matplotlib；
  - `[test]`；
  - `[lint]` 钉死 ruff 版本；
  - `[dev]` 是三者之和。
  - torch / jax 跟训练镜像走，不进 extras：抓数代码只在训练环境里跑，CI 也不装它们。
- **CI**：`tests.yml` 四个 job；`pr-check.yml` 原样保留。
  - `lint`；
  - `pytest`（3.10、3.12 矩阵）；
  - `package`：打 wheel、装成普通包、在仓库外跑 demo——抓只有在仓库根目录才跑得通的导入；
  - `generated`：重画总览与逐级大表，`.md` / `.html` 要和提交的一致；PNG 随机器字体变，不比。

### lint 与格式

- **ruff check 全仓库**，规则 `E4 E7 E9 F W B I UP`。忽略的几条各有理由（写在 `pyproject.toml` 注释里）：
  - `B023` 在这里全是「循环里定义、当轮就调用」的闭包（JAX `vjp` / `jit` 的常见写法）；
  - `B905` 要逐处判断 `strict=` 会不会改行为；
  - `E731` / `E741` 是本仓库有意的写法；
  - 模型层脚本直接用 `python` 跑、要先改 `sys.path` 再导入 janus，所以 `models/**` 免 `E402`。
- **ruff format 只对 `FMT_PATHS` 强制**（Makefile 里的列表），新模块与整体重写过的模块加进去。
- 不批量格式化老代码：
  - 试算过一次会改约 8000 行，主要是把对齐的行尾注释压成两个空格、把长表达式拆行。
  - 这些对齐注释是写给读者看的（出处、单位、为什么），压掉反而难读。
  - 同时三路并行开发都在改这些文件，批量改写会让每个人 rebase 冲突。

### 测试

- 目录照包的结构：
  - `tests/<子包>/test_<模块>.py`；
  - 模型层脚本的测试在 `tests/models/<模型>/`；
  - 文档与仓库卫生在 `tests/docs/`。
- 造数据的共用函数放 `tests/helpers.py`，不再从别的测试模块导入。
- 新增仓库卫生检查：没有内部链接、桶名只写占位、`janus/` 下每个模块都有开头说明。

### 文档

- `docs/README.md` 按读者要做的事分五类：概念 → 操作 → 规格 → 设计 → 模型指南。
- **规格与设计分开放**：
  - `docs/reference/` 只放接口规范（命令行、格式、指标、日常回归标准）；
  - 设计文档放 `docs/dev/design/`，按开写顺序编号。
  - 原来混在 `reference/` 里的三份设计文档（L0 数据与权重、L1 单算子回放、GPU 侧 kernel 检查）被各实践与 GPU 侧脚本大量引用，等并行开发停下来再一次搬完（路径记在 `CHANGES-PATHS.md`）。
  - 顺带消掉了 `levels.md` 与 `reference/l1-ops.md` 同名的困惑。
- 新增：
  - `glossary.md`（术语表的唯一出处，`principles.md` 改为引用）；
  - `architecture.md`（组件图与数据流）；
  - `guide/06-debugging.md`（原 `process.md`：对齐迭代的六步与各步产物去哪）；
  - `conventions.md`（命名规矩）；
  - `CHANGES-PATHS.md`（路径变更记录）。

### 命名

写在 `docs/dev/conventions.md`，要点：

- 文件与模块：小写下划线（Python）或小写短横线（文档、目录文件、结果文件）；
- 对齐实践：`NN-<做什么>`（2026-11-06 起；之前是 `NN-<源>-to-<目标>-<N>layers`）；
- 结果文件：`<目录名>.results.json`，全模型唯一；
- 检查 id：`L<级>.<对象>.<方面>`；
- 运行名：`<前缀>-<序号或字母>`，前缀表登记在实践的 `runs.yaml`。

### 备选方案

- **整仓 `ruff format`**：没选，理由见上。
- **`src/` 布局**：没选。
  - `src/janus` 能防止「没安装也能导入」带来的假通过。
  - 但本仓库的模型层脚本恰恰依赖从仓库根直接导入 janus（训练镜像里不 pip install）。
  - 改布局要改所有训练脚本的 `sys.path`。
- **把 `docs/reference/` 改名 `spec/`**：没选。规格留在 `reference/`、设计搬到 `design/` 已经分开，改名只多一次全仓库的链接改动。
- **justfile**：没选，要另装 `just`，make 够用。

## 影响范围

- 导入路径：`janus.alignment`、`janus.figures`、`janus.report_html` 改为 `janus.report.*`，旧名转发到 0.2。
- 命令：
  - `models/hunyuan3/analysis/*.py` 原命令照旧可用；
  - 新命令是 `janus curves|offsets|drift|fault-matrix`，另有 `janus demo`、`janus validate`。
- 测试文件路径变了（见 `docs/dev/CHANGES-PATHS.md`）；pytest 改用 `--import-mode=importlib`。

## 实施计划

分几个小 PR，每个合并前 `git pull --rebase`。顺序：

工具链 → 顶层文件与 README → analysis 进包 → report 子包 → 测试目录 → 文档索引、术语、架构、流程、命名 → 清理过期文件。

## 风险

- **并行开发冲突**：
  - 搬文件的 PR 单独提、马上合，路径变化写进 `docs/dev/CHANGES-PATHS.md`；
  - 不搬并行开发正在改的文件。
- **lint 拦住别人的 PR**：CI 的报错直接给出修法（`make fmt`）；规则集挑的是不吵的那组。

## 和原计划的偏差

实施在 2026-11-04 晚上（路径变化逐条记在 [`../CHANGES-PATHS.md`](../CHANGES-PATHS.md)）。和上面的方案比：

| 原计划 | 实际 | 原因 |
|---|---|---|
| 只有 `janus run` 与三个分析子命令 | 另加 `janus demo`、`janus validate`、`janus drift` | 见注 1 |
| `report.md` 的渲染留在 `run.py` | 移到 `janus/report/markdown.py`，`janus.run._render_report` 保留旧名 | 渲染全部归 `report/`；GPU 侧脚本按旧名导入，不用改 |
| 版本号不动 | `0.0.1` → `0.1.0.dev0`，`results.json` 另记 janus 的提交号 | 见注 2 |
| 文档只加术语表、架构、流程、命名；设计文档留在 `reference/` | 另有三项改动（见注 3） | 见注 3 |
| CI 两个 job | 四个：另加 `package` 与 `generated` | 见注 4 |
| 测试完全照包镜像 | 做到了，但分两步（见注 5） | 拆分正在改的文件会让对方 rebase 冲突 |
| 删掉没有调用方的代码 | 只删了 `dtypes.is_float`、`names.canonical_param` | 见注 6 |

- 注 1：新加的三个子命令。
  - `janus demo`（合成演示）：让快速开始在没有 POC 数据时也能跑通。
  - `janus validate`（只校验目录）：让 CI 校验仓库里全部目录。
  - `janus drift`（漂移拆解）：drift 是当晚新加的四个通用分析脚本（含点名要搬的 `group_norm_drift.py`）。
- 注 2：命令行与包结构都变了，再叫 `0.0.1` 会让之后的结果冒充旧版本。
  - `janus.commit` 规格早就有、代码一直写 `null`。
- 注 3：实际做的：
  - 另把指标速查并入 `reference/metrics.md`；
  - 加「引用的节必须存在」的 CI 检查；
  - 夜里并行开发停下后把三份设计文档搬到 `design/`。
  - 原因：复查指出指标在三处讲、节引用失联、`reference/` 里规格与设计混放且有同名文件。
- 注 4：新加的两个 job。
  - `package`（装成普通包、在仓库外跑 demo）：可编辑安装发现不了漏打包与依赖仓库路径的导入。
  - `generated`（生成的总览与大表是否最新）：提交结果忘了重画，大表就和结果对不上。
- 注 5：分两步的具体做法。
  - `tests/models/hunyuan3/test_regression.py` 先整文件搬，夜里日常回归那边停下后再按模块拆。
  - `checks/extra.py` 里 13 种检查类型没拆模块。
- 注 6：`name_map.py`、`dump/maxtext.py` 里的几个没有调用方的函数，是成对的映射接口或 TPU 侧正在改的文件，留着。

还没做、留给后面的：

- `ruff format` 扩大到 `janus/checks/`、`janus/run.py` 等老模块；
- `janus/checks/extra.py` 按族拆模块。
