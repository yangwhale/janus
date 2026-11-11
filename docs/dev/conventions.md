# 命名与写法约定

新加的文件、运行、检查照这里起名。

- **已提交的结果、运行名、检查 id 不改名**：总览、大表、案例、issue 都按名字引用它们，改名等于让旧证据失联。
- 下面「现有例外」列出不合规但保留的名字，新东西不要照抄。

本页结构：文件与目录 → 结果与图 → 运行名 → 检查 id → 代码 → 文档。

## 文件与目录

| 东西 | 写法 | 例 | 理由 |
|---|---|---|---|
| Python 模块、脚本 | 小写下划线 `snake_case.py` | `name_map.py`、`layer_replay.py` | Python 能直接 import |
| Shell 脚本 | 小写下划线 `.sh`，开头注释写用法 | `run_regression.sh` | 同上，和 Python 脚本放一起时风格一致 |
| 文档 | 小写短横线 `kebab-case.md`；读的顺序有意义的加两位序号 | `guide/01-environment-data.md`、`catalog-schema.md` | 链接里好读；序号让目录列表就是阅读顺序 |
| 检查类型的规格 | 与类型名相同（下划线） | `docs/reference/checks/tensor_tolerance.md` | 类型名就是代码里的注册名，一一对应好找 |
| 检查项目录 | `<阶段>-<主题>.yaml`，小写短横线 | `phase1-full.yaml`、`phase2-tpu.yaml`、`regression-t0.yaml` | 目录名就是结果文件名的前缀 |
| 对齐实践文件夹 | `NN-<做什么>`，用几个英文词写目的 | `01-v5p-acceptance`、`00-gpu-noise-and-drills` | 规则与理由见 [`layout.md`](layout.md)「新对齐实践的命名」 |
| 排查案例 | `cases/NN-<英文短横线简述>.md`，两位序号，按开案顺序递增 | `cases/01-loss-offset-param-norm.md` | 案例之间互相引用，序号不复用 |
| 设计说明 | `docs/dev/design/NNNN-<英文短横线简述>.md` | `0001-repo-standards.md` | 四位编号，按顺序递增 |
| 模板 | `templates/` 下与产物同名 | `templates/alignment/setup.md` | 复制即用 |

## 结果与图

| 东西 | 写法 | 例 |
|---|---|---|
| 汇总结果 | `results/<目录名>[.<变体>].results.json`，**全模型唯一** | `phase2-t-loss30.results.json`、`phase1-tp-cp.tp2.results.json` |
| 报告 | 与结果同名，`.report.md`，只从结果渲染 | `phase2-t-loss30.report.md` |
| 逐步原值 | `results/series/<运行名>.metrics.jsonl`，运行目录里的 `metrics.jsonl` 原样拷来 | `results/series/t-loss30.metrics.jsonl` |
| 分析脚本的原始输出 | `results/<案例或主题>-<内容>.json` | `results/case01-offset-attribution.json` |
| 生成的图 | 由 `dashboard/build.py` 与 `janus curves` 生成，名字由代码定（见下表），不手改 | |

结果文件名全模型唯一，是因为总览（`models/<模型>/dashboard/build.py`）按文件名找结果，重名直接报错。

`figures/` 里现在生成的几种图：

| 图 | 文件名 | 谁生成 | 例 |
|---|---|---|---|
| L2 第 1 步逐层 | `L2-step1-layers.<结果名>.png` | `build.py` → `janus/report/figures.py` | `L2-step1-layers.phase1-full-noise28.png` |
| ULP 分桶 | `<检查 id>.<结果名>.png` | 同上 | `L3.step1.grads.ulp.phase1-full-noise28.png` |
| 路由翻转 token 的分差 | `<检查 id>-margin.<结果名>.png` | 同上 | |
| 被测对 golden 的逐步曲线 | `L4.<运行名>-<字段>-<overlay｜pairdiff｜mean>.png` | `build.py` → `janus curves compare` | `L4.t-loss30-lm_loss-mean.png` |
| 按设置分组互比 | `<字段>-<overlay｜pairdiff｜mean>.png`，另有 `curves-summary.json`、`offsets.json` | `janus curves groups` | `lm_loss-mean.png` |

手画的分析图放同一目录，文件名以它说明的检查 id 或案例编号开头（如 `case01-…png`），在引用它的文档里写清怎么画的。

## 运行名

- 一次运行一个名字，**全模型唯一**，写进 `meta.json` 的 `run_name`，登记在实践的 `runs.yaml`。
- 形状：`<前缀>-<区分符>`，小写短横线。
  - 前缀说明这批运行是干什么的；
  - 区分符是字母（`a`、`b`……，同设置的重复）或短词（变了什么）。
- 每个实践在 `runs.yaml` 按批次列出前缀的含义。
  - 一个模型现用的全部前缀列在模型 README（混元 3：[`../../models/hunyuan3/README.md`](../../models/hunyuan3/README.md)「运行名前缀」）；
  - 新前缀先查那张表，不和已有的撞。

日常回归的运行名在 `models/<模型>/regression/run_regression.sh` 的命令行里给出，见那份 README 的「手动跑」一节。

## 检查 id

形状：`<组>.<对象>[.<方面>…]`，点分。组名大写（`L2`、`R08`、`DET`），其余小写，单词之间用下划线。

| 组 | 用在哪 | 例 |
|---|---|---|
| `L0`…`L4` | 迁移验收，按级 | `L0.data.fingerprint`、`L2.layer{i}.fwd`、`L3.step1.grads.ulp`、`L4.loss` |
| `R01`…`R28` | 日常回归，编号就是验收清单的行号（[`reference/regression.md`](../reference/regression.md) 第四节） | `R08.loss_tolerance`、`R15.state_final` |
| `DET` | 确定性与「只改执行方式仍逐位」：重算、抓数开关、观测点 | `DET.hooks.noop`、`DET.remat.loss` |
| `SELF` | 工具自证（poison：故意改坏的数据必须被抓到） | `SELF.poison.tensor_tolerance` |

- **对象**写比什么：`layer2`、`step1`、`router`、`weights`；逐层展开用 `foreach` 的 `{i}`。
- **方面**写怎么比或比哪一面：`fwd`、`bwd`、`grads`、`ulp`、`scale`、`bitwise`、`tol`。
- 同一个 id 在不同目录里出现时，含义必须相同：总览按 id 合并各实践的结果。

现有例外：`TF32.*`（第一阶段 TF32 开 / 关对照，按主题而不是按级起的组）。新加的检查归到 `L*` 或 `DET`。

## 代码

- 日志与报错信息用中文，说清「哪个文件 / 哪条检查 / 期望什么 / 实际什么」；技术名词保留英文。
- 模块开头写一段说明：做什么、规格在哪（`tests/docs/test_repo_hygiene.py` 检查 `janus/` 下每个模块都有）。
- 命令行入口统一用 `argparse`，`prog` 写成 `janus <子命令>`；退出码按 [`reference/cli.md`](../reference/cli.md)。
- 门槛、窗口、分母这类会影响判定的数只写在目录里，不写死在代码里；代码里的常数要注明出处或写「推断：<推导>」。

## 文档

- 中文，技术名词保留英文（golden、rel_L2、calibrated）；术语只在 [`glossary.md`](../glossary.md) 定义一次，别处链过去。
- 数字写出处（结果文件 + 检查 id，或运行名）；推出来的写「推断」并给推导；查不到的写「未确认」。
- 相对链接写到文件；CI 的断链检查会拦。
- 引用别的文档的某一节写成 `路径.md`「标题」（「A · B」表示 A 下的 B）。
  - 标题改了 CI 会拦（`tests/docs/test_section_refs.py`）。
- 不写内部链接、不写真实桶名（`tests/docs/test_repo_hygiene.py`）。
