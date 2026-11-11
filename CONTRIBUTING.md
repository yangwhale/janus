# 参与 janus

## 先说明：这是公开版

这个仓库由开发仓库定期同步生成（见 [`docs/public-edition.md`](docs/public-edition.md)），每次同步会整棵树换成新的。所以：

- 问题、建议、看不懂的结果：在本仓库开 GitHub issue。贴结果文件与检查 id，不贴截图里的数。
- 想改代码或文档：在 issue 里描述改法或附补丁。我们移植进开发仓库后，下一次同步会带出来。直接合进这里的改动会在下次同步时被覆盖，所以这里不直接合并 PR。

下面是开发 janus 时守的规矩，补丁也照这些写。

## 规矩

### 不能进仓库的东西

- 模型权重、训练数据、日志原文：只写它们在你的存储里的路径（占位写法）和 sha256。
- 真实的桶名（写成 `gs://<POC 桶>/…`）、密钥、令牌、本机绝对路径、任何公司的内部链接。详见 [`SECURITY.md`](SECURITY.md)。

### 数字与门槛

- **数字要有来源**：准出标准、门槛、示例数字都写明来自哪次运行、哪个结果文件。
  - 推出来的写「推断」并给推导。
  - 查不到的写「未确认」。
- **门槛不能比噪声严**：任何门槛都要附上它依据的噪声参照（GPU 自己重跑、TPU 只改执行方式的扰动臂），说明窗口和分母。
- **工具先自证**：
  - 比较工具上线前先喂一份故意改坏的数据，确认会报警；
  - 抓数工具先证明插桩前后输出逐位一致。
- **指标在 FP64 里算**：相近的数相减是精确的，丢精度的是后面的平方和、点积和除法。

### 版本、文档与结果

- **版本**：复现写明用的 tag，不写分支。
- **文档跟着代码走**：检查类型的规格（`docs/reference/checks/<类型>.md`）与实现一起改。
  - 发现文档与代码行为不一致，确认哪边对以后一起改齐。
- **结果里的数字不手写**：
  - `report.md` 只从 `results.json` 渲染；
  - 文档引用的数字写明出自哪个结果文件的哪条检查。
- **不改、不删已有的结果**：`results/`、`figures/`、`faults/`、`cases/` 里已提交的东西是证据。
  - 发现错了，追加更正并写明原因，不悄悄覆盖。

## 开发环境

```bash
pip install -e '.[dev]'     # 核心 + matplotlib + pytest + ruff（版本钉在 pyproject.toml）
make check                  # lint + 全部单元测试，只要 CPU
make help                   # 其余目标：test、lint、fmt、docs-check、dist、dashboard、clean
```

- 抓数代码（`janus/dump/megatron.py`、`janus/dump/maxtext.py`）要 torch / jax，只在训练镜像里跑。
  - CI 不装它们，相关测试没装时自动跳过。
- GPU / TPU 上的运行步骤在各框架的 README：
  - [`models/hunyuan3/frameworks/megatron/README.md`](models/hunyuan3/frameworks/megatron/README.md)
  - [`models/hunyuan3/frameworks/maxtext/README.md`](models/hunyuan3/frameworks/maxtext/README.md)

## 常见改动要动哪些地方

| 改动 | 要动的文件 |
|---|---|
| 加一种检查类型 | `janus/checks/<模块>.py`、`janus/run.py`、规格、测试（见注 1） |
| 改一种检查的判法 | 实现 + 规格；`CHANGELOG.md`（见注 2） |
| 加一个指标 | `janus/metrics.py`、`docs/reference/metrics.md`、`docs/reference/judging.md`、测试（见注 3） |
| 接一个训练框架的抓数 | `janus/dump/<框架>.py`、`docs/reference/dump.md`、测试（见注 4） |
| 加一个子命令 | 照 [`docs/reference/cli.md`](docs/reference/cli.md)「加一个子命令」 |
| 接一个新模型 | 照 [`docs/guide/08-new-model.md`](docs/guide/08-new-model.md) |
| 开一个对齐实践 | 照 [`docs/dev/layout.md`](docs/dev/layout.md)「加一个新实践要做的事」 |
| 提交结果文件 | `make dashboard` 重画总览与逐级大表，改动的 `.md` / `.html` 一起提交（CI 会核） |
| 写或改检查项目录 | 改完先 `janus validate <目录文件>`；CI 会校验仓库里全部目录 |
| 记一次排查 | 照 [`docs/guide/06-debugging.md`](docs/guide/06-debugging.md)，模板 [`templates/case.md`](templates/case.md) |

注 1（加一种检查类型）：

- `janus/checks/<模块>.py`（`@register`，同一族的类型放同一模块）；
- `janus/run.py` 的取数（新类型先写成纯函数加单测，再接取数）；
- 规格 `docs/reference/checks/<类型>.md` 与 `README.md` 的类型表；
- `docs/reference/catalog-schema.md` 的类型列表；
- 测试（含故意改坏会被抓到）。

注 2（改一种检查的判法）：

- 实现 + 规格一起改；
- `CHANGELOG.md` 写进对应一节（修 bug 进「Fixed」，有意改判法进「Changed」）；
- 条目前标 **[判定]**，并说明已提交的结果受不受影响。

注 3（加一个指标）：

- `janus/metrics.py`；
- `docs/reference/metrics.md` 一行（怎么算、边界、实测量级）；
- `docs/reference/judging.md` 的表一行（为什么要它、盲点）；
- 测试。

注 4（接一个训练框架的抓数）：

- `janus/dump/<框架>.py`，写同一种运行目录；
- `docs/reference/dump.md`，与 `docs/reference/dump-format.md` 有出入的地方写清；
- 测试放 `tests/dump/`。

## 代码风格

- `ruff check` 管全仓库，规则在 `pyproject.toml`；`make fmt` 自动修能修的。
- `ruff format` 只对 `Makefile` 里 `FMT_PATHS` 列出的文件强制。
  - 新写的模块、整体重写的模块加进这个列表。
  - 老代码保留对齐的行尾注释，不批量重排（理由见 [`docs/dev/design/0001-repo-standards.md`](docs/dev/design/0001-repo-standards.md)）。
- 命名照 [`docs/dev/conventions.md`](docs/dev/conventions.md)；模块开头写一段说明（做什么、规格在哪）。
