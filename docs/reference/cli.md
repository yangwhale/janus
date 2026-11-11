# 接口规范 · 命令行

装好包（`pip install -e .`）后有一个总命令 `janus`。

- `python -m janus` 与它等价，不装也能在仓库根目录用。
- 实现在 `janus/cli.py`：每个子命令是一个模块的 `main(argv) -> int`，按需导入。

```bash
janus --version          # 打印版本（与 results.json 里的 janus.version 同一个来源）
janus                    # 列出子命令
janus <子命令> --help    # 子命令的参数
```

## 子命令

| 子命令 | 模块 | 做什么 |
|---|---|---|
| `run` | `janus/run.py` | 读一份检查项目录，跑完全部检查，写结果与报告（见注 1） |
| `validate` | `janus/validate.py` | 只校验检查项目录：展开后有哪些检查、校验规则过不过；不读运行目录、不跑检查 |
| `demo` | `janus/demo.py` | 在本地造一组合成运行目录与检查项目录（不要 GPU / TPU），演示怎么判 |
| `curves` | `janus/analysis/curves.py` | 多次运行按步对标的时序图与统计（见注 2） |
| `offsets` | `janus/analysis/offsets.py` | 被测对 golden 的 loss 偏移归因（见注 3） |
| `drift` | `janus/analysis/drift.py` | 多步漂移拆解（见注 4） |
| `fault-matrix` | `janus/analysis/fault_matrix.py` | 故障演练矩阵：读每个故障运行的 `results.json`，按期望 YAML 标出「该报没报」 |

注 1（`run` 写什么）：`results.json`、`details/`、`report.md`；加 `--html` 另出 `report.html`。

注 2（`curves` 的两个子命令）：
- `compare`：被测对 golden；
- `groups`：按设置分组互比，另出各设置之间的持续偏移。

注 3（`offsets` 给什么）：
- 分段带符号均值；
- 后段分布；
- 错位自检；
- 其余逐步量谁先漂。

注 4（`drift` 的四个子命令）：
- `pairs`：几组运行两两比；
- `trend`：趋势包络与累积均值；
- `groups`：param_norm 多涨来自哪一组参数；
- `rows`：输出层逐行，按词频分桶。

### `janus run`

```bash
janus run <目录文件> [--out <结果目录>] [--test <被测运行名>] [--lkg <LKG 运行名>] [--root <运行根目录>] \
    [--html] [--render-only] [--verbose]
```

| 参数 | 作用 |
|---|---|
| `<目录文件>` | 检查项目录（YAML），格式见 [`catalog-schema.md`](catalog-schema.md) |
| `--out` | 结果目录，默认 `./janus-results` |
| `--test` | 换被测运行名，目录文件不用改（逐个跑故障注入运行时用） |
| `--lkg` | 日常回归：换 LKG 运行名，目录里 `compare: lkg` 的检查都对它比 |
| `--root` | 换运行根目录；`results.json` 里记的是实际路径 |
| `--html` | 另出 `report.html`，要 `pip install -e '.[report]'` |
| `--render-only` | 不跑检查，按 `--out` 里已有的 `results.json` 重画 `report.md`（加 `--html` 也重画 `report.html`） |
| `--verbose` | 逐条打印检查结果与耗时 |

旧写法 `python -m janus <目录文件> …`（不写 `run`）仍然认，当 `run` 处理。

### `janus validate`

```bash
janus validate <目录文件> […]          # 校验这几份
janus validate --all models [-q]       # 递归校验 models/ 文件夹下所有 catalogs/*.yaml，跳过被 include 的片段
```

和 `janus run` 开跑前用的是同一份校验（`janus.catalog.load`），包括：

- YAML；
- `include` / `vars` / `foreach` 展开；
- [`catalog-schema.md`](catalog-schema.md)「校验规则」。

输出与 CI：

- 每份打印检查条数与各级分布；有不通过的退出码 2。
- CI 里 `tests/test_validate.py` 对仓库里提交的全部目录跑一遍。

### `janus demo`

```bash
janus demo <输出目录>
janus run <输出目录>/catalog.yaml --out <输出目录>/out-good                # 全部 pass，退出码 0
janus run <输出目录>/catalog.yaml --out <输出目录>/out-bad --test t-bad    # L2.layer1.fwd 与 L4.loss fail，退出码 1
```

写出的运行：

- 6 份参照运行 `g-1`…`g-6`，`side` 都是 golden。
  - 目录里列 `g-1`、`g-2` 两份当 golden，给多 golden 的 loss 对比；
  - 6 份两两 15 对当噪声参照。
- 两份被测：
  - `t-good` 只带同量级的随机差；
  - `t-bad` 埋了一个系统错（第 1 层输出整体放大 0.3%，loss 从第 2 步起整体偏高 0.05%）。

演示什么：

- `t-bad` 的结果里第 0 层仍然 pass、第 1 层 fail，演示的是「逐层比较能把错定位到层」。
- 数据格式与真实运行相同，张量很小。
- 输出目录里已有 `runs/` 时报错，不覆盖。

### `janus curves`

```bash
janus curves compare --test <名>=<metrics.jsonl> --golden <名>=<metrics.jsonl> … --out <目录> \
    [--prefix L4] [--fields lm_loss grad_norm] [--steps 1:31] [--window N]
janus curves groups <series 目录> <输出目录> --config <分组 YAML> [--no-offsets]
```

- 每个字段出三张图（`*-overlay`、`*-pairdiff`、`*-mean`）。
- `groups` 另写 `curves-summary.json`，配置里有 `settings` 时再写 `offsets.json`。
- 分组 YAML 的键见 `janus/analysis/curves.py` 的 `GroupSpec`，例子是 `models/hunyuan3/analysis/phase1-curves.yaml`。
- 要 matplotlib。

### `janus offsets`

```bash
janus offsets --test <名>=<metrics.jsonl> --golden <名>=<metrics.jsonl> [--golden …] [--json <输出>] \
    [--segments 2:101 101:301 …] [--tail 301:1001] [--lag-window 2:1001] [--norm-steps 1 100 …]
```

- 分段、后段、取样步默认按 1000 步的运行设定。
- 输出各项的含义见模块开头的说明。
- 用法实例见 [`01-loss-offset-param-norm.md`](../../models/hunyuan3/alignments/01-v5p-acceptance/cases/01-loss-offset-param-norm.md)。

### `janus drift`

```bash
janus drift pairs  --series <名>=<metrics.jsonl> … --groups <组>=<名>,<名> … --pairs "<标签>:<组A>-<组B>" … \
                   [--fields …] [--tail 301:1001] [--norm-step 1000] [--json <输出>]
janus drift trend  --test <运行目录> --golden <运行目录> … --out <输出> [--steps 1000]
janus drift groups --test <metrics.jsonl> --ref <metrics.jsonl> [--pair <metrics.jsonl> …] [--steps 100,300,1000] [--out <输出>]
janus drift rows   --test <metrics.jsonl> --ref <metrics.jsonl> [--pair …] --counts <label_counts.npz> [--key c300] [--steps …] [--out <输出>]
```

L4 发现持续偏差以后往下拆。

- `groups` 要抓数时开了分组参数范数（`group_sq.*`、`expert_sq.*`）。
- `rows` 要逐行统计（`output_row_sq`、`output_row_u`、`embedding_row_sq`）。
- 每个子命令输出的字段见模块开头的说明；用法实例见案例 01。

### `janus fault-matrix`

```bash
janus fault-matrix <结果目录> --spec <期望 YAML> [--strict]
```

- 结果目录下每个故障运行一个子目录，里面是 `janus run --test <故障运行>` 的 `results.json`。
- 期望 YAML 写 `faults: {运行名: {what, expect: [检查 id…]}}`，例子是 `models/hunyuan3/analysis/phase1-faults.yaml`。
- `--strict`：有期望报警而没报的故障时退出码 1。

## 退出码

所有子命令共用。

- `janus run` 的 0 / 1 / 2 怎么判（哪些检查算硬检查、`inconclusive` 为什么不影响），以 [`result-schema.md`](result-schema.md)「退出码」为准。
- 这里只列命令行层面的意思：

| 码 | 意思 |
|---|---|
| 0 | 成功；`janus run` 指没有硬检查 `fail` 或 `error` |
| 1 | `janus run`：有硬检查 `fail`<br>`janus fault-matrix --strict`：有期望报警而没报的故障 |
| 2 | 用法错（参数不对、没有这个子命令）<br>目录文件不存在或不合规<br>`janus run` 有硬检查 `error` |

`inconclusive` 不影响退出码，在报告首屏单列。

## 参数写法

现有子命令的参数是各自长出来的，写法不完全一致。例如：

- `--test` 在 `run` 里是运行名、在 `offsets` / `curves compare` 里是 `名=metrics.jsonl`、在 `drift trend` 里是运行目录；
- 输出文件有的叫 `--out` 有的叫 `--json`。

旧参数不改名，免得已有命令失效。**新加的子命令照这几条**：

- 输入：
  - 一份逐步原值写 `名=metrics.jsonl`；
  - 一个运行目录直接写路径，目录名当运行名；
- 输出：一个文件用 `--out`，一个目录也用 `--out`；
- 步：
  - 范围写成和目录里 `window` 一样的 `a:b`（含 a 不含 b）；
  - 几个离散的步写成逗号分隔 `100,300,1000`；
- `--help` 的 `prog` 写 `janus <子命令>`，出错按下面的退出码返回，不抛栈。

## 加一个子命令

1. 写模块，提供 `main(argv: list[str]) -> int`，用 `argparse`，`prog` 写成 `janus <子命令>`。
2. 在 `janus/cli.py` 的 `COMMANDS` 加一行。
3. 在本页的表里加一行；`tests/test_cli.py` 会对每个子命令跑一次 `--help`。
