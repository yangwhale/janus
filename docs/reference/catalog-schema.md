# 接口规范 · 检查项目录

`schema_version: 1`。一个目录文件描述一个模型、一个阶段要过的全部检查。

- 放在哪：
  - 迁移验收：`models/<模型>/alignments/<对齐实践>/catalogs/<阶段>.yaml`；
  - 日常回归：`models/<模型>/regression/catalogs/regression-<档>.yaml`。
- `janus run <目录文件>` 读它，产出结果（见 `result-schema.md`）。

本页结构：顶层字段 → 检查项字段（select、foreach、各类型的 select 与 params）→ 说明文字与内容指纹 → 校验规则 → 最小示例。

## 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `schema_version` | int | 是 | 1 |
| `model` | str | 是 | 与 `meta.json` 的 `model` 一致 |
| `stage` | str | 是 | 如 `phase1-gpu-self`、`regression-main` |
| `description` | str | 否 | 一句话 |
| `runs` | obj | 是 | 见下 |
| `defaults` | obj | 否 | 各检查的默认参数（窗口、分母、指标），检查里可覆盖 |
| `checks` | list | 是 | 检查项列表 |
| `include` | list | 否 | 片段文件（相对本文件），见注 1 |
| `vars` | obj | 否 | 合并后的 `checks` 里 `{变量}` 换成取值，见注 2 |

注 1（`include`）：
- 片段里的 `checks` 排在本文件的 `checks` 前面。
- 几个目录共用同一批检查时用（如日常回归各档），共用的部分只写一处。

注 2（`vars`）：字符串恰好是 `{变量}` 时保留取值的类型（数字还是数字）。

### runs

```yaml
runs:
  test: t-a                                 # 被测运行名
  golden: [g1k-a, g1k-b, g1k-c]             # 一份或多份 golden
  lkg: null                                 # 回归门用：上一次通过的运行名（janus run --lkg 可在命令行换）
  noise:                                    # 噪声参照：运行对，至少 3 对才能出 calibrated 的 pass
    - [g1k-a, g1k-b]
    - [g1k-a, g1k-c]
    - [g1k-b, g1k-c]
  root: gs://<POC 桶>/janus                 # 运行目录的根
```

## 检查项字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | str | 是 | 全目录唯一，点分，如 `L3.grad.per_tensor` |
| `type` | str | 是 | 检查类型名之一，见注 1 |
| `level` | str | 是 | `L0`…`L4`、`regress` |
| `compare` | str | 否 | 比谁：`golden`（默认）、`lkg`、`self`（同一运行内，如副本一致性） |
| `select` | obj | 是 | 取哪些数据，见下 |
| `metrics` | list | 视类型 | 如 `[rel_l2, cosine, max_abs]` |
| `threshold` | str / obj | 是 | `exact`、`calibrated`、`pending`，或指标到数值的映射，见注 2 |
| `source` | str | 视情况 | `threshold` 含数值时必填：issue 评论、运行名，或「推断：<推导>」 |
| `hard` | bool | 否 | 默认 `true`，见注 3 |
| `params` | obj | 否 | 类型特有参数（如 `top_k`、`window`、`denominator`、`hard_limit`、`near_zero_floor`） |
| `notes` | str | 否 | 备注 |
| `row` / `role` / `expect` | | 否 | janus 不读，给上层脚本用，见注 4 |

注 1（`type`）：规格见 [`checks/README.md`](checks/README.md)。可选的类型名：

`bitwise_equal`、`tensor_tolerance`、`ulp_distribution`、`discrete_agreement`、`loss_series`、`multi_golden`、`scalar_series`、`optimizer_step`、`router_stats`、`noise_verdict`、`determinism`、`coverage`、`poison_selftest`、`config_diff`、`data_fingerprint`、`checkpoint_equal`、`replica_consistency`、`overlap_replay`、`artifact_check`、`health_signals`、`dtype_manifest`、`fake_fp32`、`op_replay`、`run_complete`

注 2（`threshold`）：
- `pending`（待标定）：
  - 没有噪声参照，最多 `inconclusive`；
  - 只能用在 `loss_series`、`router_stats`；
  - 可配 `pending_note` 写明等什么。

注 3（`hard`）：
- `true`：失败时 `janus run` 退出码非 0；
- `false`：只报不拦（如 FP8 软门）。
- 规格写明「默认只诊断」的类型（`ulp_distribution`）不写时默认 `false`。

注 4（`row` / `role` / `expect`，日常回归用）：
- `row` 是清单行号（R01…）；
- `role: tolerance` 标「只在 PR 声明执行级时采用的容差门」；
- `expect: {指标: 期望值}` 核单元数（R23）；
- 见 `models/<模型>/regression/regress.py`。

### select

| 键 | 说明 |
|---|---|
| `kind` | `tensor` / `metrics` / `tokens` / `checkpoint` / `config` / `artifact` |
| `step` / `steps` | 单步或范围 `"0:10"`（左闭右开，本侧步号；跨侧按 `meta.json` 的换算对齐） |
| `name` | 规范名或通配，如 `"grad.*"`、`"layers.*.mlp.router.topk_idx"` |
| `field` | metrics 的字段名，如 `lm_loss` |
| `test_field` | 序列检查里被测改读的字段（golden 与噪声对仍读 `field`），见注 1 |
| `window` | 序列检查的窗口 `"1:1000"` |
| `exclude` | 张量名通配（一个，或张量类 select 里给通配列表），从 `name` 选中的里面再剔掉一批（如专家分发缓冲区 `"*.experts.out"`） |
| `ref_rename` | `{from, to}`：参照张量名 = 被测张量名把 `from` 换成 `to`，见注 2 |

注 1（`test_field`）：如派生列 `param_norm_from_groups`（`Run.series` 由同一行的 `group_sq.*` 平方和开方现算）。

注 2（`ref_rename`）：配合 `compare: self` 在同一次运行里比两组张量（如 `<名>.recompute` 对 `<名>`）。

### foreach

一条检查可以写 `foreach: {变量: [取值…]}`。

- 条目里所有字符串中的 `{变量}` 换成每个取值，一条变多条；
- 多个变量取笛卡尔积。
- 用于逐层展开：`id: L2.layer{i}.fwd`、`name: "layers.{i}.*.out"`、`foreach: {i: [0, 1, 2, 3]}`。

### 各类型的 select 与 params

| 类型 | select | params / 其他字段 |
|---|---|---|
| `bitwise_equal`、`tensor_tolerance`、`ulp_distribution`、`coverage` | `{kind: tensor, step, name[, exclude]}` | `metrics`；`tensor_tolerance` 的 calibrated 噪声来自 `runs.noise` |
| `discrete_agreement` | `{kind: tensor, step, name}` | `mode: exact / topk`；`reference: fp64_replay`（路由选中集合对 FP64 重算） |
| `loss_series`、`scalar_series`、`multi_golden` | `{kind: metrics, field, window}` | 见注 1 |
| `router_stats` | `{kind: metrics, prefix, window}` | 例如 `prefix: router.bias_std.`，逐层统计各算一遍取最差 |
| `health_signals` | `{kind: metrics, fields}` | `window`、`spike_ratio`、`source` |
| `determinism` | `{kind: metrics, fields[, window, step, name]}` | `window` 只比窗口内的步；给了 step / name 时张量也逐位比 |
| `overlap_replay` | `{kind: metrics[, fields]}` | 被测 = 续训运行，golden = 连续运行 |
| `optimizer_step` | `{kind: tensor, step, name}`（参数名通配） | `replay_from: test / golden`、`weight_decay`（没记 `wd` 的旧运行用） |
| `config_diff` | `{kind: config}` | `mapping: [{test, ref, must_equal[, ignore]}]` 或 `[{test, equals}]`，见注 2 |
| `data_fingerprint` | `{kind: tokens, steps}` | `first_loss`（默认比首步 loss） |
| `checkpoint_equal` | `{kind: checkpoint, which: init / final[, source, test_run, ref_run]}` | 默认比参数清单，见注 3 |
| `replica_consistency` | `{kind: checkpoint, which, name, exclude}` | 被测运行里每个参数在各 rank 上的 sha256；按专家并行切开的参数要 exclude（同名不同专家，不是副本） |
| `artifact_check` | `{kind: artifact, file}` | `rules: [{name, regex, min, max, match_ref}]`、`fingerprint`；`compare: lkg` 时读 LKG 的同名文件当参照 |
| `run_complete` | `{kind: metrics}` | `steps`（计划步数，必填）、`first`（默认 1） |
| `poison_selftest` | `{kind: tensor, step, name}` | `target`、`target_kwargs`、`injections` |
| `dtype_manifest` | `{kind: checkpoint, which: init / final / [init, final][, run, source: params / state]}` | `rules: [{pattern, dtype, exclude?}]`；`run` 改查另一个运行的清单，`source: state` 改查 FP32 全量状态清单 |
| `fake_fp32` | `{kind: tensor, step, name[, exclude]}` | `max_frac`（默认 0.99；改了要写 `source`）。查更新后的主权重与动量，不查从 BF16 载入的更新前主权重 |
| `noise_verdict`、`op_replay` | — | 不能写进目录，见注 4 |

注 1（三种序列检查的 params）：
- `first_step_tol` + `first_step_source`；
- `hard_limit`、`signed_mean_se_k`；
- `denominator: golden | test | mean`（默认 `golden`）；
- `rtol` / `abs_ulp`；
- `multi_golden` 另有：
  - `noise_from: runs | golden`：默认 `runs`，有 `runs.noise` 就用它；`golden` 固定用 golden 两两组合；
  - `baseline: <golden 名>`：单 golden 模式，只对这一份比，其余 golden 只组噪声。

注 2（`config_diff`）：
- 键路径指向 `meta.json`；
- 两边都是字典时逐子键比，`ignore` 的通配子键不比。

注 3（`checkpoint_equal`）：
- 默认比参数清单：被测对 golden；给了 `test_run` / `ref_run` 就比这两个运行。
- `manifest: false` 时比 `test_dir` / `ref_dir` 两个目录逐文件。

注 4（`noise_verdict`、`op_replay`）：
- 前者是其他检查内部的判法；
- 后者由回放脚本直接写结果（见下面「校验规则」第 9 条）。

判据写死在类型里的（逐位、计数规则、自证、健康信号、dtype 规则、假 FP32）不用写 `threshold`。

## 说明文字与内容指纹

结果记的是目录的内容指纹（`catalog.content_sha256`，算法 `content-v1`，见 [`result-schema.md`](result-schema.md) 注 2），不是原文件的哈希。

- 不进指纹、随时可改的：
  - 注释；
  - 顶层的 `description`；
  - 检查项的 `description`、`notes`、`source`、`pending_note`。
- 其余都进指纹：改了就是改了检查，要用同一批运行目录重跑 `janus run` 并提交新结果。
  - 包括 `runs`（`root` 也算）、`select`、`threshold`、`metrics`、`params`（里面的 `source` 也算）、`hard`、`compare`、
    `foreach`、`vars`、`include`、`row` / `role` / `expect`。
  - 新加的键默认进指纹；确认只是说明文字，才加进 `janus/catalog.py` 的名单（`DESCRIPTIVE_TOP_KEYS`、`DESCRIPTIVE_CHECK_KEYS`），
    同时升 `HASH_ALGO`。

## 校验规则

### 载入时（第 1–6 条）

载入目录时先校验，不通过直接退出（退出码 2）：

1. `runs` 有 `test` 与 `root`；`runs.noise` 每一项是两个运行名。
2. `id` 唯一；`type` 是已注册的检查类型；`level` 合法；`select.kind` 合法；必填字段齐全。
3. `threshold` 含数值时必须有 `source`。
4. `threshold: calibrated` 时 `runs.noise` 至少 1 对。
   - `multi_golden` 例外：没给、或写了 `noise_from: golden`，就用 golden 两两组合；
   - 少于 3 对时该检查只能得到 `inconclusive` 或 `fail`。
5. `compare: lkg` 时 `runs.lkg` 不为空。
6. `threshold: pending`（待标定）只能用在 `loss_series`、`router_stats`。

这六条由 `janus.catalog.validate` 实现，`janus validate` 与 `janus run` 开跑前用的是同一份。

### 执行每条检查时（第 7–9 条）

执行每条检查时再查，不通过的那条判 `error`：

7. 引用的运行目录存在 `DONE`、`meta.json` 合规，张量 sha256 能复算。
8. 类型特有参数按该类型规格（[`checks/`](checks/README.md)`<类型>.md`）解析；缺参数、取值不合法都报 `error`。
9. 这个类型能从目录里调用。
   - `noise_verdict` 是其他检查内部用的判法；
   - `op_replay` 的结果由 GPU 回放脚本直接写 `results.json`；
   - 两者写进目录都报 `error`（「还不能从目录里调用」）。

## 最小示例

```yaml
schema_version: 1
model: hunyuan3-4l
stage: phase1-gpu-self
runs:
  test: g1k-b
  golden: [g1k-a]
  noise:
    - [g1k-a, g1k-b]
    - [g1k-a, g1k-c]
    - [g1k-b, g1k-c]
  root: gs://<POC 桶>/janus
checks:
  - id: L0.tokens
    type: data_fingerprint
    level: L0
    select: {kind: tokens, steps: "0:10"}
    threshold: exact
  - id: L3.grad.per_tensor
    type: tensor_tolerance
    level: L3
    select: {kind: tensor, step: 0, name: "grad.*"}
    metrics: [rel_l2, cosine, max_abs]
    threshold: calibrated
  - id: L4.loss
    type: multi_golden
    level: L4
    select: {kind: metrics, field: lm_loss, window: "1:1000"}
    threshold: calibrated
    params: {denominator: golden}
```
