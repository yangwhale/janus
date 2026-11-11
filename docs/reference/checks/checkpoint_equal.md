[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `checkpoint_equal`

# checkpoint_equal

**用途**：权重转换、存档往返、加载后的权重对参照。

**输入**，两种模式：

- **清单模式**（`select.manifest: true`，默认）：两个运行目录的参数清单 `params_sha/rank<R>.<which>.json`。
  - `which` 取 `init` / `final`，默认 `init`。
  - 每项 `{sha256, shape, dtype}`，抓数或转换时对每个张量的原始字节全量算出。
  - 默认比被测运行与 golden（`compare: lkg` 时比 lkg）。
  - `select.test_run` / `select.ref_run` 可以把这一条改成比两个指定的运行（例如「MaxText 检查点还原成 HF 命名」的清单对「HF 原件」的清单）。
  - `select.source: state` 改读训练状态的 FP32 全量清单 `state_sha/<which>.json`（TPU 侧 `--state-sha` 写出，见 [`docs/reference/dump-format.md`](../dump-format.md)）：
    - 主权重、两个动量、路由偏置、计数器全在里面，按存储的 dtype 算 sha256、不转 BF16。
    - 日常回归的 R07 / R15 用它——FP32 主权重上不到半个 BF16 格的差，BF16 清单看不见。
- **目录模式**（`select.manifest: false`）：`select.test_dir`、`select.ref_dir` 两个检查点目录逐文件比。

**判据**：都不抽样。

- 清单模式按 `<分组>/<名字>` 逐项比 sha256：
  - 任一项不同或只在一边 → `fail`；
  - 参照清单为空 → `error`。
- 目录模式逐文件 sha256 全部相同、文件集合相同 → `pass`。

**输出**：清单模式 `params`（两边都有的项数）、`differ`、`only_one_side`；明细列出每个不同项两边的 sha256 前 16 位。
