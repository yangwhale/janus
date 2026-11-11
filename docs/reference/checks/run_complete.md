[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `run_complete`

# run_complete

**用途**：运行完整（日常回归 R01）。

没跑完、中途重启、少写或多写了步的运行，后面每一行的结论都不可信，也永远不能当 LKG。

**输入**：被测运行目录。

- `metrics.jsonl` 里按文件顺序的步号。
  - 不经过递增校验：重启后接着追加写的运行会出现重复步，要能数出来。
- `meta.json` 的 `steps`（`first`、`last`）与 `restarts`（有就读，没有按 0）。
- `params.steps`（计划步数，必填）、`params.first`（首步，默认 1）。

**判据**：同时满足才 `pass`，否则 `fail`：

- 步号恰好是 `first … first + steps − 1`：没有重复、没有缺步、没有计划外的步；
- `meta.steps` 与计划相同；
- `meta.restarts` 为 0。

`DONE` 由读运行目录时保证（没有 `DONE` 的目录打不开，整份结果报 `error`）。`params.steps` 没写 → `error`。

**输出**：`metrics.planned`、`logged`、`duplicate`、`missing`、`extra`、`restarts`；`message` 写出具体哪几步重复或缺。

**不管什么**：作业为什么没跑完（被抢占、超时、OOM）。那要看作业日志，分类照 [`docs/reference/regression.md`](../regression.md) 第八节。
