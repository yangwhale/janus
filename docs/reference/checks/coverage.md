[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `coverage`

# coverage

**用途**：防止「漏比了也报一致」。

**输入**：`expected`（期望比较的名字，可含通配）、`compared`（实际比较过的名字）。从目录调用时两边都按 `select.name` 选、按 `select.exclude` 剔（例如 GPU 专家分发缓冲区 `*.experts.out` 在 TPU 上没有对应量，用 exclude 排除，不算缺）。

**判据**：期望里每一个都比到 → `pass`；缺任何一个 → `error`（不是 `fail`：说明证据不全，不是数值不对）。

**输出**：`coverage = {expected, compared, missing}`（`missing` 最多列 50 个）。

**边界**：期望清单为空 → `error`。
