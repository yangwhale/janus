[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `artifact_check`

# artifact_check

**用途**：开关真的生效了——在编译产物文本里计数，例如 FP8 乘法个数不为 0。

日常回归里另有两个用法：

- 计数与 LKG 相等（R20）；
- 产物指纹与 LKG 相同（R21，也用来整份比编译器生效的开关文件，R05）。

**输入**：

- 产物文本（`select.file`，运行目录里的相对路径）；
- 规则 `[{name, regex, min?, max?, match_ref?}]`；
- `compare: lkg` 时另读 LKG 运行目录里的同一个文件当参照；
- `params.fingerprint: true` 时比指纹。

**判据**：

- 每条规则的匹配次数都在 [min, max] 内（`min` 默认 0，`max` 默认无穷）；
- `match_ref: true` 的规则，匹配次数还要等于参照文件上的同一条计数。
  - 个数不写死在目录里，从 LKG 记下。
  - 编译器版本一换个数就会变，写死的数要跟着改；对 LKG 比只要求「这段时间没变」。
- `fingerprint: true`：指纹 = 去掉 `metadata={…}`（op 名、源码文件与行号）与 `stack_frame_id=…` 之后全文的 sha256。
  - 被测与参照不同即 `fail`，`details` 列出只在一边的行（各最多 10 行）。
  - 改源码行号、改函数名不改指纹；改融合、调度、算子、精度都改。
  - 只报的行（R21）写 `hard: false`。
  - `params.ignore_lines`（正则列表）：匹配到的整行在算指纹前两边都去掉，`metrics.ignored_lines` 报各去掉几行。
  - `ignore_lines` 只用于和被测平台无关的行（例：TPU 上编译器开关文件里的 `xla_gpu_*` 选项）。

结论：

- 全部满足 → `pass`。
- 有 `match_ref` 或 `fingerprint` 却没给参照（目录里没写 `compare: lkg`）→ `error`。
- 文件不存在 → `error`。

**输出**：

- `metrics.rules`、`metrics.bad`。
- 指纹模式另有 `fingerprint`、`ref_fingerprint`（前 16 位）、`fingerprint_equal`；不同时 `lines_only_test`、`lines_only_ref`。
- `details` 每条规则一行：`count`、`ref_count`（`match_ref` 时）、`min`、`max`、`ok`。
