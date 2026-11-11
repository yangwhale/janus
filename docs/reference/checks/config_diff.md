[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `config_diff`

# config_diff

**用途**：两边配置逐项对照（dropout、优化器超参、精度、数据约定等）；关掉 RNG 消费者这类「必须是某个值」的断言。

**输入**：两份配置（嵌套字典）、映射表，每项两种写法：

- 两边对照：`{test: 键路径, ref: 键路径, must_equal, transform}`；
- 等于给定值：`{test: 键路径, equals: 值, ref: 可选键路径}`。
  - 被测、参照两边都必须等于给定值（`ref` 省略时与 `test` 同路径）。
  - 用于 dropout = 0、随机路由关——两边相同还不够，两边一起开着 dropout 也「相同」。
- 整个字典对照：两边对照的写法里，键路径指到的两边都是字典时逐子键比。
  - `ignore`（通配列表）匹配到的子键不比。
  - 用于展开后的完整配置、运行进程的环境变量（日常回归 R05）：运行名、输出目录、主机名每次都不同，不比。

**判据**：

- `must_equal` 的字段不同 → `fail`；
- `equals` 任一边不等于给定值 → `fail`；
- `must_equal` / `equals` 的字段缺失 → `error`；
- 其余不同只列出。
- 比较用 `==`（0 与 0.0 相等）；`transform: float` 时先转浮点。

**输出**：

- `fields`、`must_equal_differ`（含 `equals` 不符）、`other_differ`、`missing`、`equals_rules`。
- 明细每个对照一行，`equals` 每边一行。
- 字典对照的那一行写 `keys`（参与比较的子键数）、`ignored`、`differ_count` 与 `differ`（不同的子键，最多 20 个，每个带两边的值）。
  - 不把几百个相同的子键都写出来。
