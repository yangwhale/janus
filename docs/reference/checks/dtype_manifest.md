[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `dtype_manifest`

# dtype_manifest

**用途**：导出 / 转换 / 存档后参数与 buffer 的 dtype 对不对。

典型：expert_bias 这类小张量必须 FP32，存成 BF16 后偏置更新量（远小于偏置本身）会被舍入吃掉。

**输入**：

- 被测的参数清单 `params_sha/rank<R>.<init|final>.json`（每项有 `dtype`）。
  - `select.which` 取 `init`、`final` 或两者的列表（默认 `final`）。
  - `select.run` 可指定查另一个运行的清单（默认查被测）。
  - `select.source: state` 改查 FP32 全量状态清单 `state_sha/<which>.json`（名字是 MaxText 的状态路径，分组写成 `<which>/state`）。
- `params.rules`：`[{pattern, dtype, exclude?}]`。

**判据**：

- 每个名字按规则顺序取**第一条**匹配的（fnmatch；`exclude` 匹配的不算命中这一条），dtype 不符 → `fail`。
- 没有任何规则覆盖的名字 → `error`（规则没写全，不能说清单合格）。
- 清单为空 → `error`。
- dtype 名先归一（`torch.bfloat16`、`BF16`、`bf16` 都是 `bfloat16`）。
- 逐 rank 逐项查，不抽样。

**输出**：`entries`、`wrong_dtype`、`unmatched`、`rule_hits`（每条规则命中几项，规则写错时一眼能看出命中 0）；明细列出每个不符项的实际与期望 dtype。

**目录写法**（第一阶段）：

```yaml
params:
  rules:
    - {pattern: "buffer.*expert_bias", dtype: float32}
    - {pattern: "*", exclude: "buffer.*", dtype: bfloat16}
```
