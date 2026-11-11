# 接口规范 · 结果格式

`schema_version: 2`（与 1 的区别只在 `catalog`，见注 2）。`janus run` 写出两样：

- `results.json`：汇总；
- `details/<check-id>.jsonl`：逐张量 / 逐步明细。

报告 `report.md` 只从这两样渲染，不手写数字。

## results.json

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | int | 2；独立脚本写的结果仍是 1（见注 2） |
| `janus` | obj | `{version, commit}`，见注 1 |
| `catalog` | obj | `{path, hash_algo, content_sha256, raw_sha256}`；目录用了 `include` 时另有 `includes`，每项同样四个键。见注 2 |
| `runs` | obj | 运行名 → `{path, meta_sha256}`，含 test、golden、lkg、noise 里出现的全部运行 |
| `started_at` / `finished_at` | str | ISO 8601，带时区 |
| `summary` | obj | `{pass, fail, inconclusive, error, hard_fail}`；前四个是计数，`hard_fail` 为布尔 |
| `checks` | list | 每条检查一项，见下 |

注 1（`janus`）：
- `version` 即 `janus --version`；
- `commit` 是 janus 的 git 提交号。
  - janus/ 下有未提交改动时带 `+dirty`；
  - 不是从 git 工作区装的为 `null`。

注 2（`catalog`）：
- `content_sha256` 是目录的**内容指纹**，`hash_algo` 写算法，现在是 `content-v1`（`janus/catalog.py` 的 `content_sha256`）：
  1. 解析 YAML，注释随之丢掉；
  2. 删掉纯描述的键：顶层的 `description`；`checks` 每一项的 `description`、`notes`、`source`、`pending_note`；
  3. 序列化成 JSON：键排序、没有空白、UTF-8 原文，浮点用 Python 的 `repr`（最短往返表示：`1e-3` 与 `0.001` 相同，`1` 与 `1.0` 不同）；
  4. 取 sha256。
- 名单以外的键都算内容，包括 `runs`（含 `root`）、`select`、`threshold`、`metrics`、`params`（`params` 里的 `source` 也算）、
  `hard`、`compare`、`foreach`、`vars`、`include`，以及 janus 不读、`regress.py` 读的 `row` / `role` / `expect`。
  - 所以改注释或上面那几个说明键，指纹不变，已提交的结果照样对得上；
    结果里抄下的 `threshold.source`、待标定的 `note` 仍是跑的时候的原文；
  - 改了检查本身，就用同一批运行目录重跑并提交新结果。
- 指纹按文件算，不展开 `include` / `vars` / `foreach`；片段各记一份（`includes`）。
- `raw_sha256` 是运行那一刻目录原文件的 sha256，只留作历史，不拿来核对。
- 仓库里已提交的结果都与现在的目录对得上（`tests/test_committed_results.py` 守着）。
  - 2026-11-06 之前的结果从 `schema_version` 1 迁移过来：`raw_sha256` 是原来记的 `sha256`；
    迁移脚本 `tools/migrate_catalog_content_hash.py` 在 git 历史里找到跑的那一版目录，确认它的内容指纹与现在的相同才写，
    记录在 `tools/migrate_catalog_content_hash.log`。
  - 暂时没法核对的登记在那个测试的 `HISTORICAL` 里，写明原因。
- `schema_version` 1 的 `catalog` 是 `{path, sha256}`，`sha256` 是目录原文件的哈希（改一个字就对不上）。
- 独立脚本写的结果仍是 `schema_version` 1：`catalog` 是 `{path, sha256}`，`path` 是脚本自己、`sha256` 是脚本文件的哈希，
  改了脚本就用当前脚本重跑。

### 每条检查

| 字段 | 类型 | 说明 |
|---|---|---|
| `id`、`type`、`level`、`hard` | | 同目录文件 |
| `status` | str | `pass` / `fail` / `inconclusive` / `error` |
| `threshold` | obj | `{kind: exact\|numeric\|calibrated\|pending, values, source}`，见注 1 |
| `metrics` | obj | 汇总指标：如 `{rel_l2_max: …, cosine_min: …}`；序列检查给 `{d0, signed_mean, mare, max_step, max_step_at, window, denominator}` |
| `r1`、`r2` | float | 仅 `calibrated`：被测对噪声上限的比值 |
| `worst` | list | 最差的前若干项 `{name, metric, value, location}` |
| `coverage` | obj | `{expected, compared, missing: [...]}`（适用时） |
| `evidence` | list | 读过的文件 `{path, sha256}` |
| `details` | str | 明细文件相对路径，无则省略 |
| `message` | str | 一句话结论；`error` 时写原因 |

注 1（`threshold`）：
- `calibrated` 时 `values` 是算出来的数，另给 `noise: {refs, upper}`；
- `pending`（待标定）另给 `note`，结果最多 `inconclusive`。

### 状态的判法

- `exact`：逐位相同 `pass`，否则 `fail`。
- 数值门槛：指标在门槛内 `pass`，否则 `fail`。
- `calibrated`：被测值 x、噪声参照上限 N。
  - x ≤ N `pass`；x > 2N `fail`；其间 `inconclusive`。
  - 噪声参照少于 3 对时，最多给 `inconclusive`。
- 数据缺失、校验和不符、`select` 选空：`error`（不是 `pass`）。

### 退出码

| 码 | 含义 |
|---|---|
| 0 | 没有 `hard: true` 的检查是 `fail` 或 `error` |
| 1 | 至少一条 `hard` 检查 `fail` |
| 2 | 至少一条 `hard` 检查 `error`，或目录文件校验不通过 |

`inconclusive` 不影响退出码，但在报告首屏单列。

## details/<check-id>.jsonl

一行一个比较单元（一个张量、一步、一层）：

```json
{"name": "grad.layers.3.mlp.experts.w1", "rel_l2": 3.1e-3, "cosine": 0.99999, "max_abs": 0.0078, "max_abs_at": [17, 1023, 5], "status": "pass"}
```

## 最小示例

```json
{
  "schema_version": 1,
  "janus": {"version": "0.1.0", "commit": "…"},
  "catalog": {"path": "models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/phase1-gpu-self.yaml", "sha256": "…"},
  "runs": {"hy3-4l-p1-golden-2-20261110-1": {"path": "gs://…", "meta_sha256": "…"}},
  "started_at": "2026-11-10T10:00:00+08:00",
  "finished_at": "2026-11-10T10:04:12+08:00",
  "summary": {"pass": 2, "fail": 0, "inconclusive": 1, "error": 0, "hard_fail": false},
  "checks": [
    {"id": "L4.loss", "type": "multi_golden", "level": "L4", "hard": true, "status": "inconclusive",
     "threshold": {"kind": "calibrated", "values": {"signed_mean": 4.1e-4}, "noise": {"refs": 3, "upper": {"signed_mean": 4.1e-4}}},
     "metrics": {"d0": 0.0, "signed_mean": 6.0e-4, "mare": 1.8e-3, "max_step": 9.0e-3, "max_step_at": 812, "window": "1:1000", "denominator": "golden"},
     "r1": 1.2, "r2": 1.46, "message": "带符号均值在噪声上限的 1–2 倍之间"}
  ]
}
```

示例中的数字只为说明格式，不是测量值。
