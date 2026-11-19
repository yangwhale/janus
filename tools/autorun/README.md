[janus](../../README.md) ｜ 对分支头跑数值门

# 对训练分支头跑数值门：手动触发为主

[`autorun.py`](autorun.py) 把「对被守分支的分支头跑一次数值门、出报告」包成一条命令。
**手动触发为主**（项目负责人 2026-11-18 定）：合并后需要时由人跑一次。它是确定性脚本，只用 git、gh、kubectl，
不调用任何模型或外部服务，所以也可以挂到站点的定时任务上做轮询，但**定时轮询可选、默认不开**。

```bash
python tools/autorun/autorun.py --config <站点配置> --job hy3-t1 --dry-run   # 混元 3 T1：对 hunyuan 分支头跑一次，不评论、不改状态
python tools/autorun/autorun.py --config <站点配置> --status                 # 各作业上次跑到哪个提交、结论是什么
```

`--job` 只跑这一个作业，不看它的 `enabled` 与总开关；不加 `--dry-run` 时，判不过会照作业的 `comment` 写回 PR 或提交，并记下状态。

为什么不用 GitHub Actions：org 已停用 Actions（`AGENTS.md` 第 4 节），而且门要用到 TPU 集群与本地编译缓存。

## 作业

| 作业类型 | 守什么 | 跑什么 | 资源 | 判定写回 |
|---|---|---|---|---|
| `hy3-t1` | `maxtext` 的 `hunyuan` 分支 | [混元 3 日常回归](../../models/hunyuan3/regression/README.md) T1（含 R09） | 一个 v5p-8 切片约 25 分钟；跑完删自己的 JobSet，节点池不动 | 不过、无结论 |

- 一次只处理分支头。上次跑过之后合进来的几个提交并成一次，报告里写明覆盖了几个提交。
- 判定的退出码约定：0 过、1 不过、2 工具出错、3 无结论。
- 评论只写结论与差异：分支头提交属于某个已合并的 PR 就评在 PR 下，否则评在提交下。过与工具出错不评论。
- 工具出错（集群起不来）只进日志，下一轮重试，同一提交最多 `max_attempts` 次；之后不再重试，等人看日志。

## 开关与日志

全部在站点配置里（仓库外；字段见 [`site.example.yaml`](site.example.yaml)）：

| 想做什么 | 怎么做 |
|---|---|
| 开定时轮询（可选） | 站点定时任务调 `autorun.py --config <站点配置>`（不带 `--job`），并把站点配置顶层 `enabled` 设为 `true`；只跑 `enabled: true` 的作业 |
| 停定时轮询（默认） | 站点配置顶层 `enabled: false`，或去掉定时任务；`--job` 手动跑不受影响 |
| 临时暂停 | 建 `pause_file` 指的文件；删掉即恢复 |
| 停一个作业 | 该作业 `enabled: false` |
| 不评论只跑 | 该作业 `comment: false` |
| 看状态 | `python tools/autorun/autorun.py --config <站点配置> --status` |
| 手动跑一个作业 | `… --job <名>`；加 `--dry-run` 不评论、不改状态 |

- 每次运行一个目录 `<log_dir>/<作业>/<日期>-<提交>/`：`report.md`（评论正文）、`result.json`（判定、覆盖的提交、评论链接）、`run.log`。
- 摘要一行一条追加在 `<log_dir>/history.jsonl`。
- 状态在 `<state_dir>/<作业>.json`：上次跑完的提交与判定。要让某个提交重跑，删掉这个文件或改掉 `last_done`。

## 站点上要准备的

1. 一份干净的 janus 检出（跑之前快进到 `main`），一份 `maxtext` 克隆。
2. `gh` 已登录、对 `maxtext` 有评论权限；`hy3-t1` 要 kubectl 与 `models/hunyuan3/regression/README.md` 第八节的环境变量（写在作业的 `env` 里）。
3. 同一时间只跑一个实例：轮询器自己用 `<state_dir>/autorun.lock` 加锁，手动与定时同时调起时后来的那个跳过。
