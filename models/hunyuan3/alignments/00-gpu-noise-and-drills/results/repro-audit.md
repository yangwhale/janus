# 复现审计：只照文档从零复现第一阶段

> **历史记录**：本文件记的是 2026-11-04 审计时的文档状态与当时的处理。
> - 第三节「怎么改」与第四节「处理」两栏里标「已修」的，都已合入。
> - 唯一未改的是第四节 g 条（`phase1-hooks.yaml` 头注释），原因见那一行。

- **做法**：以新接手的工程师身份，只读本仓库的文档（`README.md`、`docs/`、`models/hunyuan3/`）。
  - 在 GPU Pod 上另起一个干净目录，复现第一阶段的代表性结果；
  - 记下文档每一处缺失、含糊、错误、与实际不符的地方；
  - 再把文档改好。
- **文档版本**：`main` 的 `45fb8de`（审计开始时）。
  - janus 以 git bundle 克隆进 Pod 的 `/tmp/repro/janus`。
- **环境**：第一阶段同一个 Pod（单机 8 卡 B200，`nvcr.io/nvidia/nemo:26.10`）。
  - 运行写到 `/tmp/repro/runs`，没碰第一阶段的运行目录。
- **权重与数据**：只读引用 Pod 上已准备好的 `/data/hy3-4l`、`/data/hy3-4l-mcore`、`/data/ds`。
  - 准备它们的每一步都按文档核对了命令；
  - 能在小规模上跑的都跑了（见第二节）。
- **时间**：2026-11-04。全部 GPU 命令经 `flock /data/gpu.lock` 排队。

## 一、结论

### 复现成功

关键数字与第一阶段逐位一致。

代号：`d-a` 第一阶段确定性基线运行；`rd-golden-1` 第一阶段 1000 步噪声底里的确定性运行；`det-a` / `det-b` 本次复现的两次确定性 30 步运行；`f-weight2` / `f-w2` 第 2 层权重 ×(1+1e-2) 的埋错（前者第一阶段、后者本次复现）；`f-lr` 学习率差 1 个 FP32 ULP 的埋错。

| 项 | 文档 / 第一阶段 | 复现 | 一致 |
|---|---|---|---|
| 第 1 步 lm_loss（确定性） | `d-a`、`rd-golden-1`：11.4657564163208 | `det-a`、`det-b`：11.4657564163208 | 逐位 |
| 30 步 lm_loss、grad_norm 序列 | `d-a` 前 30 步 | `det-a` 30 步 | 逐步逐位（见注 1） |
| 确定性自证（两次 30 步，第 1 步全量抓数） | `phase1-determinism`：5 pass，张量全部逐位 | 4 pass，425 个张量全部逐位（见注 2） | 是 |
| 故障目录，被测是干净复跑 `det-b`（对照） | — | 32 / 32 pass | — |
| 故障目录，`weight_scale 1e-2`（`f-weight2`） | 25 / 32 报警 | `f-w2`：25 / 32，32 条逐项状态与第一阶段 `f-weight2` 相同 | 是 |
| 故障目录，`lr_ulp`（`f-lr`） | 7 / 32 报警 | `f-lr`：7 / 32，逐项相同 | 是 |
| `janus run --html` | 出 `report.html` | 确定性与两个故障目录都出了 HTML（带图） | 是 |

1. 跨时间，且 `det-a` 多设了三个环境变量。
2. 去掉第 1000 步那条，所以是 4 pass；`L3.step1.bitwise` 425 个张量全部逐位，退出码 0。

结果文件 `repro-audit.{determinism,faults-det-b,faults-f-w2,faults-f-lr}.results.json` 已从仓库删除：

- 它们用的是审计当时的目录副本，与现在的目录文件对不上；
- 上表是它们的全部读数，原件在 git 历史里。

### 但只照改之前的文档走不通

23 处问题里，有 3 处会直接卡住或得出错误理解：

1. **「两种设置」写错了**（`docs/04-environment.md`）。
   - 文档说两种设置都设三个确定性环境变量、默认设置「只有这三个环境变量」。
   - 查全部 51 次运行的 `meta.json`：
     - 默认设置的 18 次三个变量**都没设**；
     - 确定性设置的 33 次里有 27 次（`d-a` 确定性基线、`f-*` 故障注入、`p-*` 只换并行方式，等等）也**没设**，只开了 `deterministic_mode`。
   - 照文档用 `run.sh` 起的「默认设置」与第一阶段量噪声用的不是同一种设置。
2. **`run.sh` 依赖一套文档没写的目录布局**。
   - 它 `cd /data/jw`、`source /data/env.sh`；`env.sh` 又把 `PYTHONPATH` 指向 `/data/janus-src`。
   - 文档从没说要把仓库放到这些位置，克隆仓库后直接跑 `run.sh` 第一步就失败。
3. **没有任何一次运行的完整命令**。
   - `d-a`、`f-*` 等用了哪些参数（`--full`、`--dump-steps`、故障的参数正则），文档里没有，只能读脚本猜；
   - 故障注入的写法「种类见脚本」，`weight_scale` 的参数名正则要自己拼。

## 二、逐步复现记录

| 步 | 做法 | 结果 |
|---|---|---|
| 下权重 | 按文档的规则从 `model.safetensors.index.json` 算分片 | 10 个分片，与 Pod 上 `/data/hy3-full` 一致（见注 1） |
| 截层 | `truncate_hy3.py /data/hy3-full <新目录> 1`（截 1 层，CPU 5 秒） | 14 个张量 sha256 全部相同；**不先 `source env.sh` 时 `import torch` 失败**（见注 2） |
| 转 Megatron | `import_hy3.py` 转上面那份 1 层目录（一张卡） | 正常，`IMPORT_DONE`；4 层原件的日志写的参数数 12,386,079,744 与文档一致 |
| 数据 | `head -200 fw.jsonl` 后跑 `prep_data.sh` 里的 `preprocess_data.py` 命令 | 200 条 token 序列逐位相同；`fw.parquet` sha256 一致（见注 3） |
| 训练 | `run.sh` 的副本（只改路径），`det-a`、`det-b` 确定性 30 步 `--dump-steps 1 --full` | 每次约 3 分钟，运行目录各 41 GB |
| 故障注入 | `f-w2`（`weight_scale … 1e-2`）、`f-lr`（`lr_ulp`），参数同上 | 第一次 `--side` 写错，全部 error；改成 `golden-<n>` 重跑后正常（见注 4） |
| 检查 | 复制 `phase1-determinism.yaml` / `phase1-faults.yaml` 改 `runs.root`、golden、窗口 | 见第一节 |

1. HF 当前 revision `a960ebc…` 的 index、config、tokenizer 与 Pod 上的逐字节相同。
2. 截层的核对：
   - 14 个张量的 sha256 与 `/data/hy3-4l/tensor_sha256.json` 中同名项全部相同；
   - 配置只差 `num_hidden_layers`；
   - 不 source 时的报错：`undefined symbol: ncclCommResume`。
3. 200 条 token 序列与 `/data/ds/fw_hy3_text_document` 前 200 条逐位相同；`fw.parquet` sha256 与 Pod 上记录的一致。
4. 第一次把 `--side` 写成 `test`：
   - 训练跑完，但 `janus run` 全部 error（`side 不合法: test`）；
   - `--html` 还直接抛异常退出。

## 三、审计表

「改」一列写本 PR 怎么处理的。标「代码」的是代码问题，本 PR 只记录不改，汇总在第四节。

| # | 文档位置 | 问题 | 实际情况 | 怎么改 |
|---|---|---|---|---|
| 1 | `docs/04-environment.md`「两种设置」 | **错误**：两种设置的环境变量写错，并有一句没依据的结论（见注 1） | 默认设置三个变量都没设；确定性设置大多也没设（见注 1） | 重写表格（见注 1） |
| 2 | `gpu/README.md` 步骤表、`run.sh` | **缺失**：没说 `run.sh` 依赖的路径、会删哪些目录（见注 2） | 克隆仓库后直接跑 `run.sh` 训练不会启动，但仍打印 `RUN_DONE`（见注 2） | 新增第 0 节「路径是写死的」布局表；第 1 节给出放置命令与改路径的 `sed`（代码：路径写死） |
| 3 | `gpu/README.md`、`results/README.md` | **缺失**：没有任何一次第一阶段运行的完整命令；故障参数「见脚本」 | 实际命令在 POC Pod 的私有脚本里 | 新增「第一阶段每次运行的参数」表（见注 3） |
| 4 | `gpu/README.md` | **缺失**：没有核对用的锚点，复现者不知道自己做对了没有 | — | 每一步加锚点（见注 4） |
| 5 | `gpu/README.md`「下权重」 | **缺失**：只有一句「按 index 选分片，curl 下载」，没有命令；没说要先下 index / config / tokenizer；没钉 HF revision | 第一阶段的 index 与当前 revision `a960ebc3…` 逐字节相同 | 给可执行脚本（钉 revision）与 index 的 sha256 |
| 6 | `gpu/README.md` 截层 / 转换 / 数据 | **缺失**：没说要先 `source env.sh` | 不 source 时 `import torch` 报 `undefined symbol: ncclCommResume` | 第 0 节写明，并在每条命令前加上 |
| 7 | `gpu/README.md`「数据」、`prep_data.sh` | **缺失**：脚本不建目录、依赖分词器、parquet 没钉版本（见注 5） | — | 命令前加 `mkdir -p /data/ds`，写明依赖第 3 步，给 sha256 与文档数做核对（代码：脚本不建目录、无 `set -e`） |
| 8 | `gpu/README.md`「转 Megatron」 | **含糊**：「4 层 12,386,079,744 参数」与 HF 张量元素数对不上 | 差 576 = 3 个 MoE 层 × 192 个 `expert_bias`（见注 6） | 写明差值来源 |
| 9 | `README.md`「跑一份检查项目录」、`catalogs/` | **含糊**：示例目录依赖 POC Pod 上的运行，没说怎么换成自己的（见注 7） | 复现要复制目录改 `runs.root`、golden，确定性目录还要把 1000 步改成实际步数 | `README.md` 加说明；`gpu/README.md` 第 7 节给出改目录的 `sed`（已验证与复现所用目录一致） |
| 10 | `README.md`、`gpu/README.md` 的 `--side golden-<n>` | **缺失**：没说 `side` 是受约束的 | 只接受四种前缀；写错了训练照跑，`janus run` 全部 error（见注 8） | 两处写明（代码：训练启动时不校验；`--html` 在全部 error 时抛异常） |
| 11 | `gpu/README.md` | **缺失**：磁盘体积、`/data` 是 `emptyDir` | 一次 30 步 `--full` 运行 41 GB；删 Pod 数据全丢 | 第 0 节写明 |
| 12 | `gpu/README.md`「起 Pod」、`pod.yaml` | **缺失**：`nodeSelector` 是第一阶段集群的节点池名；私有仓库怎么进 Pod 没写 | Pod 里没有 GitHub 凭据 | 第 1 节写明改节点池名，给 `git bundle` + `kubectl cp` 的做法 |
| 13 | `docs/05-dump.md`、`results/README.md`、`coverage.md` 的「只挂统计」 | **错误**：说 `h-light2`（轻量抓数运行）只挂逐步统计、不抓张量 | 第 1 步照样抓前向张量（见注 9） | 改为「轻量抓数（第 1 步只抓前向张量，不挂梯度钩子、不包优化器）」 |
| 14 | `results/phase1-faults.md` | **含糊**：`f-gscale`（梯度在抓数之前 ×1/8 的埋错）那列有 E、`L0.data.fingerprint` 也是 F，没解释 | `f-gscale` 只跑 3 步，窗口 1–30：`L4.*` 缺步报 error，`data_fingerprint` 把缺的记录报成「token 不同」 | 表前加「读表时注意」（代码：缺步应报 error 而不是 fail） |
| 15 | `results/phase1-faults.md` | **含糊**：`L0.data.fingerprint` 在 `f-weight2` / `f-router` / `f-drop` 上报警，读者会以为数据错了 | 这三个报警是因为该检查也比首步 loss，token 本身没变 | 同上 |
| 16 | `docs/04-environment.md`「每次运行都要记下的」 | **不一致**：要求记镜像 digest 与 Megatron 版本 | `meta.json` 的 `versions.megatron_core` / `megatron_bridge` 是 `null`，`image` 是 tag 不是 digest | 写明这是第一阶段的缺口、版本以镜像为准（见注 10） |
| 17 | `README.md` `--render-only` | **不一致**：说按 `results.json` 重画报告 | 不论给不给 `--html`，`report.md` 与 `report.html` 都重画 | 写明 |
| 18 | `gpu/README.md` | **缺失**：`--janus-run` 目录重名不会清理 | 新旧文件混在一起，旧的 `DONE` 还在 | 写明每次用新目录 |
| 19 | `gpu/README.md`、`run.sh` | **缺失**：`RUN_DONE` 不论成败都打印 | 要看各 rank 日志和运行目录里的 `DONE` 判断成败 | 写明（代码：不带退出码） |
| 20 | `gpu/README.md`「两种设置」 | **缺失**：用 `run.sh` 起不出第一阶段的默认设置 | `run.sh` 无条件 `export` 三个变量 | 写明怎么起（代码：没有开关） |
| 21 | `gpu/README.md`「转 Megatron」 | **缺失**：没说要不要 GPU、多久 | 单进程一张卡，几分钟 | 写明，命令带 `CUDA_VISIBLE_DEVICES=0` |
| 22 | `gpu/README.md`「L1 单算子回放」 | **含糊**：命令写死 `/data/janus-runs2/d-a`，没说这个运行是怎么来的 | 要一次带 `--full` 的确定性运行 | 改为占位符并说明要求 |
| 23 | `results/README.md` 环境段 | **缺失**：运行目录在 POC Pod（不对外），也没指向命令 | — | 指向运行参数表、复现步骤与本文件 |

1. **第 1 条「两种设置」**：
   - 文档原文：两种设置都设三个环境变量，默认设置 = 只有三个环境变量；还有一句「只有这三个还不够：默认设置第 2 步起照样不同」。
   - 实际：
     - 默认设置 18 次运行三个变量都没设（脚本显式 `unset`）；
     - 确定性设置只有 `rd-golden-1`、`rd-det-b`、`dump-a/b`、`gd-a/b` 设了，其余 27 次没设；
     - 「只设三个变量、不开 `deterministic_mode`」从没跑过，那句结论没有依据。
   - 怎么改：
     - 重写表格：默认设置 = 什么都不设；
     - 列出每批实际设了什么；
     - 补「确定性下三个变量不改结果」的复现证据；
     - 删掉没依据的那句，改为「未确认」。
2. **第 2 条 `run.sh` 的路径**：
   - 没说 `run.sh` 依赖 `/data/jw`、`/data/env.sh`、`/data/janus-src`、`/data/hy3-4l`；
   - 也没说 `run.sh` 每次 `rm -rf /data/runs/<名>`、`/data/tlogs/<名>`。
   - 实际情况：`source /data/env.sh` 与 `cd /data/jw` 都失败（按脚本推断，未实跑）。
3. **第 3 条运行参数表**：25 行覆盖结果一览里的运行（`g1k-*`（第一阶段 GPU golden）、`i-nd`、`t-full`（这两个都是第一阶段默认设置的 GPU 附加运行）等附加运行未列），含故障正则、`r10`（只含 `r-cont` 第 10 步存档的目录）的做法。
4. **第 4 条锚点**：分片清单、`tensor_sha256.json` 的 sha256、参数数、parquet / idx 的 sha256、文档数、首步 loss 与 grad_norm、各检查期望的 pass / fail 数。
5. **第 7 条 `prep_data.sh`**：
   - 脚本 `cd /data/ds` 但不建目录（失败后在当前目录继续跑）；
   - 依赖第 3 步的分词器；
   - parquet 按 `main` 下载、没钉版本。
6. **第 8 条参数数**：HF 侧 1781 个张量共 12,386,080,320 个元素；Megatron 把 `expert_bias` 当 buffer。
7. **第 9 条示例目录**：
   - 示例 `phase1-full.yaml` 依赖 POC Pod 上的 `s-a`..`s-d` 等运行，没有它们时全部 error；
   - 没说怎么换成自己的运行（`--test` 只换被测）。
8. **第 10 条 `side` 前缀**：只接受 `golden-` / `tpu` / `tpu-perturb-` / `gpu-perturb-`。
9. **第 13 条 `h-light2`**：
   - `--dump-steps` 取默认 `1`，第 1 步照样抓前向张量（运行目录 3.8 GB）；
   - 它少的是梯度钩子和优化器包装。
10. **第 16 条版本**：复现在容器里读到 0.20.0 / 0.7.0（代码：取版本的写法）。

### 核对后没有问题的

- 10 个分片的说法；
- `truncate_hy3.py` / `import_hy3.py` / `prep_data.sh` 与 Pod 上实际用过的脚本逐字节相同；
- 文档写的镜像版本（torch、CUDA、Megatron-Core 0.20.0、Megatron-Bridge 0.7.0、TE 2.20.2）与容器里读到的一致。

## 四、发现的代码问题

| # | 位置 | 问题 | 影响 | 处理 |
|---|---|---|---|---|
| a | `models/hunyuan3/frameworks/megatron/run.sh`、`env.sh`、`prep_data.sh` | 路径写死；`run.sh` 起不出默认设置；`prep_data.sh` 不建目录（见注 a） | 换位置要复制改脚本；默认设置要另写脚本 | 已修（见注 a） |
| b | `janus/run.py` + `janus/report_html.py` | 被测 `side` 不合法时 `--html` 抛 `StoreError`，退出码 1（按约定应为 2）（见注 b） | 退出码误导 CI | 已修（见注 b） |
| c | `models/hunyuan3/frameworks/megatron/train_hy3.py` | 启动时不校验 `--side`；Megatron 两个版本恒为 `null`；`image` 记的是 tag（见注 c） | 浪费 GPU 时间；`meta.json` 缺版本 | 已修（见注 c） |
| d | `janus/checks/` 的 `data_fingerprint` | 被测缺整步记录时报 fail「token 不同」；建议同样报 error（见注 d） | 步数不够被当成检出（`f-gscale`，只跑 3 步的埋错运行） | 已修（见注 d） |
| e | `janus/run.py` | `--render-only` 不看 `--html`，总是重写 `report.html` | 轻微 | 已修：`--render-only` 只重画 `report.md`，加 `--html` 才重画 `report.html` |
| f | `models/hunyuan3/frameworks/megatron/run.sh` | `RUN_DONE` 不带退出码，失败也打印 | 要另查日志 | 已修：打印 `RUN_DONE <运行名> rc=<退出码>`，并以 torchrun 的退出码退出 |
| g | `models/hunyuan3/alignments/00-b200x8-self-4layers/catalogs/phase1-hooks.yaml` 注释 | 「只挂逐步统计（不抓张量）」与 `h-light2`（轻量抓数运行）实际不符 | 改注释会改变目录 sha256，留给下一次重跑时一起改 | 未改（见注 g） |

- **注 a**（路径写死与环境变量）：
  - 问题：
    - 路径写死（`/data/jw`、`/data/env.sh`、`/data/janus-src`、`/data/hy3-4l`、`/data/runs`）；
    - `run.sh` 无条件设三个环境变量，起不出第一阶段的默认设置；
    - `prep_data.sh` 不建 `/data/ds`、没有 `set -e`。
  - 处理：
    - 路径改成环境变量（`JANUS_ENV`、`JANUS_SRC`、`JANUS_WORK`、`JANUS_HF`、`JANUS_RUNS`、`JANUS_TLOGS`、`JANUS_DS`、`JANUS_TRITON_CACHE`、`MEGATRON_LM`），默认值仍是原路径；
    - `run.sh` 有 `--deterministic` 才设三个变量、没有就 `unset`（`JANUS_DET_ENV=1/0` 强制）；
    - `prep_data.sh` 加 `set -eo pipefail`、自己建数据目录。
- **注 b**（`side` 不合法时的退出码）：
  - 问题：每条检查 error 之后，`--html` 渲染再次打开运行目录抛 `StoreError`，进程以 traceback 退出。
  - 处理：
    - side 不合法的提示列出合法前缀；
    - `janus run` 开跑前在 stderr 报哪个运行目录不合规；
    - `--html` 遇到读不出的目录照出报告、不画图，退出码 2。
- **注 c**（`train_hy3.py`）：
  - 问题：
    - 启动时不校验 `--side`，跑完才在 `janus run` 时发现；
    - `ver("megatron.core")` 用 `__import__` 拿到的是顶层 `megatron` 包，Megatron 两个版本恒为 `null`；
    - `image` 记的是 tag 不是 digest。
  - 处理：
    - 给了 `--janus-run` 时启动即校验 `--side`（退出码 2，不占 GPU）；
    - 版本改用 `importlib.import_module` 取子包 `__version__`、退回分发包元数据（实测 0.20.0 / 0.7.0）；
    - `image` 取 `JANUS_IMAGE`，`pod.yaml` 钉住第一阶段的 digest 并设好这个变量，没带 digest 时打印警告。
- **注 d**（`data_fingerprint` 缺整步）：
  - 问题：规格只规定缺字段报 error，缺整步没有明说。
  - 处理：
    - 一侧整步没有记录报 error，消息列出缺的步（`missing_steps`）；
    - 同一步只缺部分单元仍报 fail；
    - 规格同步。
- **注 g**（未改的原因）：
  - 修复按本表的影响栏处理：改注释会让 `phase1-hooks.results.json` 里记的目录 sha256 对不上；
  - 留到下次重跑 `phase1-hooks` 时一起改；
  - `h-light2` 的实际做法见 `gpu/README.md` 运行参数表。
