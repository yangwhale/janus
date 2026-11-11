[janus](../../../../../README.md) › [混元 4](../../../README.md) › [mini](../README.md) ｜ GPU 一侧自比与演练；怎么读见指南[第 2 章 制作 GPU golden](../../../../../docs/guide/02-gpu-golden.md)

# 混元 4 · mini · GPU 一侧自比：噪声、确定性、故障演练（T1）

**这一部分干什么**：hy4-mini 的 GPU 和 GPU 自己比。语义以 TPU 一侧 MaxText 为准（[`PROTOCOL.md`](../../../PROTOCOL.md) 第 1.2 节），判定时 GPU（Megatron-Core，B200 × 8）是 golden、TPU 是被测（`PROTOCOL.md` 变更记录）。这里回答三件事：GPU 同设置重跑本来就差多少（这就是 T1 L4 的噪声尺子，`PROTOCOL.md` 第 3 节）；确定性设置下能不能逐位复现；埋了错，janus 的检查能不能抓到。做法照[混元 3 的 00](../../../../hunyuan3/alignments/00-gpu-noise-and-drills/README.md)。

**现状**：只有噪声参照的原始运行。以现行比较 tz-off-rep 为准：前 300 步 `t1zo300-l4` 的噪声上限 N 是 `g2-*` 8 份两两 28 对，1000 步 `t1zo-l4` 的是 `g3-a … g3-p` 16 份两两 120 对（`g3-i … g3-p` 是与 `g3-a … g3-h` 同一代码包、同一参数的再 8 份重跑，[`catalogs/g3x16-noise.yaml`](catalogs/g3x16-noise.yaml)）。2026-11-11 以前的 1000 步比较用 `g3-a … g3-h` 28 对，最早三代用 `g1-*` 28 对，各代见[历代比较](../../../dashboard/history.md)。第 1 步参照 `g1s-*` 3 份，`g2s-*` 6 份（`g2s-d … f` 是 2026-11-11 同设置补跑的，[`results/g2s6-meta-vs-g2sa.json`](results/g2s6-meta-vs-g2sa.json)）。各份运行的载入自检与第 1 步相对极差在 [`gpu-load-check-and-step1-spread.json`](results/gpu-load-check-and-step1-spread.json)。

| 1000 步，基线 `g3-a` | 8 份（28 对）的上限 | 16 份（120 对）的上限 |
|---|---|---|
| 主 loss（第 1–1000 步）：带符号均值 / MARE / 最大单步 | 3.78e-6 / 8.24e-6 / 5.62e-5 | 4.65e-6 / 8.95e-6 / 7.44e-5 |
| 其余 8 项 | 见 [对比](results/g3x16-noise-vs-8.json) | 放大 1.0–2.1 倍（最大的是路由偏置均值的带符号均值） |

16 份的设置逐项与 `g3-a` 相同（`meta.json` 的配置、代码指纹、镜像、并行、版本、数据，[核对](results/g3x16-noise-vs-8.json) 的 `settings_diff`）。GPU 自比与故障演练还没做；确定性模式下同设置重跑逐位在 05 做过（hy4-mini 20 步，[`ckpt-rerun-dp8.json`](../../05-gpu-scale/results/ckpt-rerun-dp8.json)），本实践的确定性两份 300 步还没跑。下面的逐级大表由结果文件生成，现在全是「进行中」或「未开始」。

每一项检查在各列上的状态总览见 [混元 4 总览](../../../dashboard/dashboard.md)；设置见 [`setup.md`](setup.md)，运行见 [`runs.yaml`](runs.yaml)，检查项目录在 [`catalogs/`](catalogs/)。

<!-- janus:table:begin（以下由脚本生成，不要手改） -->

## 逐级大表

### 来源

- 由 `models/<模型>/dashboard/build.py` 调 `janus/report/tables.py` 生成。
- 数据来自本文件夹 `results/` 下的 `results.json`，数字不手写。

### 两档门槛

- 含义见 `docs/reference/judging.md`「两档门槛」。
- **严格档**是验收判定：
  - calibrated 检查对噪声上限 N 判：≤ N 过，> 2N 不过，中间待定；
  - 参照不足 3 对最多待定；
  - 逐位类必须逐位相同；
  - 数值门槛（如 FP64 R 比）照目录里写的线。
  - 符号：✅ 过 ｜ 🟡 待定 ｜ ❌ 不过或出错 ｜ ⬜ 还没跑 ｜ — 不适用。
- **参考档**是同类 GPU→TPU 迁移项目的实践门槛：
  - 只并列报告、不决定验收。
  - 符号：✅ 过 ｜ ☑️ 记录了的都过，但有门槛项没记录 ｜ ❌ 不过 ｜ ❔ 所需指标未记录 ｜ — 无对应门槛。

### 说明栏怎么读

- 格式一律是「实测（指标）｜严格档 门槛 → 离门槛多远，判定｜参考档 门槛：实测（实测 / 门槛的倍数），判定」。
- calibrated 检查先写主判指标 rel_L2：
  - 没有 rel_L2 就写最差的指标；
  - 最差的不是它时，补一句最差的那个。
- L4 逐步量三个指标各占一行：
  - 各自按同一规则判；
  - 整条检查取最差，见报告。
- 首步差：严格档只报不判，参考档要判（MARE＝平均绝对相对差）。
- 进行中的项在证据栏写证据位置。
- 严格档里「N=…（k 对）」的 k 是算 N 用的噪声参照对数：
  - 一对是两份运行的一次比较，k 对不是 k 份运行；
  - 用了哪几份运行、怎么配对见该检查项目录的 `runs.noise`。

### 逐行结果

- 共 9 行。
- 严格档：✅ 1　🟡 4　⬜ 4
- 参考档（有对应门槛的 0 行）：无

| 级 | 检查项 | 严格档 | 参考档 | 说明 | 证据 |
|---|---|---|---|---|---|
| L0 | GPU 载入自检（每个参数等于检查点张量） | 🟡 | — | 进行中 | hy4-mini 的 g1、g2、g3 共 32 份噪声参照运行都通过（gpu-noise/results/gpu-load-check-and-step1-spread.json，读自各运行 meta.json 的 load_check）；还没写成 janus 检查 |
| L1 | 逐元素函数（FP32；mini 列是 TPU 现行开关写法，gpu 列是 B200 实测的 log；按 CUDA 文档门槛：exp ≤ 2、log ≤ 1、sigmoid / silu ≤ 4、sqrt / 除法 ≤ 1、rsqrt ≤ 2 ulp）<br>`L1.elem.log.gpu_b200` | ✅ | — | 实测 1（ulp_max）｜严格档 ≤ 1 → 1×，过｜参考档 无对应门槛 | [t1-log-ulp](../results/t1-log-ulp.results.json) |
| L1 | GPU 算子对 FP64、逐元素函数、极端输入 | ⬜ | — | 还没跑 | 待做（gpu-noise README「计划（待做）」第 6 项） |
| L3 | GPU 第 1 步参照的逐张量散布 | 🟡 | — | 进行中 | g1s 中位数 rel_L2 2.7e-3（台账 G15）；g2s 6 份两两 15 对的逐张量散布记在 01 第 1 步比较的结果文件里（gpu_noise_max；两两 rel_L2 中位数 2.48e-3，新旧两批之间没有批次差，gpu-noise/results/g2s6-meta-vs-g2sa.json），没有单独的 GPU 自比检查 |
| L3 | 只换执行方式（专家并行、梯度累积切分、重算） | ⬜ | — | 还没跑 | 待做（gpu-noise README「计划（待做）」第 4 项） |
| L4 | GPU 自比：同设置多份两两的散布（噪声尺子） | 🟡 | — | 进行中 | 现行比较的噪声尺子：前 300 步 t1zc300-l4 用 g2 8 份两两 28 对，1000 步 t1zc-l4 用 g3 16 份两两 120 对（上一代 t1zo 同一把）；2026-11-11 前的 1000 步比较用 g3 前 8 份 28 对，最早三代用 g1 8 份（各代见历代比较）；GPU 自比（catalogs/t1-gpu-noise.yaml）待做 |
| L4 | 抓数不改数值（开 / 不开全量抓数） | ⬜ | — | 还没跑 | 待做：可在确定性模式（`--deterministic`）下逐位核，05 已证明这个模式下同设置重跑逐位（gpu-noise README「计划（待做）」第 3 项） |
| L4 | 故障演练（确定性基线上埋错） | ⬜ | — | 还没跑 | 待做；确定性基线已有（05-gpu-scale/results/ckpt-rerun-dp8.json，确定性模式两次 20 步逐位），演练本身没做（gpu-noise README「计划（待做）」第 5 项） |
| L4 | 同设置重跑逐位（mini 列是 TPU 一侧；gpu 列是 GPU 一侧） | 🟡 | — | 进行中 | 确定性模式（`--deterministic`）下逐位：hy4-mini 单机 DP 8 + EP 4 两次 20 步，1620 个记录量、8 个 rank 的 1968 个参数 sha 都相同（05-gpu-scale/results/ckpt-rerun-dp8.json；2 台 PP 2 同样逐位，ckpt-rerun-pp2.json），还没写成 janus 检查。golden（g3）没开确定性模式，8 份之间不逐位（第 1 步相对极差 loss 2.9e-6、grad_norm 8.2e-5，gpu-noise/results/gpu-load-check-and-step1-spread.json） |

<!-- janus:table:end -->

## 这一阶段做了什么

### 已有的证据

| 项 | 现状 | 出处 |
|---|---|---|
| GPU 同设置 8 份的散布 | 每一代 8 份两两 28 对，每一对算一次判定量，每个判定量取各对里的最大值，就是对应 L4 比较的噪声上限 N：`g1-*`（前三代）、`g2-*`（300 步）8 份 28 对；`g3-*`（1000 步）16 份 120 对（`g3-a … g3-p`，2026-11-11 起；之前 `g3-a … g3-h` 8 份 28 对） | 01 [`../catalogs/t1t-l4.yaml`](../catalogs/t1t-l4.yaml)、[`../catalogs/t1u-l4.yaml`](../catalogs/t1u-l4.yaml)、[`catalogs/g3x16-noise.yaml`](catalogs/g3x16-noise.yaml)；各结果文件里每条检查的 `threshold.noise.upper` |
| 注意力 q·k 改 FP32 前后的散布 | `g2-*` 两两最大相对差 lm_loss 4.5e-5、grad_norm 2.2e-3；`g1-*` 3.7e-5、2.3e-3，几乎不变 | 台账 G17 |
| GPU 第 1 步参照的逐张量散布 | `g1s-*`：中位数 rel_L2 2.7e-3；第 0、1 层索引器 10 个张量三份逐位相同（`g2s-*` 6 份同样）；`g2s-*` 6 份两两 15 对：中位数 2.48e-3，批内（a–c、d–f）与跨批相同，没有批次差（[`results/g2s6-meta-vs-g2sa.json`](results/g2s6-meta-vs-g2sa.json)） | 台账 G15；01 [`../results/t1-step1-grad-vs-ref.json`](../results/t1-step1-grad-vs-ref.json)、[`../results/t1t-step1-grad-vs-ref.json`](../results/t1t-step1-grad-vs-ref.json) 的 `gpu_noise_max` |
| 默认设置下能不能逐位复现 | hy4-mini 8 卡不能：g1 / g2 / g3 各 8 份第 1 步的相对极差 loss 2.1e-6 / 1.7e-6 / 2.9e-6、grad_norm 7.3e-5 / 9.3e-5 / 8.2e-5（台账 G9 的 4.9e-6、2.4e-4 是 T1 之前关 eod 掩码的测试）；hy4-tiny B200 × 2、20 步两份逐位相同。确定性模式（`--deterministic`）下 hy4-mini 也逐位：单机 DP 8 + EP 4 两次 20 步，1620 个记录量、1968 个参数 sha 都相同 | hy4-mini：[`gpu-load-check-and-step1-spread.json`](results/gpu-load-check-and-step1-spread.json)；hy4-tiny：台账 G7（没有结果文件）；确定性模式：[`ckpt-rerun-dp8.json`](../../05-gpu-scale/results/ckpt-rerun-dp8.json)（05「存档续跑」） |
| 载入自检 | 第 1 步前每个参数等于检查点张量、FP64 平方和相等，否则退出；hy4-mini 的 `g1-*`、`g2-*`、`g3-*` 共 32 份都通过（rank 0 核 246 个参数，全部 rank 0 处不同） | 台账 G7；[`gpu-load-check-and-step1-spread.json`](results/gpu-load-check-and-step1-spread.json) |
| per-token loss 开关不改数值 | eod 关（掩码全 1）时 per-token 开 / 关第 1 步 loss 差 1.3e-6、grad_norm 差 8.5e-5，在同设置重跑的差之内 | 台账 G9 |
| 对照：TPU 重跑 | 以现行比较 tz-off-rep 为准：`tz-off-qkc` 对 `tz-off-rep` 1000 步 47 个逐步量逐位相同；旧配置两组各两份 100 步、45 个逐步量也逐位相同；只作记录，不进尺子 | 01 [`../results/t1zo-determinism.report.md`](../results/t1zo-determinism.report.md)、[`../results/hy4-mini-tpu-determinism.json`](../results/hy4-mini-tpu-determinism.json) |
| 16 份尺子重判（知识库旁注） | 1000 步尺子换成 `g3-a … g3-p` 120 对后，几份旧的 TPU 运行重判：base2 开的 `t1zg-a`、`t1zw-a` 9 项全过；base2 关的 `t1zz-a`、`tz-off-nolhs` 与 base2 开的只改执行方式对照 `ts-c-nolhs-1k` 都是过 7、待定 2（主 loss 全程、后段）。只作旁注，不改各自原来的判档 | [`results/t1zga-l4-n16`](results/t1zga-l4-n16.report.md)、[`t1zwa-l4-n16`](results/t1zwa-l4-n16.report.md)、[`t1zza-l4-n16`](results/t1zza-l4-n16.report.md)、[`tzoffnolhs-l4-n16`](results/tzoffnolhs-l4-n16.report.md)、[`tsnolhs-l4-n16`](results/tsnolhs-l4-n16.report.md) |

### 计划（待做）

| # | 做什么 | 判据 | 目录 / 运行 | 状态 |
|---|---|---|---|---|
| 1 | GPU 自比：`g2-b` 对 `g2-a`（`g3` 同理），噪声同一代：`g2` 8 份两两 28 对，`g3` 16 份两两 120 对 | 每个判定量 r ≤ 1；给出这把尺子在 GPU 一侧自比时的读数 | [`catalogs/t1-gpu-noise.yaml`](catalogs/t1-gpu-noise.yaml)；运行已有 | 待做 |
| 2 | 确定性设置下同配置两份 300 步 | 逐步量与第 1 步全部抓数逐位相同 | [`catalogs/t1-gpu-determinism.yaml`](catalogs/t1-gpu-determinism.yaml)；`g1det-a/b`（计划） | 300 步待做；20 步已在 05 做过（确定性模式两次 20 步逐位，`ckpt-rerun-dp8.json`） |
| 3 | 抓数不改数值（开 / 不开全量抓数，逐步量逐位） | 逐位相同 | 待定 | 待做 |
| 4 | 只换执行方式：专家并行 4 ↔ 8、梯度累积切分、重算开关 | 前向逐位；反向与更新后参数对噪声 | 待定 | 待做 |
| 5 | 故障演练：在确定性基线上埋错（主权重扰动、数据错位、路由改 BF16、学习率差 1 ULP、开 dropout、梯度缩放、路由偏置不更新等），看期望报警的检查有没有报 | 期望报警的都报了 | 待定；可用 05 的确定性模式作基线，但 05 那份只有 20 步（`ckpt-rerun-dp8.json`），golden `g3` 不开确定性模式，演练要先在确定性模式下跑够 300 / 1000 步 | 待做 |
| 6 | GPU 算子对 FP64、逐元素函数的 ULP、极端输入 | GPU 侧检查 | 待定 | 待做 |

### 为什么要做

- T1 的 L4 与 L3 只用 GPU 重跑的散布当尺子；它有多宽、来自哪里（原子加的顺序，还是只换执行方式也会变），直接决定判定。第 1 步比较里 GPU 三份的散布只来自原子加的顺序、不含舍入位置（`t1-step1-indexer-grad.json` 的结论），所以这把尺子对舍入位置的差偏窄。
- 故障演练和「抓数不改数值」都要在确定性基线上做：确定性模式下同设置重跑逐位（05「存档续跑」），可以作基线；golden 本身没开确定性模式。
