[janus](../../../../../README.md) › [混元 4](../../../README.md) › [mini](../README.md) ｜ GPU 一侧自比与演练；怎么读见指南[第 2 章 制作 GPU golden](../../../../../docs/guide/02-gpu-golden.md)

# 混元 4 · mini · GPU 一侧自比：噪声、确定性、故障演练（T1）

**这一部分干什么**：hy4-mini 的 GPU 和 GPU 自己比。语义以 TPU 一侧 MaxText 为准（[`PROTOCOL.md`](../../../PROTOCOL.md) 第 1.2 节），判定时 GPU（Megatron-Core，B200 × 8）是 golden、TPU 是被测（`PROTOCOL.md` 变更记录）。这里回答三件事：GPU 同设置重跑本来就差多少（这就是 T1 L4 的噪声尺子，`PROTOCOL.md` 第 3 节）；确定性设置下能不能逐位复现；埋了错，janus 的检查能不能抓到。做法照[混元 3 的 00](../../../../hunyuan3/alignments/00-gpu-noise-and-drills/README.md)。

**现状**：只有噪声参照的原始运行，三代各 8 份：`g1-*`（前三代比较）、`g2-*`（300 步：现行 `t1zga300-l4` 及之前各代的 300 步比较）、`g3-*`（1000 步：现行 `t1zga-l4` 及之前各代的 1000 步比较；各代见[历代比较](../../../dashboard/history.md)），各自两两 28 对就是 01 里对应比较的噪声上限 N；第 1 步参照 `g1s-*`、`g2s-*` 各 3 份。1000 步的尺子另有 16 份版：`g3-i … g3-p` 是同一代码包、同一参数的再 8 份重跑，与 `g3-a … g3-h` 一起两两 120 对（[`catalogs/g3x16-noise.yaml`](catalogs/g3x16-noise.yaml)）。

| 1000 步，基线 `g3-a` | 8 份（28 对）的上限 | 16 份（120 对）的上限 |
|---|---|---|
| 主 loss（第 1–1000 步）：带符号均值 / MARE / 最大单步 | 3.78e-6 / 8.24e-6 / 5.62e-5 | 4.65e-6 / 8.95e-6 / 7.44e-5 |
| 其余 8 项 | 见 [对比](results/g3x16-noise-vs-8.json) | 放大 1.0–2.1 倍（最大的是路由偏置均值的带符号均值） |

16 份的设置逐项与 `g3-a` 相同（`meta.json` 的配置、代码指纹、镜像、并行、版本、数据，[核对](results/g3x16-noise-vs-8.json) 的 `settings_diff`）。GPU 自比、确定性、故障演练还没做。下面的逐级大表由结果文件生成，现在全是「进行中」或「未开始」。

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
| L0 | GPU 载入自检（每个参数等于检查点张量） | 🟡 | — | 进行中 | hy4-mini 通过（台账 G7）；没有结果文件 |
| L1 | 逐元素函数（FP32；mini 列是 TPU 现行开关写法，gpu 列是 B200 实测的 log；按 CUDA 文档门槛：exp ≤ 2、log ≤ 1、sigmoid / silu ≤ 4、sqrt / 除法 ≤ 1、rsqrt ≤ 2 ulp）<br>`L1.elem.log.gpu_b200` | ✅ | — | 实测 1（ulp_max）｜严格档 ≤ 1 → 1×，过｜参考档 无对应门槛 | [t1-log-ulp](../results/t1-log-ulp.results.json) |
| L1 | GPU 算子对 FP64、逐元素函数、极端输入 | ⬜ | — | 还没跑 | 待做（gpu-noise README「计划（待做）」第 6 项） |
| L3 | GPU 第 1 步三份的逐张量散布 | 🟡 | — | 进行中 | g1s 中位数 rel_L2 2.7e-3（台账 G15）；g1s、g2s 的逐张量散布记在 01 第 1 步比较的结果文件里（gpu_noise_max），没有单独的 GPU 自比检查 |
| L3 | 只换执行方式（专家并行、梯度累积切分、重算） | ⬜ | — | 还没跑 | 待做（gpu-noise README「计划（待做）」第 4 项） |
| L4 | GPU 自比：同设置 8 份两两的散布（噪声尺子） | 🟡 | — | 进行中 | g1、g2、g3 各自 8 份两两 28 对就是各代 L4 比较的噪声尺子（g1：t1-l4-nozm / t1s-l4；g2：t1zr300-l4、t1zs300-l4、t1z-l4、t1v-l4、t1t-l4；g3：t1zr-l4、t1zs-l4、t1zk-l4、t1w-l4、t1u-l4 的 N）；GPU 自比（catalogs/t1-gpu-noise.yaml）待做 |
| L4 | 抓数不改数值（开 / 不开全量抓数） | ⬜ | — | 还没跑 | 待做（gpu-noise README「计划（待做）」第 3 项） |
| L4 | 故障演练（确定性基线上埋错） | ⬜ | — | 还没跑 | 待做，要先有确定性基线（gpu-noise README「计划（待做）」第 5 项） |
| L4 | 同设置重跑逐位（mini 列是 TPU 一侧；gpu 列是 GPU 一侧） | 🟡 | — | 进行中 | hy4-mini 8 卡同设置重跑不逐位（g3 8 份第 1 步的相对极差：loss 2.9e-6、grad_norm 8.2e-5）；hy4-tiny 2 卡 20 步两份逐位相同（台账 G7）；没有结果文件 |

<!-- janus:table:end -->

## 这一阶段做了什么

### 已有的证据（都不在本文件夹的结果文件里）

| 项 | 现状 | 出处 |
|---|---|---|
| GPU 同设置 8 份的散布 | 每一代 8 份两两 28 对，每一对算一次判定量，每个判定量取 28 对里的最大值，就是对应 L4 比较的噪声上限 N：`g1-*`（前三代）、`g2-*`（300 步）、`g3-*`（1000 步） | 01 [`../catalogs/t1t-l4.yaml`](../catalogs/t1t-l4.yaml)、[`../catalogs/t1u-l4.yaml`](../catalogs/t1u-l4.yaml)；各结果文件里每条检查的 `threshold.noise.upper` |
| 注意力 q·k 改 FP32 前后的散布 | `g2-*` 两两最大相对差 lm_loss 4.5e-5、grad_norm 2.2e-3；`g1-*` 3.7e-5、2.3e-3，几乎不变 | 台账 G17 |
| GPU 第 1 步三份的逐张量散布 | `g1s-*`：中位数 rel_L2 2.7e-3；第 0、1 层索引器 10 个张量三份逐位相同（`g2s-*` 同样） | 台账 G15；01 [`../results/t1-step1-grad-vs-ref.json`](../results/t1-step1-grad-vs-ref.json)、[`../results/t1t-step1-grad-vs-ref.json`](../results/t1t-step1-grad-vs-ref.json) 的 `gpu_noise_max` |
| 默认设置下能不能逐位复现 | hy4-mini 8 卡不能：g1 / g2 / g3 各 8 份第 1 步的相对极差 loss 2.1e-6 / 1.7e-6 / 2.9e-6、grad_norm 7.3e-5 / 9.3e-5 / 8.2e-5（台账 G9 的 4.9e-6、2.4e-4 是 T1 之前关 eod 掩码的测试）；hy4-tiny B200 × 2、20 步两份逐位相同 | 台账 G9、G7 |
| 载入自检 | 第 1 步前每个参数等于检查点张量、FP64 平方和相等，否则退出；hy4-mini 通过 | 台账 G7 |
| per-token loss 开关不改数值 | eod 关（掩码全 1）时 per-token 开 / 关第 1 步 loss 差 1.3e-6、grad_norm 差 8.5e-5，在同设置重跑的差之内 | 台账 G9 |
| 对照：TPU 重跑 | 逐位相同（两组各两份 100 步、45 个逐步量）；只作记录，不进尺子 | 01 [`../results/hy4-mini-tpu-determinism.json`](../results/hy4-mini-tpu-determinism.json) |

### 计划（待做）

| # | 做什么 | 判据 | 目录 / 运行 | 状态 |
|---|---|---|---|---|
| 1 | GPU 自比：`g2-b` 对 `g2-a`（`g3` 同理），噪声同一代 8 份两两 28 对 | 每个判定量 r ≤ 1；给出这把尺子在 GPU 一侧自比时的读数 | [`catalogs/t1-gpu-noise.yaml`](catalogs/t1-gpu-noise.yaml)；运行已有 | 待做 |
| 2 | 确定性设置下同配置两份 300 步 | 逐步量与第 1 步全部抓数逐位相同 | [`catalogs/t1-gpu-determinism.yaml`](catalogs/t1-gpu-determinism.yaml)；`g1det-a/b`（计划） | 待做 |
| 3 | 抓数不改数值（开 / 不开全量抓数，逐步量逐位） | 逐位相同 | 待定 | 待做 |
| 4 | 只换执行方式：专家并行 4 ↔ 8、梯度累积切分、重算开关 | 前向逐位；反向与更新后参数对噪声 | 待定 | 待做 |
| 5 | 故障演练：在确定性基线上埋错（主权重扰动、数据错位、路由改 BF16、学习率差 1 ULP、开 dropout、梯度缩放、路由偏置不更新等），看期望报警的检查有没有报 | 期望报警的都报了 | 待定；要先有第 2 项的确定性基线 | 待做 |
| 6 | GPU 算子对 FP64、逐元素函数的 ULP、极端输入 | GPU 侧检查 | 待定 | 待做 |

### 为什么要做

- T1 的 L4 与 L3 只用 GPU 重跑的散布当尺子；它有多宽、来自哪里（原子加的顺序，还是只换执行方式也会变），直接决定判定。第 1 步比较里 GPU 三份的散布只来自原子加的顺序、不含舍入位置（`t1-step1-indexer-grad.json` 的结论），所以这把尺子对舍入位置的差偏窄。
- 被测一侧没有确定性基线，就做不了故障演练，也无法证明抓数不改数值。
