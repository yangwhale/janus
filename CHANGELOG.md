# 变更记录

格式照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号照 [语义化版本](https://semver.org/lang/zh-CN/)：
0.x 期间，改了检查判法、结果格式或命令行参数的不兼容变更升次版本号，其余升修订号（详见 [`docs/design/0001-repo-standards.md`](docs/dev/design/0001-repo-standards.md)「版本」）。
公开版去掉了条目里开发仓库的 PR 号。还没有打过 tag：发版由维护者打 `v<版本>`，复现一律写 tag。

**会改变判定结果的变更**（同样的运行，结论可能不同）在条目前标 **[判定]**，旧结果要不要重判写在条目里。

## [Unreleased]

### Added

- 混元 4 mini 普查 TPU 实验二：libtpu 0.0.52 上专家并行 4、8（04 自比 V22–V24，对底 V22 都在噪声内，只换求和顺序）；推荐组合开回「索引器权重投影乘 16 位模型权重」（M27）跑 300 步一份（`x5-prod-m27`，`t1zp300m27-l4`：索引器 KD 带符号均值 −6.76e-4 → −1.98e-4，从不过变待定，`t1zp-m27-step1.json` 第 1 步逐位相同）；两个 v5p 切片走 DCN 的多切片一次（04 自比 B15，`ts-ms2-dp2-r2`：曲线在噪声内，但第 1 步 `hc_head` 三个张量离 FP64 0.51–0.57，靠 iHC 豁免单列，机制没核）。判定不变
- 混元 4 mini C 口径补数：第 1 步全量抓数 `s1f-c` 在 C 口径上重判 L3（`t1zc-step1-train-grad-vs-fp64-n6.json`、`t1zc-step1-train-grad-file-check.json`：163 个超 N，计入未定责的张量从 C 口径之前的第 3 层注意力前 `hc_scale` 换成第 3 层 FFN 前 `hc_fn`（1.14 N，R 1.43），判档仍待定）；补第 6 份 `tc-f`（scoped VMEM 48 MiB，`t1zc-repeats-6.json`：1000 步 grad_norm r 1.04 待定）；主 loss 后段残余两组实验（`mini-factorial-tpu-6.json`、`mini-factorial-c-vs-m58-6.json`、`mini-factorial-c-vs-m58-exec-only-6.json`、`mini-loss-isolation-c.json`、`mini-gpu-tpulike-branch.json`、`mini-gpu-tpulike-branch-vs-stream.json`、`mini-step1-grad-classes-c.json`：C 口径 5 份 +6.6e-6 改为 6 份 +6.9e-6）。L3 现行读数换了张量，判档不变
- 混元 4 案例 16 补核 C 口径：mini 现行比较 `tc-a` 与同配置 `tc-b … tc-e`（MaxText `8d3d7561`）按 `meta.json` 在 CPU 上重编训练步后扫 #71 的特征，5 份都是 0 处（并进 `jit-miscompile-offline-scan.json`；`8d3d7561` 没有存了 HLO 的运行可校准）；A16 补查：`tc-a`（C 口径 v5p）主头 dgrad 卷积 kernel 里没有 S，MTP 头 dgrad kernel 在 15 次存了 HLO 的程序里全部并进了 MTP 末层 RMSNorm 行和、LLO 里同样省舍入（H2 0 → 1024、V17 0 → 384 条），第 1 步 MTP 三个 norm 权重关 excess precision 前后之比 0.96–1.04（`hc-head-dgrad-fusion.json` 的 `mtp_head`，差异清单 A16、`04-tpu-self/impacts/03`）
- 混元 4 mini 补两份结果：`t1zc300-repeats.json`（C 口径 `tc-a … tc-e` 前 300 步，同一目录 `t1zc300-l4.yaml` 只换被测；`tc-b … tc-e` 9 项全过，grad_norm 0.81–0.92）、`mini-step-time.json`（C 口径与对照的步时中位数）；各处现状换成 `tc-a`，C 口径上没重做的检查标「C 口径上没重判」。判定不变
- 混元 4 案例 16 补核：mini 当时的现行比较 `tz-off-rep`（C 口径之前）与推荐组合 `x5-prod` 没存 TPU 上的 HLO，按各自 `meta.json` 在 CPU 上重编训练步后扫 #71 的特征，两者都是 0 处（`models/hunyuan4/analysis/jit_miscompile_offline_scan.py`，结果 `models/hunyuan4/results/jit-miscompile-offline-scan.json`；离线程序等于上卡程序是推断，依据是 `ts-g-a` 校准时 CPU 与 TPU 程序规范化后逐行相同）；同一 PR 补核 V17 的 hc_head 偏差
- 混元 4 mini 两项 TPU 实验：存档续跑逐位（`x4-*`，1000 步 47 个逐步量与 677 个状态张量都逐位，`t1-resume-bitwise.json`）；推荐生产组合整体跑 1000 步（`x5-prod` 3 份，`t1zp-l4`、`t1zp300-l4`、`t1zp-repeats.json`：1000 步没有不过项，但索引器 KD 三份都待定，r 1.23–1.26，当时的对齐配置 `tz-off-rep`（C 口径之前）0.77；前 300 步三份的索引器 KD 都不过，r 4.78–5.23）
- 混元 4 mini 生产配方臂 P1（YaRN factor 8、cosine 学习率、路由偏置零均值三项一起改；GPU 一侧新加 `--bias-zero-mean`、`--yarn-factor`，台账 G35；GPU `p1g-a`、`p1g-b`，TPU `p1t-a`，各 1000 步，`t1pr-*`）与 QK-Clip 真触发时 TPU 梯度累积 2 对 1 的实测（`t1zo-qkclip-accum-tau01`、`-tau005`）。诊断，不进签字
- 混元 4 mini 生产配方臂补 6 份 GPU（`p1g-c … p1g-h`），用同配方 GPU 8 份两两 28 对当尺子重判 TPU：9 项全过（`t1pr-l4-gpu8`；补跑前 GPU 只有 2 份、量不出同配方尺子，借现行尺子是不过 3、待定 1）。诊断、不进签字：TPU 1 份，8 份尺子是看过借尺结果后才定的（混元 4 `PROTOCOL.md` 第 0 节「生产配方臂」）；两种尺子的对照在 `t1pr-ruler-compare.json`
- 混元 4 mini D30 的结构消融：两边都去掉 iHC（GPU 台账 G36、`make_tiny_hf.py --no-ihc`），1000 步后段不飘；同时在旧 GPU 口径下把后段根因写成「iHC 流的 BF16 舍入点」（M58、M59），这一条随后由 #418 撤回、改成根因未定，#430 用两头各改一处的实验重新定为主因（见下面「Changed」里 #430 一条）。结果 `mini-ihc-ablation.json`
- 混元 4 案例 16：libtpu 0.0.46–0.0.52 在 v5p 上的 jit 静默算错（多输出 fusion 合进 DUS），机理、4 种 workaround 与存档训练步程序的特征扫描（`hlo_mof_dus_scan.py`）；差异清单 F8
- 混元 4 TPU 自比：RMSNorm 多舍一次在 libtpu 0.0.52 上变强的机理（`04-tpu-self/impacts/04`）
- Megatron 抓数（`janus/dump/megatron.py`）新增三个默认不改行为的选项：`gather_experts`（全量抓数步把 rank 0 所在专家并行组里其他 rank 的路由专家收到 rank 0，按全局专家号命名）、`full_kinds`（全量抓数只写指定种类）、`forward_acts`（False 时抓数步不抓前向激活）。格式见 `docs/reference/dump.md`「GPU 侧的限制」；已有运行目录不受影响
- Megatron 抓数（`janus/dump/megatron.py`）全量抓数时另存 Muon 的动量缓冲 `momentum_buffer.<参数名>`（更新后）；走 Adam 的参数照旧存 `exp_avg*`。格式见 `docs/reference/dump-format.md`；已有运行目录不受影响
- 混元 3 便宜的等价修法 `rcb`：`rr_arms.py` 新臂 `dgrad-chunk-bf16`（输出头 dgrad 按 1024 分段，每段 BF16 张量核乘法、输出 FP32，段间 FP32 累加），训练前用 `dgrad_accum_ablation.py --check-arm` 核对与离线消融 B1024 逐位相同；`rcb-a`、`rcb-b` 1000 步对 `rcc-a`（`rcc` 45 对噪声）L4 全过，步时 +6%（`rcc` +61%）。新目录 `phase2-rcb-vs-rcc`、交叉核对 `phase2-acceptance-rcb-on`、`phase2-v7x-acceptance-rcb`；正式基线不变，仍是 `rcc-a`
- 混元 3 输出头 dgrad 消融脚本 `models/hunyuan3/analysis/dgrad_accum_ablation.py`：`rcc` 去掉偏差靠的是分段，不是 FP32
- 检查清单 `models/hunyuan3/CHECKS.md`：L0–L4 每一项比什么、对象是哪些模块与张量、判法（严格档｜参考档）、v5p / v7x 结果与证据链接；L1 附「模型的每个算子由哪条检查覆盖」表，文末列已知的覆盖缺口与未列入的结果文件。骨架在 `models/hunyuan3/dashboard/checks.yaml`，数字由 `janus/report/inventory.py` 从结果文件现读，`make dashboard` 一并生成、`make dashboard-check` 一并核对；总览引用的检查在清单里缺行时生成失败
- 检查清单：`checks.yaml` 里标 `info: true` 的行（对照、可选配置、只记事实、定责用的 FP64 比较）不进规模表的判定计数，单列「只报」；格子写 rel_L2 而最差的指标离噪声上限更远时，另写「最差 …×N」
- **[判定]** TPU 扰动臂第三关的「第 1 步参数梯度」一半改成比 FP64 整步参照（臂离 FP64 不比基线远，按能量 R ≤ 1.2），不再直接比 GPU 第 1 步噪声：只改执行方式的臂会翻近似平局的路由（混元 3 当前配置 v5p，改片上内存上限的臂对基线第 3 层专家 0 梯度差 0.080），GPU 重跑不翻。当前配置 6 个臂 R 1.001 – 1.039（`models/hunyuan3/regression/results/arms-step1-vs-fp64.json`）
- 混元 3 台账 #20 的三个梯度 FP32 开关的复核：开关版 L4 第一层（`phase2-acceptance-rcc-gradfp32`、`phase2-v7x-acceptance-rcc-gradfp32` 与健康信号）、第 1 步整步并集与对 FP64（`phase2-st1-gradfp32-xnoise`、`phase2-st1-fp64-gradfp32`）、步时；对齐报告开关表加一行
- **[判定]** 混元 3 签字项 2b 的判法（`docs/reference/judging.md`「第 1 步张量：两侧执行噪声的并集」）
  - 整步第 1 步状态读的检查，噪声改取 GPU 重跑 ∪ TPU 只改执行参数的重跑（`phase2-st1-on-state-xnoise`、`phase2-v7x-st1-on-state-xnoise`）。
  - 逐层隔离回放的反向改按对 FP64 的 R ≤ 1.2 判（新增 `models/hunyuan3/analysis/fp64_backward.py`、`fwd_rratio.py --mode bwd`）。
  - 整步对 FP64 的定责工具：`fp64_step.py` 与 `fwd_rratio.py --mode step`。
  - 诊断臂（只当灵敏度参考或定责用，不进尺子）：`train_hy3.py` / `layer_replay.py --router-jitter`、`train_hy3.py --grad-fp32`。
  - 总览对应格子换到新结果；旧的 `phase2-st1-prod-state`、`phase2-l2-replay-bwd-on` 结果保留不动
- 参考档新增 `L4.param_norm`：带符号均值 ≤ 5e-6、最大单步 ≤ 1.25e-5。只影响并列报告，严格档判定不变
  - 外部没有公开门槛，是项目负责人设定的。
  - 依据：混元 3 上 GPU 生产 BF16 程序对精确轨迹的带符号均值 2.5e-6、最大单步 6.2e-6，各取约 2 倍；推导在 `reference_gates.yaml` 的 `source`。
- `multi_golden` 单 golden 模式 `params.baseline: <golden 名>`
  - 只对指定的一份 golden 算判定量，其余 golden 只组噪声带（不取平均、不取各份里最差）。
  - 不写时行为不变（取对各份里最差）。
  - 混元 3 的正式验收目录改用它，用同一批运行重判（例：v5p 对 `rcx`（生产 golden，输出头 dgrad 换 FP32 一次累加）的 loss 第 301–1000 步 7.03e-5 待定 → 对 `rcx-a` 3.65e-5 过；旧数写在 01 / 02 README）。
- 序列检查新增 `params.denominator: mean` 与 `select.test_field`；`Run.series` 新增派生列 `param_norm_from_groups`。默认行为不变
  - `params.denominator: mean`：两边均值 |t + r| / 2 作分母，交换两边只差符号，给 grad_norm 这类逐步涨落大的量用。
  - `select.test_field`：被测改读另一列，golden 与噪声对仍读 `field`。
  - `param_norm_from_groups` 由 `group_sq.*` 现算。
  - 混元 3 第一层验收目录用了这两项，原口径作只报项并排保留。
- `multi_golden` 的 `params.noise_from: runs | golden`：目录里 `runs.noise` 装的是另一种噪声（如日常回归的 TPU 执行噪声）时，这一条仍用 golden 两两组合当参照
- 参考档新增「优化器更新等价」组（`L1.optim.adamw.replay`、`L3.step1.updated`：rel_L2 ≤ 0.02 且 cos ≥ 0.9999）；记了 cosine 的检查在说明栏并列写 cos。只影响并列报告
- TPU 侧正式证据工具
  - 观测构建（`train_hy3.py --observe`）、梯度 NaN 演练（`--inject-nan-grad`）。
  - 逐层回放存算子输出（`layer_replay.py --l1`）与整网只前向（`--full-forward`）。
  - `janus/dump/maxtext.py` 全量抓数按 `--full-only` 过滤。
- 总览与逐级大表支持任意列的演练格（`<列>_expect: fail`）
- `janus` 总命令（`janus/cli.py`，console script），子命令按需导入；`janus --version`
- `janus demo`：不要 GPU / TPU 的合成演示，6 份参照运行 + 两份被测（一份埋错）
- `janus curves`、`janus offsets`、`janus fault-matrix`：逐步曲线、偏移归因、故障矩阵从 `models/hunyuan3/analysis/` 搬进 `janus/analysis/`，模型特有的分组与期望改成 YAML；`fault-matrix --strict`
- `janus validate`：只校验检查项目录（展开后的检查与校验规则），`--all` 递归；CI 校验仓库里提交的全部目录
- `janus drift pairs|trend|groups|rows`：几组运行两两比、多步趋势、分组参数范数、输出层逐行漂移从 `models/hunyuan3/analysis/` 搬进 `janus/analysis/drift.py`
- GPU 抓数（`janus/dump/megatron.py`）新的逐步字段：
  - 分组参数范数 `group_sq.*`、每专家平方和 `expert_sq.*`（`--group-norms`）
  - 输出层与 embedding 逐行统计 `output_row_sq`、`output_row_u`、`embedding_row_sq`（`--row-stats-every`）
  - `output_row_g`、`output_row_wu`
- 日常回归：v5p T0 试跑（首个 LKG、数学不变复测、改开关被拦）；T1 首个 LKG、TPU 自噪声标定工具 `models/hunyuan3/regression/calibrate.py`
- 混元 3 GPU 生产近似 golden：`train_hy3.py --routing-replay`、对齐实践 `03-b200x8-self-4layers-remat`（`rc-a`..`rc-e`）、抓数另存 `router.{logits,probs}.recompute`
- 混元 3 案例 01 的诊断工具：反向复现诊断臂、残差流探针（TPU `--resid-probe-every`）、`resid_probe_compare.py`、`gauge_analysis.py`、`adamw_step_check.py`
- 混元 3 案例 01「逐步拆分」与状态互换的工具
  - TPU：`train_hy3.py --grad-probe-at`（独立 jit 程序算生产精度 / FP32 参照的梯度与 logit 水平方向）、`--full-only`、`--export-params-at` / `--params-from` / `--first-step`（`param_io.py`）。
  - GPU：`train_hy3.py --export-params-at` / `--import-params` / `--skip-steps` / `--fp32`（`param_swap.py`）。
  - `analysis/step_decomp.py`（由 Adam 动量反推梯度、FP64 重算更新、对参照梯度）。
  - `MaxTextDump(only=…)`、`add_tensors`。
- 参考档统计：`janus/metrics.py` 的 `reference_stats`（超差元素比例、平均差、归一化最大差，容差取法是推断）
  - `tensor_tolerance` 另报 `oot_frac_max`、`mean_diff_max`、`norm_max_max`。
  - `op_replay` 另报被测对参照的 `vs_ref_*`（`vsref` / `vsref_out`）。
  - 以上都只报不判。
  - `fwd_rratio.py --mode ops --same-input` 用 `layer_replay.py --l1` 的同输入输出补算子级对照。
  - v5p 的逐层前向、逐层反向、单算子重跑成 `*-vsref` 结果（判定与原结果逐项相同，原结果保留），总览参考档的 ❔ / ☑️ 变成实判。
- 总览与逐级大表报两档门槛。不改任何检查的判定
  - 严格档：janus 自己的判定，不变。
  - 参考档：同类 GPU→TPU 迁移项目的实践门槛，`janus/report/reference_gates.yaml`；`janus/report/gates.py` 从同一份 metrics 现算，缺指标写「未记录」。
  - 说明栏统一写「实测｜严格档 门槛 → 倍数，判定｜参考档」，不再贴 `['r_ratio']` 这类原始列表。
  - 逐级大表改成 6 栏。
  - 说明见 `docs/03-metrics.md`「两档门槛」。
- 工程设施：
  - ruff lint（全仓库）与 format（`FMT_PATHS`）、`Makefile`（`make check` 等）、CI 的 lint job 与 Python 3.10 / 3.12 矩阵
  - CI 的 `package` job（装成普通包、在仓库外跑 demo）
  - CI 的 `generated` job（重画总览与逐级大表后要和提交的一致）
  - `make dist`、`make dashboard`、`make dashboard-check`
  - `.editorconfig`
- 测试：
  - 仓库卫生（内部链接、真实桶名、模块开头说明）
  - 文档里「某文件「某节」」引用的节必须存在
  - 不变式（引擎不导入 torch / jax / matplotlib；指标在 FP64 里算）
  - 补上没有直接测试的模块
- 文档：
  - 术语表、架构、对齐迭代流程、命名约定、命令行参考、设计说明 0001、路径变更记录、排查案例模板，文档索引按用途分类
  - README 重写、`CHANGELOG.md`、`LICENSE`、`SECURITY.md`、`CONTRIBUTING.md` 开发流程

### Changed

- **[判定]** 混元 4 mini 对齐配置改 C 口径：四份对齐 yml 在已开的 `ihc_force_bf16_rounding`（M36）之外再开 `hc_head_force_bf16_rounding`（M52）与 `ihc_stream_grad_gpu_order`（M58），TPU 在 iHC 与 hc_head 一带照 GPU 的写法「FP32 算、BF16 存」，GPU 不改；MaxText `8d3d7561`。现行比较从 `tz-off-rep` 换成 C 口径第一份 `tc-a`（同配置 5 份 `tc-a … tc-e` 都判、都报）。现行判定：1000 步 9 项过 8、待定 1（路由偏置均值，带符号均值 r 1.31，`t1zc-l4`），主 loss 由待定（r 1.55、1.52）变过（MARE r 0.98、0.97）；前 300 步过 8、待定 1（grad_norm MARE r 1.01，`t1zc300-l4`）。`tz-off-rep` 降为上一代，结果保留、不重判。主 loss 后段的主因定到 iHC 流梯度的舍入写法（混元 4 `ROOT-CAUSES.md` 第 30 行），取代 #418 的「根因未定」；C 口径下后段还剩约 +6.6e-6，机制待查。L3 第 1 步梯度、优化器一步、路由一致率、L1、L2、QK-Clip、健康信号、同设置重跑、存档续跑、抓数不改数值、一次实际排查 的特征扫描没在 C 口径上重判，沿用上一代读数（各文档标「C 口径上没重判」；其中 L3 第 1 步随后由 #438 在 C 口径上重判，#71 扫描由 #440 补扫 `tc-a … tc-e`）。另加 GPU 一侧诊断开关 G37（`HY4_TPU_LIKE_*`，默认不改行为）、`models/hunyuan4/analysis/mini_factorial_cells.py`，结果 `t1zc-repeats.json`、`tc-l0-settings`、`tc-l0-config`、`mini-factorial-tpu.json`、`mini-factorial-c-vs-m58.json`、`mini-gpu-tpulike.json`。变更记录见混元 4 `PROTOCOL.md` 2026-11-11
- 混元 4 mini D30 更正：主 loss 后段根因未定，范围缩到 iHC 到 hc_head 一段（#411 写的「已定位到 iHC 流的 BF16 舍入点」撤回）。判定不变
- **[判定]** 混元 4 mini L3 第 1 步梯度的 GPU 参照从 3 份 3 对补到 6 份 15 对（`g2s-d … f`），现行判定换成 6 份（`t1zo-step1-train-grad-vs-fp64-n6`）：不过 → 待定；超 N 且 R > 1.2 计入未定责的由 5 个 iHC 张量变成第 3 层注意力前 `hc_scale` 1 个，MTP `enorm` 另列，两个都按 D11 豁免规则单列（豁免规则与沿用待复核）。3 份时的结果保留。GPU 镜像的 TE / cuBLAS / cuDNN 版本改成实测。变更记录见混元 4 `PROTOCOL.md` 2026-11-11
- 混元 4 mini 优化器一步（L3）换成现行 `tz-off-rep` 第 1 步全量抓数重算（`t1zo-l3-optim-step1`），仍全过；判定不变
- **[判定]** 混元 4 mini 的对齐配置关 splash 的 base2 指数（四份对齐 yml `sa_use_base2_exp: false`，所有尺寸一致），现行比较从 `t1zg-a` 换成关 base2 的 `tz-off-rep`；同时 mini 1000 步的噪声尺子从 `g3` 8 份两两 28 对换成 16 份两两 120 对。现行判定：1000 步 9 项过 7、待定 2（`t1zo-l4`）；同一把 16 份尺子下上一代 `t1zg-a` 是 9 项全过（8 份时过 7、待定 2）。换代后另有几项结论变了：L1 低阈值 QK-Clip 第 1 步逐头最大 logit 由过变不过（第 4 层 R 0.94 → 1.55，未定责）；L3 第 1 步未定责的 iHC 张量里 3 个是新出现的；L3 第 1 步 MTP `enorm` 由过线内（R 1.18）变成超线（R 1.24），按 D11 豁免规则单列（豁免规则与沿用待复核）；前 300 步主 loss 由过变待定（带符号均值 r 1.10）。前 300 步 grad_norm（MARE r 1.23）上一代就是待定，L4 QK-Clip 全局最大 logit 1000 步 r 1.09 待定。逐项见混元 4 `PROTOCOL.md` 第 0 节表下第 2 条。旧结果不重判、原样保留，`t1zg-a` 降为上一代；L1 注意力整块、L2、路由一致率第 8 / 30 / 100 步没有关 base2 的结果，照旧引用开 base2 的一代并注明。变更记录见混元 4 `PROTOCOL.md` 2026-11-11
- **[判定]** 混元 4 TPU 自比 1000 步的尺子从 GPU `g3` 8 份两两 28 对换成 16 份两两 120 对，全部 1000 步臂重判：8 份时落在 r 2–3 的灰区臂全部落到「和 GPU 重跑分不开」，GPU 留一法最大值 2.00 → 1.82；前 300 步仍是 8 份。逐臂结果在 `models/hunyuan4/alignments/04-tpu-self/MATRIX.md`「L 长跑」，变更记录见混元 4 `PROTOCOL.md` 2026-11-10
- **[判定]** 混元 4 的曲线检查一律以参照作分母，grad_norm 不再用两边均值（`denominator: mean`）。87 个目录改动，262 份结果用同一批运行目录重跑后原名覆盖：梯度范数以外的检查不变，判定只变 1 项（`t1zz-l4` 的 grad_norm 过 → 待定），TPU 自比判档与 GPU 留一法最大值（2.47 / 1.82）不变；逐项对照在 `models/hunyuan4/results/grad-norm-denominator-rejudge.md`，变更记录见混元 4 `PROTOCOL.md` 2026-11-10。janus 本身的判法与默认值不变，`denominator: mean` 仍可用；混元 3 已签字，第一层沿用两边均值
- **[判定]** 混元 4 mini 对齐报告开关表的「波动内」线从升幅 0.45 改成 1.07：原线只取自一个对照（关掉「QK-Clip 不含偏置项」）；新线取只改执行方式的 8 份运行（base2 开、关各 4 份；每组 3 份同一配置只差 vmem 上限或延迟隐藏调度，另 1 份是早一代、少 M50），截前 300 步、用开关表同一把尺子（GPU 8 份 28 对），同组任取两份「升得最多的一项」的最大值（`models/hunyuan4/alignments/01-mini/gpu-noise/results/exec-spread-300.json`）。升幅 0.45–0.9 的 7 行退回「波动内」，单看曲线不再归到开关；超线的只剩 4 行。检查目录、结果文件与判档不变，不用重跑；拿 0.45–1.07 之间的升幅当定责依据的几处已改成如实（单份、在波动内）。去掉蒸馏 loss 一项重算，8 份尺子下最大 0.90、GPU 16 份尺子下 0.81（`exec-spread-300-nokd.json`）
- 混元 4 变更台账改号：`embedding_shard_over_data` 记作 M54（原记 M51，与精确 log 重号，）；`ihc_stream_grad_gpu_order` 记作 M58（#396 写成 M54 的运行登记与 J58 已改正）。外部按旧号引用的要对照改
- 混元 4 TPU 自比的档名「在刻度内」改叫「在噪声内」、「超出刻度」改叫「超出噪声上限」：`tpu_self_compare.py` 新写的结果用新档名，已提交结果里的旧档名不改，`build_tables.py` 读到旧档名时按新叫法显示。不改判档。变更记录见混元 4 `PROTOCOL.md` 2026-11-10
- 结果格式升到 `schema_version: 2`：`catalog` 记目录的内容指纹（`content_sha256`，`hash_algo: content-v1`），原文件 sha256 改名 `raw_sha256` 只留作历史
  - 内容指纹 = 解析 YAML（丢注释）、删掉纯描述键（顶层 `description`；检查项的 `description`、`notes`、`source`、`pending_note`）、键排序的紧凑 JSON 的 sha256（`docs/reference/result-schema.md` 注 2）。
  - 改注释和说明文字不再让已提交的结果对不上；`tests/test_committed_results.py`、`regress.py` 的 R02（`lkg.yaml` 新增 `catalog_content_sha256`）都改比内容指纹。
  - 已提交的 58 份 `janus run` 结果由 `tools/migrate_catalog_content_hash.py` 迁移：在 git 历史里找到跑的那一版目录，确认内容指纹与现在的相同才写（日志 `tools/migrate_catalog_content_hash.log`）。没迁移的 3 份不留例外：两份 CPU 冒烟（`phase2-cpu/cpu-smoke*`，目录是没进仓库的临时文件）删除，引用改指 `cpu-smoke.txt` 与真规模的结果；`phase2-t-loss30`（跑的那版与现在只差 `runs.root`，算内容）用现在的目录对同一批运行重跑，判定与数字不变，只多了 `first_differ_step` 字段
  - 独立脚本写的结果不变：仍是 `schema_version: 1`、`catalog.sha256` 是脚本文件的哈希。
  - 17 份检查项目录注释里仓库重整前的旧路径一并改成新路径（内容指纹不变）。
- 逐级大表与总览的说明写清「N=…（k 对）」的单位。只改文字，判定不变
  - k 是噪声参照的运行对数，不是运行份数。
  - 总览另列混元 3 的「k 对 = n 份两两」换算。
  - 新测试 `tests/models/hunyuan3/test_noise_pairs.py` 守住「各目录噪声对都是完全两两配对」这一前提。
- 混元 3 TPU 提交（`run.sh`、日常回归）：集群特有的提交设置移出仓库，改由站点配置文件提供。Pod 里的命令、镜像、拓扑不变
  - 站点配置 `JANUS_SITE_PROFILE`，接口见 `models/hunyuan3/frameworks/maxtext/README.md`「站点配置（`JANUS_SITE_PROFILE`）」，示例 `site-profile.example.sh`。
  - 默认提交方式下 `NODEPOOL` 改为必填。
- 版本号只写在 `janus/__init__.py`，`pyproject.toml` 读它；改为 `0.1.0.dev0`（还没发版；发版时去掉 `.dev0` 并打 tag）
- `janus run` 改用 argparse：参数不变；未知参数、缺值、目录文件不存在都返回 2，不再抛栈；`--help` 返回 0
- 渲染模块合进 `janus/report/`：
  - `report_html.py` → `report/html.py`，`alignment.py` → `report/tables.py`，`figures.py` → `report/figures.py`
  - `report.md` 的渲染从 `janus/run.py` 移到 `janus/report/markdown.py`（输出逐字节不变；`janus.run._render_report` 仍可导入）
- 测试目录：
  - 照包的结构（`tests/checks/`、`dump/`、`analysis/`、`report/`、`models/<模型>/`、`docs/`），共用造数据函数进 `tests/helpers.py`，pytest 改用 importlib 模式
  - 日常回归测试里测 janus 本身的用例按模块归位
- `janus offsets --golden` 与 `janus curves compare --golden` 写法统一：一次给几个、重复写都认（旧写法照旧可用）
- 文档：`01-principles.md` 的术语表并入 `docs/glossary.md`；指标速查并入 `docs/reference/metrics.md`；三份设计文档从 `docs/reference/` 搬到 `docs/design/` 并编号

### Fixed

- 混元 4 文档跟上 #438：L3 第 1 步各处换成 C 口径读数（未定责是第 3 层 FFN 前 `hc_fn`，`hc_scale` 改标 C 口径之前）；只开 M36 + M58 那一格写明 5 份里 2 份仍超尾部阈值，C 口径主 loss 过主要是前 100 步鼓包没了；`tc-f` grad_norm 1.04、`tc-b` 主 loss 1.14 待定上总览；总览加存档续跑逐位一行。判定不变
- 混元 4 文档与结果文件对齐：主 loss 后段按 `mini-factorial-tpu-6.json` 写明 C 口径 6 份的尾部和只开 M36 的对照分不开（p 0.32，尾部消失只在只开 M36 + M58 那一格），总览头条改为从这个文件读数；L3 改 C 口径后检查清单、总览、对齐细节的说明同步；检查清单缺口表被 YAML 折行吞掉的一行恢复；前 300 步 C 口径另 4 份（不是 5 份）；C 口径之前的格子逐处标明。判定不变
- 混元 4 TPU 自比 `tpu_self_compare.py` 的程序指纹改用跑前离线筛同一套规范化（`hlo_screen.py` 的 `canon_lines`）。已提交的 184 份结果用 `analysis/refresh_program_fingerprint.py` 从运行目录里存下的 HLO 重算 `hlo.program_fingerprint` 与 `hlo.program_equal`，其余字段不动；`program_equal` 变了的只有 C13、E11 两份（false → true）。不改判档。变更记录见混元 4 `PROTOCOL.md` 2026-11-11
- 混元 4 TPU 自比 `tpu_self_compare.py` 数 HLO 指令时漏掉 6 元及以上的元组指令（形状里的 `/*index=5*/` 注释带等号），合并后的大 all-reduce 和多输出融合都没数进去；已修并加单测。只影响结果文件里只报的 `hlo.op_counts_diff`，不改判定；已提交的结果文件不改，文档里引用过的四处计数在 `models/hunyuan4/alignments/04-tpu-self/b1-grad-reduction.md`「六」重数更正
- 文档逐条对账后的更正：台账 #1 / #5 与对齐报告第 2 节把「MXU 就近舍入」「12 万长求和上的截断模型」标为推断；#1 的 TPU 读数改为同步号的输出头回放（−2.4e-5 / −7.4e-5）；#5 与术语表的近似平局分差改为生产回放的 2.7e-5 – 6.3e-5；config-alignment 的路由翻转来源改为上游；扰动臂 FP64 R 分平台写（v5p 1.001 – 1.008、v7x 1.003 – 1.039）；术语表五级判法、principles 的 FP64 适用范围、judging 的基线位次。术语表新增截断、ŵ̄、Adam ε 区、两侧执行噪声的并集、`rcb` 与代号白话名表；台账加注 1（为什么是输出头）
- 第一阶段 5 个 GPU 脚本的结果记的是搬家前的脚本哈希，用当前脚本在 8×B200 上重跑
  - 涉及：`phase1-l1`、`phase1-kernels`、`phase1-moe-stages`、`phase1-causal-model`、`phase1-ckpt-special`。
  - 输入运行目录（`d-a`、`g61-obs`、`g61-notf32`、`g61-sv-save` / `g61-sv-load`）原件已不在，按原参数确定性重跑；第 1 步 `lm_loss`、`grad_norm` 与原记录逐位相同。
  - 状态与全部指标值逐位相同；`phase1-l1` 的两条 `L1.rratio.*` 多出参考档只报的 `vs_ref_*`，`phase1-ckpt-special` 记的训练参数多出后来新增的默认项，`phase1-causal-model` 的建模耗时不同。
  - `tests/test_committed_results.py` 的 `HISTORICAL` 清空。

- `fwd_rratio.py` 写的 5 份结果记的是旧版脚本的哈希
  - 涉及：`phase2-ff-chain`、`phase2-v7x-ff-chain`、`phase2-l1-ops-own`、`phase2-l2-replay-rratio`、`phase2-l1-ops-wcoff`。
  - 用当前脚本、同一批运行目录重跑，状态与原有指标逐项相同，只多出参考档只报的 `vs_ref_*`。
  - `tests/test_committed_results.py` 扩展到独立脚本写的结果。
  - 第一阶段 5 个 GPU 脚本的结果重跑要 B200，先登记在 `HISTORICAL`（已重跑，见上条）。
- 混元 3 实践 01 / 02 的 35 份 `janus run` 结果记的 `catalog.sha256` 与现在的目录文件对不上
  - 原因：目录之后只改了注释、`description` 或默认运行根 `runs.root`。
  - 用同一批运行目录重跑，状态与指标逐项相同，只有目录哈希、时间戳与 janus 版本号变了。
  - 新增 `tests/test_committed_results.py`：已提交结果的目录哈希对不上现在的目录就失败（有意保留旧版的写进白名单并写明原因）。
- **[判定]** `health_signals`：选中的字段没有数据（拼错、或运行没记）以前判 `pass`（「健康」），现在判 `error`
  - 已提交的 `pass` 里有没有这种情况，从 `results.json` 看不出来（只记了字段数），重跑这些目录时会暴露。
- **[判定]** `determinism`：被测某个字段一步都没有时以前判 `pass`（什么都没比），现在判 `error`
- **[判定]** `optimizer_step` 连续多步（`select.steps`）：汇总时只认 fail / error、其余一律 pass，会把 `inconclusive` 抹成 `pass`；现在按 fail > error > inconclusive > pass 汇总
- **[判定]** `ulp_distribution` 不写 `hard` 时按规格默认只报不拦（`hard: false`）；以前引擎一律默认 `hard: true`
  - 已提交的目录里只有两条没写 `hard`，它们不写门槛、只出分布，判定不变。
  - 只有这两条报 `error` 时退出码会从 2 变成 0。
- `out_of_tol_ratio` 的噪声参照与被测用同一组 `atol` / `rtol`（以前噪声那边固定用 0）；目录里还没有检查用这个指标，已提交的结果不受影响
- `janus run` 的异常处理
  - 一条检查里出了意料之外的异常，以前整次运行崩掉、没有 `results.json`、退出码 1（像「硬检查 fail」）。
  - 现在记成这条检查的 `error`，其余照跑，退出码 2。
  - `--render-only` 找不到 `results.json` 时退出码 2 并提示。
- 各子命令读不了文件、或出了意料之外的错，统一退出码 2（打印原因与栈），不再以 1 退出；`janus` 不带参数时用法打到 stderr
- `results.json` 的 `janus.commit` 一直写 `null`（规格要求写提交号）：现在记 janus 的 git 提交号，janus/ 下有未提交改动时带 `+dirty`
- `janus offsets`：参照为 0 的步跳过，不再除零中断（已提交的结果不受影响：案例 01 的 `case01-offset-attribution.json` 重算逐字节不变）

### Deprecated

- 跨模块使用的辅助函数改成公开名：`janus.checks.series.window_stats`（原 `_stats`）、`judge_series`（原 `_judge`）、`janus.checks.tensor.tensor_row`（原 `_tensor_row`）、`janus.checks.base.is_higher_better`（原 `_is_higher_better`）；旧名留作别名，0.2 删除
- 旧模块名 `janus.report_html`、`janus.alignment`、`janus.figures`：导入时给 `DeprecationWarning`，转发到 `janus.report.*`，0.2 删除

### Removed

- 混元 3 实践 00 / 01 / 02 里已被取代、也不再作为现行结论证据的结果与目录
  - 数量：37 份结果（`results.json` + `report.md`）、25 份检查项目录、4 个附属文件、4 张图。
  - 包括：
    - `rce`（精确轨迹，BF16 权重、全程 FP32 计算）、`rcec`（真精确轨迹，精确轨迹加输出头 dgrad 分段累加）与中间一步 `rcx` 的交叉核对；
    - 换代前的 `acc-health`；
    - 开关打开前的逐层反向；
    - `t-a` 旧配置的状态读与 1000 步趋势；
    - 30 步的 `t-fix` / `t-ga2` 臂；
    - 与当前配置重复的 `fa32` / `vsref` / 关窗口化的 v7x 臂；
    - 复现审计用旧目录副本跑的 4 份结果。
  - 读数留在引用它们的文档里，原件在 git 历史里。
- 没有调用方的 `janus.dtypes.is_float`、`janus.dump.names.canonical_param`
- `docs/reference/metric-cheatsheet.md`（并入 `docs/reference/metrics.md`）

## [0.0.1] - 2026-11-04

第一个可用版本：第一阶段（GPU 自比）全部完成，第二阶段（TPU）的工具与第一批结果就位。没有打 tag；按提交号 `3fff02d` 复现。

### Added

- 骨架；三份接口规范：抓数格式、检查项目录、结果格式；定时回归的设计
- 存取模块（`store.py`）与指标（`metrics.py`）；检查框架与第一批检查类型、序列检查与 `janus run`（栈式 PR，合入）
- GPU 侧：Megatron-Bridge 抓数回调（不改 Megatron 源码）与混元 3 的 GPU 脚本；逐参数梯度
- 带符号均值的标准误与可选下限
- GPU 全覆盖演练：抓数 v2、运行器全类型、故障注入、HTML 报告；并行等价演练（EP8 ↔ EP4）
- L1 单算子回放与合成用例（`op_replay`、FP64 正确舍入、GEMM 舍入模拟、R-ratio）
- GPU 补齐要新运行的检查：并行、梯度累积、NaN、重算、截断、权重往返
- 检查补强：指标、ULP 分桶、KL、假 FP32、dtype 清单、步号探测、噪声参照剔重
- 对齐总览（每项检查 × 各实践，从结果文件生成）；每个对齐实践生成逐级大表与配图
- 多次运行按步对标的时序图、按设置分组的持续偏移
- TPU 侧（MaxText）脚本与配置；TPU 原生数据加载器对拍；逐层隔离回放与第 1 步逐层对比；TPU 侧 L1 单算子回放
- GPU 侧剩余检查：TF32、专家 GEMM / 注意力回放、MoE 分阶段、poison、特殊值存 / 恢复
- 路由真 FP32 诊断开关 `--router-true-fp32`（不用于 golden）
- 日常回归：通用标准与混元 3 实例；T0 / T1 可以手动跑（FP32 状态清单、回归目录、手动入口）
- 文档：指标速查表、方法文档定稿、L0 设计、复现审计、仓库布局与接新模型指南、精度 issue 覆盖审计（公开版未收）、案例 01（TPU 1000 步 loss 稳定偏低）

### Changed

- 噪声参照从 3 份运行两两 3 对 → 6 份 15 对 → 8 份 28 对
  - 第一阶段全量目录（46 条检查）的 inconclusive 8 → 3 → 1 条（`phase1-full*.results.json`、`phase1-noise15.results.json`）。
  - 判法没变，是参照多了。
- 目录按「通用 / 模型 / 对齐实践」三层重排 #35（旧路径对照见各实践 README 末尾）
- 抓数步之后的训练用生产图；抓数分档并可逐档自检
- TPU 训练 yml 默认改为梯度累积 2

### Fixed

- 复现审计记录的代码问题，每条一个回归测试（`tests/test_audit_fixes.py`）
- 多主机 TPU：
  - 载入检查点前初始化 `jax.distributed`
  - `run.sh` 由提交机取回并上传运行目录、找写运行目录的那个 Pod 取回
  - 逐层回放修多主机反向
- `run_regression.sh` 只看同一个节点池上的作业

