[janus](../../../README.md) › [用户指南](../../../docs/guide/README.md) › [混元 3](../README.md) ｜ 日常回归的混元 3 实例；对应指南[第 7 章 日常回归](../../../docs/guide/07-daily-regression.md)

# 混元 3 · 日常回归

通用标准见 [`../../../docs/reference/regression.md`](../../../docs/reference/regression.md)（下称「标准」）。
这一页只写混元 3 的参数：被测、固定输入、平台档案、各行的具体名字与个数、噪声标定计划、上线前要补的东西、怎么手动跑。
行号 R01–R28 与标准第四节一致。

**状态**（2026-11-05）：

- **能不能跑**：T0、T1 可以手动跑（第八节），**不排程**（第九节第 3 条）。T2 及以上未实施。
- **首个 LKG**：v5p T0、T1 已建立（同一提交双跑逐位）。
  - 「数学不变」复测逐位通过、故意改开关被拦（[`results/`](results/README.md)）。
- **现行配置是 `hy3-4layers.yml` a238883b**。
  - 内容：`dense_fp32_accumulation`（FSDP 切开收缩维的投影改用 FP32 累加）与三个舍入开关，maxtext hunyuan@0622774a。
- **⚠ 仓库里现在这份 `hy3-4layers.yml` 的 sha256 是 305fa4b5，不是 a238883b**（只改了注释，数值相同）。
  - 训练脚本记的是文件字节的 sha256（`train_hy3.py` 的 `sha_file`），所以照仓库现行文件跑日常回归：R02（`config.sha256 == a238883b`）必然判 fail，噪声标定按 `config_yml_sha256` 判作废，执行级判定降为 inconclusive。
  - 两种办法：用 a238883b 那一版 yml（`git show 4db01d7a:models/hunyuan3/frameworks/maxtext/hy3-4layers.yml`，的提交）跑；或改用 305fa4b5，重建 LKG、重标噪声、改 R02 的钉值。选哪种待拍板。
  - 用旧版 yml 时不能只在工作树里换文件：`stage.sh` 只打已提交的 HEAD、要求工作树干净。要提交到单独的分支再 stage，完整步骤见[指南第 7 章 7.4 第 2 步](../../../docs/guide/07-daily-regression.md#第一次跑之前换成你自己的)。
- **现行配置下的 LKG**：T0、T1 已重建（`reg-t0-fa-a`、`reg-t1-fa-a`）。
  - 同一提交双跑逐位，各另有 3 次同程序重跑也逐位。
  - **2026-11-05 换代**（数学级换代）。
- **TPU 执行噪声**：已按新配置**重标，正式**。
  - 基线 + 7 个臂共 8 份运行，两两配对，T0、T1 各 28 对比较。
  - 绑定 yml a238883b（[`calibration.yaml`](calibration.yaml) 最新两条）。
  - R22 步时阈值也按新配置重标。
- **历史**：旧配置（7704115b）下的标定与 LKG 留作历史。
- 凡写「待实测」「待标定」的数都还没有。

## 一、被测与固定输入

| 项 | 取值 | 出处 |
|---|---|---|
| 被测 | `maxtext` 的 `hunyuan` 分支，主干每个合并提交 | [`../README.md`](../README.md#代码基准)「代码基准」 |
| 训练入口与配置 | [`../frameworks/maxtext/train_hy3.py`](../frameworks/maxtext/train_hy3.py) + [`../frameworks/maxtext/hy3-4layers.yml`](../frameworks/maxtext/hy3-4layers.yml)，命令行另加 `mtp_num_layers=0` | 01 [`setup.md`](../alignments/01-v5p-acceptance/setup.md) |
| 权重 | 截 4 层的 HF 转成的 MaxText 检查点（FP32）；清单 `tensor_sha256.json` 的 sha256 `183274dbf17e…` | 01 `setup.md`「权重」 |
| 数据 | 回放 GPU golden `g1k-a`（第一阶段 GPU golden 的第 1 份）的 `data/batches`（注 1） | 01 `setup.md`「数据」 |
| 镜像 | MaxText 官方 `tpu_pre_training`，按 digest 钉死（注 2） | [`../frameworks/maxtext/run.sh`](../frameworks/maxtext/run.sh) |
| 对齐门的 golden | `g1k-a` / `g1k-b` / `g1k-c`（[`g1k`](../../../docs/glossary.md#g1k)：B200×8，1000 步，默认设置）（注 3） | 01 [`runs.yaml`](../alignments/01-v5p-acceptance/runs.yaml) |

- 注 1：清单 sha256 `e4db6392d89a…`，`dataset_type: "synthetic"`，不走 MaxText 数据管线。
- 注 2：首个 LKG 用哪个 digest 建立时写进 [`lkg.yaml`](lkg.yaml)。
- 注 3：只给 R25 当「不变差」的参照。迁移验收的最终 golden 是 `rcc`（见第四节 R25 行）。

T0–T2、T4 都用回放数据，所以**生产的数据管线不在这几档的覆盖里**（R28 的已知缺口，由 T3 补）。

## 二、平台档案

标准不分平台，下表是每个平台的参数。
两者都是 8 个 JAX 设备，对上 GPU 8 卡（[`../frameworks/maxtext/README.md`](../frameworks/maxtext/README.md)）。

| | v5p-8 | v7x-4 |
|---|---|---|
| 拓扑 / 主机 | `2x2x2`，2 台主机 | `2x2x1`，1 台主机（4 芯片，8 个 device） |
| 主切分（T0–T2 默认变体） | FSDP 8，`per_device_batch_size: 2`、`gradient_accumulation_steps: 1` | 同 v5p（注 1）；v7x 上的回归档位还没跑过（`run_regression.sh` 没在 v7x 上跑过，没有 LKG）。有一份 T1 档参数的 1000 步运行 `reg-v7x-t1`，是观测构建的基线，换代前配置（02 `runs.yaml`）；它的步时仓库里没有读出 |
| 对应的对齐实践 | [`01-v5p-acceptance`](../alignments/01-v5p-acceptance/README.md) | [`02-v7x-acceptance`](../alignments/02-v7x-acceptance/README.md) |
| T0 步数 / 时长 | 20 步 / 一次约 10 分钟（注 2） | 20 步 / 待实测 |
| T1 步数 / 时长 | 1000 步（= golden 长度）/ 一次约 25 分钟（注 3） | 同左 / 待实测 |
| T2 抓数档位 | 由 `--probe-tiers` 选一档自检逐位的；`full` 档不用（注 4） | 待实测 |
| TPU 执行噪声 | **正式**，8 份运行两两配对得到 28 对比较，现行配置 yml a238883b（注 5） | 待标定 |
| R22 步时阈值 | T0：0.63%；T1：0.57%（注 6） | 待标定 |

- 注 1：v7x-4 也是 8 个 device，FSDP 8 沿用 v5p-8 的写法，每个 device 拿到的行就和 GPU rank 对应。
  - 见 02 [`setup.md`](../alignments/02-v7x-acceptance/setup.md)，v7x 迁移验收按这个跑。
- 注 2：提交到取回。训练循环 182 s，第 1 步含编译 17.9 s，之后每步 0.90 s（`reg-t0-lkg1-a`，旧配置的 LKG；现行配置 `reg-t0-fa-a…e` 每步中位数 0.909 s，[`calibration-v5p-fa32.json`](results/calibration-v5p-fa32.json)）。
- 注 3：提交到取回。训练循环 1075 s，之后每步 0.898 s（`reg-t1-lkg2-a`，旧配置的 LKG；现行配置 `reg-t1-fa-a…e` 每步中位数 0.906 s，同上）。
- 注 4：`full` 档在 v5p 上实测不逐位（[`../frameworks/maxtext/README.md`](../frameworks/maxtext/README.md)「自检与生产轨迹」）。
- 注 5：TPU 执行噪声的细节。
  - 8 份运行：基线 + vmem64、vmem48、vmem40、vmem56、sabwd1k、sabwd256、combo（第五节）。
  - loss 三个量（带符号均值 / MARE / 最大单步）的上限：
    - T1（`1:1001`）4.76e-5 / 1.192e-4 / 7.16e-4；
    - T0（`1:21`）4.05e-6 / 9.60e-6 / 3.25e-5。
  - 旧配置 7704115b 下的读数（T1 5.90e-5 / 1.130e-4 / 9.12e-4）留在 `calibration.yaml` 前几条。
- 注 6：R22 步时阈值的细节。
  - 现行配置，各 5 次同一程序运行 `reg-t0-fa-a…e`、`reg-t1-fa-a…e`。
  - 算法：第 3 步起步时中位数的极差 ÷ 中位数。
  - 旧配置下是 0.83% / 0.73%。

### T0 为什么取 20 步

- 第 1 步之后动量和路由偏置已经累了十几步，「第 2 步以后才显出来」的改动（动量、偏置更新）也能看到。
- 运行时间以编译为主，多跑十几步不贵。
- 推断，按实测时长调。

## 三、档位

| 档 | 混元 3 的配置 | 触发 | 状态 |
|---|---|---|---|
| T0 冒烟 | 4 层，主切分，20 步 | 目标：每次合并到 `hunyuan`；现在手动 | 可手动跑：[`tiers/t0.yaml`](tiers/t0.yaml)、[`catalogs/regression-t0.yaml`](catalogs/regression-t0.yaml) |
| T1 主力 | 4 层，主切分，1000 步；R25 对基线 `g1k-a`（第一阶段 GPU golden 的第 1 份）（注 1） | 目标：每 3 小时，有新提交才跑；现在手动 | 可手动跑：[`tiers/t1.yaml`](tiers/t1.yaml)、[`catalogs/regression-t1.yaml`](catalogs/regression-t1.yaml)；v5p 首个 LKG 已建立 |
| T2 每日 | 4 层，1000 步，第 1 步抓张量；两个并行变体；R27 在第 500 步存档续训（注 2） | 每天一次 | 未实施 |
| T3 生产形状 | 生产层数（`hunyuan3-295b.yml`，80 层）、生产切分与切片、生产数据管线，100 步 | 每周一次起步 | 未实施（注 3） |
| T4 低精度配方 | — | — | 未启用：混元 3 目前没有 FP8 配方 |
| 金丝雀 | T1 配置，libtpu 换每日构建 | 每天一次 | 未实施；换 libtpu 前要做什么，见第七节末「换 libtpu」 |
| 对照臂 | T0 配置，LKG 的提交 | 每天一次；失败时用失败那档的配置 | 未实施 |

- 注 1：`g1k-b/c` 只组噪声带；窗口 `1:1001`（第 1–1000 步），分母 golden。
- 注 2：并行变体：
  - ① 主切分；
  - ② `per_device_batch_size: 1` + `gradient_accumulation_steps: 2`（同样对得上 GPU 的行分配，[`../frameworks/maxtext/README.md`](../frameworks/maxtext/README.md#数据回放)「数据回放」）。
- 注 3：生产配置（切分、重算、数据管线）还没拿到，生产形状的切片资源也没定；两样都有了才能定这一档。

### 触发节奏

- 「触发」一栏是目标节奏。
- 现在 TPU 预算不够每天跑，**全部手动触发**（第八节）。
  - 不建任何定时任务、调度器或 CI 定时作业。
  - 有预算后再上排程（第九节第 3 条）。
- T1 每 3 小时：标准给的范围是 2–4 小时，取中间值。
  - 1000 步实测时长出来后，按「一次跑完再留出余量」重定。

### 主切分为什么显式写 GA 1

- 主切分在两份档位配置里显式写成 `per_device_batch_size=2 gradient_accumulation_steps=1`。
- `hy3-4layers.yml` 现在默认是 GA 2（对齐用，按 GPU 的 microbatch 分两次前反向）。
- 而 T2 要在第 1 步抓张量，加钩子的训练步只支持 GA 1。

### 回归守的不是正式验收的那个程序

- T0、T1 跑的是 yml 默认的重算设置（各中间量都存 `device`，等于不重算）加 GA 1（`tiers/t0.yaml`、`tiers/t1.yaml` 的 `overrides`）。
- 正式 L4 的被测 `t-rmap-on-1000` 是 GA 2，命令行另加了 rmap 重算覆盖：除层输入和 `moe_routing_maps` 外全部 `remat`（01 [`runs.yaml`](../alignments/01-v5p-acceptance/runs.yaml)）。
- 两处都改数值：重算换舍入点（第五节 `rematqkv` 一行，旧配置 yml 7704115b 下 R20 `reduce-precision` 31 → 24；现行配置下没有重测重算前后的计数），累积分组也改舍入（对齐报告台账 #12）。
- 所以 T0、T1 绿了，只说明回归这个程序没变，不说明验收签字的那个程序没变。要守验收程序，得给 GA 2 + rmap 覆盖单独建一个变体和 LKG；建不建另行确定。

## 四、各行的具体内容

只列需要按模型填的行；其余行照标准。长的取值写在表下的注里。

代号：`t-hlo`（v5p 上导出 XLA dump 的 TPU 运行）、`g1k-a`（第一阶段 GPU golden 的第 1 份）。

| 行 | 混元 3 的取值 | 出处 |
|---|---|---|
| R03 | 每步每字段 16 个单元（8 rank × 2 microbatch），4 个字段，另加每步一个非 padding 总数：每步 65 项 | 01 `phase2-t-loss30` 的 `L0.data.fingerprint`：30 步每字段 480 个单元、共 1950 项 |
| R06 | 一组「必须等于」的配置值（见注 R06） | [`../config-alignment.md`](../config-alignment.md)、`hy3-4layers.yml` |
| R07 / R15 | 训练状态的 FP32 全量清单 `state_sha/{init,final}.json`（见注 R07 / R15） | 见注 R07 / R15 |
| R16 | MoE 层是第 1、2、3 层（第 0 层稠密）；① 偏置与负载统计，② top-8 翻转数（见注 R16） | [`../config-alignment.md`](../config-alignment.md#模型结构)「模型结构」 |
| R18 | 规则：读 FP32 状态清单，训练前后各查一遍（见注 R18） | 同上「数值精度」；二阶动量 FP32 与计数器的 dtype 出自 v5p 运行 `t-hlo` 的 `modules.txt` |
| R20 | 读运行目录 `hlo/train_step.after_optimizations.txt`（见注 R20） | `hy3-4layers.yml`；对照表「数值精度」「并行与确定性」；计数出自 v5p 运行 `t-hlo` 的 XLA dump |
| R25 | 只对基线 `g1k-a` 算三个量（`params.baseline`），不取各份最差；噪声参照：3 份 golden 两两 3 对（见注 R25） | 见注 R25 |
| 单步硬上限 | 10 × golden 两两在 `1:1000` 窗口上的最大单步上限；未启用（见注「单步硬上限」） | — |
| R28 | 生产配置与 T0–T2 不同的已知项（见注 R28）；由 T3 覆盖，T3 上线前列为已知缺口 | `hy3-4layers.yml` 头注释（与 `hunyuan3-295b.yml` 只差层数）；对照表「MTP」一行 |

### 表注（四）

- 注 R06：完整清单。
  - `dropout_rate: 0.0`、`use_random_routing: false`
  - `load_balance_loss_weight: 0.0`、`z_loss_multiplier: 0.0`
  - `mtp_num_layers: 0`
  - `adam_weight_decay: 0.033`、`adamw_mask: ["norm/scale"]`
  - `learning_rate: 1.0e-5`（常数）、`gradient_clipping_threshold: 1.0`、`skip_step_on_spikes: false`
  - `routed_bias_update_rate: 0.001`
  - `matmul_precision: "default"`、`float32_gate_logits: True`、`capacity_factor: -1.0`
  - `remat_policy: "custom"` 且 `moe_routing_maps: "device"`
  - `weight_dtype: "float32"`、`mu_dtype: "float32"`、`grad_dtype: "float32"`
  - `shard_optimizer_over_data: false`（打开后梯度累积的累加器是 BF16）
  - `use_tokamax_splash: false`（打开后要同时 `sa_use_base2_exp: False`，同上）
- 注 R07 / R15
  - 清单内容：FP32 主权重、两个动量、`expert_bias`、两个优化器计数器与步数，按存储 dtype 算 sha256。
    - 工具：`train_hy3.py --state-sha`，[`janus/dump/maxtext.py`](../../../janus/dump/maxtext.py) 的 `write_state_sha`。
  - 检查 `checkpoint_equal` 加 `source: state`。
  - v5p 上训练状态共 91 个数组叶子：参数、一阶动量、二阶动量各 29 个，`expert_bias` 1 个，计数器 3 个。
  - 出处：
    - 原来的 `params_sha` 把主权重转成 BF16 再算（`write_param_sha`），看不见半个 BF16 格以内的差（00 [`README.md`](../alignments/00-gpu-noise-and-drills/README.md) 第 11 条），只留给对 GPU 比。
    - 叶子个数出自 v5p 运行 `t-hlo` 的 `modules.txt`。
- 注 R16
  - ① 每层 `router.bias_mean`、`router.bias_std`：已实施，`router_stats` 前缀 `router.bias_`。
  - ① `router.load_cv`：**暂缓**，TPU 侧还没记专家负载（[`../frameworks/maxtext/README.md`](../frameworks/maxtext/README.md)「和 GPU 抓数的不同」）。
  - ② 每层 top-8（共 192 个专家）选中集合的翻转数；参与比较的 token 数以抓数的 `meta.json` 为准。
- 注 R18
  - 参数（MaxText 只存 FP32 主权重，`weight_dtype: "float32"`）、`expert_bias`（一个 [3, 192] 的叶子，3 个 MoE 层）、一阶动量、二阶动量全部 FP32。
  - 两个优化器计数器 int32、步数 uint32。
  - 每种 dtype 的个数与 LKG 相等由 R15 保证（两边状态路径集合必须相同）。
- 注 R20
  - 读的文件：生产训练步 `jit_train_step` 优化后的 HLO，`train_hy3.py --hlo-dir` 从 XLA dump 拷来。
  - 已实施的规则：
    - ① FP8 类型 = 0；
    - ② 随机数算子 = 0；
    - ④ 分组 GEMM（megablox，按 kernel 元数据里的 `tile_m` 数）、⑤ splash attention（按指令名 `splash_mha_*`）、全部 `tpu_custom_call`、all-reduce、`reduce-precision` 的个数与 LKG 相等。
  - v5p 上 `t-hlo` 的计数（旧配置 yml 7704115b；现行配置下 all-reduce 8、`reduce-precision` 63，见第五节）：`tpu_custom_call` 15、megablox 9、splash 6、all-reduce 9、`reduce-precision` 31、FP8 0、随机数 0（规则对着这份 HLO 核过）。
  - **暂缓**：③ 路由 GEMM 的操作数 dtype 与精度、⑥ `moe_routing_maps` 作为保存点。
    - 原因：优化后的 HLO 里路由矩阵乘被融进 fusion，还没有可靠的正则。
    - 先由 R21 指纹兜底（变了会报，但不会说是哪一条）。
- 注 R25
  - 它对的是 BF16、不重算的 [`g1k`](../../../docs/glossary.md#g1k)。
  - TPU 生产对它本来就带着 GPU 输出头 dgrad 那份偏差。
    - 正式验收第一层对最终 golden `rcc` 才在噪声内（01 `phase2-acceptance-rcc-on`）。
  - 所以只用「不变差」判，不要求 `pass`。
  - 出处：
    - [`goldens.md`](../goldens.md#比较协议几份-golden-只造噪声带正式比较只对一份)「比较协议」
- 注「单步硬上限」
  - 未启用：R25 现在只报不判（`hard: false`），所以这个上限还没有按 `g1k-a/b/c` 算出。
  - R25 改成判定之前要先算出它。
  - 参考量级：第一阶段 8 份 GPU 运行两两 28 对、窗口 `1:101` 的上限是 2.34e-4（00 `phase1-full-noise28` 的 `L4.loss`）。
    - 照此是 2.3e-3（推断，窗口不同，正式值以 `1:1000` 重算为准）。
- 注 R28
  - 层数（80 对 4）。
  - MTP（回归关；模型 yml 里是 1 层，生产开不开以生产配置为准）。
  - 切分与切片（含 CP：Hy3 生产在用，不同 CP 方案会留下不同的固定偏移）。
  - 数据管线。

## 五、TPU 自噪声标定

这批运行（扰动臂）曾给迁移验收第 1 步当 TPU 噪声参照，2026-11-08 起撤回。
每个扰动臂跑 T1 的 1000 步，第 1 步抓张量。
给日常回归的 T1 / T2 当执行噪声。迁移验收的 L2 / L3 不用它：2026-11-08 起第 1 步只用 GPU 第 1 步重跑 3 对当尺子，超出部分对 FP64 定责（尺子只来自参照一侧）。

### 步骤

| 步 | 做什么 | 判据 |
|---|---|---|
| 1 | 基线（主切分）同一提交跑两次 | 逐位（这也是 v5p 上首个 LKG 的建立） |
| 2 | 候选臂逐个过标准第五节的三关 | 三关见注 1；工具 [`calibrate.py`](calibrate.py)，`--golden` 不足 10 份时第三关标「暂定」 |
| 3 | 凑够 6 份运行（基线 + 5 个臂），两两配对得到 15 对比较 | 留一法互判没有一条独撑上限 |
| 4 | 结果写进本页第二节「TPU 执行噪声」，带运行名 | — |

- 注 1：第 2 步的三关。
  - 臂自己重跑逐位。
  - R02–R07、R14、R18、R20 对基线不变。
  - 1000 步 loss 对基线不大于 GPU 自噪声的上限；第 1 步参数梯度离 FP64 整步参照不比基线远（按能量合起来 R ≤ 1.2）。
    - 第 1 步参数梯度原来比 00 `phase1-full` 的 `L3.step1.grads.all` 上限（GPU 第 1 步噪声）。2026-11-06 改成比 FP64：只改执行方式的臂会翻近似平局的路由，GPU 重跑不翻，那把尺子量不到（标准第五节第三关）。当前配置 6 个臂（参照里的 5 个加 sa256v48）R：v5p 1.001 – 1.008，v7x 1.003 – 1.039（[`results/arms-step1-vs-fp64.json`](results/arms-step1-vs-fp64.json)）。这是判据用的按能量合计值；同一文件另记逐张量的最大值：v5p 1.01 – 2.91（vmem40 最大，sa256 最小），除 sa256 外最差都是 `decoder.layers.1.pre_mlp_layernorm.weight`；v7x 1.01 – 1.61。norm scale 这类小张量只改执行方式就能挪 1.7 – 2.9 倍，和[对齐报告](../ALIGNMENT-REPORT.md)第 5 节台账注 20「量级」里 norm 张量逐张量读数不稳是同一类，单个张量超线不能单独当证据。
    - loss 的上限取**至少 10 次**默认设置 GPU 1000 步运行两两组合的最大值。
    - 现在是 `g1k-a/b/c`（第一阶段 GPU golden，不开重算，1000 步，3 份）加 `g-ctl`、`g-ctl-b`、`g-ctl-c`、`g-gn-a`、`g-gn-b`、`g-rs-a`、`g-rs-b` 共 10 份运行，两两配对得到 45 对比较。
    - 后 7 份与 golden 只差日志选项。
    - `rc-*`（GPU 生产 golden）开了整层重算与路由回放，是另一种设置，不混进来。
    - 只用 `g1k-a/b/c` 的 3 对会低估，见 [`results/README.md`](results/README.md)。

### 现行配置的标定

**现状（2026-11-05，现行配置 yml a238883b）**：配置换代后按标准第五节重标。

- 7 个在旧配置下进了参照的臂在新配置下重跑，三关全过：
  - T0 双跑都逐位；
  - 第二关 R20 计数与新基线相同（新配置下 all-reduce 8、`reduce-precision` 63）；
  - 第三关见下表（loss 一半）；第 1 步梯度一半（FP64 版，2026-11-06 补）：sabwd256、sabwd1k、combo、vmem40、vmem56 对基线 R 1.001 – 1.008（按能量合计；逐张量最大见上面注 1），全过；vmem64、vmem48 没有单独补。这一半借的是验收那批第 1 步扰动运行 `t-st1n-<臂>-gm`（当前对齐配置，2 步，命令行覆盖未登记，见 01 `runs.yaml`），不是回归参照里的 `fa-*` 臂，也不是 T1 档程序（GA 1、不重算）；臂名对应（sa256 ↔ sabwd256 等）是按开关对的。回归 T1 程序上没补。
- 基线 + 7 个臂，8 份运行两两配对得到 28 对比较，`status: 正式`。
- 留一法最大的一对 T1 1.05 倍、T0 1.03 倍。
- 注意 vmem64 的第三关 MARE 是 GPU 上限的 0.999 倍，压线进参照。
  - 它也撑起了 T1 的 MARE 上限（1.192e-4）。
- 读数在 [`results/calibration-v5p-fa32.json`](results/calibration-v5p-fa32.json)。

### 旧配置的标定

**旧配置（yml 7704115b）下的标定**：第 1–4 步都做完。

- 第 2 步试了 14 个臂，7 个进参照。
- 加上基线一共 8 份运行、两两配对得到 28 对比较，到了第 3 步「≥ 15 对」的正式启用线。
- 留一法互判：最大的一对 T1 是其余各对最大值的 1.14 倍、T0 1.21 倍，没有一条独撑上限。
- 下表是那一轮的结论（之前 4 份两两 6 对时的「暂定」两条也留着，不删）。
- 第三关的「第 1 步参数梯度」一条要抓张量，`calibrate.py` 这一版没做，只判了 1000 步 loss（现行配置下 2026-11-06 用 FP64 版补做，见上一节注 1）。

| 臂 | 改什么 | 结论 |
|---|---|---|
| vmem64、vmem48、vmem40、vmem56 | `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536 / 49152 / 40960 / 57344` | 进参照 |
| sabwd1k、sabwd256 | splash 反向块 1024 / 256（`sa_block_q_dkv`、`sa_block_kv_dkv`、`sa_block_kv_dkv_compute`、`sa_block_q_dq`、`sa_block_kv_dq`） | 进参照 |
| combo | vmem48 + sabwd1k 两个开关一起开（标准第五节「单开关不够时用组合」） | 进参照 |
| vmem24、vmem32 | `…=24576 / 32768` | 不进：第三关带符号均值是 GPU 上限的 1.12 / 1.24 倍（未证实只改执行） |
| safwd1k | splash 前向块 1024（`sa_block_q`、`sa_block_kv`、`sa_block_kv_compute`） | 不进：第三关带符号均值 1.23 倍 |
| vmem80 | `…=81920` | 不进：和 vmem64 逐位（注 2），算不上新的一份 |
| rematqkv | `query_proj/key_proj/value_proj=remat` | 不进：第二关，R20 `reduce-precision` 31 → 24（重算换了舍入点） |
| cmoff | 关掉集合矩阵乘（注 3） | 不进：第二关，R20 all-reduce 9 → 16、`reduce-precision` 31 → 26 |
| sabwd2k | splash 反向块 2048 | 没有结论：两次提交都在写运行目录之前失败，放弃 |

- 注 2：优化后的 HLO、训练后状态都相同，编译器把上限截到了同一个值。
- 注 3：`--xla_tpu_enable_windowed_einsum_for_all_gather=false`、`…_for_reduce_scatter=false`。

读数、1–2 倍区间怎么判、为什么换了集群还能合在一份参照里，见 [`results/README.md`](results/README.md#tpu-执行噪声标定正式readmemd-第五节)「TPU 执行噪声标定」。

### 参照绑定与作废

**这份参照绑定什么**（标准第五节）：写在 [`calibration.yaml`](calibration.yaml) 每一条的 `binding`。

- 平台 v5p。
- 档位配置 sha256。
- 训练配置 `hy3-4layers.yml` 的 sha256（`config_yml_sha256`，取基线运行 `meta.config.sha256`）。
- 镜像与依赖版本。

[`regress.py`](regress.py) 判每次运行时都核这几样：

- 训练配置一项对被测的 `meta.config.sha256`；老条目没记这一项的按作废处理。
- 任何一样和被测不同，参照就作废：用到它的容差门从 `pass` 降为 `inconclusive`，直到重标。

和档位目录的关系：

- 档位目录 `catalogs/regression-t{0,1}.yaml` 的 `runs.noise` 必须和 `calibration.yaml` 最新一条的 `pairs` 相同（单测核）。
- 参照运行的位置也写在那里，[`run_regression.sh`](run_regression.sh) 本地没有时按它取回。

### 怎么重标

下面的命令是旧配置（yml `7704115b`）那次标定的写法，照抄会拿旧配置的运行当基线，`regress.py` 按 `config_yml_sha256` 会判这份参照作废。现行配置（`a238883b`）那次的基线是 `reg-t1-fa-a` / `reg-t0-fa-a`，臂是 `fa-*`，步时用 `reg-t0-fa-a … e` 与 `reg-t1-fa-a … e`，输出 `results/calibration-v5p-fa32.json`（参数以那份文件里记的运行名为准）。

```bash
python models/hunyuan3/regression/calibrate.py --root <运行根> --out models/hunyuan3/regression/results/calibration-v5p.json \
    --t1-base reg-t1-lkg2-a --t1-base-pair reg-t1-lkg2-b --t0-base reg-t0-r2 \
    --golden g1k-a,g1k-b,g1k-c,g-ctl,g-ctl-b,g-ctl-c,g-gn-a,g-gn-b,g-rs-a,g-rs-b \
    --arm vmem64=cal-vmem64-a,cal-vmem64-b,cal-vmem64-t1,xla_tpu_scoped_vmem_limit_kib … \
    --yaml models/hunyuan3/regression/calibration.yaml --locations <运行名 → 占位路径.json> --date <日期> \
    --step-time-runs reg-t0-lkg1-a,reg-t0-lkg1-b,reg-t0-r2,reg-t0-st1,reg-t0-st2
```

- `--yaml` 追加一条（T0、T1 各一条），然后把两份档位目录的 `runs.noise` 改成新条目的 `pairs`。
- 训练配置（`hy3-4layers.yml`）改了也要重标。
  - 基线变了，`regress.py` 会因为 `config_yml_sha256` 对不上把旧参照判作废。
- 改了目录里的检查（内容指纹变了；只改注释与说明文字不算），LKG 要在新目录下重判逐位后重新登记。
  - [`lkg.yaml`](lkg.yaml) 里 2026-11-05 的四条就是这样来的。
- T1 的 R22 阈值另给 `--t1-step-time-runs`。

### 候选臂

候选臂：

- `remat_policy` 换一档（`moe_routing_maps` 仍存 `device`）；
- splash attention 块大小 `sa_block_q` / `sa_block_kv` 换一档；
- 编译器片上内存上限换一档。

`per_device_batch_size=1` + `gradient_accumulation_steps=2` **不当臂**：

- 第一阶段 GPU 上改累加分组，embedding 梯度差到噪声上限的 31 倍（00 README 第四节）。
- 它在 T2 里是独立的并行变体，有自己的 LKG。

单开关凑不够 5 个时，用组合（标准第五节）。

### v7x

- v7x 照同样的步骤另标一遍，不用 v5p 的数。
- 切分已定（第二节）。
- 还没做，等 v7x 的回归档位开跑（`reg-v7x-t1` 是换代前配置的观测基线，不走 `run_regression.sh`，不算回归档位）。

## 六、LKG 登记

[`lkg.yaml`](lkg.yaml)：每个「平台 × 档位 × 并行变体」一条，字段见标准第七节。

- 被取代、没批也不再补批的旧条目标 `status: superseded`，保留作历史。
  - [`run_regression.sh`](run_regression.sh) 不拿它当默认 LKG。
- `run_regression.sh --first-lkg` 打印条目，批准后追加。
- 数值没变（回归门逐位通过）时指针前移，记一条、不用批准。
- **数值变了时的换代要确认**（第九节第 1 条）。

## 七、上线前要补的东西

按顺序，前面的不做，后面的判定不可信：

| # | 要补什么 | 为什么 | 落在哪 | 状态（2026-11-05） |
|---|---|---|---|---|
| 1 | v5p 上 T0、T1 同一提交双跑逐位 | 「同程序重跑逐位」是全部逐位门的前提（注 1） | 首个 LKG | T0、T1 都已做（注 2） |
| 2 | FP32 状态清单（主权重、两个动量、`expert_bias`） | R07、R15 不能用 BF16 清单 | [`janus/dump/maxtext.py`](../../../janus/dump/maxtext.py) 的 `write_state_sha`；`checkpoint_equal` / `dtype_manifest` 的 `source: state` | 已实施，单测 [`tests/models/hunyuan3/test_regression.py`](../../../tests/models/hunyuan3/test_regression.py) |
| 3 | 从运行进程读出全部开关；从实际执行的可执行文件导出优化后 HLO | R05、R20、R21 | `train_hy3.py` 写进 `meta.json`（`flags`、`maxtext_config_full`）与运行目录 `hlo/` | 已实施；R20 的 ③⑥ 暂缓（第四节） |
| 4 | 训练代码仓库的 PR 模板加 `## 数值影响` 一栏 | 标准第三节 | `maxtext` 的 PR 模板与 `pr-check` | 已加（第九节第 4 条） |
| 5 | TPU 执行噪声标定 | 执行级判定和「不变差」都要它 | 第五节 | v5p 在现行配置（yml a238883b）下已正式标定（注 3）；v7x 未做 |
| 6 | 回归目录 `catalogs/regression-<档>.yaml`；手动入口 | 跑起来 | 本目录 | T0、T1 已实施；**排程暂缓**，等 TPU 预算（第九节第 3 条） |

- 注 1：前提见 [`../../../docs/reference/environment.md`](../../../docs/reference/environment.md#tpu-侧)「TPU 侧」；v5p 两台主机上已实测（注 2）。
- 注 2：见 [`results/README.md`](results/README.md)。另一套 v5p-8 切片上同一提交对 LKG 也逐位（`reg-t0-st1/st2`）。
- 注 3：8 份运行两两 28 对，配置换代后重标过一次。容差门 `threshold: calibrated`，参照在 [`calibration.yaml`](calibration.yaml)。

### 换 libtpu

- 现行 LKG 和全部验收运行都在同一个 libtpu 构建上（`lkg.yaml`；与混元 4 的 libtpu 0.0.46 是同一构建）。
- 换版本不能假定只换求和顺序：混元 4 从 0.0.46 换到 0.0.52 后，scatter 排序门槛和 RMSNorm 的舍入都变了（[`libtpu-052.md`](../../hunyuan4/alignments/04-tpu-self/libtpu-052.md)）。
- 照那一页的三步做：只换 libtpu 建镜像；CPU 离线编译，看训练步程序变了什么（词嵌入 scatter 排不排序、`reduce-precision` 与 all-reduce 的个数）；上卡重建 LKG，重跑 T1，并对 GPU golden 重跑 L4。

## 八、手动跑

一条命令跑一档：提交训练 → janus 对 LKG → 写回归报告 → 上传结果。

- 提交训练复用 [`../frameworks/maxtext/run.sh`](../frameworks/maxtext/run.sh)，Pod 只读、运行目录由提交机取回再上传。

```bash
export BUCKET=gs://<POC 桶> CONTEXT=<TPU 集群的 kubectl 上下文>
source <(bash models/hunyuan3/frameworks/maxtext/stage.sh <maxtext 克隆> [提交，默认 origin/hunyuan])
# 建首个 LKG：同一提交跑 <名>-a、<名>-b 两次，b 对 a 判；全部逐位才是候选，打印 lkg.yaml 条目
bash models/hunyuan3/regression/run_regression.sh T0 <名> --first-lkg
# 日常：被测对 LKG（不给 --lkg 就取 lkg.yaml 里这一档的最新一条）
bash models/hunyuan3/regression/run_regression.sh T0 <名> [--lkg <LKG 运行名>]
# PR 声明了「会改数值」：带上声明与范围（标准第三节），报告按声明改判
bash models/hunyuan3/regression/run_regression.sh T0 <名> --declare 数学级 --scope R02,R05,R15,R20
```

### 本目录的文件

| 文件 | 作用 |
|---|---|
| [`tiers/t0.yaml`](tiers/t0.yaml)、[`tiers/t1.yaml`](tiers/t1.yaml) | 档位配置：步数、`train_hy3.py` 参数、MaxText 命令行覆盖、用哪份目录、要不要取 golden。sha256 登记在 [`lkg.yaml`](lkg.yaml)，改了就是改档位定义 |
| [`catalogs/regression-common.yaml`](catalogs/regression-common.yaml) | 两档共用的检查（T0 的全部行），只写一处（注 1） |
| [`regress.py`](regress.py) | 按行汇总 janus 的结果、按 PR 声明改判、算 R22 / R23 / R25、写报告首屏、打印 `lkg.yaml` 条目（注 2） |
| [`run_regression.sh`](run_regression.sh) | 手动入口（上面的命令）（注 3） |
| [`lkg.yaml`](lkg.yaml) | LKG 登记与换代记录 |
| [`results/`](results/) | 手动试跑阶段每次运行的 `report.md` 与 `regression.json`（小文件）。排程上线后结果只进桶，这里不再放 |

- 注 1：`regression-t0.yaml` / `regression-t1.yaml` 用 `include` 引入、`vars` 给步数与窗口，T1 另加 R12、R13、R16①、R25。
- 注 2：`regress.py` 的细节。
  - 按 PR 声明改判：标准第三节的表。
  - R22（步时，只报）与 R23（覆盖）、R25「不变差」。
  - 报告首屏：标准第十一节。
- 注 3：`run_regression.sh` 的提交规则。
  - 同一个节点池上有别的 `janus-*` JobSet 在跑或排队时默认不提交（别的池上的不算）；只删自己提交的 JobSet。
  - 回归最好用专用节点池（`NODEPOOL=<回归专用节点池>`），不和对齐实验抢同一个切片。
  - 集群特有的提交设置走站点配置 `JANUS_SITE_PROFILE`（见 [`../frameworks/maxtext/README.md`](../frameworks/maxtext/README.md#站点配置janus_site_profile)「站点配置（`JANUS_SITE_PROFILE`）」）。
  - 「有别的作业在跑」与删 JobSet 也按站点配置的规则。

### 结果放在哪、怎么判

- 运行目录与结果放在桶里 `janus/hunyuan3-4l/regression/<平台>/<档>/<日期>/`：
  - 运行目录在 `runs/<运行名>/`；
  - 结果（janus 的 `results.json`、`report.md`，回归的 `regression.json`、`report.md`）在 `<运行名>/`。
- 判法：janus 跑的是「数学不变」的目录；`--declare` / `--scope` 只改报告里的判定，不改目录。
  - `role: tolerance` 的几条容差门（门槛 `calibrated`，TPU 执行噪声，第五节）只在「执行级」时采用。
  - 「数学级」时范围内的数值行只报；范围外的行不逐位就写明「PR 退回」。
- 报告首屏的「对照臂」一栏：建首个 LKG 时就是同一提交双跑。
  - 其余手动运行没跑对照臂，数值失败时先补跑一次 LKG 的提交再下结论（标准第八节）。
- 执行级声明下，R09（第 1 步范数）与 R15（训练后状态）没有容差门：不逐位时这两行是 `inconclusive`。
  - 所以执行级的 T1 现在整体最多「无结论」。
  - 容差门各自给出 pass / inconclusive / fail（[`results/README.md`](results/README.md)「容差门给真结论」）。
- 退出码：0 通过、1 数值失败、2 工具或跑不完、3 无结论（有待标定的门槛，或没有容差门的行不逐位）。

### 每一行从哪来、哪些暂缓

| 行 | 怎么实现 | 暂缓的部分 |
|---|---|---|
| R01 | `run_complete`：步数、重复、缺步、`meta.restarts` | — |
| R02 | `config_diff`：权重清单、数据清单、`hy3-4layers.yml` 的 sha256 钉死；命令行覆盖、步数、载入路径对 LKG；档位配置的 sha256 与目录的内容指纹由 [`regress.py`](regress.py) 对 `lkg.yaml` | — |
| R03 | `data_fingerprint` 对 LKG，全部步；单元数由 `expect` 核（T0 1300、T1 65000） | — |
| R04 | `config_diff`：镜像 digest、`versions`、硬件；提交号只列出 | — |
| R05 | 运行进程与编译器两份配置都对 LKG（注 1） | 「编译进程与运行进程互比」（注 1） |
| R06 | `config_diff` 的 `equals`，读完整配置（第四节的清单，另加 `enable_dropout`、主切分两项） | — |
| R07、R15 | FP32 状态清单（第四节） | — |
| R08、R09、R14 | `determinism`（R08 / R09 只比第 1 步） | — |
| R12、R13 | `loss_series` `exact`：四个量 + 第一个不同的步（T1） | — |
| R16 | ① 偏置均值 / 标准差 `router_stats` `exact`（T1） | `router.load_cv`；② 属 T2 |
| R17 | `health_signals`：非有限值、突跳 | 跳步数：不另数（注 2） |
| R18 | `dtype_manifest` 读状态清单 | — |
| R20 | `artifact_check` 计数对 LKG（第四节） | ③ 路由 GEMM 精度、⑥ 路由结果保存点 |
| R21 | `artifact_check` 指纹（只报） | — |
| R22 | `regress.py`：稳定窗口（去掉前 2 步）步时中位数对 LKG，只报；阈值 T0 0.63%、T1 0.57%（注 3） | 峰值 HBM、编译期临时区还没记 |
| R23 | `regress.py`：每行都出了结论、`expect` 的单元数对得上 | — |
| R24 | 不在档位运行里；建首个 LKG 的双跑起同样作用 | 例行对照臂随排程上线 |
| R25 | `multi_golden` 对基线 `g1k-a`（第一阶段 GPU golden 的第 1 份）（只报）；`regress.py`：逐位时继承 LKG，否则「不变差」（注 4） | — |
| R10、R11、R16②、R19、R27、R28 | 属 T2 / 每周，未实施 | 全部 |

- 注 1：R05
  - 运行进程：启动时与第一步编译后各一份环境变量与 `jax.config`、展开后的完整 MaxText 配置，逐子键对 LKG（运行名、路径、主机名不比）。
  - 编译器：XLA dump 的 `flagfile` 与 `debug_options` 整份对 LKG。
  - 暂缓「编译进程与运行进程互比」：本档位不做提前编译，两者是同一个进程，没有第二份可比。
- 注 2：R17 跳步数：`skip_step_on_spikes` 由 R06 断言为 false，训练循环里没有跳步路径，不另数。
- 注 3：R22 阈值是现行配置 yml a238883b 下的，取自 [`calibration.yaml`](calibration.yaml) 最新两条的 `step_time_threshold`；旧配置下是 0.83% / 0.73%。
- 注 4：R25
  - `multi_golden`：`baseline: g1k-a`，`g1k-b/c` 只组噪声带。
  - `noise_from: golden`：目录里的 `runs.noise` 是 TPU 执行噪声，这一条不换尺子。
  - `regress.py`：回归门逐位时继承 LKG，否则「不变差」。
    - 对 golden 的三个量，新值 − LKG 值 ≤ TPU 执行噪声上限。

### 编译缓存与试跑结果

- 持久编译缓存：MaxText 默认 `jax_cache_dir: ~/jax_cache`，在 Pod 里。
  - 每次都是新 Pod，所以回归运行都是冷编译（标准第十二节「持久编译缓存」）。
- **试跑结果**：见 [`results/README.md`](results/README.md)。

## 九、决定（2026-11-04）

1. **LKG 换代**：数值变了（会改数值、升级依赖）的每一次换代都要确认；数值没变时指针前移，不用确认。
2. **红灯期间**：某一档红了、还没结论时，只冻结声明「会改数值」与「升级依赖」的 PR。
   - 「数学不变」的 PR 照常合并。
   - 它们照样按逐位对同一个 LKG 判，报告里看第一个不同的量变没变，分清是不是同一个原因。
3. **不排程**：现在只手动跑，第三节的节奏表保留为目标。
   - 排程（定时任务、调度器、CI 定时作业）等有 TPU 预算再做。
4. **PR 声明**：`maxtext` 的 PR 模板加 `## 数值影响` 一栏，`pr-check` 核它。
   - 动了训练代码或依赖的 PR 必须三选一。
   - 「会改数值」要写幅度、范围、理由；「升级依赖」要写版本。
   - janus 自己的 PR 不加这一栏：janus 不进训练图，它的改动由 R26（工具自回归）管。
