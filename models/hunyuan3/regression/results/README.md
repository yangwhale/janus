[janus](../../../../README.md) › [用户指南](../../../../docs/guide/README.md) › [混元 3 日常回归](../README.md) ｜ 手动试跑记录

# 混元 3 · 日常回归 · 手动试跑记录

排程上线前的几次手动运行，用来确认流程能用（`../README.md` 第八节）。

- 每次运行一个子目录 `<运行名>/`，由 `../run_regression.sh` / `../regress.py` 写出：
  - `report.md`：报告首屏 + 每行状态；
  - `regression.json`：同一份数据。
- janus 的完整结果与运行目录在桶里（`janus/hunyuan3-4l/regression/v5p/T0/20261104/`），不进仓库。

本页结构：先是 T0 的四次试跑与读数、T1 首个 LKG；然后是现行配置（yml a238883b）的换代；最后是旧配置（yml 7704115b）下的噪声标定，留作历史。

## T0 试跑

- 全部是 T0（4 层、主切分、20 步），v5p-8（2x2x2），镜像与依赖同 `../lkg.yaml` 的第一条。
- `reg-t0-lkg1-a` 在对齐实验用的节点池，其余在回归专用节点池（都是 TPU v5p-8 切片）。

| # | 做什么 | 被测 → LKG | 结论 | 结果 |
|---|---|---|---|---|
| 1 | 建首个 LKG：同一提交双跑 | `reg-t0-lkg1-b` → `reg-t0-lkg1-a`（maxtext `hunyuan@4d4ffac2`） | **PASS**，15 行全部逐位 / 相等，`reg-t0-lkg1-a` 登记为首个 LKG | [`reg-t0-lkg1/`](reg-t0-lkg1/report.md) |
| 2 | 「数学不变」复测（注 1） | `reg-t0-r2`（maxtext `237a3270`）→ `reg-t0-lkg1-a` | **PASS**，逐位；LKG 指针前移到 `reg-t0-r2`（逐位，不用批准） | [`reg-t0-r2/`](reg-t0-r2/report.md) |
| 3 | 故意打开 `routed_gate_bf16_rounding=true`，按「数学不变」判 | `reg-t0-r3` → `reg-t0-r2` | **数值 FAIL（被拦）**；**训练数值本身逐位相同**（注 2） | [`reg-t0-r3/`](reg-t0-r3/report.md) |
| 3′ | 同一份结果按「会改数值 · 数学级」、范围 R02,R05,R08,R09,R15,R20,R21 改判 | 同上 | PASS：声明范围内的行只报 | [`reg-t0-r3/declared/`](reg-t0-r3/declared/report.md) |
| 4 | 另开一个精度开关 `rmsnorm_single_rounding=true`（确认数值门在 TPU 上真能拦） | `reg-t0-r4` → `reg-t0-r2` | **数值 FAIL**（注 3） | [`reg-t0-r4/`](reg-t0-r4/report.md) |

- 注 1：窗口里合进了 MaxText 一个只改 `.github` 的提交、janus 两个提交。
- 注 2：第 3 行的细节。
  - 被拦的行：R02（命令行覆盖）、R05（完整配置里正好这一个键）、R20（`reduce-precision` 31 → 33）。
  - R21 指纹变。
  - 训练数值逐位：R07–R09、R14、R15 全过。
- 注 3：第 4 行的细节。
  - R09 第 1 步 `raw_grad_norm` 不同。
  - R15 训练后 91 个状态叶子里 88 个不同。
  - R20 all-reduce 9 → 8、`reduce-precision` 31 → 56。
  - R02、R05 同上。

## 读数

### 同程序重跑逐位（第 1 行）

- 同一提交、两个节点池各跑一次。
  - 全部逐位：FP32 全量状态（训练前、训练后各 91 个叶子）、逐步 loss / 范数 / 学习率、1300 项输入指纹、优化后的 HLO、编译器的 flagfile。
  - 这是标准第七节要的「这一档在这个平台上同程序重跑逐位」的实测，v5p T0 上成立。
- 双跑里唯一不同的是 XLA `debug_options` 里一行 GPU 编译器选项 `xla_gpu_experimental_autotune_cache_mode`（a 有、b 没有）。
  - TPU 编译不读 `xla_gpu_*`，flagfile 与 HLO 指纹都相同。
  - 所以 R05 的 `debug_options` 比对改为去掉 `xla_gpu_*` 行再比（`artifact_check` 的 `ignore_lines`）。
  - 第一次判这对运行时它报了 fail，改后重判通过。

### 第 3、4 行

- **第 3 行为什么数值没变**：报告只说明「开关打开了、HLO 多了两个 `reduce-precision`、训练状态逐位相同」。
  - 原因：`matmul_precision=default` 下 TPU 的 FP32 矩阵乘本来就把操作数按 BF16 算，先把路由权重舍入成 BF16 再乘不改变结果。
  - 这一点在 v5p 上验证过：DEFAULT 的结果与「核先舍 BF16 再乘」逐位相同（01 `phase2-l1-l2-replay` 的 `router_precision_probe`，`default_eq_rounded_kernel_frac` = 1，3 层；[`../../config-alignment.md`](../../config-alignment.md) 注 10、注 21）。
  - CPU 上同一开关会改数值（CPU 冒烟里训练后 91 个叶子 88 个不同），因为 CPU 的默认精度是完整 FP32。
  - 不论原因，回归门在「数学不变」声明下把它拦住了：配置变了、编译产物变了，声明却说什么都没变。
- **第 4 行**：第 1 步 loss 逐位相同而第 1 步裁剪前梯度范数不同（第一个不同的量是 R09）。
  - 20 步 loss 对 LKG 的 MARE 2.06e-5、最大单步 6.34e-5（第 11 步）。
  - 第一个不同的 loss 在第 2 步（`R08.loss_tolerance`，门槛待标定，只报）。

### 步时

- 步时（R22，只报）：`reg-t0-r2` 第 3–20 步中位数 0.8987 s，LKG `reg-t0-lkg1-a` 0.8981 s，差 +0.07%。

## T1（1000 步）首个 LKG

| # | 做什么 | 被测 → LKG | 结论 | 结果 |
|---|---|---|---|---|
| 5 | T1 首个 LKG：同一提交（maxtext `237a3270`）双跑 | `reg-t1-lkg2-b` → `reg-t1-lkg2-a` | **PASS**，19 行全部逐位或相等（注），登记为 T1 首个 LKG | [`reg-t1-lkg2/`](reg-t1-lkg2/report.md) |

- 注：19 行包括 1000 步 loss、两个范数、6 个路由偏置统计、FP32 全量状态、65000 项输入指纹、HLO。

- **第一次尝试**：`reg-t1-lkg1-a` 跑到第 310 步左右时，回归节点池的节点被平台回收（基础设施中断）。
  - Pod 没打出结束标记，R01 不过，与被测无关，换名字重跑。
- **时长**：训练循环 1075 s（`meta.wall_seconds`，含训练前后两次 FP32 状态清单）。
  - 第 1 步 18.3 s（含编译），第 3–1000 步步时中位数 0.898 s。
  - 提交到取回约 25 分钟。
- **R25**（对 GPU golden，只报）：回归门逐位，结论继承 LKG。
  - LKG 自己对 `g1k-a/b/c` 的三个量：带符号均值 1.40e-4、MARE 1.60e-4、最大单步 6.92e-4，都超过 golden 两两的噪声上限两倍。
  - 这是预期的：`g1k` 是不重算的 BF16 GPU 程序，带着 GPU 输出头 dgrad 长求和的偏差。
  - 迁移验收对最终 golden `rcc-a`（噪声取 `rcc-a..j` 10 份重跑两两配对的 45 对比较）才在噪声内。
  - 所以 R25 只用「不变差」判（`../README.md` 第四节 R25，[对齐报告](../../ALIGNMENT-REPORT.md)第 3 节）。

## 配置换代：dense_fp32_accumulation（2026-11-05，现行配置 yml a238883b）

对齐 / 生产配置换成 `hy3-4layers.yml` a238883b。

- 打开 `dense_fp32_accumulation`（FSDP 切开收缩维的投影改用 FP32 累加）。
- 同时带上 #62 起的三个舍入开关：
  - `routed_gate_bf16_rounding`（路由核先舍 BF16 再乘）；
  - `rmsnorm_single_rounding`（RMSNorm 只舍一次）；
  - `mlp_activation_round_per_op`（门控激活按算子各舍一次）。
- maxtext 钉到 hunyuan@0622774a。
- 这是数学级改动，按标准第五、七节重建 LKG、重标噪声。
- 目录同步改动：
  - R02 钉 a238883b；
  - R06 断言这 4 个开关为 true；
  - R25 改为只对基线 `g1k-a`（第一阶段 GPU golden 的第 1 份）算（`params.baseline`，[`docs/reference/judging.md`](../../../../docs/reference/judging.md#golden-怎么用几份只造噪声带比较只对一份)「golden 怎么用」）。

### 新 LKG 与换代留档

| # | 做什么 | 被测 → 参照 | 结论 | 结果 |
|---|---|---|---|---|
| 1 | T1 新 LKG：同一提交双跑 | `reg-t1-fa-b` → `reg-t1-fa-a` | **PASS**，19 行全部逐位；`-c/-d/-e` 三次同程序也逐位 | [`reg-t1-fa/`](reg-t1-fa/report.md) |
| 2 | T0 新 LKG：同一提交双跑 | `reg-t0-fa-b` → `reg-t0-fa-a` | **PASS**，15 行全部逐位；`-c/-d/-e` 也逐位 | [`reg-t0-fa/`](reg-t0-fa/report.md) |
| 3 | 换代留档：新配置对旧 LKG，按「会改数值 · 数学级」 | `reg-t1-fa-a` → `reg-t1-lkg2-a` | 范围内的行只报；**R25 不变差 pass**；R06 / R23 报 error（注） | [`fa-vs-old-t1/`](fa-vs-old-t1/report.md)、[`fa-vs-old-t0/`](fa-vs-old-t0/report.md) |

- 注：第 3 行的细节。
  - 范围内只报的行：R02 yml、R05 配置（6 个子键）、R08 / R09 第 1 步 loss 与梯度范数、R12 / R13 / R16、R15 训练后 91 个状态叶子 88 个不同、R20 all-reduce 9 → 8、`reduce-precision` 31 → 63。
  - **R25 不变差 pass**：对 `g1k-a`（第一阶段 GPU golden 的第 1 份）的带符号均值 / MARE / 最大单步：
    - 1.30e-4 / 1.53e-4 / 6.65e-4 → 1.37e-4 / 1.74e-4 / 6.91e-4；
    - 增量 +6.99e-6 / +2.11e-5 / +2.67e-5；
    - 都小于新的 TPU 执行噪声上限 4.76e-5 / 1.19e-4 / 7.16e-4。
  - R06 / R23 报 error：旧 LKG 的 maxtext 没有 `dense_fp32_accumulation` 这个键，跨版本比较的预期结果。

### 新配置下的噪声参照

7 个臂在新配置下重跑，三关全过。

- 读数 [`calibration-v5p-fa32.json`](calibration-v5p-fa32.json)。
- `../calibration.yaml` 最新两条，`binding.config_yml_sha256` = a238883b。

| 臂 | 第三关（T1 对新基线 `reg-t1-fa-a`：带符号均值 / MARE / 最大单步，GPU 上限的倍数） |
|---|---|
| vmem64 | 5.23e-6（0.10）/ 1.192e-4（**0.999**）/ 5.09e-4（0.58） |
| vmem48 | 1.10e-5（0.20）/ 1.10e-4（0.92）/ 5.42e-4（0.62） |
| vmem40 | 2.45e-5（0.45）/ 1.10e-4（0.92）/ 4.71e-4（0.54） |
| vmem56 | 1.47e-5（0.27）/ 1.06e-4（0.89）/ 4.36e-4（0.50） |
| sabwd1k | 1.83e-5（0.34）/ 1.09e-4（0.91）/ 5.48e-4（0.63） |
| sabwd256 | 2.23e-5（0.41）/ 1.12e-4（0.94）/ 5.16e-4（0.59） |
| combo | 2.31e-5（0.43）/ 1.05e-4（0.88）/ 4.70e-4（0.54） |

| 档 | 带符号均值 | MARE | 最大单步 | 留一法（各对 r 的最大值） | R22 阈值 |
|---|---|---|---|---|---|
| T1 `1:1001` | 4.76e-5 | 1.192e-4 | 7.16e-4 | 1.05 / 1.04 / 1.01 | 0.57%（`reg-t1-fa-a…e`） |
| T0 `1:21` | 4.05e-6 | 9.60e-6 | 3.25e-5 | 1.03 / 1.00 / 1.01 | 0.63%（`reg-t0-fa-a…e`） |

- vmem64 的 MARE 压在 GPU 上限的 0.999 倍，照规则（≤ 1 倍）进参照。
  - T1 的 MARE 上限就是它这一对撑起来的。
- 第二关：7 个臂对新基线的 R20 计数都相同（all-reduce 8、`reduce-precision` 63），配置只差各自声明的键。
- `fa-vmem40-b` 第一次提交没起来（没打出结束标记，属基础设施），换名 `fa-vmem40-br2` 重跑。
- 两条新 LKG（`reg-t0-fa-a`、`reg-t1-fa-a`）写「待项目负责人确认」；2026-11-05 已经项目负责人批准。

下面是旧配置（yml 7704115b）下的标定与 LKG 记录，留作历史。

## TPU 执行噪声标定（正式，`../README.md` 第五节）

### 怎么跑

- 基线：T0 `reg-t0-r2`、T1 `reg-t1-lkg2-a`。
- 每个扰动臂跑三样：
  - T0 双跑（第一关：臂自己逐位）；
  - T0 对基线（第二关：只改执行）；
  - T1 一次（第三关：1000 步 loss 对基线不大于 GPU 自噪声上限）。
- 先跑 T1：第三关不过、或 T1 对基线的 R20 计数已经变了（第二关注定不过）的臂，不再跑 T0 双跑。

### 第三关的 GPU 自噪声上限

取至少 10 次默认设置 GPU 1000 步运行两两组合（≥ 45 对）的最大值。

- 窗口 `1:1001`，分母 golden，`calibrate.py` 算。
- 只用 3 份 golden 会低估：

代号：`g1k-a/b/c`（第一阶段 GPU golden，不开重算，1000 步，3 份）；`g-ctl*`、`g-gn-*`、`g-rs-*` 是扩充噪声带用的 GPU 对照运行（见表下注）。

| GPU 运行 | 对数 | 带符号均值 | MARE | 最大单步 |
|---|---|---|---|---|
| `g1k-a/b/c` | 3 | 3.14e-5 | 1.091e-4 | 6.34e-4 |
| `g1k-a/b/c` + `g-ctl`、`g-ctl-b`、`g-ctl-c`、`g-gn-a`、`g-gn-b`、`g-rs-a`、`g-rs-b`（注） | **45** | **5.39e-5** | **1.193e-4** | **8.72e-4** |

- 注：后 7 份与 golden 只差日志选项：`group_norms`、`row_stats_every`、`dump_steps`。
- 10 份运行各自对 `g1k-a` 在第 301–1000 步的带符号均值从 −6.5e-5（`g-ctl-b`）到 +2.4e-6（`g-ctl`）。
- `rc-a…rc-e`（GPU 生产 golden）开了整层重算与路由回放，是另一种设置，不能当这份参照。
  - 只有同一种设置的运行才互为噪声参照。

### 各臂的三关

| 臂 | 改什么 | 第一关 | 第二关 | 第三关（T1 对基线：带符号均值 / MARE / 最大单步，括号里是 GPU 上限的倍数） | 结论 |
|---|---|---|---|---|---|
| vmem64 | `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536` | 过 | 过 | 2.76e-5（0.51）/ 1.03e-4（0.86）/ 5.13e-4（0.59） | **进参照**（[`cal-vmem64-t1/`](cal-vmem64-t1/report.md)） |
| vmem48 | `…=49152` | 过 | 过 | 3.46e-5（0.64）/ 1.03e-4（0.86）/ 4.60e-4（0.53） | **进参照**（[`cal-vmem48-t1/`](cal-vmem48-t1/report.md)） |
| vmem40 | `…=40960` | 过（`cal-vmem40-a/b` 逐位） | 过 | 1.51e-5（0.28）/ 1.04e-4（0.87）/ 5.67e-4（0.65） | **进参照**（[`cal-vmem40-t1/`](cal-vmem40-t1/report.md)） |
| vmem56 | `…=57344` | 过（`cal-vmem56-a/b` 逐位） | 过 | 8.17e-6（0.15）/ 1.06e-4（0.89）/ 6.19e-4（0.71） | **进参照**（[`cal-vmem56-t1/`](cal-vmem56-t1/report.md)） |
| sabwd1k | splash 反向块 1024（`sa_block_*_dkv`、`sa_block_*_dq`） | 过 | 过（注 1） | 1.31e-5（0.24）/ 1.00e-4（0.84）/ 5.45e-4（0.63） | **进参照**（[`cal-sabwd1k-t1/`](cal-sabwd1k-t1/report.md)） |
| sabwd256 | splash 反向块 256 | 过（`cal-sabwd256-a/b` 逐位） | 过 | 2.44e-5（0.45）/ 1.02e-4（0.86）/ 4.21e-4（0.48） | **进参照**（[`cal-sabwd256-t1/`](cal-sabwd256-t1/report.md)） |
| combo | vmem48 + sabwd1k 一起开 | 过（`cal-combo-ar2` / `cal-combo-b` 逐位） | 过 | 9.16e-6（0.17）/ 1.02e-4（0.86）/ 5.08e-4（0.58） | **进参照**（[`cal-combo-t1/`](cal-combo-t1/report.md)） |
| vmem24 | `…=24576` | 过 | 过 | **6.02e-5（1.12）**/ 1.09e-4（0.91）/ 4.82e-4（0.55） | 不进参照（未证实；[`cal-vmem24-t1/`](cal-vmem24-t1/report.md)） |
| vmem32 | `…=32768` | 没跑 | 没跑（T1 上 R20 不变） | **6.70e-5（1.24）**/ 1.14e-4（0.95）/ 6.04e-4（0.69） | 不进参照（未证实；[`cal-vmem32-t1/`](cal-vmem32-t1/report.md)） |
| safwd1k | splash 前向块 1024 | 没跑 | 没跑（T1 上 R20 不变） | **6.61e-5（1.23）**/ 1.18e-4（0.99）/ 5.44e-4（0.62） | 不进参照（未证实；[`cal-safwd1k-t1/`](cal-safwd1k-t1/report.md)） |
| vmem80 | `…=81920` | — | — | 与 vmem64 逐位（注 2） | 不进参照：同一条轨迹，算不上新的一份（[`cal-vmem80-t1/`](cal-vmem80-t1/report.md)） |
| rematqkv | `query_proj/key_proj/value_proj=remat` | 没跑 | **不过**：R20 `reduce_precision` 31 → 24 | 1.98e-5（0.37）/ 9.80e-5（0.82）/ 4.45e-4（0.51） | 不进参照（[`cal-rematqkv-t1/`](cal-rematqkv-t1/report.md)） |
| cmoff | 关集合矩阵乘 | 没跑 | **不过**：R20 all-reduce 9 → 16、`reduce_precision` 31 → 26 | 2.49e-5（0.46）/ 1.01e-4（0.85）/ 5.91e-4（0.68） | 不进参照（[`cal-cmoff-t1/`](cal-cmoff-t1/report.md)） |
| sabwd2k | splash 反向块 2048 | — | — | 两次提交都在写运行目录之前失败 | 放弃 |

- 注 1：配置只差这 5 个键和由它们派生的 5 个 `local_sa_block_*`。
- 注 2：优化后的 HLO（去掉元数据）、训练后 FP32 状态清单都相同。

### 两套切片合在一份参照里

- 基线与 vmem24、vmem64 在一套 v5p-8 节点池上跑，其余臂在另一套 v5p-8 切片上跑。
- 两边能合在一份参照里：同一提交在后者上重跑，对前者上的 LKG 全部逐位。
  - T0 两次（`reg-t0-st1/st2`）、T1 三次（`reg-t1-st1/st2/st3`）。
  - 见 [`reg-t0-st1/`](reg-t0-st1/report.md)、[`reg-t1-st1/`](reg-t1-st1/report.md) 等。
- 后者读一份数据副本，但检查点仍读原路径：`meta.checkpoint.path` 进了 R02。
- 两次提交意外：
  - `cal-combo-a` 第一次提交时集群拒绝创建作业（权限接口的瞬时错误），换名 `cal-combo-ar2` 重跑；
  - `cal-sabwd256-a` 的 Pod 已经把运行目录传到桶里，只是提交机没取到，从桶里直接取回。

### 参照

基线 + 7 个臂，8 份运行两两 28 对，T0、T1 同一组臂。

- [`../calibration.yaml`](../calibration.yaml) 最新两条 `status: 正式`。
- 完整输出 [`calibration-v5p.json`](calibration-v5p.json)。

| 档 | 窗口 | 带符号均值 | MARE | 最大单步 | 最大的那一对 | 留一法互判（各对 r 的最大值） |
|---|---|---|---|---|---|---|
| T1 | `1:1001` | 5.90e-5 | 1.130e-4 | 9.12e-4 | vmem48 ~ sabwd256（前两个）、sabwd1k ~ combo（最大单步） | 1.14 / 1.00 / 1.01 |
| T0 | `1:21` | 1.45e-5 | 1.77e-5 | 4.88e-5 | 基线 ~ vmem48 | 1.21 / 1.21 / 1.07 |

留一法没有哪一对比其余各对大出 1.21 倍以上，没有一条独撑上限（标准第五节第 3 步）。

T1 的带符号均值、最大单步上限略高于 GPU 噪声上限（5.39e-5、8.72e-4；GPU 默认设置 10 份 1000 步运行两两 45 对）。这不矛盾：

- 第三关只要求每个臂**对基线**不超 GPU 上限。
- 而 T1 上限取的是 8 份 TPU 运行**两两** 28 对的最大值，两个臂恰好在基线两侧时彼此差得更远。
- 各臂对基线的带符号均值（带正负号）：vmem48 −3.5e-5、vmem64 −2.8e-5、vmem40 −1.5e-5、vmem56 +8.2e-6、combo +9.1e-6、sabwd1k +1.3e-5、sabwd256 +2.4e-5。
- vmem48 和 sabwd256 两头各一个，彼此差 5.9e-5。

其他：

- `raw_grad_norm`、`param_norm`、每层 `router.bias_*` 的上限同样在 `calibration.yaml`。
- 之前 4 份两两 6 对时的两条「暂定」条目留着，不删。

### R22 步时阈值

各 5 次同一程序，第 3 步起步时中位数的极差 ÷ 中位数：

- T0 0.83%（0.8980–0.9055 s）；
- T1 0.73%（0.8982–0.9048 s）。
- 两档都是另一套切片上的那几次偏慢。

### 1–2 倍区间

照文档和 2026-11-05 的决定，第三关只收 ≤ 1 倍的臂，规则不放宽。

- vmem24、vmem32、safwd1k 落在 1.12–1.24 倍，记「不进参照（未证实）」，不是判它们改了数学。
- 缺口靠补臂补上：这一轮补了 7 个臂，过了 4 个。

### 容差门给真结论

对已有运行在本地重判，按执行级声明，参照 8 份 TPU 运行两两 28 对（[`cal-demo/`](cal-demo/)）。

| 被测 → LKG | 容差门 | 整体 |
|---|---|---|
| `cal-vmem24-t1`、`cal-vmem32-t1`、`cal-safwd1k-t1` → `reg-t1-lkg2-a`（不在参照里的臂） | R08、R12 loss inconclusive（注）；R13 `raw_grad_norm` inconclusive；R16 pass；R25 不变差 pass | 无结论 |
| `cal-cmoff-t1`、`cal-rematqkv-t1` → `reg-t1-lkg2-a` | R08、R12、R13、R16 全部 pass；R25 pass | 无结论（只因 R09、R15） |
| `reg-t0-r4`（数学级的 rmsnorm 开关，故意按执行级声明）→ `reg-t0-st1` | R08 loss 是 TPU 上限的 1.07 / 1.16 / 1.30 倍 → inconclusive，不放行 | 无结论 |
| `reg-t0-r3`（训练数值逐位）→ `reg-t0-st1` | R08 pass | PASS |
| `reg-t1-st1/st2/st3` → `reg-t1-lkg2-a`（数学不变） | 不看容差门 | PASS（逐位） |

- 注：带符号均值在 TPU 上限的 1–2 倍。

执行级 T1 的整体结论仍是「无结论」。

- 原因：R09（第 1 步范数）和 R15（训练后状态，sha256 清单）没有容差门，不逐位就只能 inconclusive。

### LKG 重新登记

- 6 对那次（基线 + 3 个臂共 4 份运行两两配对；2026-11-05，项目负责人已批准）：
  - T1 `reg-t1-lkg2-a` 在新目录下重新登记（[`reg-t1-lkg2-cal/`](reg-t1-lkg2-cal/report.md)）；
  - T0 用 `reg-t0-st1/st2` 重建（[`reg-t0-st/`](reg-t0-st/report.md)）。
- 28 对这次（8 份运行两两配对）：只改了两份档位目录的 `runs.noise`，`R02.registry` 照样报。
  - 两对双跑在新目录下重判，只有 `R02.registry` 报这次目录变化，其余全部逐位。
  - 重新登记后重判都是 PASS（[`reg-t1-lkg2-formal/`](reg-t1-lkg2-formal/report.md)、[`reg-t0-st-formal/`](reg-t0-st-formal/report.md)）。
  - 这两条由项目负责人于 2026-11-05 批准。

## 还没做

- 第三关的「第 1 步参数梯度」一条：要抓张量，`calibrate.py` 这一版仍没实现，它只判 1000 步 loss。现行配置下另用 FP64 版手工补了 5 个臂（vmem64、vmem48 没补）；用的是验收那批第 1 步扰动运行 `t-st1n-<臂>-gm`（2 步，命令行覆盖未登记），不是这里的 `fa-*` 臂，也不是 T1 档程序，见 [`../README.md`](../README.md)第五节注 1 与「现行配置的标定」。
- v7x 的标定：切分已定（FSDP 8，同 v5p 写法，`../README.md` 第二节），等 v7x 的回归档位开跑后照第五节做。
- 对照臂（R24）随排程上线。
