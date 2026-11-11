[janus](../../../../README.md) › [用户指南](../../../../docs/guide/README.md) › [混元 3](../../README.md) ｜ 对齐实践 02；怎么读见指南[第 5 章 读结果与签字](../../../../docs/guide/05-reading-results.md)

# 混元 3 · 对齐实践 02：B200×8 → TPU v7x-4（截 4 层）

这个实践把同一套验收在 TPU v7x 上再走一遍，确认结论不只在 v5p 上成立。建议读完 01 再读这里。

- TPU v7x 4 芯片（8 个 device）对 GPU golden（B200×8，Megatron）。
- 设置、运行与读数见 [`setup.md`](setup.md)；运行清单 [`runs.yaml`](runs.yaml)。
- 检查项目录在 `catalogs/`，结果在 `results/`（照 [`../01-v5p-acceptance/`](../01-v5p-acceptance/README.md) 的结构）。

本页依次是：最终结论、排查与补齐记录（5 条）、逐级大表（脚本生成）。

## 最终结论（2026-11-06）

参照是 golden，即 GPU 生产训练程序的结果。正式迁移验收第一层用最终 golden [`rcc`](../../../../docs/glossary.md#rcc)：生产程序其余不变，只把输出头 dgrad 改成分段 FP32 累加，修掉 B200 上这次长求和的系统偏差。`rcc` 只是对参照的一处修正，TPU 一侧的精度开关、数据回放、测量口径见对齐报告第 5、6、8 节。

判法：TPU 只和事先指定的一份 [`rcc-a`](../../../../docs/glossary.md#rcc-a)（最终 golden 的第 1 份）比，噪声上限 N 取 `rcc-a..j` 10 份重跑两两配对得到的 45 对比较；r = 被测统计量 ÷ N，≤ 1 过。

- L4：当前对齐配置 `t-v7x-on-1000`（v7x 当前对齐配置 1000 步）全过。loss 全程 0.82、第 301–1000 步 0.83、grad_norm 0.85、param_norm 0.53；学习率逐位，路由偏置曲线 0.81，健康信号过；EP = 8 臂也全过。这是对事先定下的基线 `rcc-a`；十份 golden 轮流当基线时，当前配置有三份落进待定（`rcc-j` grad_norm 1.11、`rcc-e` grad_norm 1.01、`rcc-d` loss 后段 1.08），没有不过（[`case01-rebaseline-rcc.json`](../01-v5p-acceptance/results/case01-rebaseline-rcc.json)、对齐报告第 3 节）。
- L0：全过。
- L1、L2 前向：L2 前向按当前配置全过（四层 R 1.06 / 1.01 / 1.01 / 1.02）；L1 单算子除 MoE 块外全过，MoE 块回放第 2 层有 1 个 token 路由翻转（R 1.27，判不过；去掉这个 token 后 0.95）。
- 第 1 步梯度与优化器状态：严格档不过，机理查到路由近似平局翻转（对齐报告台账 #5）与第 0 层等 FP32 参数的梯度以 BF16 落地（#20）；一部分张量未达 R ≤ 1.2 的定责线：梯度逐张量最差 R 2.70，优化器状态 7.86（最差在二阶动量），更新后主权重 51.3，后两者的最差都在 128 维的 q norm 上；按能量合计梯度、动量 1.21，主权重 0.94（对齐报告第 4 节注 3）；一个专家的偏置更新方向见 #17。

全部数字见 [`../../ALIGNMENT-REPORT.md`](../../ALIGNMENT-REPORT.md)。

## 排查与补齐记录（2026-11-05 – 11-06）

下面 5 条里，第 2 条是对 `rc`（GPU 生产 golden，即原版生产程序，整层重算加路由回放，5 份）的读数，属于排查过程；`rc` 的差是已解释差异。第 3 条与第 5 条里注明 11-06 的是当前配置的结果。其余各条除注明 2026-11-06 的以外，都是下面「配置换代」之前的配置。

### 配置换代与被测

2026-11-05 配置换代：对齐配置打开 `dense_fp32_accumulation`（FSDP 切开收缩维的投影改用 FP32 累加），yml `a238883b7f91…`，maxtext：开关从 `0622774` 起有；验收钉 `0622774a`，结论只对这个提交成立；换提交先跑日常回归 T0、T1，对 LKG 逐位相同再用。原因见 [`../../config-alignment.md`](../../config-alignment.md)「收缩维被 FSDP 切开的投影的累加精度」。v7x 上生产 rmap 配置开 / 关各跑 100 步（`t-v7x-st-off-100` / `t-v7x-st-on-100`）：步时中位数 0.9615 s → 0.9798 s（+1.9%），显存峰值 23.09 GiB 不变。

换代前的被测是生产配置 1000 步 `t-v7x-rmap-1000-a`；现行正式被测是 `t-v7x-on-1000`（四个精度开关，含 `dense_fp32_accumulation`，见 [`setup.md`](setup.md)「软件版本」与 [`runs.yaml`](runs.yaml)）。`t-v7x-rmap-1000-a` 是 01 的 `t-rmap-1000`（v5p 生产配置 1000 步）的 v7x 对应：GA 2 加三个精度开关，整层重算，只存路由。v7x 这份的代码包与 yml 已登记（maxtext `237a3270`、janus `9793b367`、yml `2a76c759`，见 [`runs.yaml`](runs.yaml)）；v5p 的 `t-rmap-1000` 代码包、yml 与命令行覆盖都没登记，推断是同一份，未核（见 01 `runs.yaml`）。

### 五条记录

1. v7x 自己跑两次逐位（`phase2-v7x-rmap-repro`）

   `-a`、`-b` 各占一个切片，`metrics.jsonl` 1000 步 × 60 个字段（`step_time` 除外）全部相同，输入指纹 65,000 项相同。所以换代前配置在 v7x 上「同程序」的噪声是 0。现行验收程序 `t-v7x-on-1000` 和 v5p 的 `t-rmap-on-1000` 都没有逐步逐位的双跑结果文件（v5p 验收程序没双跑；v7x 当前配置的双跑只有汇总统计相同，见[签字清单](../../../../docs/guide/05-reading-results.md#56-签字清单现在离签字还差什么) 1b），这一条对它们是推断。
2. 对生产 golden `rc` 偏差和 v5p 同号、同量级（`phase2-v7x-acceptance-rc`）
   - 第二层，只对 GPU 生产 golden [`rc`](../../../../docs/glossary.md#rc) 的第一份 `rc-a`。被测是换代前配置 `t-v7x-rmap-1000-a`（没开 `dense_fp32_accumulation`；v5p 对应 `t-rmap-1000`）；当前配置对 `rc` 没有 janus 判定结果。
   - loss 带符号相对均值 1.20e-4（v5p 9.13e-5，待定）。这是 janus 对各份 golden 的带符号均值取绝对值后的最大值，不带方向；方向是 TPU 偏低（当前配置对 `rc` 均值 301–1000 步 −1.07e-4，[`v7x-tpu-vs-rc.json`](results/v7x-tpu-vs-rc.json)；这份 json 从没入库的 `t-v7x-on-1000` 逐步序列算出，现在复算不了）。
   - 301–1000 步 1.60e-4，同样是绝对值（v5p 1.28e-4；v7x 不过、v5p 待定）。
   - 噪声上限 6.8e-5 / 6.6e-5。
   - param_norm 2.93e-6，与 v5p 到第三位都相同（噪声上限 1.2e-7）。
3. 第一层：2026-11-06 起对最终 golden `rcc-a`
   - 检查项目录是 [`catalogs/phase2-v7x-acceptance-rcc.yaml`](catalogs/phase2-v7x-acceptance-rcc.yaml)，定义见 01 README「golden 的分工」。
   - 基线是 [`rcc-a`](../../../../docs/glossary.md#rcc-a)，噪声取 `rcc-a..j` 10 份重跑两两配对得到的 45 对比较。
   - 当前配置 `t-v7x-on-1000`（v7x 当前对齐配置 1000 步）的 r 值（≤ 1 过）：loss 0.82、loss 尾段 0.83、grad_norm 0.85、param_norm 0.53。
   - 学习率逐位，路由偏置曲线 0.81、健康信号过（`phase2-v7x-acc-health-rcc`）。
   - EP = 8 臂也全过（`phase2-v7x-acceptance-rcc-ep8`）。
   - 只有 `rcc-a..e` 5 份重跑（两两 10 对）时，grad_norm 与 param_norm 落在 1.02 / 1.24 倍。是不是 v7x 特有的数值差，没有可复算的证据：曾记过 v7x 对 v5p 的 grad_norm 带符号均值 1.3e-3、param_norm −1.1e-7，但出处未入库（没有结果文件、运行名与口径），不作依据。按 [`v7x-tpu-vs-rc.json`](results/v7x-tpu-vs-rc.json) 能间接看出：当前配置 301–1000 步 loss 对 `rc` 均值的带符号相对均值 v7x −1.07e-4、v5p −1.37e-4，v7x 比 v5p 高约 3e-5（两个均值相减，推断）；第 1000 步 param_norm 对 `rc` 均值的差两边记的是同一个数（0.0209）。这份 json 是当时从 `t-v7x-on-1000` 的逐步序列现算的（记了 `meta_sha256`），序列没入库，现在复算不了（[`../../ALIGNMENT-REPORT.md`](../../ALIGNMENT-REPORT.md) 第 3 节）。
   - 此前对精确轨迹 `rce-a`（BF16 权重、全程 FP32 计算，输出头 dgrad 仍有偏）的读数只作参考，其中 param_norm 3.53 倍不过。原因见案例 01「TPU 对精确轨迹那一成的账」。
4. L0 全过
   - 配置对照、30 步输入指纹逐位（`phase2-v7x-t-loss30`）。
   - `phase2-v7x-l0-state` 7 项全过，全部对 `g1k-a`（第一阶段 GPU golden 的第 1 份）逐位：
     - 初始参数清单 1480 项（另比生产配置的清单，同样 1480 项）；
     - 第 1 步载入后的参数 / FP32 主权重各 50 个张量；
     - expert_bias 3 个。
   - dtype 清单 2960 项、模型结构 15 条规则全符合，与 v5p 的计数相同。
   - 权重转换与 01 共用同一份检查点（`phase2-ckpt-convert`）。
   - 没比的一项是 01 的训练状态 FP32 清单（`L0.dtype.state`）。它要带 `--state-sha` 的运行。v7x 有这样的运行（日常回归同类的 `reg-v7x-t1`、观测构建 `obs-v7x-t1`，见 `runs.yaml`；v5p 这一项用的正是日常回归同类运行 `reg-t1-lkg2-a`），只是没比。v7x 与 v5p 用同一份 yml 与代码包，dtype 清单 2960 项已相同，由此推断训练状态的 dtype 也相同。用 `reg-v7x-t1` 的 `state_sha` 照 01 `phase2-l0-state` 跑一遍即可关闭，在 CPU 上就能做。注意 `reg-v7x-t1`（maxtext `237a3270`、GA 1）与 v5p 的 `reg-t1-lkg2-a`（maxtext `237a3270`，旧配置的 LKG）都是换代前的程序，不是现行验收程序（`0622774a` + yml `a238883b`）；这样关闭只证明换代前配置的训练状态 dtype 两边一致。
5. L1–L4 其余各项按 01（v5p）的同一套工具与门槛补齐（2026-11-05，见 setup.md「L1–L4 补齐」）
   - 和 v5p 的判定逐项相同，只有两处不同，见下面 5a、5b；其余见 5c。

#### 5a. 前向对 FP64 的 R-ratio 比 v5p 大，原因已定位，与 v5p 同一个

- 换代之前（`phase2-v7x-l2-replay-rratio`）：四层 1.53–1.57（v5p 1.21–1.24）。
- 单算子回放（`phase2-v7x-l1-ops-own`）里只有 qkv 1.57、稠密层 fc1 1.58 超线（v5p 1.24 / 1.29），其余算子与 v5p 相当。
- 原因和在 v5p 上查出的一样，是窗口化 collective matmul：收缩维被 FSDP 切开，分片部分积在 BF16 里累加，v7x 上偏差更大。
- 关掉窗口化核对（`r-v7x-wcoff`）：同一份回放加 `LIBTPU_INIT_ARGS=--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000`。
  - qkv 1.000、fc1 1.104，四层前向 1.059 / 1.010 / 1.012 / 1.015 全过；
  - 单算子里 qkv、qk_norm、注意力输出投影、fc1、稠密层激活与 fc2、注意力核和 v5p 关窗口化（01 的 `phase2-l1-ops-wcoff`）到第三位相同；路由 GEMM（0.017 对 0.022）、MoE 核（0.983 对 0.924）不同，都远在 1.2 以内；v5p 关窗口化的四层前向读数仓库里没有，四层无从对照；
  - 关掉后 mlp_norm 变成 1.64 超线，v5p 也是 1.64（见 01 的 `phase2-l1-ops-wcoff`）；
  - v7x 这一臂当时的 janus 输出（单算子回放与四层逐层隔离前向回放）原样收在 [`v7x-wcoff-recovered.json`](results/v7x-wcoff-recovered.json)，读数与上面一致；当时用的是旧版 `fwd_rratio.py`，回放运行目录已不在本机，没有用现行脚本重算，所以不进总览。
- 到 logits 与逐 token loss 就看不出来了（整网前向 R 1.00 / 1.02，过）。
- 2026-11-06 打开 `dense_fp32_accumulation` 后按当前配置重做（`r-v7x-l2-on`，v7x 逐层隔离回放）：
  - 四层前向 R 1.06 / 1.01 / 1.01 / 1.02，都过；单算子除 MoE 块（R 1.27，1 个 token 路由翻转）外都 ≤ 1.16（`phase2-v7x-l2-replay-rratio-on`、`phase2-v7x-l1-ops-own-on`）；v5p 当前配置四层是 1.06 / 1.01 / 1.01 / 1.09（第 3 层不同，`phase2-l2-replay-rratio-on`）。
  - 只有 MoE 块（`L1.ops.kernel.moe`）第 2 层 R 1.27：4096 个 token 里有 1 个在近似平局处和 FP64 选了不同的专家，去掉它 R 0.95（`results/v7x-l1-moe-routeflip.json`）。
  - 逐层反向现行按 FP64 R 比判，4 层全过（`phase2-v7x-l2-replay-bwd-fp64-on`）。按 GPU 噪声判的旧结果 `phase2-v7x-l2-replay-bwd-on` 与 v5p 一样：第 0 层过，1–3 层超；原因是路由翻转，而 GPU 第 1 步反向噪声极小、量不到。

#### 5b. 更新后的 expert_bias 不再逐位

这一项来自 `phase2-v7x-st1-prod-state` 的 `L3.step1.bias_after`。第 1 层 192 个专家里有 1 个（65 号）的更新方向反了（TPU +1e-3、GPU −1e-3），其余 575 个与 GPU 相同。这个专家第 1 步的负载正好落在均值附近，一个 token 的路由不同就翻号。v5p 上三层全部逐位。当前配置的 `phase2-v7x-st1-on-state-gpuonly` 同一项同样不过，读数与换代前逐位相同（第 1 层 rel_L2 0.0127、超线元素 1/192、最大差 2e-3），也是第 1 层 1 个专家翻号；结果文件没记专家号，是不是同一个 65 号没核。

2026-11-06 补：FP64 整步参照里这个专家选中 2730 个 token，比平均 2730.67 少 0.67，偏置应当加。v7x 的方向和精确算术一致，是 GPU 差一个 token 落到了均值另一边（`phase2-v7x-st1-fp64-on` 的 `L3.step1.bias_after`）。

#### 5c. 其余各项

- 优化器单步 / 连续 8 步重放过。
- 路由对 FP64 与逐层路由集合过。
- 逐层反向按 FP64 R 比判 4 层全过（按 GPU 噪声判的旧结果：第 0 层过、1–3 层超两倍，v5p 相同）。
- 第 1 步梯度、梯度分布、两个动量对 GPU 噪声不过（v5p 相同，GPU 第 1 步噪声极小）。
- 更新后主权重待定（v5p 相同）。
- 覆盖率全比到。
- 路由偏置曲线对最终 golden `rcc-a` 过（r 0.81）、健康信号过。
- NaN 演练：梯度第 5 步报警、loss 第 6 步才报（演练通过）。
- 观测构建：第三关的逐步 loss / grad_norm / param_norm 过；第 1 步梯度那一条与第二关在 v7x 上没做（v5p 上这两项都不过），所以观测构建的张量不当正式证据；「开 / 不开抓数」不逐位（v5p 相同）。
- TPU 原生数据加载器（`phase2-native-loader`）是主机侧的数据管线，结果两边共用，但理由不是「切分相同」：主机数会改全局批的行序（v7x-4 是 1 个主机进程，v5p-8 是 2 个）。主检查走单主机，正好对应 v7x；2 主机（v5p）只在单进程里模拟过（`L0.native.hosts`：用单主机换序只对上 36000/64000，按主机数换序后 64000/64000），真的多进程没跑。对齐运行回放 GPU 的 batch、不走原生加载器，不影响判定。
- 存档续训（`phase2-v7x-resume`）：第 10 步存档、续训 11–20 步，与连续训练逐位相同，5 项全过（同 v5p）。

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

- 共 121 行。
- 严格档：✅ 94　🟡 3　❌ 14　— 10
- 参考档（有对应门槛的 25 行）：✅ 17　❔ 8

| 级 | 检查项 | 严格档 | 参考档 | 说明 | 证据 |
|---|---|---|---|---|---|
| L0 | 配置逐项对照<br>`L0.config.diff.dropout` | ✅ | — | 实测 全部 6 个字段一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-t-loss30](results/phase2-v7x-t-loss30.report.md) |
| L0 | 每 rank 每步输入指纹（回放数据）<br>`L0.data.fingerprint` | ✅ | — | 实测 全部 1,950 个单元一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-t-loss30](results/phase2-v7x-t-loss30.report.md) |
| L0 | 加载后全部参数逐位<br>`L0.weights.init` | ✅ | — | 实测 全部 1,480 个参数一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-l0-state](results/phase2-v7x-l0-state.report.md) |
| L0 | 加载后全部参数逐位<br>`L0.weights.init.prod` | ✅ | — | 实测 全部 1,480 个参数一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-l0-state](results/phase2-v7x-l0-state.report.md) |
| L0 | 加载后全部参数逐位<br>`L0.weights.loaded` | ✅ | — | 实测 全部 50 个张量一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-l0-state](results/phase2-v7x-l0-state.report.md) |
| L0 | FP32 主权重逐位<br>`L0.main_params.loaded` | ✅ | — | 实测 全部 50 个张量一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-l0-state](results/phase2-v7x-l0-state.report.md) |
| L0 | 浮点 buffer（expert_bias）逐位<br>`L0.buffers.loaded` | ✅ | — | 实测 全部 3 个张量一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-l0-state](results/phase2-v7x-l0-state.report.md) |
| L0 | dtype 清单<br>`L0.dtype.manifest` | ✅ | — | 实测 全部 2,960 项一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-l0-state](results/phase2-v7x-l0-state.report.md) |
| L0 | 模型结构（实例化结果）<br>`L0.model.structure` | ✅ | — | 实测 全部规则满足｜严格档 按规则，过｜参考档 无对应门槛 | [phase2-v7x-l0-state](results/phase2-v7x-l0-state.report.md) |
| L0 | 权重转换 → MaxText 逐张量逐位（同一份检查点给 v5p、v7x 用；含 dtype 清单）<br>`L0.ckpt.convert.bitwise` | ✅ | — | 实测 全部 1,781 个参数一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-ckpt-convert](../01-v5p-acceptance/results/phase2-ckpt-convert.report.md) |
| L0 | 权重转换 → MaxText 逐张量逐位（同一份检查点给 v5p、v7x 用；含 dtype 清单）<br>`L0.ckpt.convert.dtype` | ✅ | — | 实测 全部 1,781 项一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-ckpt-convert](../01-v5p-acceptance/results/phase2-ckpt-convert.report.md) |
| L0 | 续训数据位置逐位<br>`L0.resume.data` | ✅ | — | 实测 全部 650 个单元一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-resume](results/phase2-v7x-resume.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.gpu_integrity` | ✅ | — | 实测 全部 64,000 个数组一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.index` | ✅ | — | 实测 三个索引两边重建结果逐字节相同（文件 sha256 也相同）｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.stream` | ✅ | — | 实测 全部 1,000 步一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.device_map.native` | ❌ | — | 实测 原样布局只有 2000 / 16000 个（设备 · microbatch）单元与 GPU 同号 rank 相同：MaxText 梯度累积按…｜严格档 逐位 / 逐项相同，不过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.fingerprint` | ✅ | — | 实测 全部 64,000 个单元一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.nonpad_total` | ✅ | — | 实测 全部 1,000 步一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.fields` | ✅ | — | 实测 字段约定一致：MaxText 用 int32、按全局批 [gbs, L] 存，GPU 用 int64 / float32、按 [mbs, L…｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.readers` | ✅ | — | 实测 grain_worker_count 0 / 8 与 1 逐步逐位相同｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.hosts` | ✅ | — | 实测 2 host：读进来的行按 host 交错（16000/16000 行符合 host 切分公式），用单 host 换序只对上 36000/6…｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.epochs` | ✅ | — | 实测 16000 个样本（1 个 epoch）：三个索引相同，8 个样本（含跨 epoch 边界的 0 个）逐位相同；356736 个样本（2 个…｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.eod_cases` | ✅ | — | 实测 合成数据（15 篇文档，含 1 token 空文档、恰好 seq / seq+1 长、跨两个样本的长文档，序列 16，2 个 epoch）：…｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.pad_id` | ❌ | — | 实测 含 pad id 的样本在 4 种组合下共 8 处两边不同（字段 ['labels', 'loss_mask', 'tokens']）：Me…｜严格档 逐位 / 逐项相同，不过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.negative_controls` | ✅ | — | 实测 MaxText 默认的 reset_attention_mask 与 packing：68/80 个单元不同（position）；不设 mm…｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.coverage` | ✅ | — | 实测 全部比到｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](../01-v5p-acceptance/results/phase2-native-loader.report.md) |
| L1 | 优化器单步 FP64 重放<br>`L1.optim.adamw.replay` | ✅ | ✅ | 实测 0.00213（rel_L2）｜严格档 ≤ 0.005 → 0.425×，过｜参考档 rel_L2（Frobenius） ≤ 0.02：0.00213（0.106×）；cos ≥ 0.9999：0.999999，过 | [phase2-v7x-st1-on-state-gpuonly](results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| L1 | 优化器连续 8 步重放<br>`L1.optim.adamw.8steps` | ✅ | — | 实测 0.00479（rel_L2）｜严格档 ≤ 0.005 → 0.9575×，过｜参考档 无对应门槛 | [phase2-v7x-st8-opt8](results/phase2-v7x-st8-opt8.report.md) |
| L1 | 路由选中集合对 FP64<br>`L1.router.topk.fp64.layer1` | ✅ | — | 实测 1（集合相同比例）｜严格档 ≥ 0.97 → 过｜参考档 无对应门槛 | [phase2-v7x-l2-replay-rratio-on](results/phase2-v7x-l2-replay-rratio-on.report.md) |
| L1 | 路由选中集合对 FP64<br>`L1.router.topk.fp64.layer2` | ✅ | — | 实测 0.9998（集合相同比例）｜严格档 ≥ 0.97 → 过｜参考档 无对应门槛 | [phase2-v7x-l2-replay-rratio-on](results/phase2-v7x-l2-replay-rratio-on.report.md) |
| L1 | 路由选中集合对 FP64<br>`L1.router.topk.fp64.layer3` | ✅ | — | 实测 1（集合相同比例）｜严格档 ≥ 0.97 → 过｜参考档 无对应门槛 | [phase2-v7x-l2-replay-rratio-on](results/phase2-v7x-l2-replay-rratio-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.qkv` | ✅ | ✅ | 实测 1（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.833×，过｜参考档 rel_L2 ≤ 2^-10：1.75e-04（0.179×）；归一化最大误差 ≤ 2^-7：0.005（0.64×），过（rel_L2 最大的是 `layers.0.attn.qkv.out`） | [phase2-v7x-l1-ops-own-on](results/phase2-v7x-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.qk_norm` | ✅ | ❔ | 实测 1.073（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.894×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-v7x-l1-ops-own-on](results/phase2-v7x-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.attn_proj` | ✅ | ❔ | 实测 1.001（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.834×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-v7x-l1-ops-own-on](results/phase2-v7x-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.mlp_norm` | ✅ | ❔ | 实测 1.157（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.9644×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-v7x-l1-ops-own-on](results/phase2-v7x-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.router_gemm` | ✅ | ❔ | 实测 0.0148（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.0123×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-v7x-l1-ops-own-on](results/phase2-v7x-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.dense_fc1` | ✅ | ❔ | 实测 1.118（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.9313×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-v7x-l1-ops-own-on](results/phase2-v7x-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.dense_act_fc2` | ✅ | ❔ | 实测 1.112（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.9266×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-v7x-l1-ops-own-on](results/phase2-v7x-l1-ops-own-on.report.md) |
| L1 | 专家 GEMM / 核心注意力回放、MoE 分阶段<br>`L1.ops.kernel.attn_core` | ✅ | ❔ | 实测 1.009（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.841×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-v7x-l1-ops-own-on](results/phase2-v7x-l1-ops-own-on.report.md) |
| L1 | 专家 GEMM / 核心注意力回放、MoE 分阶段<br>`L1.ops.kernel.moe` | ❌ | ❔ | 实测 1.269（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 1.057×，不过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-v7x-l1-ops-own-on](results/phase2-v7x-l1-ops-own-on.report.md) |
| L2 | 第 0 层前向<br>`L2.layer0.fwd` | ✅ | ✅ | 实测 1.059（R 比）｜严格档 ≤ 1.2 → 0.883×，过｜参考档 rel_L2 < 0.01：0.00399（0.399×）；超差元素比例 < 1%：2.65e-04（0.0265×）；均值差 < 0.01：0.00391（0.391×）；cos ≥ 0.99995：0.999992，只报，过（rel_L2 最大的是 `layers.0.mlp.fc2.out`） | [phase2-v7x-l2-replay-rratio-on](results/phase2-v7x-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 第 1 层前向<br>`L2.layer1.fwd` | ✅ | — | 实测 1.01（R 比）｜严格档 ≤ 1.2 → 0.841×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-v7x-l2-replay-rratio-on](results/phase2-v7x-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 第 2 层前向<br>`L2.layer2.fwd` | ✅ | — | 实测 1.012（R 比）｜严格档 ≤ 1.2 → 0.843×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-v7x-l2-replay-rratio-on](results/phase2-v7x-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 第 3 层前向<br>`L2.layer3.fwd` | ✅ | — | 实测 1.015（R 比）｜严格档 ≤ 1.2 → 0.846×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-v7x-l2-replay-rratio-on](results/phase2-v7x-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 各层反向（激活梯度）<br>`L2.layer0.bwd` | ✅ | ✅ | 实测 0.982（R 比）｜严格档 ≤ 1.2 → 0.818×，过｜参考档 rel_L2 < 0.01：0.0021（0.21×）；超差元素比例 < 1%：1.07e-04（0.0107×）；均值差 < 0.01：8.23e-04（0.0823×）；cos ≥ 0.99995：0.999998，只报，过（rel_L2 最大的是 `layers.0.in.grad`） | [phase2-v7x-l2-replay-bwd-fp64-on](results/phase2-v7x-l2-replay-bwd-fp64-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 各层反向（激活梯度）<br>`L2.layer1.bwd` | ✅ | — | 实测 0.9577（R 比）｜严格档 ≤ 1.2 → 0.798×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-v7x-l2-replay-bwd-fp64-on](results/phase2-v7x-l2-replay-bwd-fp64-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 各层反向（激活梯度）<br>`L2.layer2.bwd` | ✅ | — | 实测 0.8795（R 比）｜严格档 ≤ 1.2 → 0.733×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-v7x-l2-replay-bwd-fp64-on](results/phase2-v7x-l2-replay-bwd-fp64-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 各层反向（激活梯度）<br>`L2.layer3.bwd` | ✅ | — | 实测 0.9508（R 比）｜严格档 ≤ 1.2 → 0.792×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-v7x-l2-replay-bwd-fp64-on](results/phase2-v7x-l2-replay-bwd-fp64-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 各层参数梯度<br>`L2.layer0.grads` | ❌ | — | 实测 0.00276（rel_L2），cos 0.999998｜严格档 N=5.34e-04（3 对）→ 5.18×N；最差的是 1−cos 2.05e-06 = 16.4×N，不过｜参考档 无对应门槛 | [phase2-v7x-st1-on-state-gpuonly](results/phase2-v7x-st1-on-state-gpuonly.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 各层参数梯度<br>`L2.layer1.grads` | ❌ | — | 实测 0.0386（rel_L2），cos 0.999259｜严格档 N=5.27e-04（3 对）→ 73.2×N；最差的是 1−cos 7.41e-04 = 5.34e+03×N，不过｜参考档 无对应门槛 | [phase2-v7x-st1-on-state-gpuonly](results/phase2-v7x-st1-on-state-gpuonly.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 各层参数梯度<br>`L2.layer2.grads` | ❌ | — | 实测 0.0677（rel_L2），cos 0.997722｜严格档 N=1.36e-04（3 对）→ 497×N；最差的是 1−cos 0.00228 = 2.5e5×N，不过｜参考档 无对应门槛 | [phase2-v7x-st1-on-state-gpuonly](results/phase2-v7x-st1-on-state-gpuonly.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 各层参数梯度<br>`L2.layer3.grads` | ❌ | — | 实测 0.1（rel_L2），cos 0.995009｜严格档 N=3.85e-06（3 对）→ 2.6e4×N；最差的是 1−cos 0.00499 = 6.7e8×N，不过｜参考档 无对应门槛 | [phase2-v7x-st1-on-state-gpuonly](results/phase2-v7x-st1-on-state-gpuonly.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 各层路由选中集合<br>`L2.layer1.router` | ✅ | — | 实测 0.9543（集合相同比例）｜严格档 ≥ 0.942 → 过｜参考档 无对应门槛 | [phase2-v7x-l2-replay-rratio-on](results/phase2-v7x-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 各层路由选中集合<br>`L2.layer2.router` | ✅ | — | 实测 0.9688（集合相同比例）｜严格档 ≥ 0.959 → 过｜参考档 无对应门槛 | [phase2-v7x-l2-replay-rratio-on](results/phase2-v7x-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L2 | 各层路由选中集合<br>`L2.layer3.router` | ✅ | — | 实测 0.9692（集合相同比例）｜严格档 ≥ 0.965 → 过｜参考档 无对应门槛 | [phase2-v7x-l2-replay-rratio-on](results/phase2-v7x-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-v7x-st1-on-state-gpuonly.png) |
| L3 | 全部激活逐位<br>`L3.step1.act.bitwise` | ❌ | — | 实测 41 / 41 个张量不同｜严格档 逐位 / 逐项相同，不过｜参考档 无对应门槛 | [phase2-v7x-l2-replay-bitwise](results/phase2-v7x-l2-replay-bitwise.report.md) |
| L3 | 逐 token loss<br>`L3.step1.loss.per_token` | ✅ | — | 实测 1.025（R 比）｜严格档 ≤ 1.2 → 0.854×，过｜参考档 无对应门槛 | [phase2-v7x-ff-chain](results/phase2-v7x-ff-chain.report.md) |
| L3 | logits（含 KL）<br>`L3.step1.logits` | ✅ | — | 实测 0.9995（R 比）｜严格档 ≤ 1.2 → 0.833×，过｜参考档 无对应门槛 | [phase2-v7x-ff-chain](results/phase2-v7x-ff-chain.report.md) |
| L3 | 全部参数梯度<br>`L3.step1.grads.all` | ❌ | — | 实测 0.1（rel_L2），cos 0.995009｜严格档 N=5.34e-04（3 对）→ 188×N；最差的是 1−cos 0.00499 = 3.6e4×N，不过｜参考档 无对应门槛 | [phase2-v7x-st1-on-state-gpuonly](results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| L3 | 梯度分布（std 比、零比例、逐元素 p99）<br>`L3.step1.grads.dist` | ❌ | — | 实测 0.00781（零元素比例差）｜严格档 N=1.40e-05（3 对）→ 559×N，不过｜参考档 无对应门槛 | [phase2-v7x-st1-on-state-gpuonly](results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| L3 | 更新后主权重<br>`L3.step1.updated` | 🟡 | ✅ | 实测 5.24e-05（rel_L2），cos 1.000000｜严格档 N=4.13e-05（3 对）→ 1.27×N；最差的是 1−cos 1.37e-09 = 1.61×N，待定｜参考档 rel_L2（Frobenius） ≤ 0.02：5.24e-05（0.00262×）；cos ≥ 0.9999：1.000000，过 | [phase2-v7x-st1-on-state-gpuonly](results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| L3 | 优化器状态（两个动量）<br>`L3.step1.optim_state` | ❌ | — | 实测 0.1（rel_L2），cos 0.995009｜严格档 N=0.00101（3 对）→ 99.6×N；最差的是 1−cos 0.00499 = 1.1e4×N，不过｜参考档 无对应门槛 | [phase2-v7x-st1-on-state-gpuonly](results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| L3 | 更新后 expert_bias<br>`L3.step1.bias_after` | ❌ | — | 实测 0.0127（rel_L2）｜严格档 N=0（3 对）→ ∞×N，不过｜参考档 无对应门槛 | [phase2-v7x-st1-on-state-gpuonly](results/phase2-v7x-st1-on-state-gpuonly.report.md) |
| L3 | 第 1 步全部参数梯度对 FP64（R-ratio，定责用：TPU 不比 GPU 离精确算术远）<br>`L3.step1.grads.all` | ❌ | — | 实测 2.7（R 比）｜严格档 ≤ 1.2 → 2.25×，不过｜参考档 无对应门槛 | [phase2-v7x-st1-fp64-on](results/phase2-v7x-st1-fp64-on.report.md) |
| L3 | 比全了没有（覆盖率）<br>`L3.step1.coverage` | ✅ | — | 实测 全部比到｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-coverage](results/phase2-v7x-coverage.report.md) |
| L3 | 存档续训重叠步逐位<br>`L3.resume.state` | ✅ | — | 实测 全部 91 个参数一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-resume](results/phase2-v7x-resume.report.md) |
| L3 | 存档续训重叠步逐位<br>`L3.resume.step11` | ✅ | — | 实测 全部 256 个张量一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-resume](results/phase2-v7x-resume.report.md) |
| L3 | 存档续训重叠步逐位<br>`L3.resume.coverage` | ✅ | — | 实测 全部比到｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-resume](results/phase2-v7x-resume.report.md) |
| L4 | 设置核对：学习率逐步（学习率是设置的常数，不是观测量；检查 id 仍是 L4.lr）<br>`L4.lr` | ✅ | — | 实测 逐步在容差内｜严格档 rtol ≤ 0、abs_ulp ≤ 0，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | — | ✅ | 实测 +1.75e-05（首步差）｜严格档 只报，不判｜参考档 首步差 ≤ 2e-4：1.75e-05（0.0873×），过 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | ✅ | ✅ | 实测 5.12e-05（带符号均值）｜严格档 N=9.47e-05（45 对）→ 0.54×N，过｜参考档 带符号均值 ≤ 2e-4：5.12e-05（0.256×），过 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | ✅ | — | 实测 1.13e-04（MARE）｜严格档 N=1.38e-04（45 对）→ 0.817×N，过｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | ✅ | — | 实测 4.60e-04（最大单步）｜严格档 N=9.75e-04（45 对）→ 0.471×N，过｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss.tail` | — | ✅ | 实测 +1.75e-05（首步差）｜严格档 只报，不判｜参考档 首步差 ≤ 2e-4：1.75e-05（0.0873×），过 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss.tail` | ✅ | ✅ | 实测 6.32e-05（带符号均值）｜严格档 N=1.21e-04（45 对）→ 0.521×N，过｜参考档 带符号均值 ≤ 2e-4：6.32e-05（0.316×），过 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss.tail` | ✅ | — | 实测 1.24e-04（MARE）｜严格档 N=1.50e-04（45 对）→ 0.831×N，过｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss.tail` | ✅ | — | 实测 4.60e-04（最大单步）｜严格档 N=5.91e-04（45 对）→ 0.778×N，过｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | grad_norm<br>`L4.grad_norm` | — | — | 实测 -2.37e-04（首步差）｜严格档 只报，不判｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | grad_norm<br>`L4.grad_norm` | ✅ | — | 实测 0.0015（带符号均值）｜严格档 N=0.00453（45 对）→ 0.331×N，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | grad_norm<br>`L4.grad_norm` | ✅ | — | 实测 0.0281（MARE）｜严格档 N=0.033（45 对）→ 0.851×N，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | grad_norm<br>`L4.grad_norm` | ✅ | — | 实测 0.265（最大单步）｜严格档 N=0.459（45 对）→ 0.578×N，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | param_norm<br>`L4.param_norm` | — | — | 实测 -1.26e-07（首步差）｜严格档 只报，不判｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | param_norm<br>`L4.param_norm` | ✅ | ✅ | 实测 9.01e-08（带符号均值）｜严格档 N=2.27e-07（45 对）→ 0.397×N，过｜参考档 带符号均值 ≤ 5e-6：9.01e-08（0.018×），过 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | param_norm<br>`L4.param_norm` | ✅ | — | 实测 9.48e-08（MARE）｜严格档 N=2.33e-07（45 对）→ 0.407×N，过｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | param_norm<br>`L4.param_norm` | ✅ | ✅ | 实测 2.58e-07（最大单步）｜严格档 N=4.86e-07（45 对）→ 0.531×N，过｜参考档 最大单步 ≤ 1.25e-5：2.58e-07（0.0207×），过 | [phase2-v7x-acceptance-rcc](results/phase2-v7x-acceptance-rcc.report.md) |
| L4 | 路由偏置 / 负载曲线<br>`L4.router.bias_std` | ✅ | — | 实测 0.001（带符号均值）｜严格档 N=0.00193（45 对）→ 0.52×N，过｜参考档 无对应门槛 | [phase2-v7x-acc-health-rcc](results/phase2-v7x-acc-health-rcc.report.md) |
| L4 | 路由偏置 / 负载曲线<br>`L4.router.bias_std` | ✅ | — | 实测 0.00266（MARE）｜严格档 N=0.00353（45 对）→ 0.753×N，过｜参考档 无对应门槛 | [phase2-v7x-acc-health-rcc](results/phase2-v7x-acc-health-rcc.report.md) |
| L4 | 路由偏置 / 负载曲线<br>`L4.router.bias_std` | ✅ | — | 实测 0.013（最大单步）｜严格档 N=0.016（45 对）→ 0.809×N，过｜参考档 无对应门槛 | [phase2-v7x-acc-health-rcc](results/phase2-v7x-acc-health-rcc.report.md) |
| L4 | 健康信号（NaN、突跳）<br>`L4.health` | ✅ | — | 实测 健康｜严格档 spike_ratio ≤ 3，过｜参考档 无对应门槛 | [phase2-v7x-acc-health-rcc](results/phase2-v7x-acc-health-rcc.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss` | — | ✅ | 实测 +1.75e-05（首步差）｜严格档 只报，不判｜参考档 首步差 ≤ 2e-4：1.75e-05（0.0873×），过 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss` | ✅ | ✅ | 实测 1.69e-05（带符号均值）｜严格档 N=9.47e-05（45 对）→ 0.179×N，过｜参考档 带符号均值 ≤ 2e-4：1.69e-05（0.0846×），过 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss` | ✅ | — | 实测 1.07e-04（MARE）｜严格档 N=1.38e-04（45 对）→ 0.778×N，过｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss` | ✅ | — | 实测 7.52e-04（最大单步）｜严格档 N=9.75e-04（45 对）→ 0.771×N，过｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss.tail` | — | ✅ | 实测 +1.75e-05（首步差）｜严格档 只报，不判｜参考档 首步差 ≤ 2e-4：1.75e-05（0.0873×），过 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss.tail` | ✅ | ✅ | 实测 8.26e-06（带符号均值）｜严格档 N=1.21e-04（45 对）→ 0.068×N，过｜参考档 带符号均值 ≤ 2e-4：8.26e-06（0.0413×），过 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss.tail` | ✅ | — | 实测 1.09e-04（MARE）｜严格档 N=1.50e-04（45 对）→ 0.731×N，过｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss.tail` | ✅ | — | 实测 3.92e-04（最大单步）｜严格档 N=5.91e-04（45 对）→ 0.664×N，过｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm` | — | — | 实测 +1.60e-05（首步差）｜严格档 只报，不判｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm` | ✅ | — | 实测 4.73e-04（带符号均值）｜严格档 N=0.00453（45 对）→ 0.105×N，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm` | ✅ | — | 实测 0.0322（MARE）｜严格档 N=0.033（45 对）→ 0.9743×N，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm` | 🟡 | — | 实测 0.482（最大单步）｜严格档 N=0.459（45 对）→ 1.049×N，待定｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm.golden_den` | — | — | 实测 +1.60e-05（首步差）｜严格档 只报，不判｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm.golden_den` | ✅ | — | 实测 0.00139（带符号均值）｜严格档 N=0.00614（45 对）→ 0.227×N，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm.golden_den` | ✅ | — | 实测 0.0325（MARE）｜严格档 N=0.0335（45 对）→ 0.9675×N，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm.golden_den` | 🟡 | — | 实测 0.635（最大单步）｜严格档 N=0.559（45 对）→ 1.14×N，待定｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm` | — | — | 实测 -1.26e-07（首步差）｜严格档 只报，不判｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm` | ✅ | ✅ | 实测 3.20e-08（带符号均值）｜严格档 N=2.27e-07（45 对）→ 0.141×N，过｜参考档 带符号均值 ≤ 5e-6：3.20e-08（0.0064×），过 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm` | ✅ | — | 实测 4.76e-08（MARE）｜严格档 N=2.33e-07（45 对）→ 0.205×N，过｜参考档 这个量无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm` | ✅ | ✅ | 实测 2.57e-07（最大单步）｜严格档 N=4.86e-07（45 对）→ 0.529×N，过｜参考档 最大单步 ≤ 1.25e-5：2.57e-07（0.0206×），过 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm.raw` | — | — | 实测 -1.04e-07（首步差）｜严格档 只报，不判｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm.raw` | ✅ | — | 实测 3.73e-09（带符号均值）｜严格档 N=2.27e-07（45 对）→ 0.0164×N，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm.raw` | ✅ | — | 实测 5.33e-08（MARE）｜严格档 N=2.33e-07（45 对）→ 0.229×N，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm.raw` | ✅ | — | 实测 2.44e-07（最大单步）｜严格档 N=4.86e-07（45 对）→ 0.503×N，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.lr` | ✅ | — | 实测 逐步在容差内｜严格档 rtol ≤ 0、abs_ulp ≤ 0，过｜参考档 无对应门槛 | [phase2-v7x-acceptance-rcc-gradfp32](results/phase2-v7x-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的路由偏置曲线与健康信号<br>`L4.router.bias_std` | ✅ | — | 实测 0.0013（带符号均值）｜严格档 N=0.00193（45 对）→ 0.675×N，过｜参考档 无对应门槛 | [phase2-v7x-acc-health-rcc-gradfp32](results/phase2-v7x-acc-health-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的路由偏置曲线与健康信号<br>`L4.router.bias_std` | ✅ | — | 实测 0.00237（MARE）｜严格档 N=0.00353（45 对）→ 0.672×N，过｜参考档 无对应门槛 | [phase2-v7x-acc-health-rcc-gradfp32](results/phase2-v7x-acc-health-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的路由偏置曲线与健康信号<br>`L4.router.bias_std` | ✅ | — | 实测 0.0106（最大单步）｜严格档 N=0.016（45 对）→ 0.664×N，过｜参考档 无对应门槛 | [phase2-v7x-acc-health-rcc-gradfp32](results/phase2-v7x-acc-health-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的路由偏置曲线与健康信号<br>`L4.health` | ✅ | — | 实测 健康｜严格档 spike_ratio ≤ 3，过｜参考档 无对应门槛 | [phase2-v7x-acc-health-rcc-gradfp32](results/phase2-v7x-acc-health-rcc-gradfp32.report.md) |
| L4 | 梯度 NaN 先于 loss 报警<br>`L4.health.grad_norm` | ✅ | — | 演练（期望报警）：按期望报警｜本条原判定 不过：grad_norm@5=非有限 | [phase2-v7x-t-nan8](results/phase2-v7x-t-nan8.report.md) |
| L4 | 观测不改数值（开 / 不开抓数）<br>`DET.hooks.noop` | ❌ | — | 实测 不可复现：['lm_loss@1', 'grad_norm@1', 'param_norm@12']，张量不同 0｜严格档 逐位 / 逐项相同，不过｜参考档 无对应门槛 | [phase2-v7x-obs-gate](results/phase2-v7x-obs-gate.report.md) |
| L4 | 存档续训重叠步逐位<br>`L4.resume.overlap` | ✅ | — | 实测 重叠步全部逐位相同｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-v7x-resume](results/phase2-v7x-resume.report.md) |

<!-- janus:table:end -->
