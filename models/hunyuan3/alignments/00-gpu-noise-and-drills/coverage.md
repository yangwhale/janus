# 混元 3 · 检查覆盖表（第一阶段：GPU 对 GPU）

对照的是本仓库要求的全套验证标准：L0–L4 + 噪声判定 + 确定性 + 存档 / 硬件 + 工具自证 + 编译产物 + 门禁设计。

> 状态列是**第一阶段（GPU 自比）结束时**的分档，保留作记录。
> - 其中「待 TPU」「部分」「待做」各行之后在第二阶段（TPU v5p / v7x）的去向，在说明栏以「第二阶段：」标出。
> - TPU 对 GPU 的最终结论见[对齐报告](../../ALIGNMENT-REPORT.md)。

## 怎么读这张表

### 状态分档

第一阶段只有 GPU，所以每条检查按在 GPU 上能做到哪一步分档：

| 状态 | 含义 |
|---|---|
| **已跑** | GPU 自比（默认设置两次运行互为被测 / golden，噪声参照为同设置运行两两）真跑出了结论 |
| **演练** | 在确定性基线上往训练里真埋错，检查能报警（见 `faults/phase1-faults.md`）；真正的结论要等 TPU 侧 |
| **待做** / **演练待做** | GPU 上能做，还没做 |
| **部分** | 做了一部分（第二阶段的准备工作，在 CPU 上跑），缺什么写在说明里 |
| **待 TPU** | 只有 TPU 侧才有意义（XLA 编译产物、TPU 编译开关、跨栈数据管线…） |
| **FP8 后** | 开 FP8 之后再做 |
| **流程** | 不是数值检查，落在报告格式、门禁规矩、目录写法里 |

### 汇总

下面表里共 62 行，一行常常合并了几条同类要求：

- 已跑 53 行（其中 11 行另做了故障演练）；
- 部分 2、演练 3、待做 2、待 TPU 2；
- 另有编译产物（待 TPU）、FP8（FP8 后）、门禁设计（流程）三类整体列在最后。

### 目录与脚本

- 检查项目录在 `models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/`：
  - `phase1-full.yaml`（自比全量）；
  - `phase1-faults.yaml`（故障注入）；
  - `phase1-inject.yaml`（逐层注入）；
  - `phase1-determinism.yaml`（确定性自证）；
  - 其余 `phase1-*.yaml` 各对应一组补充运行（见 `README.md` 的结果一览）。
- L1 单算子要 GPU 回放，用独立脚本 `frameworks/megatron/l1_ops.py`（做法见 `docs/dev/design/0003-l1-op-replay.md`）。
- kernel 回放、MoE 分阶段、poison / 因果性、特殊值存恢复用 `frameworks/megatron/l1_kernels.py` 等（`docs/dev/design/0004-l1-gpu-kernels.md`）。
- 结果在 `results/`。

## L0 地基：数据、权重、配置（必须逐位）

| 检查 | 类型 | 目录条目 | 状态 | 说明 |
|---|---|---|---|---|
| 每个 rank 每个 microbatch 收到的 tokens / labels / loss_mask / position_ids 逐位比，窗口覆盖全部步 | `data_fingerprint` | `L0.data.fingerprint` | 已跑 · 演练 | 每 rank 每 microbatch 记 sha256；数据错位一格被抓 |
| 每步全局非 padding token 数逐步相等 | `data_fingerprint`（`<步>.nonpad_total`） | 同上 | 已跑 | |
| 同一份初始权重、同一 batch，首步 loss 逐位相同 | `data_fingerprint`（first_loss） | 同上 | 已跑 | 不同就先查数据和权重，不查数值 |
| 加载后全部参数逐位与参照相同（全量，不抽样），含 buffer（expert_bias）与 FP32 主权重 | `checkpoint_equal`（参数清单） / `bitwise_equal` | `L0.weights.init`、`L0.weights.loaded`、`L0.main_params.loaded`、`L0.buffers.loaded` | 已跑 · 演练 | buffer 不在 `named_parameters()` 里，单独列；主权重上小于 BF16 半格的错在 BF16 参数里看不见，必须单独比 |
| 配置逐项 diff（模型、优化器、调度器、并行、数据、版本） | `config_diff` | `L0.config.diff` | 已跑 · 演练 | 开 dropout、改学习率、改路由精度都被抓 |
| 关掉所有随机消费者（dropout、随机路由），从同一份权重起步 | `config_diff`（equals） | 同上 | 已跑 | dropout 两边都必须等于 0（两边相同还不够）；随机路由类开关 `meta.json` 没记录，抓数补上后再加断言 |
| 模型结构真的长成配置说的样子（路由层数、分组专家、共享专家、q/k 归一化） | `artifact_check`（`modules.txt`） | `L0.model.structure` | 已跑 | 看实例化出来的模块，不看配置 |
| HF → Megatron → HF 往返逐张量 sha256 + 张量数 | `checkpoint_equal` + `coverage` | `phase1-ckpt-roundtrip` | 已跑 | 1781 / 1781 逐位；往返对上证明不了映射表本身（互逆的错查不出），映射表靠 L2 |
| 转换后 dtype 清单（expert_bias 等必须 FP32） | `artifact_check` / `dtype_manifest` | `phase1-ckpt-roundtrip`、`L0.dtype.manifest` | 已跑 | 往返后 dtype 与原件一致；`dtype_manifest` 按规则查参数清单 init / final 逐 rank 共 2960 项：expert_bias 48 项 FP32，其余 BF16，没有规则覆盖的名字报 error |
| 续训数据位置：恢复后每 rank 的 token 与连续训练相同 | `data_fingerprint` | `phase1-resume` 的 `L0.resume.data` | 已跑 | |
| 配比索引表、tokenizer 资产、跨栈数据管线逐字段 | `data_fingerprint` / `config_diff` | `phase2-native-loader` 的 `L0.native.index`、`L0.native.fields` | 部分 | 索引逐字节相同、四个字段逐元素相同；混合表没测（见注 1） |
| TPU 原生数据加载器与 GPU 逐样本一致（设备与 rank 对应、混合 / 打乱索引表逐位、续训位置、多机多进程） | `data_fingerprint` | `phase2-native-loader`（`models/hunyuan3/data/`） | 部分 | CPU 上换序后 64000 个单元逐元素对上 GPU；TPU 首步 loss、续训位置、真多机未做（见注 2） |
| Megatron ↔ MaxText 权重直接互转（含 FP32 主权重与优化器状态） | `checkpoint_equal` | — | 待做 | 现状经 HF 中转（往返已逐位）；第二阶段仍未做（见注 3） |

1. **配比索引表等**：
   - MaxText 现建的 doc / sample / shuffle 索引与 Megatron 缓存逐字节相同；
   - 四个字段逐元素相同；
   - 混合表没测（只有一个数据集）；
   - 分词器 sha256 未写进运行元信息。
2. **TPU 原生数据加载器**：
   - MaxText `mmap_npy` 在 CPU 上读 1000 步：换序后 64000 个单元逐元素对上 GPU。
   - 原样的设备分配在梯度累积下与 Megatron 不同（MaxText 没有配置项，换序在 `janus.data_layout`）。
   - 已比：跨 epoch 索引、合成 EOD 边界、读取进程数、模拟多 host。
   - 未做：TPU 首步 loss、TPU 续训位置、真多机。
   - 逐项见 `docs/dev/design/0002-l0-data-weights.md`。
   - 第二阶段：对齐运行不用原生加载器，TPU 原样回放 GPU 每一步每个 rank 的 batch。
     - 「每 rank 每步输入指纹」0 / 1950 个单元不同，见[对齐报告](../../ALIGNMENT-REPORT.md)第 4 节；
     - 所以未做的这几项不影响对齐结论；
     - 改用原生加载器训练时要补。
3. **权重直接互转**：
   - 直接互转的设计与检查见 `docs/dev/design/0002-l0-data-weights.md`。
   - 第二阶段：仍未做（0002「直接互转，含完整训练状态」标未落地）。
   - 本次对齐两边都从同一份 HF 权重起步，用不到它；要接着对方的训练状态续训时才需要。

## L1 算子与优化器

| 检查 | 类型 | 目录条目 | 状态 | 说明 |
|---|---|---|---|---|
| 优化器单步 FP64 重放：用抓下的 FP32 主权重、梯度、实际生效的超参重算一步 AdamW，比更新量与两个动量 | `optimizer_step` | `L1.optim.adamw.replay` | 已跑 · 演练 | 抓出实际权重衰减是 0.033 而不是配置的 0.1（调度器默认值覆盖）；梯度在优化器前被缩放也被抓 |
| 路由选中集合对 FP64 重算（同一份输入、权重、expert_bias） | `discrete_agreement`（fp64_replay） | `L1.router.topk.fp64` | 已跑 | GPU FP32 路由第 1 步 3 个 MoE 层（第 1–3 层）翻转率为 0 |
| 单算子真实输入回放对训练逐位、对 FP64 真值（rel_L2、逐位相同率、ULP）：路由 GEMM、注意力输出投影、qkv、q / k / MLP 前归一化 | `op_replay` | `L1.gemm.*`、`L1.norm.replay`（`phase1-l1`） | 已跑 | 回放全部逐位等于训练；路由 GEMM 是「BF16 入 FP32 出」（见注 1） |
| 降精度复现：CPU 模拟 GPU 舍入方式逐位命中 | `op_replay`（emulate） | `L1.emulate.gemm` | 已跑 | 张量核累加模型（16 个一组、对齐保留 25 位、向零截断）复现路由 logits 与 BF16 出 GEMM 100% |
| R-ratio 三方比；逐元素初等函数 ULP、加 / 乘 / 类型转换逐位；极端输入对 FP64 公式 | `op_replay` | `L1.rratio.*`、`L1.elem.*`、`L1.extreme.*` | 已跑 | 多舍一次 R = 1.41 被门槛 1.2 抓到而 cosine 看不出；exp / rsqrt ≤ 2 ULP、sigmoid / silu ≤ 3；RMSNorm 平方和溢出时所有实现静默出 0 |
| 打平用例（GPU top-k 对稳定排序规则）与真实路由分数的打平行数 | `op_replay`（expect） | `L1.ties.*` | 已跑 | GPU `torch.topk` 打平时下标小者优先；CPU `torch.topk` 不是；真实数据第 8 / 9 名打平 0 行 |
| 随机 cotangent 的 VJP 对 FP64 解析梯度（RMSNorm、路由分数函数、整条路由） | `op_replay` | `L1.vjp.*` | 已跑 | FP32 rel_L2 ≤ 1.9e-7，BF16 ≤ 1.8e-3 |
| kernel 回放真实输入：专家分组 GEMM（前向、dgrad、wgrad，对逐专家循环）、核心注意力（前向、真实 dO 的反向），先对训练逐位再对 FP64 | `op_replay` | `L1.kernel.*`（`phase1-kernels`） | 已跑 | 前向与反向全部与训练逐位；分组与逐专家循环逐位（见注 2） |
| 合成 poison：分组 GEMM 尾块 / 组边界、注意力 THD padding 填 NaN / Inf；因果性：只改未来 token（注意力 kernel、整网真实模块） | `op_replay` | `L1.poison.*`、`L1.causal.*`（`phase1-kernels`）、`L2.causal.model.*`（`phase1-causal-model`） | 已跑 | poison 全部逐位隔离；未来填 NaN 会漏进过去（见注 3） |
| TF32 开 / 关：关法实测、训练里哪些 GEMM 输入是 FP32、关掉后第 1 步全部张量与 30 步 loss | `op_replay` / `bitwise_equal` / `tensor_tolerance` / `loss_series` | `TF32.*`（`phase1-kernels`、`phase1-tf32`） | 已跑 | 只有 `NVIDIA_TF32_OVERRIDE=0` 关得掉 TE 的 FP32 GEMM；受影响的只有路由反向两个 GEMM；前向逐位、参数梯度到噪声 4.95 倍、loss 在噪声内 |
| TPU 编译开关归因（highest、excess precision、CPU 白盒复现） | — | — | 待 TPU | 第二阶段：已测 excess precision 与 highest；没有 CPU 逐位复现（见注 4） |

1. **单算子回放**：
   - 回放全部逐位等于训练；
   - 路由 GEMM 走 TE `general_gemm`（BF16 入 FP32 出），不是「先升 FP32 再 matmul」；
   - 路由 logits 对正确舍入逐位相同率 0.002%–0.04%，BF16 出的 GEMM 99.5%–99.8%。
2. **kernel 回放**：
   - 3 层专家、4 层注意力的前向与反向全部与训练逐位；
   - 分组与逐专家循环逐位；
   - 对 FP64：GEMM 1.66e-3（BF16 出）、wgrad ≤ 3.5e-6、注意力输出 ≤ 1.95e-3、dq/dk/dv ≤ 4.0e-3。
3. **poison 与因果性**：
   - poison 全部逐位隔离；
   - 注意力只改未来 token 时过去逐位不变，但未来填 NaN 会漏进过去（dk/dv 全污染）；
   - 整网因果性见 `README.md`。
4. **TPU 编译开关归因**（第二阶段）：
   - excess precision 关掉后 param_norm 漂移不变、loss 偏移小一半（偏差台账 #10）；
   - `matmul_precision: highest`（矩阵乘用最高精度，FP32 输入真按 FP32 乘）+ `logits_dot_in_fp32`（输出头的输入和权重升到 FP32 再乘）已测，即开关表 logmm 行；
   - TPU 舍入方式没有做 CPU 逐位复现，「MXU 就近舍入累加」是推断（[`goldens.md`](../../goldens.md)「输出头 dgrad」参考行）。

## L2 模块 / 逐层（第 1 步，同一份权重）

| 检查 | 类型 | 目录条目 | 状态 | 说明 |
|---|---|---|---|---|
| 每层每个子模块前向输出（qkv、q/k 归一化、核心注意力、输出投影、MLP 归一化、专家、共享专家、层输出） | `tensor_tolerance`（calibrated） | `L2.layer{i}.fwd` | 已跑 | rel_L2、cosine、均值偏移、max_abs；默认设置第 1 步前向逐位相同 |
| 每层子模块输出的梯度（反向） | `tensor_tolerance`（calibrated） | `L2.layer{i}.bwd` | 已跑 | 反向起不再逐位 |
| 每层逐参数梯度 | `tensor_tolerance`（calibrated，含 norm_ratio） | `L2.layer{i}.grads` | 已跑 · 演练 | `f-gscale`（抓数之前 ×1/8）由范数比报警，cosine 看不出缩放 |
| 每层路由选中集合 | `discrete_agreement`（calibrated） | `L2.layer{i}.router` | 已跑 | |
| 整网串跑找台阶：第一个从逐位相同变成不同的点 | `bitwise_equal`（逐层） | `phase1-faults` 的 `L2.layer{i}.fwd` | 演练 | 第 2 层权重埋错时第 0、1 层逐位、第 2 层起不同 |
| 逐层注入：每层输入换成参照的层输入，先证注入生效，再看误差是否被隔离到单层 | `bitwise_equal` | `phase1-inject` | 演练 | 第 2 层埋错时只有第 2 层输出不同；专家分发缓冲区（`mlp.experts.out`）行数随其他 rank 变，不能跨运行逐位比 |
| DP 副本一致性（每个参数在所有持有它的 rank 上 sha256 相同） | `replica_consistency` | `L2.dp.replica` | 已跑 | |
| 并行等价（只换并行方式） | `bitwise_equal` / `ulp_distribution` / `tensor_tolerance` / `loss_series` | `phase1-parallel`、`phase1-parallel-ext`、`phase1-tp-cp`、`phase1-ga` | 已跑 | EP8↔EP4 前向逐位、反向超噪声（embedding 梯度）；TP2 / CP2 整网标量在门槛内；GA 1↔2 embedding 梯度差 2.1% |
| MoE 层内分阶段（router → scores → top-k → permute → GEMM1 → 激活 → GEMM2 → unpermute → 加共享专家）对 FP64 | `op_replay` | `L2.moe.*`（`phase1-moe-stages`） | 已跑 | 观测点 `--observe moe`；两个台阶：GEMM1（BF16 出，1.66e-3）、激活（本阶段 2.8e-3–3.0e-3，链式升到 3.7e-3）；搬运逐位、top-k 与 FP64 全同 |
| 截断模型对拍：截 N 层与全深度的前缀逐位 | `bitwise_equal` | `phase1-truncate` | 已跑 | 2 层对 4 层，前 2 层前向全部逐位 |
| 优化器连续多步重放 | `optimizer_step`（steps） | `phase1-opt8` | 已跑 | 第 1–8 步连续 FP64 重放，误差在 FP32 存储舍入量级 |

## L3 第 1 步整网

| 检查 | 类型 | 目录条目 | 状态 | 说明 |
|---|---|---|---|---|
| 第 1 步全部激活逐位（默认设置之间前向确定） | `bitwise_equal`（软） | `L3.step1.act.bitwise` | 已跑 | |
| 逐 token loss、logits | `tensor_tolerance` | `L3.step1.loss.per_token`、`L3.step1.logits` | 已跑 · 演练 | |
| 全部梯度、梯度 ULP 分布、梯度分布统计 | `tensor_tolerance` / `ulp_distribution` | `L3.step1.grads.all`、`L3.step1.grads.dist`、`L3.step1.grads.ulp` | 已跑 | ULP 只诊断不判，报 1 ULP 比例与 0 / 1 / 2–4 / 5–8 / 9–16 / 17+ 分桶；`grads.dist` 判 std 比、零比例差、逐元素相对误差 p99 |
| 更新后主权重、Adam 两个动量、更新后 expert_bias | `tensor_tolerance` | `L3.step1.updated`、`L3.step1.optim_state`、`L3.step1.bias_after` | 已跑 · 演练 | |
| 比全了没有 | `coverage` | `L3.step1.coverage` | 已跑 | 零覆盖不打印通过 |
| logits KL（写明方向，正负对照标定） | `tensor_tolerance`（`kl`） | `L3.step1.logits` | 已跑 | KL(参照 ‖ 被测) 行平均；GPU 自比第 1 步 KL = 0；正负对照见注 1 |
| 假 FP32：标 float32 但低 16 位全 0 | `fake_fp32` | `L3.step1.fake_fp32.main_param`、`L3.step1.fake_fp32.optim_state` | 已跑 | 门槛 0.99（推断）；更新后主权重最高 2.75e-3、动量最高 1.84e-5（见注 2） |

1. **logits KL 的对照**（`results/phase1-kl-controls.results.json`）：
   - GPU 自比第 1 步前向确定，KL = 0；
   - 正对照 f-weight2（一个权重缩放 1e-2）对 d-a（第一阶段确定性基线运行）为 7.4e-5（rel_l2 9.1e-3）；
   - f-drop（开 dropout 0.1）为 0.196；
   - 路由改 BF16 1.6e-4、数据错位 1.12。
2. **假 FP32**：更新前主权重从 BF16 检查点载入，本来就 100% 在 BF16 网格上，不查。

## L4 多步

| 检查 | 类型 | 目录条目 | 状态 | 说明 |
|---|---|---|---|---|
| loss：首步差、带符号均值（判定）、MARE、最大单步及步号，多份 golden，分母写明，固定窗口 | `multi_golden` | `L4.loss` | 已跑 · 演练 | 带符号均值配标准误下限（`signed_mean_se_k`）；被测按 `meta.steps.convention` 换算步号，`auto_offset` 探测偏移——约定故意写错一格时首步检查与探测都红 |
| 学习率逐步对齐 | `scalar_series` | `L4.lr` | 已跑 · 演练 | 差 1 个 FP32 ULP 被抓 |
| grad_norm（裁剪前）、param_norm 逐步 | `loss_series` | `L4.grad_norm`、`L4.param_norm` | 已跑 | |
| 每层 expert_bias 标准差、负载变异系数逐步 A/B | `router_stats` | `L4.router.bias_std`、`L4.router.load_cv` | 已跑 · 演练 | 偏置不更新（`f-bias`）被抓 |
| 健康信号：非有限值、突跳 | `health_signals` | `L4.health`、`phase1-spike` | 已跑 · 演练 | 学习率临时 ×300 时第 16 步报 grad_norm 突跳 |
| 每步 num_zeros、实际 wd 记录 | 抓数 | `metrics.jsonl` | 已跑 | |
| 分项 loss（aux / z）单独对拍 | `loss_series` | — | 待做 | 第一阶段配置没开 aux / z loss（每步 loss 只有 lm_loss），开了以后每项一条 |
| 梯度 NaN：健康检查先于 loss 报；跳过更新的步状态不变 | `health_signals` | `phase1-nan` | 演练 | grad_norm 第 5 步报、loss 还正常；BF16 下框架没跳过这次更新（参数被污染），跳步不变这条无从验证 |

## 噪声参照与判定

| 检查 | 类型 | 状态 | 说明 |
|---|---|---|---|
| GPU A/A：同设置 ≥ 3 次运行两两当噪声参照 | `noise_from_pairs` | 已跑 | **只用同一种设置的运行对**（见注 1） |
| calibrated 判定（≤ N pass、> 2N fail、其间 inconclusive，参照 < 3 最多 inconclusive） | 全部 calibrated 检查 | 已跑 | |
| 均值的标准误（分批均值与一阶自相关取大），\|均值 / 标准误\| < 2 写「分辨不了」 | `mean_se` | 已跑 | 报告里分段表与累积均值图都带 |
| TPU 扰动臂（只改执行方式的 TPU 运行） | `noise_verdict` | 待 TPU | 第二阶段：v5p 5 份全部 ≤ 0.85；v7x 4 份里 1 份 grad_norm 待定（见注 2） |

1. **同设置噪声参照**：
   - 「设置」含确定性开关与分布式优化器开关；
   - 实测「分布式优化器开 + 默认」一组对其余组有持续偏移（结果说明第 6 条）；
   - 逐位相同的参照对自动剔除并列出；
   - 留一法互判（每条用其余参照的上限判）只报不判。
2. **TPU 扰动臂**（第二阶段）：
   - 只改 splash 注意力块大小、片上内存上限，造了 v5p 5 份、v7x 4 份扰动运行；
   - 对最终 golden 的基线 [`rcc-a`](../../../../docs/glossary.md#rcc-a)：
     - v5p 5 份四个 L4 量 r 全部 ≤ 0.85；
     - v7x 4 份里 3 份 ≤ 0.85、1 份 grad_norm 待定（MARE 1.02 倍、最大单步 1.15 倍）。
   - 出处：[对齐报告](../../ALIGNMENT-REPORT.md)第 3 节，[`case01-rcc-ensemble.json`](../01-v5p-acceptance/results/case01-rcc-ensemble.json)。

## 确定性与复现

| 检查 | 类型 | 状态 | 说明 |
|---|---|---|---|
| 同配置确定性重跑：全部字段与抓数逐位相同 | `determinism` / `bitwise_equal` | 已跑 | `phase1-determinism` |
| 观测不改数值：开 / 不开抓数逐位相同 | `determinism` | 已跑 | `phase1-hooks`、`phase1-obs` 都逐位（见注 1） |
| 激活重算：重算的前向与首次前向逐位 | `bitwise_equal`（ref_rename） | 已跑 | `phase1-remat`：**不逐位**；开路由回放后逐位（见注 2） |
| 存档恢复 vs 连续训练：续训后数据位置、重叠步全部标量、续训首步全部抓数逐位 | `overlap_replay` / `data_fingerprint` / `bitwise_equal` | 已跑 | `phase1-resume`：第 10 步存档续训 10 步，全部逐位相同 |
| 特殊值存 / 恢复：存档前埋 −0.0、FP32 / BF16 次正规数，恢复后逐位 | `op_replay`（bitref） | 已跑 | `phase1-ckpt-special`：BF16 参数、FP32 主权重、两个动量、FP32 buffer 共 72 个张量逐位保真（不开分布式优化器） |

1. **观测不改数值**：
   - `phase1-hooks`：抓数 v2 全量钩子对轻量抓数（第 1 步只抓前向张量，不挂梯度钩子、不包优化器），30 步全部字段逐位（两边优化器配置要一致）；
   - `phase1-obs`：新加的可选观测点（注意力、MoE 各阶段、专家权重、GEMM 清单）对 `d-a`（第一阶段确定性基线运行）第 1 步 425 个张量逐位。
2. **激活重算**（`phase1-remat`）：
   - 首次前向对重算 9 / 51 个张量不同（MoE 层专家输出及其下游），路由选中不变；
   - 首次前向对不开重算的基线 26 / 46 不同（`DET.remat.fwd_vs_baseline`）；
     - 分母是第 1 步全部 `*.out`：44 个层内输出，加 `embed.out`、`final_norm.out`；
     - 另一条「重算对基线」26 / 51 是不同的比较，见 [README 第四节注 2](README.md#四只换执行方式哪些仍逐位)。
   - 原因是不建图时 top-k 不排序、权重归一化顺序变了；
   - 开路由回放后逐位（`03` 的 `remat-replay`）。

## 工具自证

| 检查 | 类型 | 状态 | 说明 |
|---|---|---|---|
| 注入已知错误必须红（每条数值门槛配一次） | `poison_selftest` | 已跑 | 另有 8 个真训练故障注入，见 `faults/phase1-faults.md` |
| 选空、缺张量、缺字段一律 error，不静默通过 | 全部类型 | 已跑 | 单测覆盖；`index.json` 张量重名、输入指纹缺期望字段也报 error |
| ULP 按原始 dtype；指标在 FP64 里算；非有限值位置不一致直接 fail | 指标层 | 已跑 | 单测覆盖 |

## 编译产物、FP8、门禁

### 编译产物

HLO / LLO 舍入点、融合、编译开关、AOT 产物与配置一致。

- 第一阶段分档为待 TPU。
- GPU 侧对应的是 `L0.model.structure` 这类「看实例化结果不看配置」。
- 第二阶段：
  - TPU 侧实例化出来的结构与 dtype 清单已查（01 / 02 的 `phase2-l0-state`、`phase2-v7x-l0-state` 里的 `L0.model.structure`）；
  - HLO / LLO 舍入点与融合没有做成检查项；
  - 加抓数输出会改变融合、第 1 步 loss 就不逐位，这一点是实测的；
  - 所以正式 L4 用不加钩子的生产构建（[对齐报告](../../ALIGNMENT-REPORT.md)第 4 节「观测不改数值」行）。

### FP8

先证明真走了 FP8、FP8 对 BF16 与 FP8 对 FP8 分开判、软门 + 逐步硬限。

- 分档：FP8 后。

### 门禁设计

对齐门与回归门分开、改门槛的规矩、数学不变的改动附对照、升级依赖单独过数值闸。

- 分档：流程，见 `docs/reference/regression.md`。
