[janus](../../README.md) › [混元 4](README.md) ｜ 事先登记的比较协议

# 混元 4 · 比较协议（事先登记，反向方案）

- 术语见[知识库的术语对照表](knowledge.md#术语对照表)。
- 本文件在**任何 GPU 对 TPU 的比较结果出现之前**定稿并冻结。以后要改，只能追加到文末「变更记录」：写明改了什么、为什么、是在看到哪些结果之前还是之后，改前、改后的判定并列。看到结果之后的改动不能把「不过」改成「过」。
- 判法见 [`docs/reference/judging.md`](../../docs/reference/judging.md)。本文件写混元 4 的具体选择。
- 语义以 TPU 一侧 MaxText 的混元 4 实现（`maxtext` 混元 4 集成分支，建于 `d723fa2d`）为准；GPU 一侧（Megatron-Bridge + Megatron-Core）补齐对上它。MaxText 的写法符合生产语义时 GPU 改；不符合时 MaxText 加默认关的开关，并记一行变更。
- 正文是事先登记的版本；之后的改动在文末变更记录，现行口径汇总在第 0 节。

## 0. 现行口径（按变更记录汇总）

| 项 | 现行 | 变更记录 |
|---|---|---|
| 角色 | GPU 是 golden / 基线，TPU 是被测（T0–T2、T1m 都一样） | 2026-11-07（角色） |
| T1 比较 | TPU `tz-off-rep`（对齐配置全部开关，含 M50，`sa_use_base2_exp: false`）1000 步对 GPU `g3-a`；同一运行前 300 步对 `g2-a` | 2026-11-07（g3）、2026-11-09（M50）、2026-11-11（base2 关） |
| splash base2 | 所有尺寸的对齐 yml 都是 `sa_use_base2_exp: false`（M43）；TPU 自比的 golden `ts-g-a` 仍是开 base2 的一代，不改 | 2026-11-11 |
| 噪声 N | 只用 GPU：T1 1000 步 `g3-a … g3-p` 16 份 120 对、前 300 步 `g2-a … g2-h` 28 对；T2、mid 各自 8 份 | 2026-11-07（尺子）、2026-11-11（T1 1000 步 16 份） |
| L3 噪声 | GPU 第 1 步三份 `g2s-a/b/c`（q·k 在 FP32 里算）；超出的对 CPU FP64 整网重算定责；TPU 梯度以训练步实际用的为准 | 2026-11-07（G17）、2026-11-08（FP64 参照）、2026-11-09（训练步梯度） |
| L4 窗口 | T1：第 1–1000 步、第 101 步起；前 300 步另判。T2：登记判定窗口 300 步，1000 步另写目录（支线报告读 1000 步） | 2026-11-07、2026-11-08 |
| 判定对象 | 只对 GPU 基线一份；对 8 份各比一次只作诊断 | 2026-11-09 |
| 曲线分母 | L4 所有曲线（含 grad_norm）的逐步相对差一律以参照（基线）作分母 | 2026-11-10（分母） |
| L3 分辨率下限与文件完整性 | N = 0 且两边对真值都小于一次 BF16 舍入的张量按下限过（优化器一步没有 N，两边都 < u 即按下限过）；TPU 梯度文件逐张量核平方和，损坏的不可判 | 2026-11-08、2026-11-09 |
| L3 灌选择 | 灌 GPU 的路由与 top-k 只作诊断，正式判定用 TPU 自选 | 2026-11-09 |
| 路由一致率 | 对 CPU FP64 判：每层 R = TPU 的翻转率 ÷ GPU 三份的最大值，≤ 1.2 过 | 2026-11-09 |
| 接受 | 记入变更记录的待定项写作已定责、已接受，格子判定照结果文件；有定责证据但没记入的写作「定责证据已有，未记为已接受」 | 2026-11-09 |
| 前瞻验证 | 三条事后改的判法（L3 分辨率下限、路由一致率按真值口径、优化器一步的下限）原样用于真实权重截层下一批没看过结果的运行，判定事先固定 | 2026-11-09 |

- L3 分辨率下限、路由一致率（对真值口径）、优化器一步的下限三条在看到结果之后改。路由一致率、优化器一步两条在现行数据上把改前不过的格子改判为过；L3 分辨率下限在改的当时翻过判定，现行数据上按它过的 5 个张量离精确值本来只有 GPU 的 0.72–0.95 倍，不改判定。前两条与本文件开头「看到结果之后的改动不能把『不过』改成『过』」一条冲突；改前、改后的判定在第 7 节与 [mini 对齐细节「判法变更记录」](alignments/01-mini/DETAILS.md#判法变更记录)并列。

## 1. 参照与设置

### 1.1 阶段与运行

| 阶段 | 模型 | 做什么 | 每侧份数 | 步数 | 基线 |
|---|---|---|---|---|---|
| T0 | hy4-tiny（MaxText `hy4-tiny`，约 538 万参数） | FP32 逐层前向 / 反向、第 1 步整步，两侧对 FP64 | 第 1 步 1 份 | 1 | — |
| T1 | hy4-mini（MaxText `hy4-mini`，427,783,489 参数，随机初始化） | 生产式训练，L0–L4 | TPU：同设置 2 份 `t1-a`、`t1-b`（TPU 已实测逐位相同，第二份只核这一点）+ 只改执行参数的扰动 4 份 `t1p-1 … t1p-4`（只作诊断）；GPU 8 份 `g1-a … g1-h`（L4 噪声只用这 8 份）；第 1 步全量抓数每侧 3 份 | 300 | 见第 2 节 |
| T2 | 真实权重截层：B（第 0–1 层，12,078,550,957 参数）、D1（第 0 层 + MTP，12,153,673,625 参数） | 真实性终检：L0、L3、L4；TPU 配置 `hy4-t2-b.yml` / `hy4-t2-d1.yml`（T1 现行开关原样） | GPU 8 份 `gb2-a … gb2-h` / `gd2-a … gd2-h`（L4 噪声只用这 8 份）+ 第 1 步全量抓数 3 份 `gb2s-*` / `gd2s-*`；TPU 2 份 `tb2-a`、`tb2-b` / `td2-a`、`td2-b` + 第 1 步诊断 `tb2diag` / `td2diag` | 300（1000 可选） | 见第 2 节 |
| T1m | hy4-mid（MaxText `hy4-mid`，5,189,678,849 参数，随机初始化；hy4-mini 的结构，宽 6144、64 头、q_lora 2048、稠密 FFN 18432、16 个路由专家 top-4 宽 512） | 生产式训练，L0–L4；介于 T1 与 T2 之间：只换尺寸、不换权重 | GPU 8 份 `gm-a … gm-h`（L4 噪声只用这 8 份）+ 第 1 步全量抓数 3 份 `gms-a … gms-c`；TPU `tm-a` + 第 1 步诊断 `tmdiag` | 300 | 见第 2 节 |

- hy4-mini：hy4-preview 的结构与每头维度（qk_nope 192、qk_rope 64、v 256、kv_lora 512、索引器 32 × 128、top-k 2048），真实词表 120832；宽度 1024、8 头、q_lora 512、稠密 FFN 3072、16 个路由专家（top-4）宽 256 + 1 个共享专家；1 稠密 + 6 MoE（一个 IndexCache 组加 2 层余数）+ 1 层 MTP。参数由 HF 张量形状数出；同一数法给 hy4-preview 不含 MTP 769.91B，对上公开的 770B。
- hy4-mini 的初始权重是一份随机初始化的 HF 目录（`make_tiny_hf.py --config mini --seed 0`，权重矩阵标准差 0.006），两边都从它载入：清单 `tensor_sha256.json` 的 sha256 `1d869124db698cc51d3fb8fa1c2d83a00ab51066701afd81acf9a0a8af176433`。TPU 一侧转换后反向取回 212 / 212 个张量逐位相同。
- 检查点读法各尺寸统一为 vLLM / SGLang 的读法：q_pe / k_pe 交错存放、转换时去交错，latent norm eps 1e-5，索引器 k_norm eps 1e-6（hy4-tiny 也改成这种读法）。
- 代号：`t1-a` 是 T1 的 TPU 第 1 份，`g1-a` 是 T1 的 GPU 第 1 份；`g1s-*` 是 GPU 第 1 步全量抓数参照；TPU 第 1 步诊断叫 `t1diag*`。TPU 一侧按设置分代：`t1-*` / `t1p-*`（最初）、`t1n-*` / `t1np-*`（路由偏置零均值关）、`t1s-*` / `t1sp-*`（再开五个舍入开关）、`t1t-*` / `t1tp-*`（再改输出头精度、重算策略与 QK-Clip 统计），见变更记录。
- 同一阶段同一侧的各份完全同设置（同代码提交、同配置、同初始权重、同数据），只因基础设施失败补跑，不因数值补跑；有效份数少于 6 份（15 对）时该侧判「参照不足」。

### 1.2 训练设置

目标是 GPU 与 TPU 对齐，不是复现生产。规则：
- 生产已知、两边做到不费事的，照生产（β2 0.95、ε 1e-10、eod 不算 loss 等）。
- 生产语义**未知**的，**照现状**：按 MaxText 现在的行为做，GPU 对上。表里标「照现状」，以后要改两边一起改。
- 模型结构的缺口：GPU 照 MaxText 补。优化器与 loss 的细节：MaxText 加默认关的开关去对上原版 Megatron-Core，GPU 保持原版。每一处改动记在 [`CHANGES-LEDGER.md`](CHANGES-LEDGER.md)。

| 项 | 取值 | 出处 / 状态 |
|---|---|---|
| 优化器 | Muon：momentum 0.95、Newton-Schulz 5 步、Nesterov 开 | 生产 |
| NS 系数 | simple（3.4445, −4.7750, 2.0315）× 5 | 照现状（MaxText 写死） |
| Nesterov 的写法 | 原版 Megatron-Core：不带偏差修正，g' = β·mu_t + (1−β)·g_t | 优化器与 loss 细节由 MaxText 加开关对上原版 Megatron-Core（台账 M15） |
| 更新缩放 | MaxText 默认的宽度缩放 √max(1, fan_out / fan_in)（`muon_consistent_rms` 不设；即 Megatron-Core 的 `shape_scaling`） | 照现状 |
| NS 归一化 | x / max(‖x‖, 1e-7)（原版 Megatron-Core） | 同上（台账 M17） |
| MLA 的 q_b / kv_b / 输出门在 Muon 里怎么切 | q_b、kv_b 各自所有头拼成一个矩阵；输出门每头一个矩阵（`muon_attention_gate_whole_matrix: false`） | 照现状（生产 `muon_split_mla_q / _kv / _gate`、`further_split_mla_head` 语义未知） |
| 超连接（iHC）矩阵 | 走 Adam（`muon_exclude_ihc: true`） | 生产 |
| GLU 的 gate / up | 拼成一个矩阵正交化（`muon_fuse_glu: true`；Megatron-Core 原生融合 fc1） | 登记值 |
| 路由权重 | 走 Muon | 两边默认 |
| 标量参数的 Adam | 普通 Adam（原版 Megatron-Core；MaxText 关掉 Adam 分支的 Nesterov），β1 0.9、β2 0.95、ε 1e-10 | β、ε 照生产；普通 Adam 见台账 M16 |
| 梯度裁剪 | 1.0 | 生产 |
| 权重衰减 | 0（Muon 与 Adam） | 照现状（MaxText 默认；生产值未知） |
| 学习率 | 常数 1e-5（无 warmup、无衰减） | 登记值 |
| eod 掩码下 aux loss 的权重 | 索引器 KL 按每个 microbatch 的查询行数加权、MTP loss 每行（= 每个设备）按自己的原 / 滚动 token 数重标，都再除以全局有效 token 数（原版 Megatron-Core 的 per-token 写法） | 台账 M21、M22 |
| QK-Clip | 开，alpha 0.5、threshold 100；每层每头一个缩放，分母不加 1e-6（原版 Megatron-Core）；梯度累积时最大 logit 按 microbatch 取最大；MTP 层的注意力两边都裁 | 生产；分母与累积见台账 M18、M19；MTP 一条是 GPU 补的胶水（G7） |
| 索引器 | KD 训练，系数 0.1，`indexer_sparse_training: true`；各产出层（full 层与 MTP）的 KD loss 取平均再 × 0.1（GPU 每层系数 0.1 ÷ 产出层数）；KD loss 用原版 Megatron-Core 的写法：精确 log_softmax 与 log(max(p, 1e-10))，p = 0 的项不计 | 系数照生产；平均照 MaxText（GPU 已对上）；eps 写法见台账 M20 |
| MTP | 1 层，loss 系数 0.1 | 生产 |
| 路由 | sigmoid、缩放 2.827、辅助损失 0；路由偏置**不做**零均值更新（`routed_bias_zero_mean_update: false`，原版 Megatron-Core 没有这个选项），更新率 1e-3 | 生产；更新率是登记值（DeepSeek-V3 的值） |
| SwiGLU 激活截断 | 10 | 生产 |
| 可学习 sink | 初值来自共同的 HF 目录；`log_ratio` 照现状 | 照现状 |
| 重算 | 整层重算、路由索引存下不重选（`remat_policy: custom`，除层输入外全部重算，`moe_routing_maps: device`；`full` 下路由索引不存，见变更记录） | 生产 |
| 精度 | BF16 计算，FP32 主权重与优化器状态，路由 FP32 | 生产 |
| GPU 一侧的舍入位置 | TE GEMM FP32 累加只舍一次（`dense_fp32_accumulation`）；TE RMSNorm 只舍一次（`rmsnorm_single_rounding`）；路由 BF16 权重在 FP32 里乘（`routed_gate_bf16_rounding`）；不融合 SwiGLU 按算子舍入（`mlp_activation_round_per_op`）；索引器真 FP32（`indexer_fp32_scores`） | GPU 一侧（Megatron-Core 0.20、TE 2.20）的实际写法，TPU 用开关对上（台账 M8–M10、M12、M23） |
| 输出头 | FP32 权重 × FP32 隐状态，真 FP32 矩阵乘（`logits_dot_in_fp32` + `logits_matmul_precision: highest`；GPU 关 TF32） | GPU 一侧实际写法 |
| QK-Clip 的最大 logit | 只在选中且因果的位置上取缩放后 q·k 的最大值，不含 sink（`qk_clip_max_logit_exclude_sink`） | 原版 Megatron-Core |
| RoPE | theta 1e7，YaRN factor 1 | 照现状（生产 factor 8.0） |
| loss 掩码 | eod 位置不算 loss（Megatron `eod_mask_loss` 的写法），由 GPU 写进每步数据的 `loss_mask`，TPU 原样回放 | 生产；两边用同一份 `loss_mask` |
| 序列 / 批 | 4096；全局批 16 = 8 个设备 × microbatch 1 × 累积 2；GPU B200 × 8，数据并行 8、专家并行 4；TPU v5p 2x2x2（8 个设备） | 序列要大于 top-k 2048；设备 d 的第 m 个 microbatch 两边相同 |
| 数据 | fineweb-edu `sample/10BT` 第一个分片，混元 4 分词器（与真实权重截层同一份 `tokens.npy`） | — |

TPU 一侧的配置文件：[`frameworks/maxtext/hy4-mini-t1.yml`](frameworks/maxtext/hy4-mini-t1.yml)。

### 1.3 已登记的 MaxText 改动

都在 MaxText 混元 4 集成分支，开关默认关、关时行为不变（单测证明），T1 的配置把需要的开关打开。

| 改动 | 原来 | 现在 | 开关 / 默认 | 证据 |
|---|---|---|---|---|
| QK-Clip 在层扫描下按层缩放 | 扫描块里各层共用一个缩放（对扫描维也取最大），logit 小的层被多裁 | 每层每头一个缩放 | 修 bug，无开关 | 1 稠密 + 10 MoE 的 hy4-tiny 上实测（[`results/hy4-tiny11-qk-clip-scan.json`](results/hy4-tiny11-qk-clip-scan.json)）；单测 `qk_clip_scan_test` 修前 4 条全不过 |
| MLA 输出门在 Muon 里整块正交化 | 每头一个矩阵 | 一个 [emb, 头 × v] 矩阵 | `muon_attention_gate_whole_matrix`，默认关；T1 不开（照现状） | `muon_hy4_switches_test` |
| 超连接矩阵走 Adam | 走 Muon | 各层 pre / post / res alpha 与 HC head 的 hc_fn 走 AdamW | `muon_exclude_ihc`，默认关；T1 打开 | 同上 |
| GLU 拼接正交化 | wi_0、wi_1 各自正交化 | 在输出轴上拼成一个矩阵做 NS，更新拆回 | `muon_fuse_glu`，默认关（只支持 optax_muon）；T1 打开 | 同上；与手工拼接的参照逐元素一致 |

Muon 分区（哪个参数走 Muon、NS 作用在哪个矩阵上）由 `frameworks/maxtext/muon_partition.py` 从 MaxText 自己的判定写出，见 [`results/hy4-tiny-muon-partition.md`](results/hy4-tiny-muon-partition.md)；GPU 一侧每次运行写 `optimizer_partition.json`，L0 逐项核两边一致。

## 2. 被测与基线（登记时的口径，现行见第 0 节）

- T1：TPU 是参照一侧，基线 `t1-a`；被测是 GPU 的 `g1-a`，只和 `t1-a` 逐步比，不取平均、不挑最差。GPU 另外 7 份各自对 `t1-a` 判一遍，作旁证。
- T2：GPU 是 golden / 基线（B `gb2-a`、D1 `gd2-a`），TPU 是被测（`tb2-a`、`td2-a`），只和基线逐步比；GPU 另外 7 份各自对被测判一遍作旁证（变更记录 T2 一行）。
- 诊断用的臂随时可以跑，不改变被测的判定。

## 3. 噪声与判定（GPU 运行代号是登记时的，现行见第 0 节）

- 噪声上限 N：**只用 GPU**。GPU 同设置重跑两两配对，每一对各算三个判定量，每个判定量取各对的最大值（`judge_calibrated`）。份数按窗口：T1 前 300 步 8 份 28 对（现行 `g2-a … g2-h`；最早一代是 `g1-a … g1-h`），1000 步 16 份 120 对（`g3-a … g3-p`，2026-11-11 起）；T2、mid 各自 8 份 28 对（第 1 节「噪声 N」一行）。GPU 是参照答案，TPU 的波动不能进尺子（TPU 不能自证）。
- TPU 同设置重跑是否逐位相同照样记录（T1 两份 100 步、45 个逐步量逐位相同，[`hy4-mini-tpu-determinism.json`](alignments/01-mini/results/hy4-mini-tpu-determinism.json)），但不改变噪声的取法。
- TPU 只改执行参数的扰动运行**只作诊断**，不进任何门槛（只改执行、不改数学）：`t1p-1` 注意力块大小 256（`sa_block_*` 全部 256，默认 512）；`t1p-2` 专家并行 4 × FSDP 2（`ici_expert_parallelism: 4`）；`t1p-3` MoE 走不分组的稠密实现（`sparse_matmul: false`、`megablox: false`）；`t1p-4` 重算策略改成 `custom`、全部命名张量存 HBM（即 `hy4-4layers.yml` 的设置）。
- r = 被测对基线的判定量 ÷ N：r ≤ 1 过，r > 2 不过，之间待定；每项取三个判定量里最差的 r。
- 测量口径：grad_norm 的相对差以两边均值作分母；param_norm 两边都比更新后参数范数的精确值（GPU `param_norm_fp64`；TPU `param_norm_exact`，每 128 个元素一段 FP32 求平方和、段间 FP64 合成，不含路由偏置；台账 J13）——整棵参数树的 FP32 全局求和对精确值逐步误差 −4.4e-7 至 +2.9e-7，与 param_norm 的 N 同量级。不加标准误下限。

## 4. 指标与窗口（窗口是登记时的，现行见第 0 节）

| 指标 | 判定量 | 窗口（T1，300 步） | 窗口（T2，300 步；登记前原写 100 步） |
|---|---|---|---|
| 主 loss | 带符号均值、MARE、最大单步（首步不进带符号均值） | 第 1–300 步；第 101–300 步 | 第 1–300 步；第 101–300 步 |
| MTP loss | 同上 | 同上 | D1：同上 |
| 索引器 KD loss | 同上 | 第 1–300 步 | 第 1–300 步 |
| grad_norm（裁剪前） | 同上 | 第 1–300 步 | 第 1–300 步 |
| param_norm | 同上 | 第 1–300 步 | 第 1–300 步 |
| 路由偏置曲线（每层均值、标准差）与负载 CV | 同 loss | 全程 | 全程（B 第 1 层；D1 MTP 层） |
| QK-Clip：每步最大 logit、被裁的头数 | 同 loss；被裁头数两边一致 | 全程 | 全程 |
| 健康信号（NaN / Inf、裁剪是否触发、MoE 是否丢 token） | 两边一致 | 全程 | 全程 |

学习率不在上表：它是第 1.2 节登记的常数，不是观测量。改作设置核对，归 L0（第 5 节），两边每一步记下的值都要等于登记值 1e-5（目录 [`t1-l0-settings.yaml`](alignments/01-mini/catalogs/t1-l0-settings.yaml)），不计入 L4 的过 / 不过 / 待定。

## 5. 第 1 步与逐层（L0–L3）

- L0：逐位相同，没有容差：设置核对（逐步学习率等于登记值）、初始权重（两边从同一份 HF 目录载入，各自反向取回对 `tensor_sha256.json`）、输入数据指纹、配置逐项对照（含第 1.2 节每一行）、Muon / Adam 分区逐参数一致。负对照：不去交错的读法必须被判不一致。
- T0：hy4-tiny、FP32，逐层前向 / 反向与第 1 步更新后参数，两侧各对 FP64；R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64) ≤ 1.2 过。
- L1、L2（T1、T2）：对 FP64 的 R 比 ≤ 1.2 过；单层上的 top-k 翻转与路由翻转只统计、不设过线，按分差分桶，每个翻转对 FP64 定责（整网的翻转率由下面的「路由一致率」检查判）。
- L3 第 1 步整步的梯度、Muon 动量、Adam 状态与更新后参数：噪声只用 GPU 第 1 步参照（`g1s-*`，≥ 3 份两两），超出的项对 FP64 定责（R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64)）。FP64 参照在 CPU 上算：GPU golden 同一套 Megatron-Core 代码与 loss 语义，全程 FP64（`frameworks/megatron/step1_ref.py`），与 TPU 不同源；被测一侧（TPU）产出的数不能作参照（变更记录 2026-11-08）。离散选择（DSA top-k、MoE 路由）由 FP64 自己选，翻转只统计。L3 的正式判定一律用 TPU 自选的路由与 DSA top-k；把 GPU 的选择灌进 TPU 或 FP64 的结果只作诊断（说明差来自离散翻转还是连续误差），不作判定或定责依据：生产时 TPU 不可能回放 GPU 的选择（变更记录 2026-11-09）。
- L3 分辨率下限：GPU 第 1 步噪声 N = 0、且 rel_L2(TPU, FP64) 与 rel_L2(GPU, FP64) 都小于 BF16 单位舍入 u = 2⁻⁸ 的张量，记「低于分辨率（按下限过）」，不用 R 判——两边都不到一次舍入时 R 只是两种舍入实现之比（例：第 1 层索引器 `k_norm.weight` 两代写法 R 1.60 / 0.72）。
- 路由一致率（正式检查，2026-11-09 起）：TPU 自选的 MoE 路由（每个 token 选中的专家集合，不计顺序）与 DSA 索引器 top-k（每个 query 选中的位置集合）逐层比。**判定对真值**：每层 R = TPU 对 CPU FP64 自选的翻转率（DSA 为不重合率：选中位置里 FP64 没选的比例，对 query 平均）÷ GPU 各份（≥ 3 份）对 FP64 的同一量的最大值，≤ 1.2 过、≤ 2 待定、> 2 不过，整体取最差层（同 L3 的 R）；FP64 参照每行都存路由与 DSA top-k（`step1_ref.py --dsa-topk-all`）。对 GPU 的翻转率与 GPU 两两的同一量照报不判：GPU 重跑之间舍入位置相同，前几层两两逐位相同、尺子为 0，它量的是确定性，不是离真值多远。翻转按分差（FP64 加偏置分数第 k 名减第 k+1 名）分桶。测第 1 步（与 L3 同一批 token）和第 8、30、100 步（三方从同一份重建状态出发：GPU `g2w-b` 该步的主权重与路由偏置，各自选）。实现 `analysis/routing_agreement.py`，抓数文件先过完整性自检（DSA top-k 有重复或空位的 query 直接报错）。
- 第 1 步测不到权重侧的开关：从 BF16 检查点起训时，主权重与输出头输入第 1 步都在 BF16 格点上，只作用在权重一侧舍入或精度的开关此时是空操作；这类开关用训练后权重的重放或多步 A/B 判（变更记录 2026-11-08，[案例 13](cases/13-weight-side-switch-step1-blind-spot.md)）。

## 6. 停止规则

- 一个阶段在下面两种情况之一出现时结束：该阶段 L0 逐位、L1 / L2 全过、L4 每一项 r ≤ 1；或者每一个不过 / 待定的项在台账里都有定责证据，且已接受（理由写在台账）。
- 不在看到结果之后加份数、换基线、换窗口、换指标；要换就记一行变更，新旧判定并列。
- 查出某一侧实现与生产语义不符时，只修那一处（MaxText 加开关、默认关），该侧重跑，新旧两套判定都留在报告里。

## 7. 变更记录

| 日期 | 改了什么 | 为什么 | 在看到哪些结果之前 / 之后 |
|---|---|---|---|
| 2026-11-07 | 初版 | — | 之前：此时没有任何 Hy4 的 GPU 对 TPU 比较结果 |
| 2026-11-07 | 第 3 节加一条：GPU 重跑逐位相同时，L4 噪声改用两侧执行噪声的并集 | 初版没有覆盖「尺子宽度为 0」的情况；GPU 程序换成 HF 参考实现后可能出现 | 之前：此时还没有任何 Hy4 GPU golden 跑完，也没有任何 GPU 对 TPU 的比较结果 |
| 2026-11-07 | 阶段 C、D 的 golden 用 HF 默认的专家实现 `grouped_mm`（`--experts-impl grouped_mm`）；A、B 维持已跑完的 `eager`。C 的第 1 步参照改为 `gc1-b … gc1-d` | HF 默认（golden 以 HF `hy_v4` 参考实现为准，不额外指定专家实现时它走 `grouped_mm`） | 之前：此时没有任何阶段 C、D 的 L4 golden，也没有任何 GPU 对 TPU 的比较结果 |
| 2026-11-07 | 第 1 节的 golden（`ga`…`gd`）作废，等按生产训练配置重订后再登记 | golden 用的 AdamW 不是生产优化器（生产是 Muon + QK-Clip） | 之前：此时没有任何 GPU 对 TPU 的 L4 比较结果；只有阶段 A 的 L1 单算子 R（TPU 一侧），按新 golden 重做 |
| 2026-11-07 | **全文按反向方案重写**（旧版见本分支 `a88b93e` 的 `PROTOCOL.md`）：语义以 TPU 一侧 MaxText 混元 4 分支为准，GPU 一侧 Megatron-Bridge 补齐对上；阶段改为 T0 / T1 / T2；第 1.2 节登记训练设置（未知项与待定项单列）；第 1.3 节登记四项 MaxText 改动的决定 | MaxText 分支实现最完整；生产用 Muon + QK-Clip + 索引器 KD 训练 | 之前：此时没有任何混元 4 的 T0 / T1 / T2 运行，也没有任何 GPU 对 TPU 的比较结果 |
| 2026-11-07 | 第 1.2 节：生产语义未知的一律「照现状」（MaxText 现在的行为，GPU 对上），撤掉「待定」清单；学习率改为常数 1e-5；两边设备数取 8 | 目标是两边对齐，不是复现生产 | 之前：此时没有任何 T0 / T1 / T2 运行，也没有任何 GPU 对 TPU 的比较结果 |
| 2026-11-07 | 第 1.2 节：Nesterov 写法、标量参数的 Adam、NS 归一化、KD loss 的 eps 写法、QK-Clip 的分母与梯度累积，改为 MaxText 加开关对上原版 Megatron-Core（T1 打开） | 优化器与 loss 细节以原版 Megatron-Core 为准，GPU 保持原版 | 之前：此时没有任何 T0 / T1 / T2 运行，也没有任何 GPU 对 TPU 的比较结果 |
| 2026-11-07 | **T1 冻结**。第 1.1 节：TPU 同设置 2 份 + 只改执行参数的扰动 4 份（定义见第 3 节），GPU 8 份；第 1.2 节：加 eod 掩码下 aux loss 权重一行（台账 M21、M22）；第 3 节：写明 T1 用并集噪声。T1 的 TPU 配置是 `frameworks/maxtext/hy4-mini-t1.yml`（MaxText `8611ea6b`），GPU 配置见第 1.2 节与台账 G1–G11 | TPU 已实测逐位相同，8 份同设置没有信息 | 之前：此时 T1 没有任何正式运行，也没有任何 GPU 对 TPU 的比较结果 |
| 2026-11-07 | 第 3 节测量口径：T1 的 param_norm 两边改读第 s 步**更新之后**的 FP32 全局范数（TPU `param_norm`，GPU `param_norm`），不用分组合成 | GPU 的 `param_norm` / `param_norm_fp64` 是更新之后的值，TPU 的 `param_norm_grouped` 是更新之前的值，而且 TPU 的分组漏了若干小参数（更新前的初始值 211.287616，HF 目录逐张量 FP64 算出 211.287661；漏掉的平方和约 0.019）。两边 FP32 全局范数的测量误差各自进各自一侧的噪声对 | 之前：此时还没有算任何 GPU 对 TPU 的检查结果（只看过两边第 1–3 步各字段的取值，用来核语义） |
| 2026-11-07 | 第 1.2 节：路由偏置改为不做零均值更新（TPU `routed_bias_zero_mean_update: false`），T1 的 TPU 6 份重跑。第一次比较（`t1-l4.results.json`，零均值开）里路由偏置均值不过（r 19）；原版 Megatron-Core 0.20 没有零均值更新，协议原来写的「两边都零均值」GPU 做不到。新旧判定并列：旧结果文件原样保留，新结果另存 `t1-l4-nozm.results.json`；偏置整体平移不改 top-k、门控权重用不带偏置的分数，预期只影响偏置均值这一项（推断，以新结果为准） | 规则：优化器与 loss 细节 MaxText 对上原版 Megatron-Core，GPU 保持原版 | **之后**：已看到第一次 T1 比较的全部结果 |
| 2026-11-07 | 第 1.2 节加一行「GPU 一侧的舍入位置」：TPU 打开 `dense_fp32_accumulation`、`rmsnorm_single_rounding`、`routed_gate_bf16_rounding`、`mlp_activation_round_per_op`，以及 `indexer_fp32_scores`（索引器真 FP32，原版 Megatron-Core 的写法）。T1 的 TPU 6 份重跑，新结果另存，旧结果保留。第 1 步逐张量梯度比较里，TPU 对 GPU 的差普遍是 GPU 散布的 2–3 倍；按 GPU 一侧给出的舍入位置打开这四个开关后，逐张量比值中位数 2.67 → 1.65、超过 2 倍的张量 139 → 52（`t1-step1-precision-switches.json`） | 规则（优化器 / loss / 数值细节 MaxText 对上原版 Megatron-Core）；是精度差就加开关、记台账、T1 打开 | **之后**：已看到 T1 两次比较与第 1 步逐张量比较 |
| 2026-11-07 | 报告口径：学习率从 L4 的比较项移到 L0 设置核对（新目录 `t1-l0-settings.yaml`：两边每一步都等于登记的常数 1e-5），不再计入 L4 的过 / 不过 / 待定；第 4 节表格删去学习率一行，第 5 节 L0 加上这一项。janus 的 `scalar_series` 加 `equals`（两边都要等于登记值）。`t1-l4`、`t1-l4-nozm` 两份结果按新目录重出，其余各项判定不变（`t1-l4-nozm`：过 2、不过 2、待定 5；设置核对：过 1） | 学习率是设定的常数，不该当成观测指标打分 | 报告口径变更，不改协议实质：判定规则、噪声、窗口都没动；已看到 T1 比较结果 |
| 2026-11-07 | MaxText 一侧的代码位置改为`maxtext` 的混元 4 集成分支（分支名带个人账号，不写进仓库；建于 `d723fa2d`，与 `feat/68-hy4-reverse` 当时的头相同）；之后的 MaxText 提交都进这条分支，`feat/68-hy4-reverse` 停用；`stage.sh` 默认打包 `MAXTEXT_BRANCH` 指定的分支 | | 只改代码位置，不改任何数值：此前各运行记的 MaxText 提交号仍然有效 |
| 2026-11-07 | 第 1 节代号说明改写：原来写「`-1` 后缀（如 `t1s-a`）是第 1 步全量抓数参照」，与后来 `t1s-*` 表示「打开舍入开关后的重跑」冲突；改为按设置分代列出代号，GPU 第 1 步参照仍叫 `g1s-*` | 文字更正，代号与已有运行目录一致 | 不改任何设置或判定 |
| 2026-11-07 | 第 1.2 节四处 TPU 设置改动，T1 的 TPU 6 份重跑（`t1t-a`、`t1t-b`、`t1tp-1..4`，扰动定义同 `t1np-*`），新结果另存、旧结果保留：（1）输出头加 `logits_matmul_precision: highest`：T1 原来只开 `logits_dot_in_fp32`，FP32 矩阵乘在 TPU 默认精度下只做一遍 BF16，而 GPU 的输出头是真 FP32（关 TF32）；（2）重算改为 `remat_policy: custom`（除层输入外全部重算，路由索引 `moe_routing_maps: device` 存下）：`full` 下 MaxText 不读 `moe_routing_maps`，反向重算时会重选路由，与本节「路由索引存下不重选」的写法不符；（3）`qk_clip_max_logit_exclude_sink: true`（MaxText 新开关，默认关）：TPU flash 内核记的最大 logit 从 sink 起算，原版 Megatron-Core 不含 sink。（4）`indexer_weights_proj_bf16_weight: true`（MaxText 新开关，默认关）：索引器 weights_proj 原来乘 FP32 主权重，原版 Megatron-Core 乘 BF16 的模型权重、FP32 输出。T1 里 QK-Clip 从未触发（τ 100，两边最大 logit 约 0.14），第（3）条预计不改数值（推断） | 规则（数值细节 MaxText 对上原版 Megatron-Core / GPU 实际写法） | **之后**：已看到 T1 三次比较 |
| 2026-11-07 | 第 3 节、第 5 节：**撤回两侧并集的规则**。L4 噪声只用 GPU 8 份两两 28 对，每个判定量取各对的最大值；L3 噪声只用 GPU 第 1 步参照；TPU 只改执行参数的扰动运行只作诊断，不进门槛。T1 已有的三次 L4 比较（`t1-l4`、`t1-l4-nozm`、`t1s-l4`）按新尺子重判，结果文件覆盖，旧判定见本行之前的提交 | GPU 是参照答案，TPU 的波动不能进尺子（不能自证） | **之后**：已看到 T1 三次比较的全部结果 |
| 2026-11-07 | 第 5 节 T0 一行：R 写成 rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64) | R 统一为 TPU ÷ GPU；文字更正，已有数值本来就是这个方向 | 不改任何设置或判定 |
| 2026-11-07 | GPU 打开 `hy4_attn_scores_fp32`（注意力分数在 FP32 里算再进 softmax；台账 G17），GPU 重跑 8 份 `g2-a … g2-h` 与第 1 步参照 `g2s-a/b/c`；之后的比较以 `g2-*` 为 GPU 参照、噪声取 `g2-*` 两两 28 对。比较目录的角色改为：GPU 是 golden / 基线，TPU 是被测（`t1t-l4`：被测 `t1t-a`，基线 `g2-a`）。| 分数先舍成 BF16 再 softmax 是 janus eager 参照路径的 bug，生产的融合核保留 FP32；R 统一为 TPU ÷ GPU（GPU golden、TPU 被测） | **之后**：已看到 T1 前三次比较 |
| 2026-11-07 | 测量口径：param_norm 两边都改读更新之后参数范数的精确值（GPU `param_norm_fp64`；TPU 新列 `param_norm_exact`，每 128 个元素一段 FP32 求平方和、FP64 合成，不含路由偏置）。原来比的是两边各自的 FP32 全局求和：TPU 的 `learning/param_norm` 整棵树在 FP32 里一路相加，对精确值偏低 1e-7 至 3e-7（第 1 步 −1.39e-7），而两边精确值第 1 步只差 1.6e-9。`t1t-l4` 的被测换成 `t1t-c`（与 `t1t-a` 同设置、同代码路径，除 `step_time` 与新列外逐位相同）。| 每一项不对齐都要有根因；param_norm 的系统偏差先拆出测量误差 | **之后**：已看到 `t1t-l4` 的结果 |
| 2026-11-07 | 文字更正：「GPU 一侧的舍入位置」那一行写的「打开这四个开关后中位数 2.67 → 1.65、超过 2 倍 139 → 52」是**三个**开关（`dense_fp32_accumulation`、`rmsnorm_single_rounding`、`routed_gate_bf16_rounding`）的结果；四个都开（再加 `mlp_activation_round_per_op`）是 1.72 / 62（`t1-step1-precision-switches.json`）。T1 的设置不变 | 结果文件与本表不一致，按结果文件更正 | 不改设置或判定 |
| 2026-11-07 | 第 1.2 节：TPU 打开 `ihc_stream_sum_unrolled`（iHC 四路流加权和改为逐路 FP32 乘加，同 GPU 的写法；台账 M28），从下一代 TPU 运行起生效（`t1t-*`、1000 步的 `t1u-*` 仍是旧设置） | 每一项不对齐都要有根因 | **之后**：已看到 t1t 第 1 步逐张量结果 |
| 2026-11-07 | 加 1000 步比较：GPU g2 设置 8 份 `g3-a … g3-h`（1000 步，第 1–300 步数据与 g1 / g2 逐条相同），TPU 现行 t1t 设置 2 份 `t1u-a`、`t1u-b`（回放 g3-a 的数据）；目录 `t1u-l4.yaml`（窗口第 1–1000 步、第 101–1000 步，噪声 g3-* 两两 28 对）与 `t1u-l0-settings.yaml`。判定规则不变 | 补 1000 步比较 | 之后：已看到 300 步的各次比较 |
| 2026-11-07 | 报告口径：L3 第 1 步逐张量梯度按第 5 节原文「超出的项对 FP64 定责」判，超出 N 的张量（不只超出 2N 的）都要对参照定责：有超 2N 而 R > 1.2 的判不过，只有 1–2N 而 R > 1.2 的判待定；另按算子分组各判一次，给检查清单「模型的每个算子由哪条检查覆盖」用。总览与检查清单加 1000 步一列（`t1u-l4`、`t1u-l0-settings`）与现行 t1t 的第 1 步（`t1t-step1-grad-vs-ref`）。`t1s`、`t1t` 两代的整网判定不变（都不过）；t1s 的未定责张量由 2 个变为 5 个（多出的 3 个在 1–2N） | 原来的生成脚本只对超出 2N 的张量定责，比第 5 节的写法松 | 报告口径变更，不改协议实质；已看到 t1s、t1t 第 1 步逐张量结果 |
| 2026-11-07 | **T2 登记（配置与运行）**。第 1.1 节 T2 一行由「每侧 4 份、100 步」改为：GPU 同设置 8 份 `gb2-a … gb2-h`（B）/ `gd2-a … gd2-h`（D1）各 300 步、第 1 步全量抓数 3 份 `gb2s-*` / `gd2s-*`；TPU 2 份 `tb2-a`、`tb2-b` / `td2-a`、`td2-b`（第二份只核逐位）与第 1 步诊断 `tb2diag` / `td2diag`。TPU 配置 `frameworks/maxtext/hy4-t2-b.yml`、`hy4-t2-d1.yml`：T1 现行 `hy4-mini-t1.yml` 的全部数值与优化器开关原样（含台账 M28 `ihc_stream_sum_unrolled`），只换 `model_name: hy4-preview`，单测逐键核对；初始权重是真实权重截层转换的检查点（B 用 AdamW 时期的转换，D1 新转换）。GPU 参数与 g2 / g3 相同（q·k 在 FP32 里、输出头 dgrad 每 1024 段、Muon、QK-Clip、索引器 KD、eod 掩码、专家并行 4），只改层数与 MTP；第 1 步参照不抓全部路由专家（janus 默认每层两个）。作业规格见 `alignments/02-real-weight/README.md`。第 2 节 T2 一行改为 GPU 基线、TPU 被测（与 T1 现行角色一致） | T2 用 300 步、噪声只用 GPU 8 份（28 对），与 T1 同一把尺子；原来的「每侧 4 份」是 TPU 也进尺子时期的写法，原来的「基线是 TPU 第 1 份」是改角色之前的写法 | 之前：此时没有任何 T2 运行 |
| 2026-11-07 | **T2 登记（窗口与噪声）**。第 4 节 T2 窗口改为 300 步：主 loss 第 1–300 步与第 101–300 步，其余第 1–300 步；MTP loss 只有 D1；路由偏置 B 是第 1 层、D1 是 MTP 层。噪声上限 N 只用 GPU 同设置 8 份两两 28 对，每个判定量取最大值（第 3 节）；L3 噪声只用 `gb2s-*` / `gd2s-*` 三份两两。1000 步可选（GPU 各 8 份、TPU 各 1 份），另写目录，不进 T2 的登记判定。目录 `alignments/02-real-weight/catalogs/t2b-l4.yaml`、`t2d1-l4.yaml` | 与 T1 同口径；300 步是 T1 已验证过的窗口 | 之前：此时没有任何 T2 运行 |
| 2026-11-07 | **T2 登记（检查）**。L0 载入：TPU 检查点用当前 MaxText 提交反向取回、逐张量对 HF 清单 `tensor_sha256.json`，RoPE 去交错逐位，dtype 清单（`model/t2_l0.sh`，B 用 `l0-ckpt-convert-b.yaml`、D1 用 `l0-ckpt-convert-d1.yaml`）；GPU 每份运行的载入自检通过、`meta.json` 记的清单 sha256 等于同一份清单；两边对同一份清单逐位即初始权重逐位相同。L0 设置：两边每步学习率等于 1e-5（`t2b-l0-settings.yaml`、`t2d1-l0-settings.yaml`）；TPU `meta.json` 的 `yml_keys_not_effective` 为空；GPU `meta.json` 的设置与作业规格逐项相同。L0 数据：TPU 输入指纹对 GPU 基线逐位。L3：第 1 步逐张量梯度，超出 GPU 三份参照散布的张量对 FP32 / FP64 参照定责（第 5 节，判法同 T1 现行口径）。L4：第 4 节各项，判法同第 3 节 | 与 T1 同一套检查；真实权重下多出来的只有检查点来源 | 之前：此时没有任何 T2 运行 |
| 2026-11-07 | 第 1.2 节优化器一行：TPU 打开 `muon_mcore_ns_bf16`（MaxText 新开关，默认关，台账 M30、M31）：Muon 的 Newton–Schulz 在 BF16 张量上迭代，同 GPU 的 emerging_optimizers（`muon_fp32_matmul_prec` medium）。下一代 TPU 运行起生效，旧结果保留。GPU 的 Muon 增量离 FP64 2.7%–13%（g2s-a、g2o-a），TPU 默认（FP32 迭代）约 3e-3；连续 8 步回放里开关关时 TPU 对 GPU 的 Muon 增量差最多 12.8%、开时 1.6%（`results/t1-optim-replay-tpu.json`）。只看 R，开关关时 Muon 张量多数已经过 | 规则：优化器细节 MaxText 对上原版 Megatron-Core，GPU 保持原版（GPU 是 golden） | **之后**：已看到优化器回放 `tw-optim8` 的结果 |
| 2026-11-07 | 第 1.2 节数值细节：TPU 打开 `transcendental_accuracy_highest`（MaxText 新开关，默认关，台账 M32、M33）：FP32 的 exp / log / sigmoid / silu（索引器 KD、MLA 输出门、iHC 门与 hc_head、路由、MLP 与路由专家的 silu、主 loss 与 MTP loss 的交叉熵）用 accuracy=HIGHEST。下一代 TPU 运行起生效，旧结果保留。TPU v5p 默认的 FP32 exp / log / sigmoid / silu 实测 ulp_max 105 / 62 / 83 / 86（`results/tpu-elementwise-ulp.json`），GPU 约 1–2；第 1 步诊断里 KD 的 LSE −0.953 → +0.025 格、第 0 层 KD 第 0 行对 FP64 −1.155e-3 → +9.7e-5、第 0 层 KD 对 GPU −1.37e-3 → −4.1e-5（`results/t1-step1-transcendental-accuracy.json`）；L1 交叉熵 R 2.95（开关关时）。第 1 步梯度里第 3 层 FFN 的 hc_base 不受影响（R 8.7 → 6.7） | 规则（数值细节 MaxText 对上 GPU 实际写法，GPU 是 golden） | **之后**：已看到 L1、第 1 步诊断与 T1 历次比较 |
| 2026-11-08 | T2 的 TPU 配置暂关 `qk_clip_max_logit_exclude_sink`（台账 M25）：阶段 B 打开时第 1 步更新后第 1 层 wq_b / wkv_b 变 NaN，关掉后正常（tb2dbg / tb2dbg2）。QK-Clip 在 T2 不触发（GPU 最大 logit B 16.4、D1 60.1，τ 100），关掉不改任何数值；根因待查。| 运行发散，来自这个统计量开关 | **之后**：已看到 T2 B 第一次 TPU 运行 |
| 2026-11-08 | 第 1.2 节：TPU 打开 `ihc_force_bf16_rounding`（台账 M36，iHC 分支输入与更新后的流用 reduce_precision 舍成 BF16，同 GPU 的 eager 写法），从 t1zr 起生效。打开 M28 后 TPU 1000 步主 loss 系统偏高（5 个运行同号），推断是 XLA 在 excess precision 下省掉了 iHC 输出的 BF16 往返（开关效果支持，HLO 未核）；打开后 t1zr-a 对 GPU 均值的偏差 +8.6e-6 → +3.5e-6，1000 步与前 300 步都没有不过 | 每一项不对齐都要有根因 | **之后**：已看到 t1zk 的 1000 步结果 |
| 2026-11-08 | 第 1.2 节数值细节更正：`transcendental_accuracy_highest` 在 MaxText `b8d4ba8b` 之前对 sigmoid / silu 实际没起作用（`lax.logistic` 带 accuracy=HIGHEST 在 TPU v5p 上与默认逐位相同，`results/tpu-elementwise-ulp.json`，`tw-elem3`：sigmoid 83、silu 86 ULP 不变；exp / log 降到 1 / 3 ULP）。`b8d4ba8b` 起开关里的 sigmoid / silu 改用 HIGHEST 的 exp 搭（台账 M37），iHC 门、hc_head、路由 sigmoid、FP32 silu 的数值随之变化；T1 / T2 从 `t1zs` 起生效，此前的运行（含 `t1z*`、`t1zr`）只有 exp / log 一半，旧结果保留 | 开关原意就是这几处对上 GPU（M32）；实测没生效 | **之后**：已看到 `tw-opmini-a` / `tw-opmini-b` 逐位相同与 `tw-elem3` 的 ULP |
| 2026-11-08 | 第 1.2 节：TPU 打开 `muon_adam_accurate`（台账 M34：标量 Adam 的除法用带修正的商、sqrt 加一次牛顿迭代、1−β^t = −expm1(t·log β)），T1 与 T2；T2 重新打开 `qk_clip_max_logit_exclude_sink`（M25，阶段 B 的 NaN 由 M35 修好）。从下一次打包的运行起生效 | TPU 8 步优化器回放里几个小张量的 Adam 更新 R 到 3.8，CPU 上同样的算法 R ≤ 1.02：差在 TPU FP32 除法与 1−β^t 的精度 | **之后**：已看到 T1 / T2 的比较与优化器回放 |
| 2026-11-08 | 第 5 节（L3 第 1 步逐张量）加**分辨率下限**，从本日起对后续判定生效，并按同一规则重标已有结果：一个张量若 GPU 第 1 步噪声 N = 0，且 rel_L2(TPU, 参照) 与 rel_L2(GPU, 参照) 都小于 BF16 的单位舍入 u = 2⁻⁸ ≈ 3.91e-3，则记「低于分辨率（按下限过）」，不再用 R 判。理由：两边的误差都不到一次 BF16 舍入，R 是两次舍入实现之比，实测在同一张量上随 TPU 写法在 0.62–1.59 间摆动（t1-step1-small-tensor-variants.json） |  | **之后**：已看到 t1t / t1v / t1z 第 1 步逐张量结果 |
| 2026-11-08 | **hy4-mid 登记（配置、运行、检查）**。新阶段 T1m（第 1.1 节）：hy4-mid，随机初始化 HF 目录 `make_tiny_hf.py --config mid --layers 7 --mtp 1 --seed 0`（权重矩阵标准差 0.006，同 hy4-mini），两边从它载入；TPU 配置 `frameworks/maxtext/hy4-mid-t1.yml`：T1 现行 `hy4-mini-t1.yml` 的全部数值与优化器开关原样，只换 `model_name: hy4-mid`（单测逐键核对）；MaxText 加模型 `hy4-mid`（集成分支，不影响其他模型）。GPU 参数同 g2 / g3（只换 HF 目录），8 份 `gm-a … gm-h` 各 300 步、第 1 步全量抓数 `gms-a … gms-c`；TPU `tm-a`（300 步，回放 `gm-a`）与第 1 步诊断 `tmdiag`。检查同 T1：L0（权重反向取回逐位、RoPE、GPU 载入自检、学习率 `tm-l0-settings.yaml`、数据指纹）、L3（第 1 步逐张量对 `gms-*`，超出的对 FP32 参照定责，含分辨率下限）、L4（`tm-l4.yaml`：主 loss 第 1–300 / 101–300 步、MTP loss、索引器 KD、grad_norm、param_norm、路由偏置；噪声只用 `gm-*` 两两 28 对） | T1 与 T2 之间同时换了尺寸与权重，T2 不过的项分不清是哪一样带来的；hy4-mid 只换尺寸 | 之前：此时没有任何 hy4-mid 运行 |
| 2026-11-08 | T2 加一代 TPU 运行 `tb4-a`、`td4-a`（300 步，回放 `gb2-a` / `gd2-a`），用 T1 现行全部开关（M34、M36、M37，M25 重新打开），目录 `t2b4-l4.yaml`、`t2d14-l4.yaml` 与设置核对 `t2b4-l0-settings.yaml`、`t2d14-l0-settings.yaml`；判定规则、窗口、噪声同 `t2bz-l4` / `t2d1z-l4`，旧结果保留。另加诊断（不进门槛）：`tb-fp32-30`、`td-fp32-30`，同一批数据上 30 步的 TPU FP32 轨迹（`dtype=float32 matmul_precision=highest sparse_matmul=false`），逐步比 TPU 与 GPU 8 份谁离它更近（同 T1 的 `t1-early-grad-norm-vs-fp32.json`）。`t2d1z-l4` 主 loss 带符号均值（N 6.47e-6，TPU 1.92e-5，r 2.97）、MTP loss 带符号均值（N 2.43e-5，TPU −1.04e-4，r 4.28）不过，`t2bz-l4` 主 loss MARE（N 1.85e-5，TPU 3.75e-5，r 2.03）不过 | 每一项不对齐都要有根因；先确认第 2–100 步的差在哪一侧 | **之后**：已看到 `t2bz-l4`、`t2d1z-l4` 与第 1 步对 FP32 的结果 |
| 2026-11-08 | 第 1.2 节 MoE 一行：TPU 打开 `moe_mcore_prob_before_fc2`（MaxText 新开关，默认关，台账 M40）：路由专家在 wo 之前乘 FP32 的路由权重（act×up 已舍 BF16 → × prob → 舍一次），top-k 输出不再加权、在 BF16 里逐个相加，同 GPU（Megatron-Core 的 experts.py 与 scatter_add_ 的 unpermute）。T1（`hy4-mini-t1.yml`）、T2（`hy4-t2-b.yml`、`hy4-t2-d1.yml`）与 hy4-mid（`hy4-mid-t1.yml`）一起打开，下一代 TPU 运行起生效，旧结果保留。GPU 核对（台账 G26）读到乘法位置不同；单测里整块 MoE 开、关两组各自对自己顺序的参照前向反向逐位一致 | 规则（数值细节 MaxText 用默认关的开关对上原版 Megatron-Core，GPU 保持原版） | **之后**：已看到 G26 的 GPU 核对与 M40 的单测 |
| 2026-11-08 | 第 1.2 节 MLP 一行：TPU 打开 `mlp_fused_fc1_dgrad`（MaxText 新开关，默认关，台账 M41）：SwiGLU 两个上投影的输入梯度在 FP32 里相加后只舍一次，同 GPU 的融合 linear_fc1（稠密层、共享专家、路由专家三处）。T1、T2 与 hy4-mid 的 yml 一起打开，下一代 TPU 运行起生效，旧结果保留 | 规则（同上）；MoE 输入梯度与「fc1 融合」写法差约一次 BF16 舍入（dX · rel_L2：MlpBlock 3.1e-3，路由 MoE 整块 2.7e-3），打开后为 0 | **之后**：已看到 M40 / M41 的单测 |
| 2026-11-08 | 第 5 节 L3：第 1 步逐张量梯度的定责参照由 TPU 一侧的 FP32 + HIGHEST 运行（`t1diag18b` 等：MaxText 同一张图改 dtype float32、matmul_precision highest）改为 CPU 上的 FP64 整网重算（`frameworks/megatron/step1_ref.py`：GPU golden 同一套 Megatron-Core 代码与 loss 语义，参数、激活、梯度全程 FP64，逐算子核过没有 FP64 以外的浮点输出；同一份初始权重、同一批第 1 步数据）。判定规则（超出 N 的定责、R ≤ 1.2、分辨率下限）不变，T1 的 t1t、t1v、t1z 三代按新参照重判，旧结果文件保留作对照。离散选择由 FP64 自己选；灌 GPU 选择的版本只做得了第一行（GPU 只抓了 rank 0 第 0 个 microbatch 的选择），作灵敏度诊断，不用于定责。| 参照不能由被测一侧产出；旧参照与本节原文「对 FP64 定责」不一致。实测旧参照离 FP64 的 rel_L2 中位数 2.71e-3，是 GPU 离 FP64（6.83e-3）的 0.41 倍，205 个张量里 28 个超过一半（`results/t1z-step1-grad-vs-fp64.json`） | **之后**：已看到 t1t、t1v、t1z 第 1 步逐张量结果。新旧并列（t1z）：旧参照下不过，未定责 3 个；FP64 下仍不过，未定责 5 个（第 3 层 FFN 前 `hc_base` 转为已定责，新增第 2 层注意力前 `hc_scale`、第 4 层 FFN 前 `hc_base` 与 `hc_fn`） |
| 2026-11-08 | 报告口径（不改判定规则）：L1 的 KD 输入、低阈值 QK-Clip 最大 logit、L2 逐层回放、t1s 的索引器梯度这四项仍只有对 TPU FP32 + HIGHEST 参照的 R，总览与检查清单把它们记成「只报」（🟡），不再按 R ≤ 1.2 判过 / 不过；等 CPU FP64 参照后重判 | 同上一行：参照不能由被测一侧产出，对它的 R 不作定责 | **之后**：已看到四项结果。之前 QK-Clip 最大 logit 判不过（第 3 层 R 1.29），L2 判过（最差 R 1.15），KD 输入判过（0.94–0.99），t1s 索引器梯度判过（0.78–1.03）；现在都记「只报」 |
| 2026-11-08 | 第 5 节（方法说明，不改判定规则）：从 BF16 检查点起训时，第 1 步的主权重与输出头输入都在 BF16 格点上，只作用在权重一侧舍入或精度的开关（M13、M27、M8）第 1 步必然逐位不变；L3 第 1 步的比较只能为激活与梯度一侧的舍入定责。权重侧开关的效果与定责，用训练后权重的单算子重放（换上第 N 步的 FP32 主权重）或多步单开关 A/B（`tq-*`）给出；第 0 / 1 步逐位相同不作为「开关无效」的证据 | 实测 M13 第 1 步前向逐位、300 步关掉后主 loss 最大单步 r 0.61 → 6.03（N 4.53e-5，TPU 2.75e-5 → 2.73e-4）（[案例 13](cases/13-weight-side-switch-step1-blind-spot.md)） | **之后**：已看到逐开关第 1 步结果（`tw-sw-fwd2`、`tw-sw-m13`）与 `tq-m13` |
| 2026-11-08 | 第 5 节 L3 的 CPU FP64 参照同样用于 hy4-mid（T1m）：`step1_ref.py` 加逐层重算（主干与 MTP 层，torch 的非重入 checkpoint）与专家 BF16 存储（路由专家权重存检查点的 BF16、用时精确转 FP64，权重梯度在 FP64 里算后累加到磁盘上的 FP64 文件），两者在 hy4-mini 上与不用时 205 个张量逐位相同；判定规则同 T1 | hy4-mid 全 FP64 单进程超过 250 GB 的内存上限 | 之前：此时没有任何 hy4-mid 对 CPU FP64 的比较结果 |
| 2026-11-08 | 第 1.2 节注意力一行：TPU 打开 `splash_dsinks_fp32`（MaxText 新开关，默认关，台账 M42）：tokamax splash 的 sink 梯度改在 FP32 里算（dsink = −Σ_q exp(sink − lse)·di），同 GPU（autograd 穿过 FP32 softmax，p_sink 与求和在 FP32）。T1、T2 与 hy4-mid 的 yml 一起打开，下一代 TPU 运行（t1zv）起生效，旧结果保留。hy4-mid 第 1 步 sink 梯度对 CPU FP64 同向 R 1.25–1.42；TPU 精度探针（`results/t1-sink-probe-tpu.json`）里 sink 梯度的 R 关 1.62–2.88 → 开 0.61–2.00，第 6 层与 MTP 仍超线（di 用的是 BF16 的 o） | 规则（同上） | **之后**：已看到 TPU 精度探针 |
| 2026-11-08 | 第 5 节 L3 的 CPU FP64 参照用于 T2（B、D1）：`step1_ref.py --expert-store ram`（路由专家权重存检查点的 BF16、用时精确转 FP64，权重梯度在内存里 FP64 累加；hy4-mini 上与不用时 205 个张量逐位相同），输出只留 GPU 全量抓数里有的路由专家（每层 0、1 号），grad_norm 按全部专家算；TPU 一侧用现行 T2 yml 的第 1 步诊断 `tb6diag`、`td6diag`。判定规则同 T1 | T2 截层全 FP64 的参数加梯度约 194 GB，加激活超过单进程 250 GB 的上限 | 之前：此时没有任何 T2 对 CPU FP64 的比较结果 |
| 2026-11-08 | 报告口径（不改判定规则）：上面「只报」那一行的四项改对 CPU FP64 参照重判、恢复按 R ≤ 1.2 判过 / 不过。KD 输入：`t1zdiag2` 对 GPU `g1p-kdd`（`results/t1-l1-kd-qkclip-vs-fp64.json`）；低阈值 QK-Clip 最大 logit：同一文件；L2 逐层回放：`tw-l2-g` 对 `layer_ref.py` 的 FP64（`results/t1-l2-layer-vs-fp64.json`）；t1s 的索引器梯度：`t1s-step1-grad-vs-fp64.json`。总览与检查清单的门槛栏写「对 CPU FP64 的 R 比 ≤ 1.2」 | FP64 参照到了，与 TPU 不同源，R 能定责 | **之前**：四项记「只报」。**之后**：KD 输入过（R 0.96 / 0.94 / 0.98），L2 过（最差是第 2 层层输入梯度 R 1.16），t1s 索引器梯度过（0.81–1.05），低阈值 QK-Clip 第 3 层最大 logit 不过（R 1.29） |
| 2026-11-08 | 第 5 节 L3（第 1 步逐张量梯度）加 TPU 梯度文件完整性检查：`step1_vs_fp64.py` 给了 `--tpu-diag` 就逐张量核文件里的平方和对设备上算的（`diag_step1.json` 的 `per_tensor.sq`），相对差 > 1e-10 的记为落盘损坏，不进统计与判定，有这种张量时整体至少判待定。D1 `td6diag` 有 5 个、hy4-mid `tm2diag` 有 9 个（`t2d1-step1-grad-vs-fp64.json` 已按此重判，原先列为未定责的 MTP 层输出门 R 7.82、输出头 R 2.31 撤销）；`tb6diag`、`tmdiag`、`t1zdiag2`、`t1vdiag` 全部对得上 | 诊断文件从 Pod 分块取回时流偶尔被截断，截断的那块在拼好的文件里留下整段 0（干净张量的相对差在 1e-14 量级，损坏的 1.5e-7 到 6.6e-3，全是文件偏小）；取回一侧逐块核字节数、整体核 sha256，本条在其后再核一次 | 之前：TPU 梯度文件直接拿来比，没有核 |
| 2026-11-09 | 第 1.2 节：对齐用的四份 yml（T1 / T2 B / T2 D1 / mid）把 `splash_dsinks_fp32`（M42）关回去，现行比较维持 `t1zu-a`。M42 单开让整网离 GPU 更远（1000 步主 loss 带符号均值 r 0.65 → 1.32，路由偏置均值 r 0.77 → 1.61），在照 GPU 写法的 M42b（dp 逐位置舍 BF16、sink 路与求和 FP32）验证之前，从对齐配置撤下 | 对齐配置的目标是照 GPU 的写法 | **之后**：已看到 `t1zv-b` 的 1000 步与前 300 步结果 |
| 2026-11-09 | 第 1.2 节：对齐用的四份 yml 打开 `splash_dsinks_gpu_order`（M42b：sink 梯度照 GPU 的写法，dp 逐位置舍 BF16、p 与求和 FP32；M42 保持关），现行比较从 `t1zu-a` 换成 `t1zw-a`，旧代（`t1zr`–`t1zv`）的结果保留可查 | 照 GPU 写法、整网收拢：单层 5 层对 GPU 写法的 rel_L2 1.1e-6–2.5e-6、对 FP64 的 R 1.000–1.001（`tw-sinkprobe2`）；`t1zw-a` 1000 步主 loss 首次全过（全程 MARE r 0.98，带符号均值 r 0.36）。 | **之后**：已看到 `t1zw-a` 的 1000 步与前 300 步结果 |
| 2026-11-09 | 更正（只更正说法，不改判定规则）：上面 2026-11-09 打开 M42b 那一行写的「`t1zw-a` 1000 步主 loss 首次全过」不准确：M28 之前的 `t1u-a`（1000 步）主 loss 也过过。准确说法是「自打开 M28 以来第一次」，并且这是观测结果，收拢是否全来自 M42b 待多步回放证实 | 原说法不准确 | **之后**：已看到 `t1u-a`、`t1zw-a` 的 1000 步结果 |
| 2026-11-09 | 第 1.2 节：对齐用的四份 yml（T1 / T2 B / T2 D1 / mid）写明 `sa_use_base2_exp: false`（M43：splash 不走 base2 指数，与 GPU 的自然指数同；M42b 保持开）。单独排一代 `t1zz-a`（`t1zw-a` 只再关 base2，1000 步）验证，现行比较在它判定之前仍是 `t1zw-a`；mid 与 T2 下一代同步 | 对真值更近、无步时代价：前向 logsumexp 对 FP64 的最大绝对差小约 30 倍（`tw-sinkprobe`）；T2 B 第 1 步两层 `q_a_layernorm` 梯度对 CPU FP64 R 2.89 / 1.58 → 0.88 / 0.79（`tqa-nb2`）；mini 步时 0.4767 → 0.4762 s。反向证据：单层 sink 梯度多数层略差、mini 300 步 A/B 两项由过变待定（各一次运行），由 `t1zz-a` 的 1000 步验证。 | **之后**：已看到 `tq-base2`（300 步）与 `tqa-nb2`（第 1 步）结果；`t1zz-a` 未跑 |
| 2026-11-09 | 第 5 节 L3：**灌选择只作诊断**。L3 的正式判定一律用 TPU 自选的路由与 DSA top-k；灌 GPU 选择的结果（`tw-fr*`、`tw-fri1`、`t1-step1-cont-noise-tw-fri1.json`、`tw-x*` 等）改标「诊断」，只说明差从哪来（离散翻转 vs 连续误差），不作判定或定责依据，已写成「过」或「已定责」的地方改回按自选路由的判定。新增正式检查「路由一致率」（定义见第 5 节） | 生产时 TPU 不可能回放 GPU 的选择，判定必须看 TPU 自己算出来的路由。 | **之后**：已看到灌选择的第 1 步与多步回放结果；路由一致率尚未测 |
| 2026-11-09 | 第 1.2 节：对齐用的四份 yml 把 `sa_use_base2_exp` 改回 `true`（撤回同日「写明 false」那一行），现行比较维持 `t1zw-a`；`t1zz-a` 降为诊断代。生产建议仍是关（对真值更准，根源一览第 24 行） | `t1zz-a` 1000 步主 loss 带符号均值对 8 份 GPU 的 r 1.62–2.62（中位数 2.06），`t1zw-a` 0.07–1.07（中位数 0.51），GPU 28 对留一 r 中位数 0.36、最大 1.09（`t1-l4-vs-8gpu.json`）：关 base2 让整网离 GPU 更远，不是只对 `g3-a` 的偶然。 | **之后**：已看到 `t1zz-a` 1000 步与 8 份 GPU 的比较 |
| 2026-11-09 | 第 5 节：「路由一致率」的正式判定改为对真值口径（每层 TPU 对 CPU FP64 自选的翻转率或 DSA 不重合率 ÷ GPU 各份对 FP64 的最大值，≤ 1.2 过、≤ 2 待定、> 2 不过，同 L3 的 R）；原登记的「对 GPU 两两」口径照报不判。DSA 同口径，FP64 参照每行存 DSA top-k | 第 1 步 GPU 三次重跑在第 1 个 MoE 层的路由与 DSA 第 0、1 层逐位相同，两两的尺子为 0：它量的是确定性，TPU 只要有一个 token 选得不同就不过，与离真值多远无关。对真值口径与 L3 的 R 同一逻辑。 | **之后**：已看到第 1 步路由的两种口径（对真值每层 R 0.89–0.99，对 GPU 两两第 1–5 层不过）；DSA 对真值与第 8、30、100 步当时未出 |
| 2026-11-09 | 第 3 节：L4 加报对 8 份 GPU（8 个 r 的中位数与最大值，旁列 GPU 28 对留一 r）；过线规则三版候选待定，正式判定仍只对基线，两种并报 | 只对一份 GPU 的 r 偶然性大：`t1zw-a` MTP loss 带符号均值对 `g3-a` 0.35、对 8 份中位数 1.06；`t1zv-b` 路由偏置均值对 `g3-a` 1.61、中位数 0.79（`t1-l4-vs-8gpu-rules.json`）；改报法 | **之后**：已看到四代 T1 对 8 份 GPU 的 r |
| 2026-11-09 | 撤回同日第 3 节「L4 加报对 8 份 GPU」一行：不采用对 8 份的报法与 C1–C3 候选规则，L4 判定维持只对单一 golden（T1 `g3-a`，T2、mid 各自的 a 份），GPU 多份重跑只用来定噪声上限 N；`analysis/l4_vs_all_goldens.py` 留作诊断，结果文件只报不判、不进检查清单 | — | **之后**：已看到四代 T1 与 T2、mid 对 8 份 GPU 的 r 与三版候选规则的判定 |
| 2026-11-09 | 第 1.2 节：对齐用的 yml 打开 `indexer_kd_generalized_kl`（台账 M50：索引器 KD 改记广义 KL，差值经 stop_gradient 加回，梯度逐位不变）；T1 的现行比较从 `t1zw-a` 换成 `t1zg-a`（`t1zga-l4`），旧结果保留。| MaxText 的 KD 记录写法在 FP32 下病态，M50 只改记录值，单步验收记录值对 FP64 在 1 个 FP32 ulp 以内 | **之后**：已看到 `t1zg-a` 1000 步：过 7、待定 2，索引器 KD 由待定（带符号均值 r 1.06）变过（最差判定量 MARE r 0.85；带符号均值 0.69），41 个逐步量与 `t1zw-a` 逐位相同（KD、总 loss、param_norm 的记录值有末位差，参数逐位相同） |
| 2026-11-09 | L3：MTP `enorm`（第 1 步超出 N 1.45 倍、对 CPU FP64 R 1.24，两边误差都小于一次 BF16 舍入）接受，按分辨率下限的道理定责；L3 一格的判定照结果文件不变 | 两边误差都小于一次 BF16 舍入 | **之后**：已看到 `t1z` 设置第 1 步的结果 |
| 2026-11-09 | 报告口径（不改判定规则）：L3 全部参数梯度一格把接受的张量（MTP `enorm`）算已定责，未定责计数 5 → 4，判定仍是不过（`build.py` 的 `STEP1_ACCEPTED`）；取代上一行「L3 一格的判定照结果文件不变」里的计数口径 | 同一张量在格子里算未定责、说明里写已接受，两种说法并存 | **之后**：已看到 `t1z` 设置第 1 步的结果 |
| 2026-11-09 | L4：grad_norm 与路由偏置标准差两项待定接受为已定责（grad_norm：前 10 步的差在 GPU 一侧，第 11–100 步与第 30 步同一状态 GPU 偏低同号，尖峰段是轨迹差，同一状态单步 6 次里 5 次 TPU 更近 FP64、第 100 步 R 1.29；路由偏置标准差：同一状态 35 处（5 个状态 × 7 层）偏置更新方向里 TPU 与 FP64 不同 2 处、GPU 每份 1 处，都在近平局处，多步偏离是轨迹差），格子判定照结果文件不变（🟡），说明写「已定责、已接受」 | 证据 `t1zg-grad-norm-attribution.json`、`t1zg-bias-std-windows.json`、`t1-bias-update-sign.json`；尖峰段第 849、900 步同一状态单步：R 0.69、0.07（`t1-spike-same-state.json`） | **之后**：已看到 `t1zg-a` 1000 步的结果 |
| 2026-11-09 | 第 5 节 L3：第 1 步的 TPU 梯度以训练步实际用的为准（由第 1 步动量反推，相对误差约 1e-7），诊断程序（`--diag-step1`）的梯度只作旁证 | 两个程序在 TPU 上分开编译，同一状态、同一批数据的第 1 步梯度不逐位相同：205 个张量相对差中位数 2.9e-4（`t1-l3-optim-step1.results.json` 的 `tpu_diag_vs_train_grad`）；按训练步梯度重判，未定责仍是同一批 6 个 iHC 张量 | **之后**：已看到现行配置第 1 步的两种梯度 |
| 2026-11-09 | 第 5 节 L3：加「优化器一步」：两边各自的 FP64 优化器作用在自己的梯度上，比更新后参数与 Adam 二阶动量对 FP64 的 R ≤ 1.2；这项没有 GPU 噪声，两边对 FP64 都 < u 的张量按分辨率下限过（不带 N = 0 的条件） | 第 1 步更新后的状态混着梯度的差，单独比优化器一步才分得出优化器本身 | **之后**：规则与结果 `t1-l3-optim-step1` 同一次提交 |
| 2026-11-09 | 第 0 节「接受」：第 1 步 6 个 iHC 张量（[mini 对齐细节 D11](alignments/01-mini/DETAILS.md#d11)）、L4 QK-Clip 全局最大 logit（D29）、FP32 log 3 ulp（D14）记为已定责、已接受；格子判定照结果文件 | D11：整网的差经各层传播、叠加路由翻转，落在病态小参数上；单算子在线内，单层回放 6 个里 5 个 R ≤ 1.2。D29：同一状态单步四次 TPU 都更近 FP64，GPU 偏差各份共有。D14：GPU 实测 1 ulp，TPU 现行 3 ulp、真实输入上 1 ulp；1 ulp 写法（M51）打开会换轨迹，不进对齐配置 | **之后**：已看到 `t1zg-l2-param-grads.json`、`t1-qkclip-same-state.json`、`t1zla-l4` |
| 2026-11-09 | **前瞻验证登记（看结果之前固定判定）**：三条事后改的判法——L3 分辨率下限（N = 0 且两边对 CPU FP64 都 < u = 2⁻⁸ 的张量按下限过）、路由一致率按真值口径（每层 R = TPU 的翻转率 ÷ GPU 三份的最大值，≤ 1.2 过）、优化器一步（两边各自的 FP64 优化器作用在自己的梯度上，更新后参数与 Adam 二阶动量 R ≤ 1.2，两边都 < u 的按下限过）——原样用于真实权重截层 B、D1 下一批还没看过结果的现行配置运行：TPU 用 `frameworks/maxtext/hy4-t2-b.yml`、`hy4-t2-d1.yml`（本仓库 `ce475a6` 的版本，sha256 前缀 `6d07981e`、`4379e561`，含 M42b、M50），第 1 步梯度取训练步实际用的（由第 1 步动量反推，同 mini）；GPU 参照 `gb2s-a … c`、`gd2s-a … c`；CPU FP64 参照用 `step1_ref.py` 同设置重算；路由一致率比第 1、8、30、100 步，第 8、30、100 步从 GPU 基线（B `gb3-a`、D1 `gd3-a`）同一份重建状态起步。运行的 MaxText 提交在运行时写进 `runs.yaml`。这批运行的判定只按本条，出现不过就记不过，不再改规则 | 三条判法都在看到 mini 结果之后改，把改前不过的格子改判为过；用没看过结果的运行验证它们 | **之前**：B、D1 的这批运行还没跑 |
| 2026-11-09 | 更正（只更正说法，不改判定规则）：上面 2026-11-07 第 1.2 节那一行写 `qk_clip_max_logit_exclude_sink`（M25）「预计不改数值（推断）」不准确：只关它的 300 步对照（`tq-m25`）里梯度范数从第 1 步起就与全开不同，主 loss 带符号均值 0.48 → 0.93 倍 | 原说法不准确；推断是开关改了 TPU 编译结果 | **之后**：已看到 `tq-m25` 与 `tq-base` |
| 2026-11-09 | **事后解读（不改判定规则）**：TPU 自比（`alignments/04-tpu-self/`）的「略超」另给一列解读。GPU 同设置 8 份做留一法（56 种被测 × golden 组合，波动上限用被测之外 7 份两两 21 对，检查与 TPU 自比相同）：300 步最差 r > 1 的占 0.59、> 2 的占 0.11、最大 2.47；1000 步 > 1 的占 0.64、> 2 的占 0.00、最大 2.00（`results/gpu-rerun-loo-r-{300,1000}.json`）。TPU 自比的臂若曲线最差 r 不超过对应窗口的 GPU 最大值、第 1 步没有定不了责的张量，解读为「与重跑噪声不可区分」 | 一次比较 8 条检查 × 3 个判定量，任一过 1 即略超；单个量超过 28 对最大值的概率约 1 / 29，合起来纯 GPU 重跑也有一半以上落到略超。不看这一点会把换序当成问题 | 看到 TPU 自比各臂结果之后（多数换序臂 r 1.0–1.3） |
| 2026-11-09 | **前瞻登记（看结果之前固定判定）**：上一行「与重跑噪声不可区分」作为解读列继续保留，不改正式判档；从现在起的 TPU 自比新臂、以及真实宽度截层的比较，事先按这条规则给出解读：曲线最差 r ≤ 同设置、同窗口的 GPU 留一法最大值（hy4-mini：300 步 2.47、1000 步 2.00；别的设置先按 `analysis/gpu_loo_r.py` 算出再用），且第 1 步没有梯度离 CPU FP64 远 1.2 倍以上（被测离 FP64 ÷ max(对照离 FP64, GPU 三份离 FP64 的中位数) ≤ 1.2，iHC 小参数按 01-mini D11 单列）→「与重跑噪声不可区分」，否则「可区分」。之后看这条规则在新结果上站不站得住（例如故意改坏的臂是否都落在「可区分」、只换顺序的臂是否都落在「不可区分」），再定要不要升为正式判档 | 规则是看到结果之后定的，先作解读，用之后的新结果验证 | 看到 TPU 自比约 60 个臂之后；对之后的新臂与真实宽度截层是事先登记 |
| 2026-11-10 | TPU 自比 1000 步的刻度从 GPU `g3` 8 份两两 28 对换成 16 份两两 120 对（`g3-a` … `g3-p`，同一代码包与参数），全部 1000 步臂重判；上一行前瞻登记里 hy4-mini 1000 步的留一法最大值随之从 2.00 改为 1.82（16 份，240 种组合，`results/gpu-rerun-loo-r-1000.json`），规则本身不变。两种「留一法」口径不同，数字不能互比：① TPU 自比用的（`analysis/gpu_loo_r.py`）：n 份里任取有序的一对当（被测, golden），共 n × (n − 1) 种，上限用被测之外 n − 1 份两两重算，每种组合取全部 8 条检查 × 3 个判定量里最大的 r，再对所有组合取最大——是「整次比较最差的那一项」的最大值，对应 TPU 自比里「曲线最差 r」；② GPU 一侧的（`analysis/noise_leave_one_run.py` 与 janus 的 `noise_loo`）：golden 固定为 `g3-a`，每份重跑轮流当被测、上限用不含它的其余运行两两，**每条检查各自**取最大 r（另一种是每个噪声对 ÷ 其余各对的最大值）——是单条检查的最大值，16 份时在 1.0–1.35。① 同时对检查和 golden 取最大，所以比 ② 大 | 8 份时 1000 步上纯换顺序的臂到过 2.33、2.50，超过 2.00，当时把 1000 步 r 2–3 记灰区、事先说好等 16 份刻度重判。16 份下上限放大 1.0–2.1 倍（梯度范数带符号均值 1.70 倍），灰区六臂落到 1.18–1.62，全部「与重跑噪声不可区分」 | 扩份数是看到灰区后定的；16 份的留一法与各臂重判同时算出，判定规则没有因结果改动 |
| 2026-11-10 | **曲线分母统一**：L4 与 TPU 自比、GPU 多机等所有曲线检查（loss、grad_norm、param_norm、路由偏置）的逐步相对差一律以参照作分母；grad_norm 不再用两边均值 \|a + b\| ÷ 2（第 3 节「测量口径」第一句作废）。混元 4 下 87 个目录去掉 `denominator: mean`，用同一批运行目录重跑全部 262 份受影响的结果（原名覆盖，改前改后逐项对照入库：[`results/grad-norm-denominator-rejudge.md`](results/grad-norm-denominator-rejudge.md)）；分析脚本 `noise_leave_one_run.py`、`kd_gn_windows.py`、`l4_grad_norm_attribution.py`、`plot_l4_figures.py` 同步改分母并重跑。顺带更正：2026-11-09、2026-11-10 两行写的「8 条检查 × 3 个量」只对 300 步成立，1000 步目录是 9 项（多一条 MTP loss 第 101 步起） | 便于解释，所有检查同一规则。两边均值是混元 3 第一层为对称性和 +E[d²] 偏置引入的；混元 4 上实测偏置远小于噪声（纯数据并行一臂带符号均值只挪 2e-7，N 9.9e-5）。实测影响：262 份里梯度范数以外的检查一项不变；判定只变 1 项（`t1zz-l4` 的 grad_norm 过 → 待定，r 0.917 → 1.006，历史诊断比较）；最差 r 在 5 以内的 252 份改前改后之差中位数 1.1e-4、最大 0.10（带符号均值的 r 最大挪 0.43，`t2d15b-l4` 0.55 → 0.98）；现行比较 `t1zga-l4` grad_norm 1.25 → 1.23（仍待定、已接受，2026-11-09 的定责证据按新分母重跑，结论不变）；TPU 自比 171 个比较判档全不变；GPU 留一法最大值 300 步 2.47、1000 步 1.82 不变 | **之后**：已看到全部现有结果。改动由项目负责人拍板，理由是统一规则而不是改判；改后唯一的判定变化是由过变待定（变严） |
| 2026-11-10 | **只改叫法，不改判定**：噪声那把尺子统一叫「尺子 / 噪声上限 N」，与混元 3、混元 4 的对齐文档一致；「刻度」只指 FP32 记录值的分辨率。TPU 自比的档名「在刻度内」改为「在噪声内」、「超出刻度」改为「超出噪声上限」（`tpu_self_compare.py` 新写的结果用新档名）。已提交结果文件里的 `verdict.label` / `verdict.band` 是历史证据不改，`build_tables.py` 读旧档名时按新叫法显示；新旧对应写在知识库术语对照表。本表此前各行的旧叫法不改 | 同一个东西两种叫法，「刻度」又和 FP32 记录分辨率撞名 | 之后（不涉及任何判定，所有比较的判档不变） |
| 2026-11-11 | **登记，不改判档**：收窄 iHC 小参数的豁免（TPU 自比 `tpu_self_compare.py` 的 `ihc_accepted`；GPU↔TPU 第 1 步的 [D11](alignments/01-mini/DETAILS.md#d11) 同理）。提议的新口径：iHC 张量照样先算 R = 离 FP64 ÷ max(golden, GPU)；R ≤ 5 且被测与 FP64 逐元素同号的才单列为「已接受」，R > 5 或有元素与 FP64 反号的记「未定责」，进判档。按提议口径重判（只按 R，逐元素同号这一条要另抓元素，未算），现有 TPU 自比里会从「已接受」变成「未定责」的：`hc_head` 三个张量的那几次（D3 及其 1000 步、关小矩阵乘改写、v7x 默认及其 1000 步、RMSNorm ε 改小），层内 iHC 张量 R > 5 的只有 RMSNorm ε 改小（70.8）与两次灵敏度注入（loss 系数 2 倍 553、1.01 倍 5.3）——这三次本来就是真改动，改判正是想要的；其余运行层内 iHC 张量 R 都 ≤ 5。等前瞻验证的下一批运行出来后再定是否生效 | `hc_head` 三个参数第 1 步梯度在这几次里离 FP64 18–46 倍、`hc_head_base` 前三个元素变号，一直被豁免盖住。单算子回放（`alignments/04-tpu-self/results/night3-hc-head-replay.json`）量了病态程度：这个梯度八成是末层 RMSNorm 的 ε 项（ε/(均方 + ε) ≈ 2.2e-3，FP64 的 6.87–11.18e-6 里 8.8–9.2e-6 来自它），偏差约为 ε 项的 1.26 倍。D11 登记时写「病态程度没量」，现在量了，原来「1.5–2 倍的小偏差」的理由不覆盖这种同号常数偏差 | **之后**：已看到全部现有 TPU 自比臂的 `ihc_accepted`。同日补充：这几次的偏差已定到输出头反向那颗卷积 kernel 里 RMSNorm 反向两项用的余切舍入不一致（`ts-n3-d3final-barrier` 加 optimization_barrier 后回到 golden），是真实的数值缺陷，不是可以豁免的小偏差，更支持收窄 |
| 2026-11-11 | 第 1.2 节：对齐用的四份 yml（T1 / T2 B / T2 D1 / mid）把 `sa_use_base2_exp` 改成 `false`（M43，所有尺寸一致），撤回 2026-11-09「改回 `true`」那一行。T1 现行比较从 `t1zg-a` 换成 `tz-off-rep`（`t1zo-l4`、`t1zo300-l4`），`t1zg-a` 降为上一代，结果保留。`tz-off-rep` 是关 base2 的那份原始运行：默认执行方式、代码包含 M50，主 loss、MTP、grad_norm 1000 步逐位等于 `t1zz-a`；没有选只改执行方式的几份重复（`tz-off-nolhs`、`tz-off-vm32`、`tz-off-vm64`），以免挑判档好的那份。同时 T1 1000 步的噪声 N 从 `g3-a … g3-h` 28 对换成 `g3-a … g3-p` 16 份 120 对（TPU 自比 2026-11-10 已换）。L3 第 1 步梯度换成 `tz-off-rep` 训练步实际用的（`t1zo-step1-train-grad-vs-fp64`），路由一致率第 1 步换成 base2 关的 `tw-ra1z`；L1 低阈值 QK-Clip 第 1 步逐头最大 logit 换成 `tz-off-qk01`，L4 QK-Clip 曲线、健康信号、同设置重跑逐位换成 `tz-off-qkc`；路由一致率第 8、30、100 步与 L1 注意力整块、L2 里没有关 base2 结果的项照旧引用开 base2 的一代并注明；优化器一步只比算术，仍用 `mz-opt2` | ① 真实宽度（06-rw3）上关掉更准：第 1 步主 loss 对 FP64 −9.87e-5 → −1.24e-6（R 1.85 → 0.02），逐张量梯度 102 个里 86 个更近；1000 步对 GPU 不过 3 → 0。② mini 上「关掉离 GPU 更远」（`t1zz-a`，r 0.36 → 1.91）已证为轨迹偶然：开、关各 4 份 1000 步（3 份只改执行方式），16 份尺子下主 loss r 开 0.90–1.69、关 0.93–1.55，两组交叠（`mini-base2-loo.json`）。③ 各尺寸配置一致，不用登记尺寸之间的差别。④ 生产本来就建议关（根源一览第 24 行） | **之后**：已看到 mini 开、关各 4 份 1000 步、真实宽度 250 步与 1000 步两臂、第 1 步对 FP64 的全部结果；改的时候已知 `tz-off-rep` 在 16 份尺子下主 loss 待定（r 1.55）、其余 7 项过；L3 第 1 步与 L1 低阈值 QK-Clip、L4 QK-Clip 的关 base2 结果是改配置之后才跑出来的（L1 低阈值 QK-Clip 第 4 层 R 1.55 不过） |
