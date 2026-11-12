[janus](../../../../README.md) › [混元 4](../../README.md) › [mini](README.md) ｜ 配置对照表

# 混元 4 · 两边配置对照表（T1：TPU MaxText ↔ GPU Megatron）

T1（hy4-mini，427,783,489 参数，7 层 + 1 层 MTP，随机初始化）两边的配置，逐项对照；各对齐实践的 `setup.md` 只写和这里不一样的地方。

- **对照的是哪一版**：TPU 一栏写的是 `hy4-mini-t1.yml` 现在的取值（M28、M31、M33、M34、M36 都开，M37 随 M32 的开关生效；M38 的七个按位置覆盖项默认跟随总开关；M39 的反向写法随 MaxText `1d9cc8c7` 起生效；M40、M41、M42b、M50 对齐配置打开；M52、M58 2026-11-11 起打开（C 口径）；M42 默认关、对齐配置不开；M43 `sa_use_base2_exp` 2026-11-11 起设 `false`）。**现行比较 `tc-a` 就是这一版**（MaxText `8d3d7561`，跑的时候用覆盖参数开 M52、M58，实际生效的配置与这一版相同，见 `tc-l0-config`；PROTOCOL 变更记录 2026-11-11「C 口径」一行）；C 口径之前的上一代 `tz-off-rep`（MaxText `b87a55aa`）与它只差 M52、M58，再上一代 `t1zg-a`（MaxText `47d31984`）比 `tz-off-rep` 再多开 base2。各代运行与它的差别见 [`goldens.md`](goldens.md)「TPU 运行」与[历代比较](../../dashboard/history.md)，以各运行的 `meta.json` 为准。GPU 一栏是 `g2-*` / `g3-*` 的设置（`g1-*` 只差注意力 logits 一处，台账 G17）。
- **两个角色**（[`PROTOCOL.md`](../../PROTOCOL.md)）：语义以 TPU 一侧 MaxText 为准，模型结构的缺口 GPU 照它补，优化器与 loss 的细节 MaxText 加默认关的开关对上原版 Megatron-Core；判定时 GPU 是 golden、TPU 是被测。
- 训练设置的登记在 [`PROTOCOL.md`](../../PROTOCOL.md) 第 1.2 节；两边对上游 / 原版的每一处改动（开关、提交、测试、数值核验）在 [`CHANGES-LEDGER.md`](../../CHANGES-LEDGER.md)，表里写「台账 M#」（MaxText）、「G#」（GPU 一侧）、「J#」（janus 工具）。
- 每一项检查比了什么、现在什么结果：[检查清单 `CHECKS.md`](CHECKS.md)。

本页先说明出处与取值约定，再按模型结构、数值精度、优化器与调度、数据、并行与确定性逐类对照，最后列出为对齐改过的代码。

## 出处与取值约定

出处约定：

- 「T1 yml」：[`frameworks/maxtext/hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml)。
- 「模型 yml」：`maxtext` 混元 4 集成分支的 `src/maxtext/configs/models/hy4-mini.yml`；「MaxText 默认」：同一分支 `src/maxtext/configs/base.yml` 与 `configs/types.py` 的默认值。三处都按 `d723fa2d` 读（台账 M24 的提交，T1 诊断运行用的版本；协议冻结时 T1 的 TPU 配置是 `8611ea6b`）。C 口径之前的运行用 `b87a55aa`：与 `d723fa2d` 相比，`base.yml` 与 `types.py` 只新增键，新增的全部默认 false 或 null（如 `qk_clip_max_logit_exclude_sink`、`indexer_weights_proj_bf16_weight`、`splash_dsinks_gpu_order`），已有键的默认值没有变，模型 yml 没有改（`git diff d723fa2d b87a55aa` 核过），所以表里「默认」一栏对 `b87a55aa` 仍成立。现行比较 `tc-a` 用 `8d3d7561`：与 `d723fa2d` 相比同样只新增键（含 M52、M58、M59 与一批诊断键），新增的都默认 false、null 或空串（唯一例外 `ihc_stream_grad_gpu_order_parts` 默认 `"all"`，只在 M58 打开时起作用），已有键的默认值没有变，模型 yml 没有改（`git diff d723fa2d 8d3d7561 -- src/maxtext/configs/base.yml src/maxtext/configs/types.py src/maxtext/configs/models/hy4-mini.yml` 核过，2026-11-11），表里「默认」一栏对现行运行也成立。每一份运行实际用的 MaxText 提交记在它的 `meta.json`。
- 层叠顺序：MaxText 按 `base.yml` → T1 yml → 模型 yml → 命令行合并，**模型 yml 压过 T1 yml**（`pyconfig.py`）。两处都设、取值不同的键，由 [`frameworks/maxtext/mt_common.py`](../../frameworks/maxtext/mt_common.py) 的 `forced_overrides` 补到命令行；T1 里只有 `indexer_loss_scaling_factor`（模型 yml 0、T1 yml 0.1）一处。`--stage mini` 另给 `base_num_decoder_layers=7 mtp_num_layers=1`。`train_hy4.py` 开训前逐键核对 T1 yml 的值实际生效，不一致就停。
- 「GPU」：Megatron-Core 0.20 + TE（NeMo 26.10 镜像所带；`meta.json` 不记 TE 版本，同一镜像实测是 TE 2.20.2，见[软件版本](setup.md#软件版本)） 之上的 janus 程序 [`frameworks/megatron/`](../../frameworks/megatron/README.md)（`hy4_mcore.py` 的模型 config、`train_hy4mg.py` 的训练入口）。GPU 一侧取值按协议第 1.2 节与台账 G 行。现行 1000 步 golden `g3-*` 的完整附加参数（含 `--opt-kv`）记在 [`05-gpu-scale/setup.md`](../05-gpu-scale/setup.md) 的 `G3`，与 `g3-a` 日志的 `RUN_START` 行逐项相同，没写的参数取 `train_hy4mg.py` 的默认值；`g1-*`、`g2-*` 的命令行没有入库，只能读各自 `meta.json` 记下的设置。
- 「库版本」：表里只比开关，不比库版本。现行比较 TPU 一侧 jax 0.11.1、libtpu 0.0.46，GPU 一侧 NeMo 26.10 镜像，逐项与出处见 [`setup.md`「软件版本」](setup.md#软件版本)。同一开关换了 libtpu 效果可能不同（[`libtpu-052.md`](../04-tpu-self/libtpu-052.md)）。
- 「HF 目录」：hy4-mini 的随机初始化目录（`model/make_tiny_hf.py --config mini --layers 7 --mtp 1 --seed 0`，清单 sha256 见协议第 1.1 节），两边都从它载入。

取值约定：拿不准的写「待核」，推出来的写「推断」并给理由。TPU 一栏写「`键`：默认 → T1」；默认与 T1 相同时只写一个值。

对齐方式：**相同**（同一个值、同一种算法）／**等价**（数学相同、写法或舍入位置不同）／**照现状**（生产语义未知，两边按 MaxText 现在的行为，协议第 1.2 节）／**不同**（已知不一致，写原因）。

状态：已对齐（有检查结果，括号里写到哪一级）／已对齐（配置）（只有配置或单测核对）／有差异／待核。

## 模型结构

| 类别 | 参数 / 开关 | GPU（Megatron） | TPU（MaxText）：默认 → T1 | 对齐方式 | 状态 | 台账 / 证据 |
|---|---|---|---|---|---|---|
| 结构 | 模型、层数、MTP | HF 目录的 config：7 层（1 稠密 + 6 MoE）+ 1 层 MTP（`--layers 7 --mtp`） | `model_name: hy4-mini`（台账 M6）；`--stage mini` → `base_num_decoder_layers=7`、`mtp_num_layers=1`；`override_model_config: false → true` | 相同 | 已对齐（L0：反向取回 212 / 212 逐位，台账 J3） | M6、J3 |
| 结构 | 维度 | 宽 1024、8 头、q_lora 512、kv_lora 512、qk_nope 192、qk_rope 64、v 256、稠密 FFN 3072、词表 120832（HF config） | `base_emb_dim 1024`、`base_num_query_heads 8`、`q_lora_rank 512`、`kv_lora_rank 512`、`qk_nope_head_dim 192`、`qk_rope_head_dim 64`、`v_head_dim 256`、`base_mlp_dim 3072`、`vocab_size 120832`（模型 yml） | 相同 | 已对齐（形状） | M6 |
| 结构 | MoE | 16 个路由专家 top-4、宽 256、1 个共享专家；第 0 层稠密（`moe_layer_freq`） | `num_experts 16`、`num_experts_per_tok 4`、`base_moe_mlp_dim 256`、`shared_experts 1`、`first_num_dense_layers 1`（模型 yml） | 相同 | 已对齐（形状） | M6 |
| 注意力 | MLA 两个潜空间 norm 的 eps | `attention_latent_norm_epsilon` = `rms_norm_eps` 1e-5（项目定下的读法） | `mla_lora_norm_epsilon`：-1 → 1e-5（模型 yml） | 相同 | 已对齐（配置） | M7 |
| 注意力 | RoPE | θ 1e7、factor 1；检查点交错存放，运行时去交错（`mla_rotary_interleaved`） | `rope_type: yarn`、`rope_factor 1`（= 普通 RoPE）、`rope_max_timescale 1e7`；`rope_interleave: false`、`checkpoint_rope_interleave: true`（转换时去交错，跑拼接式） | 等价（同一个置换，q·k 逐项相同） | 已对齐（L0：去交错 32 / 32；负对照在 hy4-tiny） | M7、J3；协议「RoPE」行：照现状（生产 factor 8.0） |
| 注意力 | 门控 MLA | `out × sigmoid(linear_gate(x))`，逐元素 | `attention_output_gate: elementwise`（模型 yml） | 相同 | 已对齐（配置）；精度差别见「数值精度」 | G1 |
| 注意力 | 可学习 sink | 每头一个 FP32 logit，拼成 softmax 的一列，不带 value | `attention_sink: true`（模型 yml） | 照现状（生产 `learnable_sink_log_ratio` 语义未知） | 已对齐（配置） | 台账「照现状」表 |
| 索引器 | DSA 索引器 | 32 头 × 128、top-k 2048；k_norm eps 1e-6；RoPE 在每头后 64 维；full / shared 按 HF `indexer_types` | `indexer_n_heads 32`、`indexer_head_dim 128`、`indexer_topk 2048`、`indexer_k_norm_epsilon 1e-6`、`indexer_rope_position: last`、`indexer_share_interval 4`（模型 yml） | 相同 | 已对齐（配置相同；C 口径 `s1f-c` 上 L3 索引器 20 个张量 R 0.78–1.01（`t1zc-step1-train-grad-vs-fp64-n6`）；C 口径之前的 `tz-off-rep` L3 索引器各张量 R ≤ 1.05，第 0、1 层 `k_norm` / `weights_proj` 5 个张量 N = 0 且两边误差都小于 u，按分辨率下限过（R 0.78–0.97）；更早 t1z / t1v 一代第 1 层 `k_norm.weight` R 1.58 / 1.59） | G1、M7 |
| 索引器 | 主模型走稀疏 top-k 注意力 | `dsa_indexer_use_sparse_loss=True` | `indexer_sparse_training: true`（模型 yml） | 相同 | 已对齐（配置） | G4 |
| 超连接 | iHC | 4 路，post 幅度 2.0，eps 1e-6，混合系数 FP32 | `mhc_expansion_rate 4`、`hc_variant: ihc`、`hc_post_magnitude 2.0`、`hc_eps 1e-6`、`use_hc_head: true`、`hc_head_norm_epsilon 1e-5`（模型 yml） | 相同 | 有差异（配置相同；L1 阶段 A 的 iHC 混合、hc_head R 1.000，C 口径 `s1f-c` 上 L3 对 CPU FP64、GPU 第 1 步参照 6 份时有 1 个 iHC 张量超线、未定责（第 3 层 FFN 前 `hc_fn`，1.14 N，R 1.43；`t1zc-step1-train-grad-vs-fp64-n6`），按 [D11 豁免规则](DETAILS.md#d11-rule) 单列、计入（base2 开那一代登记接受的 6 个之一，接受理由没有数值门槛、复核不了；是否沿用待拍板）；C 口径之前的 `tz-off-rep` 是第 3 层注意力前 `hc_scale`（1.05 N，C 口径下 0.66 N），GPU 3 份时是 5 个（第 0 层注意力前 `hc_base`，第 2、3 层注意力前 `hc_scale`，第 6 层 FFN 前 `hc_base`、`hc_fn`；第 0、6 层这 3 个是关 base2 后新出现的）；base2 开那一代的 6 个（第 2 层注意力前 `hc_scale`，第 3 层 FFN 前 `hc_base`、`hc_fn`，第 3 层注意力前 `hc_scale`，第 4 层注意力前 `hc_base`、`hc_scale`）当时已接受；`tz-off-rep` 上只剩第 3 层注意力前 `hc_scale` 超线，按 [D11 豁免规则](DETAILS.md#d11-rule) 单列、计入（base2 开那一代的接受理由没有数值门槛，复核不了）；见下面「iHC 四路流加权和」一行） | G1 |
| MLP | SwiGLU 截断 | 只截路由专家（gate ≤ 10、up ∈ [−10, 10]）；稠密层与共享专家用不截断的 config | `mlp_activations_limit 10`、`mlp_activations_limit_inclusive_grad: true`（模型 yml）；稠密 MLP 不截 | 相同 | 已对齐（配置） | G1 |
| 路由 | 打分与选择 | sigmoid、带偏置选择、top-4 归一化、缩放 2.827、FP32 路由（`moe_router_dtype: fp32`），辅助损失 0 | `routed_score_func: sigmoid`、`routed_bias: true`、`routed_weight_normalize: true`、`routed_scaling_factor 2.827`、`n_routing_groups -1`（模型 yml）；`load_balance_loss_weight 0`、`use_random_routing: false`（T1 yml） | 相同 | 有差异：L4 路由偏置均值上一代 t1zr 1000 步待定（带符号均值的绝对值：N 0.501，TPU 0.591，r 1.18），诊断代 t1zt 不过（TPU 1.27，r 2.54），加 M39 后回到噪声内，再上一代 t1zu 过（TPU 0.385，r 0.77），上一代 t1zg 过（TPU 0.404，r 0.81），C 口径之前的 tz-off-rep 过（16 份尺子：TPU 0.206，N 1.04，r 0.20）；现行 tc-a 带符号均值 r 1.31 待定、未定责（C 口径另 5 份 0.36–0.80，见「路由偏置更新」行）；L1：配置原样时 hy4-mini 第 1 层 MoE 一个近乎平局的 token 路由翻转（对 FP64 的 rel_L2：GPU 4.85e-3、TPU 9.97e-3，R 2.06），来自 TPU 近似的 sigmoid，开关全开（M37）后不再翻转（TPU 4.05e-3，R 0.84，对齐细节 D18）。另有一处舍入点不同：路由权重 TPU 乘在专家 down 投影之后、GPU 乘在之前，开关 `moe_mcore_prob_before_fc2`（M40，默认关）已加，T1 / T2 配置已打开；整网：开 M41 时关 M40（`t1zu-c`）→ 开 M40（`t1zu-a`），1000 步主 loss 带符号均值 N 3.78e-6，TPU 7.20e-6 → 2.45e-6，r 1.91 → 0.65（根源一览第 19 行） | 协议「路由」行 |
| 输出 | 输出头 | 不与词表共享，FP32 权重与输出（`install_fp32_head`） | `logits_via_embedding: false`、`logits_dot_in_fp32: true`（模型 yml） | 相同（精度见「数值精度」输出头一行） | 已对齐（L1 输出头 R 0.18，阶段 A） | M13 |
| 输出 | MTP | 1 层，loss 系数 0.1（`mtp_loss_scaling_factor`）；MTP 吃的是主干经 hc_head 与最后的 RMSNorm 之后的隐状态（Megatron-Core 0.20.0 `GPTModel._postprocess` 把解码器输出交给 `self.mtp`，解码器末尾已过 `final_layernorm`；混元 4 在这个位置放的是 `Hy4FinalNorm` = norm(hc_head(·))） | `mtp_num_layers 1`、`mtp_loss_scaling_factor 0.1`、`mtp_input_post_final_norm: true`（模型 yml） | 相同 | 已对齐（配置） | G1 |
| 其他 | dropout | 0（`hidden_dropout`、`attention_dropout`） | `dropout_rate 0`（T1 yml）、`enable_dropout: false`（模型 yml） | 相同 | 已对齐（配置） | — |

## 数值精度

| 类别 | 参数 / 开关 | GPU（Megatron） | TPU（MaxText）：默认 → T1 | 对齐方式 | 状态 | 台账 / 证据 |
|---|---|---|---|---|---|---|
| 精度 | 计算、主权重、梯度、优化器状态 | BF16 计算（`params_dtype` BF16），FP32 主权重与优化器状态（meta `dtypes`） | `dtype: bfloat16`、`weight_dtype: float32`、`grad_dtype: float32`、`mu_dtype: "" → float32`（T1 yml） | 相同 | 已对齐（配置） | 协议「精度」行 |
| 精度 | 矩阵乘默认精度 | TF32 关（`allow_tf32 = False`，启动时对 FP64 实测写进 meta） | `matmul_precision: default`（T1 yml） | 不同（硬件决定：TPU 默认单遍 BF16、FP32 累加） | 有差异（需要真 FP32 的几处单独设精度：输出头、索引器、iHC） | M13、M23 |
| 精度 | GEMM 累加只舍一次 | TE GEMM：FP32 累加、输出只舍一次 | `dense_fp32_accumulation: false → true` | 相同（TPU 用开关对上） | 已对齐（第 1 步梯度诊断：四个舍入开关都开后，逐张量比值中位数 2.67 → 1.72） | M12；协议变更记录「GPU 一侧的舍入位置」 |
| 精度 | RMSNorm 舍入次数 | TE 融合 RMSNorm：按 TE 2.20.2 源码（`common/normalization/rmsnorm/rmsnorm_fwd_kernels.cuh`），FP32 里求均方、`rsqrtf`，乘 γ 后输出只舍一次（TE 版本按镜像推断；没设 `NVTE_NORM_FWD_USE_CUDNN` 时走这条内核，镜像的环境变量未核） | `rmsnorm_single_rounding: false → true` | 相同（TPU 用开关对上） | 已对齐（按源码；TE 版本与 cuDNN 开关两处未核） | M9 |
| 精度 | 路由 GEMM | BF16 权重 × BF16 激活，FP32 输出 | `routed_gate_bf16_rounding: false → true`（权重先舍 BF16，再在 FP32 里乘）；`float32_gate_logits: true`、`router_matmul_precision: highest`（模型 yml） | 相同（TPU 用开关对上） | 已对齐（配置） | M8 |
| 精度 | 门控激活舍入 | 非融合 SwiGLU：silu 舍一次、乘积再舍一次 | `mlp_activation_round_per_op: false → true` | 相同（TPU 用开关对上） | 已对齐（配置；阶段 A 稠密 MLP 的 L1 R 0.95） | M10 |
| 精度 | 索引器打分 | 真 FP32：`weights_proj` 输出 FP32、q·k 与按头求和 `.float()`、KD 教师 q·k 同样；k_norm 权重留 FP32 | `indexer_fp32_scores: false → true`（`weights_proj` HIGHEST、q·k FP32 操作数） | 相同（TPU 用开关对上）；`weights_proj` 乘 BF16 舍入后的权重、FP32 输出，同原版（`indexer_weights_proj_bf16_weight: false → true`，`t1t-*` 起，M27） | 已对齐（配置；L1 KD 输入 R ≤ 0.99、L3 索引器梯度对 CPU FP64 R 0.81–1.05，都是 t1s 一代；L3 在 C 口径 `s1f-c` 上 R 0.78–1.01，见上面「DSA 索引器」一行） | M23、M27 |
| 精度 | iHC 混合系数 | FP32，autocast 关 | `hc_matmul_precision: "" → highest`（模型 yml） | 相同 | 已对齐（L1 iHC R 1.000，阶段 A） | — |
| 精度 | iHC 四路流加权和 | `torch.sum(pre · x)`：FP32 逐元素乘、FP32 相加（`Hy4HyperConnection`，autocast 关） | `ihc_stream_sum_unrolled: false → true`（`t1v-*` / `t1w-*` 起；关时用 HIGHEST einsum，编译后是 FP32 逐元素乘 + 配对相加，相加顺序与 GPU 不同，案例 11） | 相同（TPU 用开关对上，`t1v-*` / `t1w-*` 起）；`t1t-*`、`t1u-*` 是关 | 已对齐（`t1v-*` / `t1w-*` 起）：第 1 步 iHC 梯度对 CPU FP64 R > 1.2 的张量 11 → 8（`t1t-step1-grad-vs-fp64.json`、`t1v-step1-grad-vs-fp64.json`）；L4 param_norm 300 步 r 1.60 → 0.88、1000 步 1.06 → 0.43；同时 1000 步主 loss r 0.92 → 1.88（转待定；下一行的 M36 修掉大部分，对齐细节 [D26](DETAILS.md#d26)） | M28 |
| 精度 | iHC 分支输入与更新后的流的 BF16 舍入 | eager：两处都存成 BF16（`Hy4HyperConnection`） | `ihc_force_bf16_rounding: false → true`（台账 M36，`t1zr-*` 起）：先 reduce_precision 再舍 BF16。关时 astype 的 FP32 → BF16 → FP32 往返在 HLO 里还在，后端在 excess precision 下省掉一部分（v5p 最小复现的 LLO：RMSNorm 均方那颗用没舍的 FP32，[TPU 自比影响逐件](../04-tpu-self/impacts/05-ihc-bf16-rounding.md)） | 相同（TPU 用开关对上，`t1zr-*` 起）；`t1z-*` 及之前是关 | 已对齐（`t1zr-*` 起；开 M28 带来的同号偏高记为已修，对齐细节 [D26](DETAILS.md#d26)）：1000 步主 loss 带符号均值 r 2.12 → 0.76（`t1zk-l4` → `t1zr-l4`），对 GPU 均值的偏差 +8.6e-6 → +3.5e-6（推算）；剩下的偏高是 TPU 主 loss 共同偏高，记在 [D30](DETAILS.md#d30) | M36 |
| 精度 | iHC 与 hc_head 的流梯度、分支输出的 BF16 舍入 | autograd：BF16 的流在一层里有三处用到，三份梯度各舍 BF16、按「加权和 + 门控」再加「残差」的顺序相加，每加一次舍一次（hc_head 两处）；分支输出是 BF16 张量（`Hy4TransformerLayer`、`Hy4FinalNorm`；CPU 上逐位核过，`analysis/ihc_stream_grad_order_probe.py`） | `ihc_stream_grad_gpu_order: false → true`（台账 M58，2026-11-11 起，C 口径）：custom_vjp 把流拆成几份，余切逐份舍 BF16 按 GPU 的顺序相加；分支输出走一次 reduce_precision。关时这几处在省舍入下按 FP32 走 | 相同（TPU 用开关对上，`tc-*` 起）；`tz-off-rep` 及之前是关 | 已打开（`tc-*` 起）：C 口径之前 mini 后段的主因（[根源一览第 30 行](../../ROOT-CAUSES.md#r30)）；后段尾部在 C 口径下没有显著变小（C 口径 6 份第 551–1000 步 +6.9e-6，只开 M36 的对照 +9.4e-6，p 0.32，[`mini-factorial-tpu-6.json`](results/mini-factorial-tpu-6.json)），现行比较 `t1zc-l4` 主 loss 过主要是前 100 步的鼓包没了（M52），见 [D30](DETAILS.md#d30) | M58 |
| 精度 | hc_head 收拢之后的 BF16 舍入 | FP32 加权和后 `.to(bfloat16)` 再进末层 RMSNorm（`Hy4FinalNorm`） | `hc_head_force_bf16_rounding: false → true`（台账 M52，2026-11-11 起，C 口径）：reduce_precision，转置同时舍余切。关时 `astype` 在省舍入下可能留 FP32 | 相同（TPU 用开关对上，`tc-*` 起）；`tz-off-rep` 及之前是关 | 已对齐（`tc-*` 起）：mini 前段鼓包（第 51–100 步）的来源（[根源一览第 28 行](../../ROOT-CAUSES.md#r28)） | M52 |
| 精度 | 输出头 | 真 FP32（FP32 权重 × FP32 激活，TF32 关） | `logits_dot_in_fp32: true`（模型 yml）；`logits_matmul_precision: "" → "highest"`（`t1t-*` 起；只开前者时 TPU 默认精度只做一遍 BF16） | 相同（TPU 用开关对上） | 已对齐（配置；第 1 步诊断里设成 `highest` 不改变逐张量比较的结论） | M13 |
| 精度 | 输出头 dgrad | `--head-chunk` 默认 1024：每 1024 个词一段、段内 FP32 矩阵乘、段间 FP32 相加（`hy4_head_chunk`；`g1-*`、`g2-*` 的 `meta.json` 都记 `head_dgrad_chunk: 1024`） | `logits_matmul_precision: highest` 下 XLA 的 FP32 点积；词表维的累加顺序由 TPU 后端决定，没核（没看 LLO；「按矩阵单元分块累加」是推断） | 等价（求和顺序不同）；不分段时 B200 上 FP32 长求和沿输出层行均值方向有朝 0 的系统偏差（混元 3 实测，B200、CUDA 13.4、cuBLAS 13.8、词表 120832；混元 4 没量，推断沿用，见[对齐细节第 6 节「GPU 一侧」](DETAILS.md#gpu-一侧只影响-golden-的制作不影响-tpu-训练) G19 那行） | 已登记（GPU 设置） | G19 |
| 精度 | FP32 超越函数（exp / log / sigmoid / silu） | CUDA 的 FP32 `expf` / `logf` 等：exp ≤ 2、log ≤ 1 ULP（CUDA 文档）；sigmoid / silu 是 PyTorch eager 的 FP32 opmath，B200 实测最大 3.4 ulp（台账 G26，[`frameworks/megatron/README.md`](../../frameworks/megatron/README.md)「sigmoid / silu 各处的实际算法」） | `transcendental_accuracy_highest: false → true`（台账 M33，`t1z-*` 起；t1v、t1w 没有）：索引器 KD、输出门、iHC 门与 hc_head、路由、silu、主 loss 与 MTP loss 的交叉熵改用 `accuracy=HIGHEST`（交叉熵见 MaxText `94007054`）；sigmoid / silu 从 MaxText `b8d4ba8b` 起改用 HIGHEST 的 exp 搭（台账 M37，`t1zs-*` 起，此前 `lax.logistic` 的 HIGHEST 不起作用；MaxText `1d9cc8c7` 起这两个函数的反向改用 PyTorch 的解析式，M39；七处可以分别覆盖，M38）；注意力内核里的 exp 不在其内 | 默认不同（TPU 默认是近似实现，ulp_max 105 / 62 / 83 / 86，`tw-elem3`）；开关写法 exp 1、log 3、sigmoid 3、silu 4（`tw-elem4`，`results/tpu-elementwise-ulp.json`） | exp / log：`t1z-*` 起已打开（M33）；第 1 步诊断：打开后 KD 的 lse 偏差 −0.953 → +0.025 格，梯度不看 N、R > 1.2 的张量个数（GPU 取 `g2s` 三份的中位数） 7 → 7（成员有变），`results/t1-step1-transcendental-accuracy.json`。sigmoid / silu：打开 M37 的诊断代 `t1zs` TPU loss 偏高变多，待查。log 仍超 1 ULP；对齐细节 D14 | 台账 M32、M33、M37 |
| 精度 | 输出门 sigmoid | `torch.sigmoid` 在 BF16 上算、BF16 输出 | FP32 里算、舍成 BF16 后乘 | 不同（读代码所得） | 待核（只在 L1 注意力整块里间接比） | CHECKS「逐元素函数的精度」 |
| 精度 | Newton–Schulz 的矩阵乘 | `muon_fp32_matmul_prec: medium`：X 转 BF16 张量迭代（emerging_optimizers 0.3.0），A、B、X 每步各舍一次 BF16 | `muon_mcore_ns_bf16: false → true`（台账 M30、M31；原来默认精度下迭代量留 FP32） | 对上 GPU（连续 8 步回放：Muon 增量对 GPU 差 ≤ 1.6%，`results/t1-optim-replay-tpu.json`） | 已对齐（优化器 8 步回放里 Muon 的差 12.8% → 1.6%）；T1 打开（M31），从 t1v / t1w 之后的`t1z-*` 起；300 步 L4 见 `t1z-l4`（grad_norm 仍待定 1.31）；现行 `tc-a` grad_norm 1000 步过（MARE r 0.79），前 300 步待定（MARE r 1.01，C 口径另 5 份 `tc-b … tc-f` 0.81–0.92、都过；`tc-f` 0.90 见 [`mini-config-c-vs-m58.json`](results/mini-config-c-vs-m58.json)）；C 口径之前的 `tz-off-rep` 是 0.81、1.23 | 台账 M30、M31 |
| 精度 | Adam 的 FP32 除法、sqrt 与偏差修正 | PyTorch 的 FP32 除法与 sqrt（CUDA 默认编译选项下这两个是正确舍入；PyTorch 的编译选项没有直接核，间接证据：TPU 改成正确舍入的写法后，8 步回放里这些项与 GPU 逐位相同，见本行状态栏），偏差修正在 Python 里用 FP64 算 | `muon_adam_accurate`：默认 false，T1 / T2 true（台账 M34，2026-11-08 打开，`t1zs` 之后的运行起：倒数加一步牛顿修正、sqrt 一步牛顿修正、1 − β^t = −expm1(t·log β)） | 开关打开后相同（TPU 默认除法 2、sqrt 3、1 − β^t 108 ULP，`tw-elem3`；开关写法 1 / 1 / 2，`tw-elem4`） | 已打开；整网：诊断代 `t1zt` 1000 步路由偏置均值不过（N 0.501，TPU 1.27，r 2.54），两边的差比 t1zr 大，原因没有查；C 口径之前的 `tz-off-rep` 路由偏置均值过（r 0.20，`t1zo-l4`）；现行 `tc-a` 1000 步待定（带符号均值 r 1.31，未定责；C 口径另 5 份 0.36–0.80（只改执行方式的 4 份 0.39–0.80，0.36 是改了重算策略的 `tc-e`），`t1zc-repeats-6.json`），挡签字条件 2a；8 步回放 `tw-optim8b`（第一版 `64652880`）里 Adam 小张量 R 3.82 → ≤ 1.03（`results/t1-optim-replay-adam-acc-tpu.json`）；最终版 `7c95fabe` 的回放 `tw-optim8c` 里这些项与 GPU 逐位相同（`results/t1-optim-replay-adam-acc-r2-tpu.json`，对齐细节 D17） | 台账 M34 |
| 精度 | 三个梯度 FP32 开关 | 不适用（GPU 的 norm、输出头、嵌入梯度本来就是 FP32） | `rmsnorm_scale_grad_fp32`、`logits_dense_wgrad_fp32`、`embedding_grad_fp32`：false（T1 不开） | 不同（T1 不开） | 有差异（单项诊断：`embedding_grad_fp32` 在 hy4-mini 第 1 步，默认设置下词嵌入梯度 R 0.96 → 0.92（根源一览第 22 行）；`rmsnorm_scale_grad_fp32` 在真实权重 B 第 1 步，norm 梯度 R 中位数 1.60 → 0.98（第 23 行）；`logits_dense_wgrad_fp32` 混元 4 上没单独量） | M12 |

## 优化器与调度

| 类别 | 参数 / 开关 | GPU（Megatron） | TPU（MaxText）：默认 → T1 | 对齐方式 | 状态 | 台账 / 证据 |
|---|---|---|---|---|---|---|
| 优化器 | 类型 | Megatron-Core Muon（emerging_optimizers 0.3.0），`--optimizer muon` | `opt_type: adamw → muon`、`muon_type: optax_muon`（T1 yml） | 相同 | 已对齐（配置） | 协议「优化器」行 |
| 优化器 | 动量、NS 步数与系数、Nesterov | momentum 0.95；NS 5 步 simple 系数、Nesterov 开（`train_hy4mg.py` 在 `--optimizer muon` 时默认设上） | `muon_beta 0.95`；NS 5 步、系数 (3.4445, −4.7750, 2.0315) 与 Nesterov 在 MaxText 里写死 | 相同 | 已对齐（配置） | G7 |
| 优化器 | Nesterov 的写法 | 原版：β · mu_t + (1 − β) · g_t，不做偏差修正 | `muon_mcore_nesterov: false → true` | 相同（TPU 用开关对上） | 已对齐（单测：与 Megatron-Core 逐步比，M15–M17 全开时相对差 ≤ 1.3e-6） | M15 |
| 优化器 | 标量参数的 Adam | 普通 Adam，β1 0.9、β2 0.95、ε 1e-10 | `muon_adam_plain: false → true`（不再是 NAdam）；`adam_b1 0.9`、`adam_b2 0.95`、`adam_eps: 1e-8 → 1e-10`、`adam_eps_root 0` | 相同（TPU 用开关对上）；GPU 实际生效的是 `config.adam_eps` 1e-10（`meta.json` 里的 `config.eps` 1e-8 是写死的过时字段，`t1-param-norm-locus.json` 的 notes） | 已对齐（单测：与 torch Adam 相对差 ≤ 1e-5） | M16 |
| 优化器 | NS 前的归一化 | x / max(‖x‖, 1e-7) | `muon_mcore_ns_normalize: false → true`（原来 x / (‖x‖ + `adam_eps`)） | 相同（TPU 用开关对上） | 已对齐（单测） | M17 |
| 优化器 | Muon 分区：iHC 矩阵 | `attn_hc.fn`、`ffn_hc.fn`、hc_head 的 `hc_fn` 标 `use_muon=False`（`--muon-exclude-ihc ihc+head`） | `muon_exclude_ihc: false → true` | 相同 | 已对齐（L0 逐参数对拍 `L0.optim.partition`：239 个参数一致；GPU rank 0 上 140 Muon / 99 Adam，TPU 分区表见下） | M3、G5 |
| 优化器 | Muon 分区：gate / up | Megatron-Core 的 `linear_fc1` 本来是融合的一个矩阵 | `muon_fuse_glu: false → true`（只支持 `optax_muon`） | 相同（TPU 用开关对上） | 已对齐（单测：与手工拼接逐元素一致） | M4 |
| 优化器 | Muon 分区：输出门 | 每头一个矩阵（`--muon-gate per_head`，Megatron-Core 的 QKV 拆分路径） | `muon_attention_gate_whole_matrix: false`（T1 不开） | 照现状 | 已对齐（单测：与逐头单独跑逐位相同） | M2、G6 |
| 优化器 | Muon 分区：q_b、kv_b | 各自所有头拼成一个矩阵 | 同（MaxText 现状） | 照现状（生产 `muon_split_mla_*` 语义未知） | 已对齐（配置） | 台账「照现状」表 |
| 优化器 | Muon 分区：路由权重 | 走 Muon | `muon_include_routers: true` | 相同 | 已对齐（配置） | 协议「路由权重」行 |
| 优化器 | 权重衰减 | 0（`--wd` 默认 0.1，T1 传 0：`g2-a`、`g3-a` 的 `meta.json` 记 `weight_decay_effective` 0.0，作用于全部参数） | `muon_weight_decay 0`、`adam_weight_decay: 0.1 → 0` | 相同（照现状） | 已对齐（配置；GPU 读 `meta.json`） | 台账「照现状」表 |
| 优化器 | 梯度裁剪 | 全局范数 1.0；grad_norm 记裁剪前 | `gradient_clipping_threshold 1.0` | 相同 | 已对齐（配置） | — |
| QK-Clip | 开关、α、阈值 | 原版 `clip_qk`，α 0.5、τ 100，每层每头；MTP 层也裁（`--qk-clip-mtp on`，胶水） | `use_qk_clip: false → true`、`qk_clip_threshold 100`；α 0.5 在 MaxText 里写死 | 相同 | 已对齐（单测）；τ 100 下 1000 步没触发过；L4 全局最大 logit 曲线在 C 口径之前的 `tz-off-qkq` 上待定（1000 步 MARE r 1.09，`t1zo-l4-qkclip`；C 口径上没重判） | M1、G7 |
| QK-Clip | 分母 | min(1, τ / max_logit)，不加 eps | `qk_clip_mcore_denominator: false → true` | 相同（TPU 用开关对上） | 已对齐（单测） | M18 |
| QK-Clip | 最大 logit 的取法 | 只在选中且因果的位置上取缩放后 q·k 的最大值，不含 sink | `qk_clip_max_logit_exclude_sink: false → true`（`t1t-*` 起；关时 flash 内核的 running max 从 sink 起算） | 相同（TPU 用开关对上） | 已对齐（单测） | M25 |
| 注意力 | logits 精度 | `hy4_attn_scores_fp32: False → True`（训练入口 `--attn-scores-fp32`，`g2-*` 起）：q·k 在 FP32 里算再进 softmax；概率舍成 BF16 后乘 V | tokamax splash：q·k FP32 累加不舍入，softmax 在 FP32 | 相同（GPU 对上） | 已开，GPU 重跑 `g2-*` | G17 |
| Muon | 更新缩放 | `muon_scale_mode: shape_scaling`（g1-a 的 meta.json）：√max(1, 输出维 / 输入维) | `muon_consistent_rms` 不设：√max(1, fan_out / fan_in) | 相同 | 已对齐（单测参照即按 `shape_scaling` 生成） | G18 |
| QK-Clip | 梯度累积时的最大 logit | 按 microbatch 取最大 | 修 bug（无开关）：原来各 microbatch 的 aux 一律求和 | 相同 | 已对齐（单测；TPU 低阈值实测：阈值 0.1 让 QK-Clip 真触发，累积 2 对累积 1 第 1 步逐头最大 logit 比值 0.997–1.003，前 30 步两者之差与 GPU 两份重跑同一量级，判为只差求和顺序，[`t1zo-qkclip-accum-tau01.json`](results/t1zo-qkclip-accum-tau01.json)、[案例 07](../../cases/07-qk-clip-bugs.md)） | M19 |
| QK-Clip | 扫描层下的缩放 | 每层每头 | 修 bug（无开关）：原来同一扫描块的各层共用一个缩放 | 相同 | 已对齐（单测与 hy4-tiny 11 层实测） | M1 |
| 调度 | 学习率 | 常数 1e-5（`--lr` 默认 1e-5，`min_lr` 同值） | `learning_rate: 3e-5 → 1e-5`、`lr_schedule_type: cosine → wsd`、`warmup_steps_fraction: 0.1 → 0`、`wsd_decay_steps_fraction: 0.1 → 0`、`learning_rate_final_fraction: 0.1 → 1.0` | 相同 | 已对齐（L0 设置核对：两边每一步都等于 1e-5） | J7；协议「学习率」行（照现状，生产 2e-5 cosine） |
| 调度 | 跳过突跳步 | 不跳 | `skip_step_on_spikes: false` | 相同 | 已对齐（配置） | — |
| 路由 | 路由偏置更新 | 更新率 1e-3（`--bias-rate`），bias += rate · sign(平均负载 − 负载)，不减均值（原版没有这个选项；janus 训练入口另加了 `--bias-zero-mean`，默认关，只在生产配方臂里开，见下面「生产配方臂」小节） | `routed_bias_update_rate: 0 → 0.001`（T1 yml 与模型 yml 同值）；`routed_bias_zero_mean_update: false` | 相同 | 标准差已对齐（✅）；均值以现行 tc-a 为准待定（r 1.31，未定责，C 口径另 5 份 0.36–0.80；C 口径之前的 tz-off-rep ✅），见「Adam 的 FP32 除法、sqrt 与偏差修正」一行 | G14；协议变更记录「路由偏置」行。**生产建议：两边都打开**（混元 4 生产配方；GPU 一侧需在偏置更新后加一行减均值，原版 Megatron-Core 没有这个选项）；对齐时两边都关只因为 golden 用原版，见 [ROOT-CAUSES.md](../../ROOT-CAUSES.md) 第 1 行 |
| 索引器 | KD 系数与按层平均 | 每层系数 0.1 ÷ 产出层数（`--indexer-loss 0.1`） | `indexer_loss_scaling_factor: 0 → 0.1`（经 `forced_overrides`；MaxText 对各产出层取平均后乘它） | 相同 | 已对齐（`indexer_kd_check.py`：与 MaxText 的转写一致到 2.5e-8） | G4 |
| 索引器 | KD loss 的写法 | 原版 `compute_dsa_indexer_loss`：L1 归一化 1e-10 截断、精确 log_softmax、log(max(p, 1e-10))、p = 0 的项不计 | `indexer_kd_mcore_log: false → true` | 相同（TPU 用开关对上） | 已对齐（单测）；L4 KD loss：现行 `tc-a` 1000 步过（MARE：N 3.90e-4，TPU 2.94e-4，r 0.75，16 份尺子）；C 口径之前的 `tz-off-rep` 过（TPU 2.98e-4，r 0.77）；上一代 `t1zg-a`（打开 M50，只改记录值）1000 步过（MARE：N 3.49e-4，TPU 2.96e-4，r 0.85，8 份尺子）；上一代 `t1zw-a` 带符号均值待定（N 1.83e-4、TPU +1.93e-4，r 1.06），原因是记录写法在 FP32 下病态（对齐细节 D1、根源一览第 25 行） | M20 |
| 索引器 | eod 掩码下 KD 的权重 | 按查询行数加权，再 ÷ 全局有效 token 数（per-token） | `indexer_kd_mcore_weight: false → true` | 相同（TPU 用开关对上） | 已对齐（单测） | M21、G10 |
| MTP | eod 掩码下 MTP loss 的重标 | 每个 rank 按自己的原 / 滚动 token 数 | `mtp_loss_mcore_per_rank: false → true` | 相同（TPU 用开关对上）；GPU 另记 MaxText 口径 `mtp_loss_mt`，L4 两边都读它 | 已对齐（单测） | M22、G11、G12 |
| 损失 | z-loss | 无 | `z_loss_multiplier 0` | 相同 | 已对齐（配置） | — |

### 生产配方臂（P1）多出的三个开关

T1 不开。生产配方臂（GPU `p1g-a … p1g-h` 8 份，c–h 是补跑的尺子；TPU `p1t-a`；2026-11-11）三项一起改，结果见[对齐报告「适用范围」](ALIGNMENT-REPORT.md#适用范围测过什么没测什么)。

| 项 | GPU（janus 训练入口，台账 G35；默认都关） | TPU（MaxText 键） | T1 | P1 | 核对 |
|---|---|---|---|---|---|
| 学习率调度 | `--lr 2e-5 --lr-schedule cosine --warmup-steps 200 --lr-final-fraction 0.1`（照 MaxText `create_learning_rate_schedule` 在 FP32 里算） | `learning_rate`、`lr_schedule_type`、`warmup_steps_fraction`、`learning_rate_final_fraction` | 常数 1e-5（见上表「学习率」行） | 峰值 2e-5、cosine、warmup 200（TPU `warmup_steps_fraction: 0.2`）、终点 0.1 | 逐步学习率 1000 步 702 步逐位相同，最大相对差 1.0e-6 |
| YaRN | `--yarn-factor 8`（只换 RoPE 频率，不乘 mscale） | `rope_factor`（`rope_type: yarn`，`rope_attention_scaling: false`） | 1 | 8 | 频率在 CPU 上与 MaxText 写法逐位相同；12 个 RoPE 模块都换了 |
| 路由偏置零均值 | `--bias-zero-mean on`（偏置更新量先减对专家的均值） | `routed_bias_zero_mean_update` | 关 | 开 | 1000 步各层偏置均值漂移 TPU ≤ 1.9e-8、GPU 8 份 ≤ 4.3e-8 |

上表「核对」一列 GPU 8 份都核过，每份读数相同（学习率）或在所写上限内（偏置漂移）。8 份 GPU 的 meta.json 对 `p1g-a` 逐字段相同（配置、代码、镜像、并行、版本、数据、模型、步数）。证据：[`t1pr-settings.json`](results/t1pr-settings.json)、[`t1pr-gpu8-meta-diff.json`](results/t1pr-gpu8-meta-diff.json)。

## 数据

| 类别 | 参数 / 开关 | GPU（Megatron） | TPU（MaxText）：默认 → T1 | 对齐方式 | 状态 | 台账 / 证据 |
|---|---|---|---|---|---|---|
| 数据 | 来源 | fineweb-edu `sample/10BT` 第一个分片，混元 4 分词器（与真实权重截层同一份 `tokens.npy`）；`--save-batches` 写下每个 rank 的 tokens / labels / loss_mask | `dataset_type: synthetic`；`train_hy4.py` 按 GPU 写下的 batch 原样回放 | 相同 | 已对齐（L0 `L0.data.fingerprint`：`t1zg-a` 回放的输入对 `g3-a` 65,000 个单元 0 处不同；现行 `tc-a` 与 C 口径之前的 `tz-off-rep` 都回放同一份 `g3-a` 数据，没有单独出指纹；阶段 A 4,800 个单元 0 处不同） | G9 |
| 数据 | loss 掩码 | eod（`eos_token_id`）位置 `loss_mask` = 0（Megatron `eod_mask_loss`），MTP 按滚动后的掩码 | 原样回放 GPU 的 `loss_mask` | 相同 | 已对齐（配置） | G9；协议「loss 掩码」行 |
| 数据 | 主 loss 平均 | `calculate_per_token_loss`：loss 和 ÷ 全局批有效 token 数 | 手动梯度累积：xent 和 ÷ 全局 token 数 | 相同 | 已对齐（配置；GPU per-token 开 / 关差在同设置重跑内） | G9 |
| 数据 | 序列、批 | 4096；全局批 16 = 8 卡 × microbatch 1 × 累积 2（`DP = 卡数`、`GA = 16 ÷ DP`） | `max_target_length: 2048 → 4096`、`per_device_batch_size: 12 → 1`、`gradient_accumulation_steps: 1 → 2` | 相同（设备 d 第 m 个累积步 = GPU rank d 的 microbatch m） | 已对齐（配置） | G7、J4 |
| 数据 | 种子 | `model_parallel_cuda_manual_seed(1234)`；不随机初始化（从 HF 目录载入） | `init_weights_seed: 0 → 1234`、`data_shuffle_seed: 0 → 1234`（不起作用：权重全部从检查点载入，L0 初始权重逐位；数据走回放，L0 输入指纹逐位；`train_hy4.py` 每步不传随机数，`dropout_rate: 0`） | 不适用 | — | — |

## 并行与确定性

| 类别 | 参数 / 开关 | GPU（Megatron） | TPU（MaxText）：默认 → T1 | 对齐方式 | 状态 | 台账 / 证据 |
|---|---|---|---|---|---|---|
| 并行 | 切分 | B200 × 8：数据并行 8、专家并行 4（`--ep` 默认 8，T1 传 4：`g1-a`、`g2-a`、`g3-a` 的 `meta.json` 记 `parallelism.ep` 4），TP 1、CP 1；token 分发 alltoall | v5p 2x2x2（8 个设备）：`ici_fsdp_parallelism -1`（= 8）、`ici_expert_parallelism 1`、`ici_tensor_parallelism 1`、`ici_data_parallelism 1`、`dcn_data_parallelism: -1 → 1` | 不同（执行方式不同、数学相同；TPU 只改执行参数的诊断运行 `t1np-2` 用专家并行 4） | 已对齐（配置） | 协议第 3 节 |
| 并行 | MoE 实现 | grouped GEMM（`moe_grouped_gemm`） | `megablox: true`、`sparse_matmul: true`、`capacity_factor -1`（不丢 token） | 等价 | 已对齐（配置；诊断运行 `t1np-3` 改用稠密实现） | 协议第 3 节 |
| 重算 | 激活重算 | 整层重算（`--recompute`：full / uniform / 1） | `remat_policy: full → custom`（`t1t-*` 起；除层输入外全部 remat，等价于整层重算） | 相同 | 已对齐（配置） | 协议「重算」行、M26 |
| 重算 | 路由在重算里重选与否 | 整层重算，路由跟着重算（读代码：full 重算把整层前向再跑一遍；Megatron-Core 的路由回放 `moe_enable_routing_replay` 默认关，`train_hy4mg.py` 不开） | `moe_routing_maps: device`（模型 yml），`custom` 下生效，路由索引存下不重选；`full` 下不读这个设置，T1 在 `t1t-*` 之前会重选 | 写法不同；路由索引是整数，重算输入相同时选择相同，不改数值（推断） | 已按协议写法对齐（TPU） | M26 |
| 注意力 | 实现 | Megatron-Core 稀疏注意力 + sink（`sparse_sink_attention`，FP32 softmax） | `attention: autoselected → flash`（tokamax splash，`use_tokamax_splash: true`）；块大小 `sa_block_*` 512；`sa_use_base2_exp: true → false`（2026-11-11 起，M43，见下面开关表） | 不同（末位注定不同；诊断运行 `t1np-1` 把块大小改成 256） | 已对齐（logits 精度见「优化器与调度」表里的「logits 精度」行，台账 G17）；L1 注意力整块与 L2 逐层回放是 base2 开时测的，关 base2 后没有重测 | 协议第 3 节 |
| 层 | 扫描 | 不适用 | `scan_layers: true` | — | — | M1（扫描下 QK-Clip 的修正） |
| 确定性 | 同设置重跑 | 不逐位（`deterministic_mode: False`；8 份重跑第 1 步的相对极差：loss 2.1e-6 / 1.7e-6 / 2.9e-6、grad_norm 7.3e-5 / 9.3e-5 / 8.2e-5（`g1` / `g2` / `g3`，按各自 `metrics.jsonl` 算）；台账 G9 的 4.9e-6、2.4e-4 是 T1 之前关 eod 掩码的测试） | 逐位（两组各两份 100 步、45 个逐步量） | — | L4 噪声只用 GPU 重跑两两（前 300 步 8 份 28 对，1000 步 16 份 120 对，协议第 3 节）；TPU 逐位只作记录（C 口径之前的配置两份 1000 步 47 个逐步量逐位相同，`t1zo-determinism`；C 口径上没做） | G9；[`hy4-mini-tpu-determinism.json`](results/hy4-mini-tpu-determinism.json) |
| 观测 | 日志、存档 | janus 运行目录（`JanusDump`），逐步记 lm_loss、grad_norm、param_norm、param_norm_fp64、mtp_loss、mtp_loss_mt、indexer_loss、max_attention_logit、路由偏置统计；QK-Clip 逐层逐头的最大 logit、η 与被裁头数（G21） | `log_period: 100 → 1`、`enable_checkpointing: true → false`；`train_hy4.py` 另记 `param_norm_exact`（`t1t-c`、`t1u-*` 起） | — | — | G12、G13、G21、J13 |
| 观测 | param_norm 的口径 | `param_norm_fp64`：更新之后各参数平方和在 FP64 里合成（精确值） | `param_norm_exact`：更新之后每 128 个元素一段 FP32 求平方和、主机 FP64 合成，不含路由偏置（同 GPU）；MaxText 自己的 `learning/param_norm` 整棵树在 FP32 里一路相加，对精确值逐步误差在 −4.4e-7 至 +2.9e-7 之间、平均偏低约 6e-8（`t1t-c`，由同一运行的两列算出） | 等价（都是精确值；L4 两边比这两列） | 口径已对齐（`t1t-l4` 起；数值上 `t1v-l4`、`t1w-l4` 过） | PROTOCOL 变更记录「param_norm」两行；J13 |
| 诊断 | 诊断开关 | `--diag-kd-dump`（默认关） | `debug_sow_indexer: false`（T1 不开，只在诊断运行里开） | — | 不改数值 | G16、M24 |

## 为对齐改过的代码

逐项的提交、测试与数值核验在 [`CHANGES-LEDGER.md`](../../CHANGES-LEDGER.md)，这里只列改了什么、对上谁、生产上怎么用。用途分类：A 对齐专用（生产关掉）、B 修 bug / 建议生产也开、C 取舍；口径和步时的量法见[根源一览表前的图例](../../ROOT-CAUSES.md)，生产建议见[对齐细节第 6 节](DETAILS.md#6-开关表怎样做到在可解释范围内对齐)。

| 改动 | 哪一侧 | 对上谁 | 开关（T1） | 台账 | 用途分类 | 生产关掉时的精度影响 | 性能影响（开 vs 关） |
|---|---|---|---|---|---|---|---|
| QK-Clip 在扫描层下按层缩放 | MaxText | 修 bug | 无 | M1 | B（修 bug，无开关） | 不修：QK-Clip 触发时裁错 | 不适用 |
| 输出门整块正交化 | MaxText | 备用 | `muon_attention_gate_whole_matrix`（关） | M2 | 备用，T1 不开 | — | 未测 |
| iHC 矩阵走 Adam | MaxText | 生产 `muon_exclude_ihc` | `muon_exclude_ihc`（开） | M3 | 设置：按训练配方定 | 非精度项 | 未测 |
| gate / up 拼接正交化 | MaxText | Megatron-Core 融合的 `linear_fc1` | `muon_fuse_glu`（开） | M4 | 设置：按训练配方定 | 非精度项 | 未测 |
| `maxtext_muon` 的维度判定读 config | MaxText | 修 bug | 无（T1 不用） | M5 | B（修 bug，无开关） | — | 不适用 |
| 新模型 hy4-mini；hy4-tiny 改 vLLM 读法 | MaxText | — | 模型名 | M6、M7 | — | — | — |
| 四个精度开关与输出头精度 | MaxText | GPU（Megatron-Core + TE）的舍入位置 | `routed_gate_bf16_rounding`、`rmsnorm_single_rounding`、`mlp_activation_round_per_op`、`dense_fp32_accumulation`（开）；`logits_matmul_precision`（`highest`，`t1t-*` 起） | M8–M10、M12、M13 | M12、M9：B；M13：C（取舍，生产看情况）；M8、M10：A | 见[对齐细节第 6 节](DETAILS.md#6-开关表怎样做到在可解释范围内对齐) | M12 开 0.4767 s/步、关 0.4729 s/步，开着贵 0.8%（`tq-base` / `tq-m12`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss MARE r 0.89 → 0.97；M9 开 0.4767 s/步、关 0.4777 s/步，开着省 0.2%（`tq-base` / `tq-m9`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss 带符号均值 r 0.48 → 1.24；M8 开 0.4767 s/步、关 0.4763 s/步，开着贵 0.1%（`tq-base` / `tq-m8`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss MARE r 0.89 → 1.01；M10 开 0.4767 s/步、关 0.4770 s/步，开着省 0.1%（`tq-base` / `tq-m10`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss MARE r 0.89 → 0.96；M13：只关它 300 步主 loss r 6.03 不过，步时 +24%（hy4-mini `tq-m13` 0.3854 → `tq-base` 0.4767 s；hy4-mid 2.2610 → 2.8125 s） |
| Muon 的 Nesterov、标量 Adam、NS 归一化 | MaxText | 原版 Megatron-Core | `muon_mcore_nesterov`、`muon_adam_plain`、`muon_mcore_ns_normalize`（开） | M15–M17 | A（语义约定：已有在 GPU 上调好的超参时开） | 非精度项 | 未测 |
| QK-Clip 的分母；梯度累积时最大 logit 取最大 | MaxText | 原版 Megatron-Core；修 bug | `qk_clip_mcore_denominator`（开）；无 | M18、M19 | 分母：A（语义约定）；M19：B（修 bug） | 非精度项 | 未测 |
| 索引器 KD 的写法与 eod 权重；MTP loss 的 eod 重标 | MaxText | 原版 Megatron-Core | `indexer_kd_mcore_log`、`indexer_kd_mcore_weight`、`mtp_loss_mcore_per_rank`（开） | M20–M22 | A（语义约定） | 非精度项 | 未测 |
| 索引器真 FP32 | MaxText | 原版 Megatron-Core | `indexer_fp32_scores`（开） | M23 | B（代价见右栏） | 见[对齐细节第 6 节](DETAILS.md#6-开关表怎样做到在可解释范围内对齐) | 开 0.4767 s/步、关 0.4484 s/步，开着贵 6.3%（`tq-base` / `tq-m23`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss 带符号均值 r 0.48 → 1.08 |
| 诊断：索引器 KD 输入 | MaxText | — | `debug_sow_indexer`（关） | M24 | 诊断，生产不开 | — | — |
| QK-Clip 最大 logit 不含 sink；`custom` 重算存路由索引；weights_proj 乘 BF16 权重 | MaxText | 原版 Megatron-Core；PROTOCOL 第 1.2 节「重算」行 | `qk_clip_max_logit_exclude_sink`（开）、`remat_policy: custom`、`indexer_weights_proj_bf16_weight`（开）；都从 `t1t-*` 起 | M25–M27 | M25、M26：B（语义）；M27：A | 见[对齐细节第 6 节](DETAILS.md#6-开关表怎样做到在可解释范围内对齐) | M25 开 0.4767 s/步、关 0.4723 s/步，开着贵 0.9%（`tq-base` / `tq-m25`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss 带符号均值 r 0.48 → 0.93；M26 开 0.4767 s/步、关 0.4768 s/步，开着省 0.0%（`tq-base` / `tq-m26`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss MARE r 0.89 → 0.80；M27 开 0.4767 s/步、关 0.4763 s/步，开着贵 0.1%（`tq-base` / `tq-m27`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss MARE r 0.89 → 0.86，不过：indexer_loss |
| iHC 四路流加权和逐路 FP32 乘加 | MaxText | GPU 的 `torch.sum` 写法 | `ihc_stream_sum_unrolled`（开，`t1v-*` / `t1w-*` 起） | M28 | B | 见[对齐细节第 6 节](DETAILS.md#6-开关表怎样做到在可解释范围内对齐) | 开 0.4767 s/步、关 0.4757 s/步，开着贵 0.2%（`tq-base` / `tq-m28`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss 带符号均值 r 0.48 → 1.25 |
| iHC 分支输入与流强制舍成 BF16 | MaxText | GPU 的 eager 写法 | `ihc_force_bf16_rounding`（开，`t1zr-*` 起） | M36 | A | 见[对齐细节第 6 节](DETAILS.md#6-开关表怎样做到在可解释范围内对齐) | 开 0.4767 s/步、关 0.4763 s/步，开着贵 0.1%（`tq-base` / `tq-m36`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss 带符号均值 r 0.48 → 1.01 |
| iHC 与 hc_head 的流梯度、分支输出照 GPU 舍 BF16 | MaxText | GPU 的 autograd 写法 | `ihc_stream_grad_gpu_order`（开，`tc-*` 起，C 口径） | M58 | A | 未测（生产不开：只为照 GPU，离精确值远近没测） | 1000 步运行步时中位数 C 口径 0.4963 s、只开 M36 的对照 0.4960 s（`tc-a` / `ti-o-a`，第 11–1000 步，单次运行，不是专门的 A/B，两份的 MaxText 提交也不同：`8d3d7561` / `751fdc6d`；[`mini-step-time.json`](results/mini-step-time.json)） |
| hc_head 收拢之后强制舍 BF16 | MaxText | GPU 的 `.to(bfloat16)` | `hc_head_force_bf16_rounding`（开，`tc-*` 起，C 口径） | M52 | A | 未测（生产不开：只为照 GPU） | 同上一行（C 口径整体） |
| QK-Clip 逐层逐头记录 | MaxText | 诊断（低阈值 QK-Clip 逐头比 GPU） | `qk_clip_log_per_head`（只在诊断运行里开） | M29 | 诊断，生产不开 | — | — |
| Muon 的 Newton–Schulz 在 BF16 张量上迭代 | MaxText | 原版 Megatron-Core（emerging_optimizers） | `muon_mcore_ns_bf16`（开，`t1z-*` 起） | M30、M31 | A | 关（FP32 迭代）：Muon 增量对 FP64 中位数 3.0e-3；开（同 GPU）：中位数 3.95e-2 | 开 0.4767 s/步、关 0.4715 s/步，开着贵 1.1%（`tq-base` / `tq-m31`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss MARE r 0.89 → 0.94 |
| FP32 exp / log / sigmoid / silu 与交叉熵用 `accuracy=HIGHEST`；sigmoid / silu 改用 HIGHEST 的 exp 搭 | MaxText | GPU 的 FP32 超越函数（exp ≤ 2、log ≤ 1 ULP，CUDA 文档；sigmoid / silu B200 实测 ≤ 3.4 ulp，G26） | `transcendental_accuracy_highest`（开，exp / log `t1z-*` 起，sigmoid / silu `t1zs-*` 起） | M32、M33、M37 | exp / log：B（代价见右栏）；sigmoid / silu：待定 | 见[对齐细节第 6 节](DETAILS.md#6-开关表怎样做到在可解释范围内对齐) | 开 0.4767 s/步、关 0.4627 s/步，开着贵 3.0%（`tq-base` / `tq-m33`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss 带符号均值 r 0.48 → 0.99，不过：indexer_loss |
| Adam 的精确 FP32 除法、sqrt 与偏差修正 | MaxText | GPU 的 PyTorch Adam | `muon_adam_accurate`（开，`t1zs` 之后的运行起） | M34 | B | 见[对齐细节第 6 节](DETAILS.md#6-开关表怎样做到在可解释范围内对齐) | 开 0.4767 s/步、关 0.4762 s/步，开着贵 0.1%（`tq-base` / `tq-m34`，300 步运行第 11–300 步中位数）；只关这一个开关的 300 步 L4：主 loss MARE r 0.89 → 0.93 |
| QK-Clip 对最大 logit 不是正数的头缩放取 1 | MaxText | 修 bug（η 为负时开根号得 NaN） | 无 | M35 | B（修 bug，无开关） | 不修：最大 logit 全为负的头触发裁剪就 NaN | 不适用 |
| sigmoid / silu 按位置的覆盖项；精确 sigmoid / silu 的反向用解析式 | MaxText | GPU 各处 sigmoid / silu 的实际写法（G26） | `accurate_sigmoid_*` / `accurate_silu_*`（默认 null，跟随 `transcendental_accuracy_highest`）；反向无开关 | M38、M39 | 待定（随 M37） | 见对齐细节第 6 节 | 未测 |
| 路由权重在 fc2 之前乘、top-k 逐个加并舍 BF16 | MaxText | 原版 Megatron-Core | `moe_mcore_prob_before_fc2`（开，janus `9d20039` 起；`t1zu-a`、`t1zw-a` 带，`t1zr` 没有） | M40 | A | 关：少几次 BF16 舍入，推断同样准或更准 | 开 0.4767 s/步、关 0.4756 s/步，开着贵 0.2%（`tq-base` / `tq-m40`，300 步运行第 11–300 步中位数，[`tq-step-time.json`](results/tq-step-time.json)） |
| gate / up 的输入梯度在 FP32 里相加、只舍一次 | MaxText | GPU 融合的 `linear_fc1` | `mlp_fused_fc1_dgrad`（开，janus `2679a83` 起；`t1zu-a`、`t1zu-c`、`t1zw-a` 带，`t1zr` 没有） | M41 | B（少一次舍入） | 关：dX 多舍一次，对融合写法的 rel_L2 约 3e-3（CPU 单测） | 开 0.4767 s/步、关 0.4853 s/步，开着省 1.8%（`tq-base` / `tq-m41`，同上，[`tq-step-time.json`](results/tq-step-time.json)） |
| 注意力 sink 的梯度全程 FP32（不照 GPU 写法） | MaxText | 对 FP64 更准 | `splash_dsinks_fp32`（M42，默认关；对齐配置不开：2026-11-09 撤下，诊断代 `t1zv-b` 带） | M42 | B（生产建议开，对 FP64 sink 梯度更准） | 关：sink 梯度在 BF16 里累加，单层各层离 FP64 是 GPU 写法的 1.6–2.9 倍 | 未测 |
| 注意力 sink 的梯度照 GPU 写法：dp 逐位置舍 BF16，p 与求和 FP32 | MaxText | GPU（Megatron-Core autograd 穿过 FP32 softmax） | `splash_dsinks_gpu_order`（M42b，默认关；对齐配置 2026-11-09 起开） | M47 | A（只为照 GPU 写法；生产用 M42） | 关：同上一行 | +3.9%（单开关 A/B，台账 M47） |
| 索引器 KD 改记广义 KL（差值经 stop_gradient 加回，梯度逐位不变） | MaxText | 对真值（CPU FP64）：原写法在 FP32 下病态 | `indexer_kd_generalized_kl`（开，2026-11-09 起） | M50 | B（只改记录值） | 关：KD 记录值偏高，1000 步 KD 待定（r 1.06） | 无（只改记录值） |
| splash 不走 base2 指数（q 不乘 log2(e) 再舍 BF16），与 GPU 的自然指数同 | MaxText | GPU（自然指数，q·k 在 FP32） | `sa_use_base2_exp`（MaxText 默认 true；对齐配置 2026-11-11 起关，所有尺寸一致；整网结果：关 base2、C 口径之前的 `t1zo-l4` 主 loss r 1.55 与 base2 无关，见对齐细节 D30；现行 C 口径 `t1zc-l4` 主 loss 过，MARE r 0.98） | M43 | B（生产建议关：前向 lse 与 q 一路梯度对 FP64 更准；对齐配置关） | 开着：lse 离 FP64 约大 30 倍；T2 B 两层 `q_a_layernorm` 梯度 R 2.89 / 1.58（关后 0.88 / 0.79） | 用 base2（`true`）比不用慢 0.1%（0.4767 对 0.4762 s/步，`tq-base` / `tq-base2`） |
| 混元 4 接到 Megatron-Core 上（门控 MLA、sink、iHC、索引器、截断、FP32 输出头、MTP） | GPU | MaxText / HF 的模型结构 | 无 | G1 | GPU 参照 | — | 不涉及 TPU |
| 索引器 KD 训练；iHC 不走 Muon；输出门按头正交化 | GPU | MaxText 的写法 | `--indexer-loss 0.1`、`--muon-exclude-ihc ihc+head`、`--muon-gate per_head` | G4–G6 | 设置：按训练配方定 | 非精度项 | 不涉及 TPU |
| 注意力 q·k 在 FP32 里算再进 softmax | GPU | 生产融合内核与 TPU 内核 | `--attn-scores-fp32`（开，`g2-*` 起） | G17 | B（只改 GPU 参照路径） | 关：q·k 多一次 BF16 舍入 | 不涉及 TPU |
| 输出头 dgrad 分段累加（登记现状） | GPU | 避开长求和偏差 | `--head-chunk 1024` | G19 | B（GPU 参照路径） | 关：FP32 长求和朝 0 的系统偏差（混元 3 实测，混元 4 推断沿用） | 不涉及 TPU |
| Muon 动量缓冲、QK-Clip 逐层记录、全量抓数指定步、逐算子抓数 | GPU | 诊断（优化器回放、QK-Clip、param_norm、单算子定位） | `--qk-dump-steps`、`--full-steps`、`--l1-capture-step` | G20–G23 | 诊断 | — | — |
| 训练入口：载入自检、MTP 层 QK-Clip、eod 掩码与 per-token loss、janus 运行目录 | GPU | 协议第 1.2 节 | `--qk-clip-mtp on`、`--eod-mask on`、`--per-token-loss on` | G7、G9、G12、G13 | 设置 / 工具 | 非精度项 | 不涉及 TPU |
| 第 1 步梯度导出成 HF 布局；KD 原始输入导出 | GPU | 第 1 步诊断 | 工具 | G15、G16 | 工具 | — | — |
| Muon 分区表、QK-Clip 扫描检查、hy4-mini 的 HF 目录与转换、T1 的 TPU 配置、第 1 步诊断、`scalar_series` 的 `equals`、检查脚本入口与逐元素 ULP 脚本、TPU 的 `param_norm_exact` | janus | — | 工具 | J1–J24 | — | — | — |
