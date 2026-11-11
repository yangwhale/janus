[janus](../../README.md) › [用户指南](../../docs/guide/README.md) ｜ 配置对照表；对应指南[第 3 章 TPU 侧准备](../../docs/guide/03-tpu-setup.md)

# 混元 3 · 两边配置对照表（GPU Megatron ↔ TPU MaxText）

模型一层的对照表，所有对齐实践共用。
每个实践的 `setup.md` 只写本实践和这里不一样的地方（硬件、切分、镜像与提交号），其余引用这一页。

相关页面：

- 模板见 [`../../templates/config-alignment.md`](../../templates/config-alignment.md)。
- GPU golden 每一份是怎么产生的，见 [`goldens.md`](goldens.md)：
  - 最终 golden `rcc` 对上游默认的全部改动；
  - 每处改动两侧的机制；
  - 试过没用的开关；
  - 只对一份 golden 比的协议。

本页先说明出处与取值约定，再按模型结构、数值精度、优化器与调度、数据、并行与确定性逐类对照，最后列出为对齐改过的代码。各表格里只放短结论，「见注 N」的长说明紧跟在该表下面。

### 出处与取值约定

出处约定：

- 「GPU meta」指 GPU golden 运行目录里 `meta.json` 的对应字段（`model_config`、`optimizer_config`、`scheduler_config`、`dtypes`），由 [`frameworks/megatron/train_hy3.py`](frameworks/megatron/train_hy3.py) 写出。
  - 表里的取值按第一阶段 golden [`g1k-a`](../../docs/glossary.md#g1k)（不开重算，1000 步，3 份里的第 1 份）的 meta 读。
  - 最终 golden [`rcc`](../../docs/glossary.md#rcc) 和它只差整层重算、路由回放、输出头 dgrad 分段累加与关分布式优化器（[`goldens.md`](goldens.md)「从 g1k 到 rcc」）。
  - 这几项在「激活重算」「路由是否在重算里重选」「输出头 dgrad」「分布式优化器」各行单列。
- 「TPU yml」指 [`frameworks/maxtext/hy3-4layers.yml`](frameworks/maxtext/hy3-4layers.yml)。
- 「MaxText 模型 yml」指 `maxtext` 的 `src/maxtext/configs/models/hunyuan3-smoke.yml`。
- 「MaxText 代码」指同一仓库的 `src/maxtext/layers/moe.py`、`layers/linears.py`。
  - 以上都按 v5p [`t-loss30`](../../docs/glossary.md#run-prefixes)（TPU 30 步冒烟）用的 commit `64cd9a64` 读。
  - 精度开关 #94–#97 按 `4d4ffac2` 读，`dense_fp32_accumulation`（FSDP 切开收缩维的投影改用 FP32 累加，#102）按 `0622774` 读。
  - 现行 `hy3-4layers.yml`（sha256 `a238883b…`）打开了这四个开关，所以要 `hunyuan@0622774` 或更新（[`frameworks/maxtext/README.md`](frameworks/maxtext/README.md)）。
- 「HF」指 `tencent/Hy3` revision `a960ebc3…` 的 `config.json`，经 [`truncate_hy3.py`](model/truncate_hy3.py) 截成 4 层、去 MTP。

取值约定：拿不准的写「待查」，推出来的写「推断」并给理由。

对齐方式：

- 相同（同一个值、同一种算法）
- 等价（写法不同、数学相同，要给证据）
- 不可避免差异（硬件或框架决定，只能定责）

状态：

- 已对齐（有检查结果证明，括号里写证明到哪一级）
- 已对齐（配置）（只有配置核对，没有数值检查专门测它，括号外写为什么不另测）
- 有差异（已知不同，写量级与定责）
- 不适用（当前配置不会发生）
- v5p、v7x 两个实践用同一份 yml；只写一个平台的证据时注明。

## 模型结构

| 类别 | 参数 / 开关 | 为什么关键 | GPU（Megatron）怎么配 | TPU（MaxText）怎么配 | 对齐方式 | 状态 | 证据 |
|---|---|---|---|---|---|---|---|
| 结构 | 隐藏维 | 所有权重形状 | `hidden_size: 4096`（GPU meta） | `base_emb_dim: 4096`（模型 yml） | 相同 | 已对齐 | 01 `phase2-ckpt-convert`（1781 个张量形状与逐位） |
| 结构 | 层数 | 截层 | `num_layers: 4`（GPU meta；HF `num_hidden_layers: 4`） | `base_num_decoder_layers: 4`（模型 yml） | 相同 | 已对齐 | 同上 |
| 结构 | 注意力头 / KV 头 / 每头维度 | GQA 形状、qkv 拆分 | `num_attention_heads: 64`、`num_query_groups: 8`、`kv_channels: 128`（GPU meta） | `base_num_query_heads: 64`、`base_num_kv_heads: 8`、`head_dim: 128`（模型 yml） | 相同 | 已对齐 | 同上；qkv 交错布局见 01 `results/phase2-cpu/verify-megatron.txt` |
| 结构 | 稠密 FFN 维 / 前几层稠密 | 第 0 层是稠密层 | `ffn_hidden_size: 13312`（GPU meta）；稠密层数由 AutoBridge 从 HF `first_k_dense_replace: 1` 读，meta 未记 | `base_mlp_dim: 13312`、`first_num_dense_layers: 1`（模型 yml） | 相同 | 已对齐 | 00 `L0.model.structure`（路由层数 3）；01 `phase2-ckpt-convert` |
| 结构 | 专家数 / top-k | 路由 | `num_moe_experts: 192`、`moe_router_topk: 8`（GPU meta） | `num_experts: 192`、`num_experts_per_tok: 8`（模型 yml） | 相同 | 已对齐（形状） | 01 `phase2-ckpt-convert` |
| 结构 | 专家 FFN 维 | 专家 GEMM 形状 | `moe_ffn_hidden_size: 1536`（GPU meta） | `base_moe_mlp_dim: 1536`（模型 yml） | 相同 | 已对齐 | 同上 |
| 结构 | 共享专家 | 每层多一条 FFN | `moe_shared_expert_intermediate_size: 1536`（GPU meta；HF `num_shared_experts: 1`） | `shared_experts: 1`（模型 yml；维度推断取 `base_moe_mlp_dim`，转换脚本按 name_map 搬的形状已核对） | 相同 | 已对齐（形状） | 01 `phase2-ckpt-convert` |
| 结构 | 词表 | embedding / 输出层形状 | 由 AutoBridge 从 HF `vocab_size: 120832` 读，meta 未记 | `vocab_size: 120832`（模型 yml） | 相同 | 已对齐（形状） | 01 `phase2-ckpt-convert` |
| 结构 | 输入输出 embedding 是否共享 | 输出层是不是单独一份权重 | HF `tie_word_embeddings: false`；GPU meta `untie_embeddings_and_output_weights: null`（字段没取到值，实际按 HF，推断不共享） | `logits_via_embedding: False`（模型 yml） | 相同（推断） | 已对齐（形状） | 01 `phase2-ckpt-convert`（输出层单独一个张量逐位） |
| 结构 | 路由打分函数 | sigmoid 与 softmax 的分差、翻转不同 | `moe_router_score_function: sigmoid`（GPU meta） | `routed_score_func: "sigmoid"`（模型 yml） | 相同 | 已对齐（L1、L2） | top-8 集合对 FP64 每层一致率 ≥ 0.9995（见注 1） |
| 结构 | 专家偏置与更新率 | 无辅助损失的负载均衡；偏置更新量小 | `moe_router_enable_expert_bias: true`、`moe_router_bias_update_rate: 0.001`（GPU meta） | `routed_bias: True`、`routed_bias_update_rate: 0.001`（模型 yml） | 相同 | 已对齐（配置） | 01 `phase2-t-loss30` 的 `L0.config.diff.dropout`（更新率相等） |
| 结构 | top-k 权重归一与缩放 | 专家输出加权 | `moe_router_topk_scaling_factor: 2.826`（GPU meta）；公式见注 2 | `routed_scaling_factor: 2.826`、`norm_topk_prob: False`（模型 yml）；公式见注 2 | 等价（同一个公式，两边读代码） | 已对齐（L1） | v5p `L1.ops.kernel.moe` R 0.94 过；v7x R 1.27 是 1 个 token 选错专家（见注 2） |
| 结构 | 辅助损失 / z-loss | 改 loss 与梯度 | `moe_aux_loss_coeff: 0.0`、`moe_z_loss_coeff: null`（GPU meta） | `load_balance_loss_weight: 0.0`、`z_loss_multiplier: 0.0`（TPU yml） | 相同 | 已对齐（配置） | 两边系数都是 0，没有数值检查专门测它（间接证据见注 3） |
| 结构 | MTP | 多一个预测头和 loss | `mtp_num_layers = None`（`train_hy3.py`） | 命令行强制 `mtp_num_layers=0`（`mt_common.py` 的 `FORCED`；模型 yml 里是 1，只在命令行覆盖时生效） | 相同 | 已对齐（L0） | v5p / v7x `L0.model.structure` 规则「没有 MTP」：模型变量清单里不含 `mtp`（`phase2-l0-state`、`phase2-v7x-l0-state`） |
| 结构 | RoPE | 位置编码 | `rotary_base: 11158840.0`、`position_embedding_type: rope`（GPU meta） | `rope_max_timescale: 11158840`、`rope_type: "default"`（模型 yml） | 相同 | 已对齐（L1） | `L1.ops.kernel.attn_core` R 1.009，v5p、v7x 都过（见注 4） |
| 结构 | 归一化与 eps | RMSNorm 末位 | `normalization: RMSNorm`、`layernorm_epsilon: 1e-05`（GPU meta） | `normalization_layer_epsilon: 1.0e-5`（模型 yml） | 相同 | 已对齐（配置 + L1） | 取值两边都是 1e-5；L1 两项都过（见注 5） |
| 结构 | qk 归一 | 注意力前对 q、k 做 RMSNorm | `qk_layernorm: true`（GPU meta） | `use_qk_norm: True`（模型 yml） | 相同 | 已对齐（L0、L1） | `L0.model.structure` 规则「q 归一化」「k 归一化」（每头维度 128 的 scale，1 + 3 层）；`L1.ops.op.qk_norm` R 1.07 过（v5p、v7x） |
| 结构 | 注意力偏置 | 多一组参数 | 无（参数清单里没有偏置张量，见注 6） | `attention_bias: False`（模型 yml；见注 6） | 相同 | 已对齐 | 01 `results/phase2-cpu/verify-megatron.txt`、`verify-maxtext.txt`、`phase2-ckpt-convert` |
| 结构 | dropout | 必须为 0 才可比 | `hidden_dropout: 0.0`、`attention_dropout: 0.0`（GPU meta） | `dropout_rate: 0.0`（TPU yml）；`enable_dropout: False`（模型 yml） | 相同 | 已对齐 | 01 `phase2-t-loss30` 的 `L0.config.diff.dropout` |

- 注 1（路由打分函数）：
  - 喂 golden 的输入，TPU 选中的 top-8 集合对 FP64 的一致率每层 ≥ 0.9995。
    - 检查项 `L1.router.topk.fp64.layer1..3`，v5p `phase2-l2-replay-rratio-on`、v7x `phase2-v7x-l2-replay-rratio-on`。
  - 打分函数换成 softmax 会改路由概率，MoE 块输出就不会过 `L1.ops.kernel.moe`（推断）。
  - v5p 这一项 R 0.94 过；v7x 的 R 1.27 是 1 个 token 的路由翻转（[`ALIGNMENT-REPORT.md`](ALIGNMENT-REPORT.md) 第 4 节）。
- 注 2（top-k 权重归一与缩放）：
  - GPU：sigmoid 时 top-k 分数除以它们的和再乘缩放。
    - 出处：Megatron-Core `moe_utils.py` 的 `topk_routing_with_score_function`，按上游 `core_r0.15.0` 读，容器里的 0.20.0 没有逐字核对。
  - TPU：sigmoid 时由 `deepseek_scale_weights` 先除以 top-k 权重和再乘缩放（MaxText 代码 `moe.py`）。
  - 证据：MoE 块单独回放（输入用 golden 的 `mlp_norm.out`）。
    - v5p `L1.ops.kernel.moe` R 0.94 过（`phase2-l1-ops-own-on`）。
    - v7x 同项 R 1.27，去掉近似平局处选错专家的 1 个 token 后 0.95（[`v7x-l1-moe-routeflip.json`](alignments/02-v7x-acceptance/results/v7x-l1-moe-routeflip.json)）。
- 注 3（辅助损失 / z-loss）：
  - 间接证据：第 1 步优化器 FP64 重放用的是 TPU 自己反推的梯度。
  - 这与 L4 对 [`rcc-a`](../../docs/glossary.md#rcc-a)（最终 golden 的第 1 份，验收的比较基线）全部在噪声内（[`ALIGNMENT-REPORT.md`](ALIGNMENT-REPORT.md) 第 3 节）相容。
- 注 4（RoPE）：
  - 注意力核心单独回放（RoPE + 缩放 + 注意力，输入用 golden 的 q / k / v）。
  - `L1.ops.kernel.attn_core` R 1.009，v5p、v7x 都过（`phase2-l1-ops-own-on`、`phase2-v7x-l1-ops-own-on`）。
  - 基数或配对方式不同是 O(1) 的差。
- 注 5（归一化与 eps）：
  - 取值两边都是 1e-5。
  - L1：MLP 前归一化 R 1.16、q / k 归一化 R 1.07，v5p、v7x 都过（`L1.ops.op.mlp_norm`、`L1.ops.op.qk_norm`，同上两个文件）。
  - 推断：eps 本身差一个数量级时，影响量级是 Δeps / mean(x²)，L1 不一定分得出来，取值以配置为准。
- 注 6（注意力偏置）：
  - GPU：meta 未记这个字段。
    - GPU 每个 rank 的参数清单 185 个，与 HF 按 name_map 拼出的逐个相同、没有多出的张量。
    - HF 里没有注意力偏置张量。
  - TPU：MaxText 抽象模型的 30 个变量与 name_map 推出的逐个一致，没有偏置变量。

## 数值精度

代号：`rc`（GPU 生产 golden）、`rcc`（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加）、`rcc-a`（最终 golden 的第 1 份，验收的比较基线）。

| 类别 | 参数 / 开关 | 为什么关键 | GPU（Megatron）怎么配 | TPU（MaxText）怎么配 | 对齐方式 | 状态 | 证据 |
|---|---|---|---|---|---|---|---|
| 精度 | 计算 dtype | 激活与 GEMM 输入 | `bf16: true`（GPU meta）；`dtypes.compute: bf16` | `dtype: "bfloat16"`（TPU yml） | 相同 | 已对齐（L1、L2 前向） | 两边 meta 记的计算 dtype 都是 BF16；L1 单算子 9 项、L2 逐层前向 4 层的 R 比都 ≤ 1.2（见下面「收缩维被 FSDP 切开」行），算术精度和 GPU 同档 |
| 精度 | 参数与主权重 | 更新精度；小扰动在 BF16 里看不见（00 README 第 11 条） | BF16 参数 + FP32 主权重：`main_params_dtype: torch.float32`（GPU meta） | 只存 FP32 主权重，计算时转 BF16：`weight_dtype: "float32"`（TPU yml） | 等价 | 已对齐（L0、L3） | 载入后的主权重对 golden 逐位；更新后不是 BF16 冒充的 FP32（见注 7） |
| 精度 | 优化器状态 | Adam 两个动量 | `exp_avg_dtype`、`exp_avg_sq_dtype: torch.float32`（GPU meta） | `mu_dtype: "float32"`（TPU yml）；二阶矩 `nu` 跟参数 dtype，即 FP32（optax `scale_by_adam`） | 相同 | 已对齐（L0、L3） | 两个动量全是 FP32，更新后也不是 BF16 冒充（见注 8） |
| 精度 | 梯度 dtype 与规约 | 规约顺序与精度 | `dtypes.grad_reduce: fp32`（GPU meta） | `grad_dtype: "float32"`（TPU yml）；但专家、embedding、归一化层的梯度在 8 个设备之间按 BF16 规约（案例 01 第 6 节 4c） | dtype 配置相同；规约顺序不同（切分不同，见下）；部分梯度 BF16 规约（不可避免） | 有差异（已核对无影响） | GPU 反过来改成 BF16 规约，仍在 GPU 自己的散布里（见注 9） |
| 精度 | 路由 GEMM | 路由 logits 离 FP64 本来就远；但翻转不从这里来：喂两边相同的路由输入，翻转是 0，翻转来自上游注意力舍入造成的输入差 | `moe_router_dtype: fp32`（GPU meta），在这版 Megatron 里走 TE `general_gemm`，BF16 入、FP32 出，不是先升 FP32 再乘 | `float32_gate_logits: True`（模型 yml）；DEFAULT 档下 FP32 操作数先舍 BF16 再乘（见注 10） | 相同（操作数）；累加方式不同（不可避免） | 已对齐（L1） | TPU 对 FP64 4.8e-8–5.7e-8，GPU 3.8e-6–4.0e-6；翻转 0（见注 10） |
| 精度 | 路由打分 dtype | sigmoid 在哪个精度算 | 路由 logits FP32，sigmoid 在 FP32 上（推断，按 `moe_router_dtype: fp32`） | sigmoid 与加偏置都在 FP32 上（见注 11） | 相同（GPU 一侧推断） | 已对齐（L1） | top-8 集合对 FP64 每层一致率 ≥ 0.9995（见注 11） |
| 精度 | expert_bias dtype | 存成 BF16 后更新量会被舍入吃掉 | FP32 buffer（00 `L0.dtype.manifest` 规则 `buffer.*expert_bias`） | FP32（MaxText `MoEBiasVar`；检查点里存 FP32） | 相同 | 已对齐（检查点） | 01 `phase2-ckpt-convert` 的 `L0.ckpt.convert.dtype`（3 个 FP32） |
| 精度 | TF32 | FP32 输入的 GEMM 会被降成 TF32 | 默认开，只有 `NVIDIA_TF32_OVERRIDE=0` 能全关（见注 12） | 没有 TF32。路由反向的 GEMM 按 `matmul_precision: "default"` 算，FP32 操作数推断是单遍 BF16 | 不可避免差异（推断） | 有差异 | 00 `phase1-kernels` 的 `TF32.gemm_inventory`、`TF32.switch.matrix`；00 `phase1-tf32`（关 TF32 后参数梯度最大差是噪声上限的 4.95 倍，30 步 loss 在噪声内） |
| 精度 | TPU matmul 精度 | 决定 FP32 操作数的 dot 走几遍 BF16 | — | `matmul_precision: "default"`（TPU yml，与 base.yml 默认相同）；影响范围见注 13 | — | 已核对（路由 GEMM 前向） | 01 `phase2-l1-l2-replay` 的 `router_precision_probe`；`train_hy3.py` 把 `matmul_precision` 写进 `meta.json` |
| 精度 | 收缩维被 FSDP 切开的投影的累加精度（注意力 q / k / v、稠密 MLP 的 wi） | 投影输出多舍几次，L1 / L2 里 TPU 离 FP64 比 GPU 远两到三成的就是这两处 | TE `LayerNormLinear`：GEMM FP32 累加、输出舍一次（见注 14） | 窗口化 collective matmul 在 BF16 里累加；开关 `dense_fp32_accumulation` 改成 FP32（见注 14） | 相同（开关打开） | 已对齐（L1、L2 前向，v5p、v7x）；2026-11-05 决定打开，对齐配置 yml 从 `a238883b7f91…` 起 | 打开后 L1 单算子 9 条、L2 逐层前向 4 层全过；L4 不需要它（见注 15） |
| 精度 | 注意力 softmax | 末位差异的主要来源之一；逐层隔离回放里第 1–3 层第一个超过 BF16 末位量级的就是 `attn.core.out` | cuDNN fused attention，内部 softmax 精度未查（cuDNN 不公开内核写法；不影响判定，见证据栏） | 旧 Splash attention，softmax 在 FP32 里算（见注 16） | 不可避免差异（两种内核） | 已对齐（L1）；对 GPU 逐元素有差异 | `L1.ops.kernel.attn_core` R 1.009（v5p、v7x）；1000 步换 FP32 点积注意力也过（见注 16） |
| 精度 | logits 与 loss | 交叉熵精度 | logits BF16（golden 第 1 步 logits 张量 dtype 是 BF16）；融合交叉熵，反向写出 BF16 的 dlogits；确定性设置下关 `cross_entropy_loss_fusion`（`train_hy3.py`） | `cast_logits_to_fp32: true`、`logits_dot_in_fp32: false`（TPU yml）：logits 在 BF16 出、转 FP32 算交叉熵，反向的 dlogits 也是 BF16 | 等价（两边都是 BF16 logits、FP32 交叉熵、BF16 dlogits） | 已对齐（L4） | 第 1 步 loss 相对差 d0 2.5e-6（v5p）/ 9.7e-6（v7x）；GPU 换写法无影响（见注 17） |
| 精度 | excess precision | TPU 默认在融合内部跳过中间的 BF16 舍入；GPU 每个算子各舍一次。影响前向与重算、DP 与 FSDP 的舍入位置 | 无此机制：PyTorch / Megatron 每个算子输出舍到 BF16 | 没设，XLA 默认开（`--xla_allow_excess_precision` 默认 true）；开关与取舍见注 18 | 不可避免差异（保持默认开；要对齐的舍入点用下面两行的开关显式写） | 有差异（已核对：不进对齐配置也在噪声内） | 现行配置（开着）对 `rcc-a` L4 全部在噪声内（关掉的几臂见注 18） |
| 精度 | RMSNorm 舍入次数 | 默认写法比全程 FP32 多舍两次（归一化结果、scale 各转一次 BF16）；关 excess precision 后 25.29% 元素不是正确舍入 | TE RMSNorm：对 FP64 正确舍入 99.9988%–99.9993%（`../../docs/dev/design/0003-l1-op-replay.md` 第 5 条） | `rmsnorm_single_rounding: true`（TPU yml）；v5p 实测默认写法已经只舍一次（见注 19） | 相同 | 已对齐（L1、L2 前向） | 打开后各层 `in.grad` 对 golden 的 rel_l2 略降（见注 19） |
| 精度 | 门控激活舍入（silu·up） | 「SiLU 舍一次再乘」与「整体舍一次」27.48% 元素不同 | 路由专家：`silu` 在 BF16 里舍一次、乘积再舍一次（见注 20） | `mlp_activation_round_per_op: true`（TPU yml）：按算子舍入（见注 20） | 相同（按算子舍入） | 已对齐（L1）；L2 见证据 | 整层回放里打开后 `fc2.out`、`shared.out`、各层 `in.grad` 都变小（见注 20） |
| 精度 | 路由核 BF16 舍入 | TPU 的核是 FP32 主权重，GPU 是 BF16 参数 | BF16 权重 × BF16 激活、FP32 累加 | `routed_gate_bf16_rounding: true`（TPU yml）：核先舍 BF16 再乘，梯度直通（见注 21） | 相同 | 已对齐（L1） | `default_eq_rounded_kernel_frac` = 1（见注 21） |
| 精度 | 路由概率乘在哪 | 数学相同、舍入点不同 | 概率乘在激活上、fc2 之前：`(silu·up) * probs` 升 FP32 再转回 BF16（Megatron `experts.py` 的 `bias_act_func`；00 `L1.kernel.experts.fwd.replay` 逐位） | 乘在 fc2 输出上：`weight_sum`，`float32_weight_sum: true` 时在 FP32 里加权求和（MaxText 代码 `moe.py`） | 不同（等价变换） | 有差异 | 未改。MaxText 只有 ring-of-experts 的 TC 路由有「乘在激活上」的开关（`moe_tc_ragged_weights_on_activation`），本配置用不上 |
| 精度 | 注意力 kernel 与 base2 | Tokamax Splash 默认 `sa_use_base2_exp=True`，多一次 BF16 舍入，TPU loss 系统性偏高约 3.5e-4 | — | `use_tokamax_splash: false`（base.yml 默认，TPU yml 没改），走旧 Splash，不经过 base2。改用 Tokamax 时同时设 `sa_use_base2_exp: False` | — | 不适用（当前配置） | `meta.json` 的 `maxtext_config.use_tokamax_splash` |
| 精度 | 输出头 dgrad（dh = dlogits · W，收缩长度 = 词表 120832） | 「所有 logits 一起升降」这个方向的梯度全靠这一次长求和（见注 22） | GPU 生产 `rc` 在张量核上一次乘完，系统地朝 0 偏；最终 golden `rcc` 改成分段 FP32 累加（见注 22） | MXU：BF16 乘、FP32 累加（`matmul_precision: default`），沿 ŵ̄ 对 FP64 −2.4e-5 / −7.4e-5（第 1 / 100 步，输出头回放） | 相同（`rcc` 与 TPU 都是 BF16 乘积在 FP32 里累加，只剩求和顺序）；对 `rc` 有差异 | 已对齐（L4，对 `rcc-a`）；对 `rc` 有差异（已解释） | 对 `rcc-a` v5p、v7x L4 四项都过，最差 r 0.85；对 `rc` 是已解释差异（见注 23） |

- 注 7（参数与主权重）：
  - 检查点是 FP32（01 `phase2-ckpt-convert` 的 dtype 清单）。
  - 载入后的主权重对 golden 逐位，50 个张量 0 个不同（`L0.main_params.loaded`，v5p `phase2-l0-state`、v7x `phase2-v7x-l0-state`）。
  - `L0.model.structure` 规则「主权重 FP32（全部 29 个参数叶子）」。
  - 更新后的主权重不是 BF16 冒充的 FP32（`L3.step1.fake_fp32.main_param`，50 个张量，`phase2-st1-prod-state`、`phase2-v7x-st1-prod-state`）。
- 注 8（优化器状态）：
  - `L0.dtype.state`：`opt_state/0/mu`、`nu` 各 58 项全是 FP32（v5p `phase2-l0-state`）。
  - 第 1 步更新后两个动量不是 BF16 冒充的 FP32（`L3.step1.fake_fp32.optim_state`，100 个张量，v5p、v7x）。
- 注 9（梯度 dtype 与规约）：
  - GPU 反过来改成 BF16 规约（`g-bf16grad`，1000 步）：param_norm 对 g1k-a（第一阶段 GPU golden 的第 1 份）+6.6e-4。
  - 同设置的 `g-ctl`（扩充噪声带用的 GPU 对照运行）+8.2e-4，在 GPU 自己的散布里（[`goldens.md`](goldens.md)「梯度 BF16 规约」行）。
  - L4 对 [`rcc-a`](../../docs/glossary.md#rcc-a) 全部在噪声内。
- 注 10（路由 GEMM）：
  - TPU 配法：`float32_gate_logits: True`（模型 yml），因此 GateLogit 的 dtype 是 FP32。
    - 输入和核都转 FP32，再按 `matmul_precision: default` 做 `dot_general`（MaxText 代码 `moe.py` 的 `GateLogit`）。
  - **v5p 实测：DEFAULT 档下 FP32 操作数就是两边先舍 BF16、单遍乘、FP32 累加**：
    - 用不能用 BF16 表示的核，DEFAULT 的结果与「两个操作数先舍 BF16 再点积」逐位相同（3 层，各 4096 × 192 个元素，100%）。
    - 对 FP64 的 rel_l2 1.5e-4–3.1e-4，HIGHEST 是 8.0e-8–8.3e-8。
    - 原来的推断成立，与另一次独立测量（DEFAULT 2.34e-3、HIGHEST 9.86e-8）一致。
  - 证据：01 `phase2-l1-l2-replay`，输入用 golden 的 `mlp_norm.out`。
    - TPU logits 对 GPU 的 rel_l2 3.8e-6–4.0e-6。
    - TPU 对 FP64 4.8e-8–5.7e-8，GPU 对 FP64 3.8e-6–4.0e-6（GPU 张量核的累加截断，`docs/dev/design/0003-l1-op-replay.md` 第 4 条）。
    - 按 golden 偏置重选 top-8，翻转 0。
  - GPU 写法：00 `phase1-l1` 的 `L1.gemm.router.replay`。
- 注 11（路由打分 dtype）：
  - TPU：GateLogit 输出 FP32，sigmoid 与加偏置都在 FP32 上（MaxText 代码 `moe.py` 的 `GateLogit.__call__`）。
    - 专家输出加权 `float32_weight_sum: true`（base.yml 默认）。
  - 证据：喂 golden 的输入，TPU top-8 集合对 FP64 每层一致率 ≥ 0.9995（`L1.router.topk.fp64.layer1..3`，v5p、v7x）。
  - sigmoid 若在 BF16 里算，近似平局处的翻转会多出一截（推断，没有单独做 BF16 臂）。
- 注 12（TF32，GPU 一侧）：
  - 容器设了 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，默认开。
  - 训练里输入是 FP32 的 GEMM 只有路由反向：rank 0 第 1 步共 12 次（dgrad、wgrad 各 6 次）。
  - 只有 `NVIDIA_TF32_OVERRIDE=0` 能全关（torch 开关管不到 TE）。
  - 第一阶段 golden `g1k`、GPU 生产 `rc` 与最终 golden `rcc` 都是 TF32 开。
  - `rcc` 只在 `dgrad-chunk` 那一次 FP32 乘法里临时关（[`goldens.md`](goldens.md)「TF32」行）。
- 注 13（TPU matmul 精度）：
  - `matmul_precision: "default"` 传给所有 `DenseGeneral`（`linears.py` 的 `_compute_dot_general_nnx`）和 GateLogit。
  - 没设 `jax_default_matmul_precision`（`train_hy3.py` 与 MaxText 源码里都没有）。
  - BF16 操作数的 dot 不受这一档影响（推断）。
  - 受影响的是 FP32 操作数，即路由 GEMM 前向与反向。
    - v5p 上 DEFAULT 对 FP32 操作数是单遍 BF16（实测，见「路由 GEMM」行）。
  - megablox 分组 GEMM 不看这一档：
    - 全 FP32 配置下路由专家输出对 FP64 仍是 3.2e-3，和全部舍 BF16 一样。
    - 推断：Pallas 核里的 `dot_general` 没给 precision，FP32 操作数按单遍 BF16 乘（案例 01「TPU FP32 参照本身」行）。
    - 改走 XLA `ragged_dot` 并设 `jax_default_matmul_precision: highest` 才是真 FP32。
- 注 14（收缩维被 FSDP 切开的投影：两边怎么配）：
  - GPU：TE `LayerNormLinear`：归一化输出舍一次到 BF16，GEMM FP32 累加、输出舍一次。
    - golden 的 qkv.out 与这样的主机复现 99.7% 逐位（第 0、1 层各 1024 个 token）。
  - TPU：XLA 把这类 dot 编成窗口化的 collective matmul（v5p T1 配置优化后 HLO 里 41 个 `windowed_dot_general` 循环）。
    - 每一轮的分片部分积以 BF16 输出、在 BF16 累加器里相加（q 投影的循环累加器是 `bf16[2,4096,64,128]`）。
  - 开关 `dense_fp32_accumulation`（默认关）：DenseGeneral 用 `preferred_element_type=float32`，最后舍一次。
  - 另一种办法是 `LIBTPU_INIT_ARGS=--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000` 关掉窗口化。
    - 它前向、反向和输出层都关，改性能；开关只管前向的注意力与 MLP 投影。
- 注 15（收缩维被 FSDP 切开的投影：证据）：
  - 01 逐层隔离回放，qkv.out 对 GPU rel_L2：
    - [`r-wc-base`](../../docs/glossary.md#run-prefixes)（不开）2.36e-3；
    - `r-fa32`（开）1.8e-4；
    - `r-wc-off-r4`（关窗口化）1.7e-4。
  - v5p FP64 R 比：
    - 单算子 qkv 1.235 → 1.000、第 0 层 fc1 1.294 → 1.117，9 条全过（`phase2-l1-ops-own` 与 `-own-on`）；
    - 逐层前向 4 层 R 1.06 / 1.01 / 1.01 / 1.09，全过（`phase2-l2-replay-rratio` 与 `-on`）。
  - v7x 同样打开后：
    - 单算子 qkv 1.570 → 1.000、第 0 层 fc1 1.582 → 1.118（`phase2-v7x-l1-ops-own` 与 `-own-on`）；
    - 逐层前向 1.57 / 1.53 / 1.56 / 1.55 → 1.06 / 1.01 / 1.01 / 1.02（`phase2-v7x-l2-replay-rratio` 与 `-on`）。
  - 关窗口化那一臂（`phase2-l1-ops-wcoff`）的 MLP 前归一化 R 1.64。
    - 残差加直接吃了没舍的投影输出，比 GPU 少舍一次，所以用开关不用 flag。
  - L4 不需要它：打开之前的配置对 `rcc-a` 也全过，最差 r v5p 0.79、v7x 0.78（[`case01-tpu-arms-vs-rcc.json`](alignments/01-v5p-acceptance/results/case01-tpu-arms-vs-rcc.json)）。
  - 排查期的旁证：[`t-wcoff-1000`](../../docs/glossary.md#run-prefixes)（生产配置 + 关窗口化，1000 步）。
    - 第 1000 步 param_norm 对当时的参照 `rcx`（生产 golden，输出头 dgrad 换 FP32 一次累加）+8.50e-3，基线 +8.74e-3，几乎不动（案例 01）。
- 注 16（注意力 softmax）：
  - TPU 配法：`attention: "flash"`（TPU yml，旧 splash attention）。
    - `float32_logits: false`（base.yml，只管 dot-product 路径）。
    - 旧 Splash 的 softmax 在 FP32 里算。
  - 判定：注意力核心单独回放对 FP64 不比 GPU 远，`L1.ops.kernel.attn_core` R 1.009（v5p、v7x）。
  - TPU 对 GPU 逐元素的差（不判，只描述）：01 `phase2-l1-l2-replay`，输入用 golden 的层输入。
    - `attn.core.out` 对 GPU 的 rel_l2 第 0 层 1.6e-3、第 1–3 层 4.3e-3–6.2e-3。
    - 下游 `mlp_norm.out` 3.4e-3–3.9e-3。
    - 路由每层翻转 3.5%–5.4%。
    - dot-product + `float32_logits` 的诊断臂见同一文件。
  - 1000 步：TPU 换成 FP32 点积注意力（[`t-v7x-attn32-1000`](../../docs/glossary.md#run-prefixes)）对 `rcc-a` 四项都过，最差 r 0.78（[`goldens.md`](goldens.md)「注意力 FP32 点积」行）。
- 注 17（logits 与 loss）：
  - 第 1 步 loss 相对差 d0 2.5e-6（v5p）/ 9.7e-6（v7x）（`phase2-t-loss30`、`phase2-v7x-t-loss30` 的 `L4.loss`，对 g1k）。
  - GPU 交叉熵换成 MaxText 写法（`rr_arms.py` 的 `ce-mt`）、关融合（`g-cefuse-off`）都无影响（[`goldens.md`](goldens.md)「试过、没有采用的开关」）。
  - dlogits 舍 BF16 两边共有，见「输出头 dgrad」行。
- 注 18（excess precision）：
  - TPU 一侧：
    - `run.sh` 透传调用方的 `XLA_FLAGS`，关掉用 `XLA_FLAGS=--xla_allow_excess_precision=false`。
    - 关掉为的是舍入位置确定，不是更准（有些算子默认更接近正确舍入）。
    - v5p 上默认开时，RMSNorm 的中间舍入、门控激活 `silu` 后的舍入都被省掉（见下面两行）。
    - 需要和 GPU 一样各舍一次的地方用 `reduce_precision` 显式写，不靠这个开关。
  - 证据：
    - 实际取值记在 `meta.json` 的 `determinism.XLA_FLAGS`。
    - 关掉的两臂见 01 `phase2-precision-fixes.md`：`r-l2-noexcess`（TPU 逐层隔离回放关 excess precision，L1 + L2）与 `t-ga2-noexcess-30`（TPU GA 2 关 excess precision，30 步）。
    - 1000 步关掉的 `t-noexc-1000`：param_norm 漂移不变、loss 偏移变小一半（案例 01「TPU excess precision」行，排查期对 `rc` 的读数）。
    - 现行配置（开着）对 `rcc-a` L4 全部在噪声内。
- 注 19（RMSNorm 舍入次数）：
  - TPU 开关：FP32 里算完只舍一次，scale 先 `reduce_precision` 到 BF16。
  - v5p 实测默认写法已经只舍一次：XLA 把中间那次省掉了。
    - 单独编译时对 FP64 正确舍入 99.999%、与 golden 逐位 99.9987%（3 个 MoE 层的 MLP 前归一化）。
    - 整层回放里开关打开前向逐位不变，只改反向（显式 FP32 路径的梯度不经过 BF16 转换）。
  - 证据：01 `phase2-l1-l2-replay`，开关打开后各层 `in.grad` 对 golden 的 rel_l2 从 2.09e-3 / 4.82e-3 / 5.78e-3 / 6.92e-3 降到 2.06e-3 / 4.79e-3 / 5.73e-3 / 6.89e-3。
- 注 20（门控激活舍入）：
  - GPU：路由专家是非融合 GLU，`silu` 在 BF16 里舍一次、乘积再舍一次（00 `phase1-kernels` 的 `L1.kernel.experts.fwd.replay` 逐位）。
    - 稠密层与共享专家同一份配置，推断相同。
  - TPU 开关：`silu` 输出 `reduce_precision` 到 BF16，再乘、再舍一次。
  - v5p 实测默认写法编出来是「FP32 算完舍一次」：
    - 第 0 层 golden `fc1.out` 喂进去，对 FP64 正确舍入 99.98%（`activations_in_float32` 99.99%）。
    - 与 GPU 写法只有 73.7% 逐位（rel_l2 2.8e-3）；按算子舍入与 GPU 写法 100% 逐位。
    - `activations_in_float32`（MaxText 修改后稀疏路径也读）在 v5p 上和默认一样只舍一次，没用。
  - 证据：01 `phase2-l1-l2-replay`，整层回放里打开后：
    - `fc2.out` 4.17e-3 → 4.08e-3；
    - `shared.out` 6.67e-3 / 5.22e-3 / 5.36e-3 → 6.51e-3 / 4.93e-3 / 5.01e-3；
    - 各层 `in.grad` 也都变小。
- 注 21（路由核 BF16 舍入）：
  - TPU 开关：核先 `reduce_precision` 到 BF16 再乘，梯度直通。
    - #94 的 `astype` 写法会把梯度也舍成 BF16，#97 修掉。
  - v5p 上 DEFAULT 档本来就先舍 BF16，开不开逐位相同。
    - 不能用 BF16 表示的核也一样。
    - 回归试跑 20 步开 / 关整条训练逐位相同，编译后只多 2 个 reduce-precision。
    - 开着是为了不依赖后端，不改 loss。
  - 证据：01 `phase2-l1-l2-replay` 的 `router_precision_probe`（`default_eq_rounded_kernel_frac` = 1，3 层）。
  - 另一次独立测量在 v7x 上测得 DEFAULT 下 FP32 matmul 相对误差 2.34e-3、`HIGHEST` 9.86e-8。
- 注 22（输出头 dgrad：为什么关键、GPU 怎么配）：
  - 为什么关键：这一次长求和有系统偏差，就是每步都同向偏的一股力，几百步累积成 logits 整体水平、输出层冷门词行的范数和 loss 的偏移（案例 01）。
  - GPU 生产 [`rc`](../../docs/glossary.md#rc)：BF16 dlogits × BF16 参数在张量核上一次乘完，结果系统地朝 0 偏。
    - 第 1、100 步沿 ŵ̄ −6.5e-4 / −6.1e-4，正负输入对称、随 K 增大。
    - 原因是张量核累加截断，合成输入复现见 [`gemm_accum_bias.py`](analysis/gemm_accum_bias.py)。
  - 最终 golden `rcc`：同一份 BF16 dlogits 和 BF16 参数升到 FP32。
    - 词表按 1024 一段、关 TF32 在 CUDA 核上做 FP32 乘法、段间 FP32 累加，结果舍回 BF16（[`rr_arms.py`](frameworks/megatron/rr_arms.py) 的 `dgrad-chunk`）。
    - `rcc-a` 训练自检对 FP64 rel_L2 6.9e-7 – 7.7e-7。
- 注 23（输出头 dgrad：证据）：
  - 对最终 golden `rcc-a`（噪声取 `rcc-a..j` 10 份重跑两两配对的 45 对）：
    - v5p、v7x 当前配置 L4 的 loss 全程、loss 301–1000 步、grad_norm、param_norm 都过。
    - 四项里最差的比值 r 0.85（`phase2-acceptance-rcc-on`、`phase2-v7x-acceptance-rcc`；[`ALIGNMENT-REPORT.md`](ALIGNMENT-REPORT.md) 第 3 节）。
  - 对 GPU 生产 `rc`：写成已解释差异（ALIGNMENT-REPORT 偏差台账 #1、#2）。
    - TPU 第 1000 步 param_norm 比 `rc` 五份均值多 +2.09e-2。
    - 301–1000 步 loss 相对差 −1.37e-4（v5p）/ 约 −1.08e-4（v7x）。
    - v7x 那个数是推断：v7x 对 `rcc-a` 的 +6.32e-5 加上 `rcc-a` 对 `rc` 均值的 −1.71e-4；v7x 当前配置的逐步序列没有入库，没有直接算。
  - 为什么分段：关 TF32 的 cuBLAS FP32 一次累加整个词表（`dgrad-exact` 的写法）在 B200 上也有偏。
    - 沿 ŵ̄ 第 1 步 −1.8e-6、θ₂₉₉（第 300 步开始时的状态）−2.6e-4（[`case01-cublas-fp32-longk.json`](alignments/01-v5p-acceptance/results/case01-cublas-fp32-longk.json)）。
    - 分段累加才精确。
  - 历史（已取消，只作排查记录）：
    - 2026-11-05 曾以 `rcx-a..e`（生产 golden，输出头 dgrad 换 FP32 一次累加，即 `dgrad-exact`）为第一层；
    - 当晚改以精确轨迹 `rce-a` 为第一层、噪声取 `rcx-a..e` 5 份两两 10 对；
    - 查到上面的 cuBLAS 偏差后，11-06 定为 `rcc`。
    - 当时的读数与「输出头 dgrad 约占六成」的拆分已被 `rc`、`rcc`、`rcec`（真精确轨迹，精确轨迹加输出头 dgrad 分段累加）三级阶梯取代（[`goldens.md`](goldens.md)「从 g1k 到 rcc」，案例 01 §6）。

## 优化器与调度

| 类别 | 参数 / 开关 | 为什么关键 | GPU（Megatron）怎么配 | TPU（MaxText）怎么配 | 对齐方式 | 状态 | 证据 |
|---|---|---|---|---|---|---|---|
| 优化器 | 算法 | 解耦权重衰减 | `optimizer: adam`（GPU meta；Megatron 的 adam 是解耦权重衰减的 AdamW） | `opt_type: "adamw"`（TPU yml） | 等价 | 已对齐（L1，两侧） | 两侧都做了 FP64 重放，在门槛内（见注 24） |
| 优化器 | β1 / β2 / ε | 更新量 | `adam_beta1: 0.9`、`adam_beta2: 0.95`、`adam_eps: 1e-08`（GPU meta；ε 加在 sqrt 外面） | `adam_b1: 0.9`、`adam_b2: 0.95`、`adam_eps: 1.0e-8`、`adam_eps_root: 0.0`（TPU yml） | 相同 | 已对齐（L1） | 上一行的两组重放用的就是这几个值（见注 25） |
| 优化器 | 实际生效的权重衰减 | 只读配置会对错超参 | 配置 `weight_decay: 0.1`，但调度器 `start_weight_decay` / `end_weight_decay: 0.033` 覆盖了它（GPU meta 的 `scheduler_config`；00 README 第 10 条） | `adam_weight_decay: 0.033`（TPU yml） | 相同（按实际值） | 已对齐（配置） | 01 `phase2-t-loss30` 的 `L0.config.diff.dropout`（TPU `weight_decay` = GPU `start_weight_decay`） |
| 优化器 | 哪些参数衰减 | 一维参数不衰减 | Megatron 默认对一维参数（各 RMSNorm 的 scale）不衰减 | `adamw_mask: ["norm/scale"]`（TPU yml）：扫描层把一维 scale 堆成二维，只能按名字排除 | 等价 | 已对齐（L1，两侧） | 两侧重放都按「一维 wd = 0、其余 0.033」吻合（见注 26） |
| 优化器 | 梯度裁剪 | 第 1 步就裁（裁剪系数 0.057） | `clip_grad: 1.0`（GPU meta） | `gradient_clipping_threshold: 1.0`（TPU yml） | 相同 | 已对齐（L1） | 第 1 步优化器重放按 `max_norm / (grad_norm + 1e-6)` 算出的裁剪系数 0.0570（v5p、v7x），带着它重放的更新量在门槛内（`phase2-st1-prod-state`、`phase2-v7x-st1-prod-state` 的 `clip_coeff`） |
| 优化器 | 学习率与调度 | 常数学习率 | `lr = min_lr = 1e-5`、`lr_decay_style: constant`、`lr_warmup_iters: 0`（GPU meta） | `learning_rate: 1.0e-5`、`lr_schedule_type: "wsd"`、`warmup_steps_fraction: 0.0`、`wsd_decay_steps_fraction: 0.0`、`learning_rate_final_fraction: 1.0`（TPU yml） | 等价 | 已对齐 | 01 `phase2-t-loss30` 的 `L4.lr`（30 步逐位） |
| 优化器 | 跳步 | 尖峰时是否跳过更新 | 不跳（golden `skipped_iter` 全 0） | `skip_step_on_spikes: false`（TPU yml） | 相同 | 已对齐（配置） | 两边都关；1000 步健康信号都过（见注 27） |

- 注 24（优化器算法）：
  - TPU 侧：生产配置第 1 步状态读，按被测自己 meta 里记的 β1 / β2 / ε / 权重衰减 / 裁剪阈值重放一步。
    - 用和 GPU 侧同一个 FP64 公式（ε 加在根号外，[`janus/replay.py`](../../janus/replay.py) 的 `adamw_step`）。
    - 比更新量和两个动量：48 个参数 × 3 = 144 个张量，rel_l2 最大 2.13e-3（v5p）/ 2.12e-3（v7x），门槛 5e-3，cosine 1.0（`L1.optim.adamw.replay`，`phase2-st1-prod-state`、`phase2-v7x-st1-prod-state`）。
    - 连续 8 步重放（第 0 层注意力、各层路由权重、全部归一化权重，66 个张量）rel_l2 最大 4.80e-3 / 4.79e-3（`L1.optim.adamw.8steps`，`phase2-st8-opt8`、`phase2-v7x-st8-opt8`）。
    - 梯度从一阶动量反推，所以一阶动量按构造吻合，有信息量的是更新量与二阶动量。
  - GPU 侧：00 `L1.optim.adamw.replay`（FP64 重放，141 个张量在门槛内）。
- 注 25（β1 / β2 / ε）：
  - 上一行的两组重放用的就是 TPU meta 里记的这几个值。
  - 第 1 步偏差修正把 β 消掉了，更新量只检验公式。
  - β2 由二阶动量 v₁ = (1 − β2)·ĝ² 检验。
  - β1、β2 一起由连续 8 步重放检验（第 2–8 步的更新量与 v 都依赖它们）。
  - ε 只在 √v̂ 接近 1e-8 的元素上起作用，重放对它的取值不敏感（推断），取值以配置为准。
- 注 26（哪些参数衰减）：
  - TPU 侧：上面两组重放按「一维参数 wd = 0、其余 0.033」算（`janus/replay.py` 的 `no_weight_decay`，按 Megatron 布局的形状判）。
    - 全部在门槛内，8 步那组含全部归一化权重。
  - 推断：每元素的衰减项与 Adam 更新量之比约为 wd·\|p\|。
    - 归一化 scale 若被衰减，这一比值是 3% 量级（\|p\| 取 1），会超过 5e-3 的门槛。
    - 二维矩阵 \|p\| 小，衰减项落在门槛以下，0.033 这个取值靠上面「实际生效的权重衰减」行的配置检查。
  - GPU 侧：00 `L1.optim.adamw.replay`（norm 按 wd = 0、二维按 0.033 吻合）。
- 注 27（跳步）：
  - 两边都关。
  - TPU 侧 `metrics.jsonl` 的 `skipped_iter` 是 `train_hy3.py` 写死的 0，不能当证据。
  - 间接证据：1000 步健康信号（非有限值、突跳）v5p、v7x 都过（`phase2-acc-health-rcc`、`phase2-v7x-acc-health-rcc` 的 `L4.health`）。

## 数据

代号：`t-ga2-30`（TPU 改成 GA 2 的 30 步运行）。

| 类别 | 参数 / 开关 | 为什么关键 | GPU（Megatron）怎么配 | TPU（MaxText）怎么配 | 对齐方式 | 状态 | 证据 |
|---|---|---|---|---|---|---|---|
| 数据 | 序列长度 | — | `--seq 4096`（GPU meta `config.args`） | `max_target_length: 4096`（TPU yml） | 相同 | 已对齐 | 01 `phase2-t-loss30` 的 `L0.data.fingerprint` |
| 数据 | 全局批 / microbatch / 梯度累积 | 累加分组变了，舍入点就变（00 README 第四节：梯度累积切分让 embedding 梯度差到噪声上限的 31 倍） | `gbs 16`、`mbs 1`，8 卡 DP，每卡 2 个 microbatch 分两次前反向再累加 | `per_device_batch_size: 1`、`gradient_accumulation_steps: 2`（TPU yml，janus 67208dd 起）；分法与之前的排法见注 28 | 相同（分组方式）；累加顺序与精度未逐位核对 | 已对齐（对齐配置的一部分） | 01 [`t-ga2-30`](../../docs/glossary.md#run-prefixes)：输入指纹逐个 microbatch 逐位；只改这一项 loss 没有变好（见注 28） |
| 数据 | 数据来源 | 两边吃同一批 token | Megatron `GPTDataset` 读 `fw_hy3_text_document`（`prep_data.sh`） | 不用 MaxText 数据管线：`dataset_type: "synthetic"`，由 `train_hy3.py` 按 golden 的 `data/batches` 原样回放（`replay.py`） | 相同（回放） | 已对齐 | 01 `phase2-t-loss30` 的 `L0.data.fingerprint`（1950 个单元逐位） |
| 数据 | EOD / pad id | 文档边界、pad 处理 | EOD `120026`、pad `120002`（HF）；`GPTDataset` 把 tokens / labels 里的 pad id 换成 0、label 为 pad 的位置 loss_mask 置 0 | 回放时与 GPU 相同；原生加载器（Grain `mmap_npy`）原样保留 pad id | 回放：相同；原生加载器：有差异 | 回放：已对齐；原生加载器：有差异 | 01 `phase2-native-loader` 的 `L0.native.pad_id`、`L0.native.fields` |
| 数据 | 注意力 / 位置 / loss 掩码重置 | 是否按文档切分注意力 | `reset_attention_mask = reset_position_ids = false`、`eod_mask_loss = false`（`data/README.md`） | 回放：整条序列一个因果段；原生加载器要设 `reset_attention_mask: false`（MaxText 默认 true） | 相同 | 已对齐（原生加载器） | 01 `phase2-native-loader` 的 `L0.native.eod_cases` |
| 数据 | 样本 → 设备的分配 | 原生加载器下谁拿哪一行 | Megatron 每个 microbatch 取连续 dp × mbs 行 | 原生加载器按 `reshape(gbs//ga, ga)` 交错取行；`janus/data_layout.py` 给出换序 | 等价（换序后） | 有差异（原样）／已对齐（换序后） | 01 `phase2-native-loader` 的 `L0.native.device_map.native`、`L0.native.hosts` |
| 数据 | 种子 | — | `seed 1234`（GPU meta） | `init_weights_seed: 1234`、`data_shuffle_seed: 1234`（TPU yml）；两个种子都不影响数值（见注 29） | 相同 | 不适用（种子不进数值） | 生产 HLO 里随机数算子 0 个（`regression/README.md` R20 ②） |

- 注 28（全局批 / microbatch / 梯度累积）：
  - 现行分法：设备 d 第 m 个累积步拿 GPU rank d 的 micro m，和 GPU 一样分两次前反向再累加。
  - 之前的运行是 `per_device_batch_size: 2` + GA 1，一次处理两行（各运行 `meta.json` 的 `maxtext_config`）：
    - [`t-loss30`](../../docs/glossary.md#run-prefixes)（TPU 30 步冒烟）、`t-a`（TPU 第 1 步全量抓数的 1000 步运行）、`t-nohooks`（不加抓数钩子的对照运行）。
  - 证据：01 `t-ga2-30`，输入指纹逐个 microbatch 逐位。
    - 30 步 loss 对 g1k-a（第一阶段 GPU golden 的第 1 份）的 MARE 2.07e-5，旧排法 `t-loss30` 1.89e-5，只改这一项没有变好（01 `phase2-precision-fixes.md`）。
- 注 29（种子）：
  - 权重从检查点载入、数据回放，两个种子都不影响数值。
  - 因此 `unsafe_rbg` 与 `--xla_tpu_spmd_rng_bit_generator_unsafe` 也不影响（它们只在随机初始化、dropout 时起作用）。

## 并行与确定性

代号：`g1k`（第一阶段 GPU golden）、`rc`（GPU 生产 golden）、`rcc`（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加）、`rcc-a`（最终 golden 的第 1 份，验收的比较基线）、`t-loss30`（TPU 30 步冒烟）。

| 类别 | 参数 / 开关 | 为什么关键 | GPU（Megatron）怎么配 | TPU（MaxText）怎么配 | 对齐方式 | 状态 | 证据 |
|---|---|---|---|---|---|---|---|
| 并行 | 切分方式 | 规约顺序；EP 度会改数值（00 README 第四节） | DP 8 + EP 8，TP / PP / CP 1（GPU meta `parallelism`），`moe_token_dispatcher_type: alltoall` | v5p：FSDP 8，其余 1（TPU yml 的 `ici_fsdp_parallelism: -1`；[`t-loss30`](../../docs/glossary.md#run-prefixes) 的 `meta.json`） | 不可避免差异 | 有差异（对 `rcc-a` 两种切分都在噪声内） | 只换 EP 度反向就不逐位；对 `rcc-a` 两种切分都在噪声内（见注 30） |
| 并行 | collective matmul | FSDP 下 all-gather 与 matmul 重叠会改变 matmul 结果：单个线性层 37.98% 元素与不重叠时不同 | — | 按默认阈值开（`LIBTPU_INIT_ARGS` 默认空）；不能当扰动臂（见注 31） | 相同（开关打开后，见「收缩维被 FSDP 切开的投影」行） | 已对齐（L1、L2 前向） | 见「收缩维被 FSDP 切开的投影」行；关窗口化试过不过（见注 31） |
| 并行 | ZeRO-1 + 梯度累积 | `shard_optimizer_over_data=True` 且有梯度累积时，参数在累积前转 BF16、累加器也是 BF16 | — | `shard_optimizer_over_data: false`（base.yml 默认）：TPU yml 有 `gradient_accumulation_steps: 2`，累加器随参数是 FP32（读 `gradient_accumulation.py`） | — | 不适用（当前配置） | 打开 ZeRO-1 前要先改 `gradient_accumulation.py` 的 ZeRO-1 分支 |
| 并行 | 分布式优化器 | 「分布式优化器开 + 默认设置」这组对其余三组有持续偏移（00 README 第 6 条） | 全部 golden 关（`use_distributed_optimizer: false`）：`g1k-*` 由 `--full` 强制，最终 golden `rcc-*` 带 `--no-dist-opt`；全量抓数要求关 | 无对应开关；FSDP 下优化器状态按切分存，AdamW 逐元素更新，切分不改公式（推断） | — | 不适用（TPU 无此开关） | TPU 的逐元素更新由优化器 FP64 重放核对（「优化器 · 算法」行）；GPU 侧的偏移见 00 `figures/offsets.json` |
| 确定性 | 确定性开关 | 噪声参照只能用同一种设置 | 默认设置（不设三个确定性变量），重跑自然散开，这就是噪声来源（见注 32） | 无对应开关；同程序同输入重跑逐位相同 | 不可避免差异（GPU 默认设置不确定、TPU 确定） | 已核对 | TPU 双跑逐位，散布靠扰动运行来造（见注 32） |
| 重算 | 激活重算 | 首次前向与重算不逐位（00 README 第四节）；生产训练都开重算 | `rcc-*`、`rc-*` 每层整层重算；`g1k-*` 不开（见注 33） | `remat_policy: "custom"`，存层输入、各投影等，其余重算（见注 33） | 不可避免差异：两边存的东西不同（TPU 存得多）；路由选中这一项相同（见下一行） | 有差异 | 重算不是案例 01 偏移的来源（见注 33） |
| 重算 | 路由是否在重算里重选 | 不回放时 GPU 重算会重跑 top-k，专家输出与首次前向不逐位（见注 34） | `rcc-*`、`rc-*` 回放 top-k 专家下标；`g1k-*` 不重算，不涉及（见注 34） | `moe_routing_maps: "device"`：top-k 下标与排序置换跨重算边界保存（见注 34） | 相同（只存下标、权重重算；TPU 权重一侧是推断） | 已对齐（GPU 侧）；TPU 侧只读了代码 | GPU 57 / 57 个张量逐位；TPU 没有在编译后的程序里核实（见注 34） |
| 专家执行 | 丢 token 与分组 GEMM | 丢 token 会改数值 | dropless（Megatron 默认） | `capacity_factor: -1.0`、`megablox: true`、`sparse_matmul: true`（TPU yml） | 相同（不丢 token） | 已对齐（L1） | MoE 块单独回放对 FP64 不比 GPU 远（v5p `L1.ops.kernel.moe` R 0.94；v7x 见「top-k 权重归一与缩放」行）。推断：被丢掉的 token 那一份专家输出是 O(1) 的差，会让这一项不过 |

- 注 30（切分方式）：
  - 00 `phase1-parallel`、`phase1-parallel-ext`：只换 EP 度，反向就不逐位。
  - 并行方式对 L4 的影响（2026-11-05）：TPU 生产配置只改成和 GPU 一样的 EP = 8（`ici_fsdp_parallelism=1`、`ici_expert_parallelism=8`）。
    - 1000 步对精确轨迹的 param_norm 带符号均值：v5p `t-ep8-1000` 比当前 FSDP 配置少 18%（第 1000 步 +2.35e-3 降到 +1.86e-3），v7x `t-v7x-ep8-1000` 反而多 17%（3.74e-7 对 3.19e-7）；两者都在 TPU 执行噪声上限 9e-8 以内。
    - 仍是约 3 倍噪声，不过。
    - 这是排查期对 `rce-a`（精确轨迹，BF16 权重、全程 FP32 计算）的读数，`rce` 现在只作参考；读数见案例 01 第 0 节「并行方式」行。
    - 结论：切分方式只解释 TPU 剩下那一份的一小部分。
  - 对最终 golden `rcc-a`：FSDP 8（当前配置）与 EP = 8 两种切分在 v5p、v7x 上 L4 都在噪声内。
    - 目录：`phase2-acceptance-rcc-on` / `-ep8`、`phase2-v7x-acceptance-rcc` / `-ep8`。
    - 切分方式不构成可测的差。
  - 已知限制：GPU 换 FSDP 的对照没做成（两种失败方式都记在 [goldens.md](goldens.md)「并行」行）：
    - Megatron-FSDP 载入检查点时报 `MegatronFSDP` 没有 `sharded_state_dict`（Bridge 要求 FSDP 用 `fsdp_dtensor` 格式的检查点，现有的是 `torch_dist`）；
    - torch FSDP2 在专家权重上报 `Invalid sharding pattern`（EP 与 FSDP 的切分叠在一起）。
    - TPU 一侧已说明切分方式只占一小部分，这项降为确认性实验（案例 01 第 0 节「并行方式」行）。
- 注 31（collective matmul）：
  - 它改的是舍入（分片部分积在 BF16 里累加），不是只改执行，所以不能当扰动臂。
  - 对齐配置用 `dense_fp32_accumulation` 把部分和留在 FP32，不关窗口化。
  - 关窗口化（`--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000`）试过，MLP 前归一化 R 1.64 不过（`phase2-l1-ops-wcoff`）。
  - 日常回归里关集合矩阵乘的候选臂 `cmoff` 第二关不过（all-reduce 9 → 16，`regression/README.md` 第五节）。
- 注 32（确定性开关）：
  - GPU：最终 golden `rcc-a..j` 与第一阶段 `g1k-a/b/c` 都是默认设置（不设三个确定性变量，`deterministic_mode: false`）。
    - 反向的原子加不确定，重跑自然散开，这就是噪声来源。
    - `g1k-det` 是确定性设置，只用来归因，不和默认设置混作参照。
    - 出处：00 README 第 6 条。
  - TPU：v7x 生产配置 1000 步双跑，`lm_loss`、`grad_norm`、`param_norm` 逐步逐位，输入指纹 65000 个单元 0 个不同（`phase2-v7x-rmap-repro`）。
    - v5p 同一提交双跑逐位（日常回归首个 LKG `reg-t0-lkg1-a` / `-b`，记在 `phase2-v7x-rmap-repro.yaml` 文件头）。
    - 所以 TPU 的散布要靠只改执行参数的扰动运行来造（v5p 5 份、v7x 4 份，ALIGNMENT-REPORT 第 3 节）。
- 注 33（激活重算）：
  - GPU：
    - 最终 golden `rcc-*` 与 GPU 生产 `rc-*`：`recompute_granularity: full`、`recompute_method: uniform`、`recompute_num_layers: 1`（每层单独整层重算，只存层输入；GPU meta 的 `recompute`）。
    - 第一阶段 `g1k-*`（归因参照）：不开（`recompute: false`）。
    - 这套重算设置是假设的，真实的生产重算配置还没拿到，拿到后可能重跑（03 `setup.md`）。
  - TPU：`remat_policy: "custom"`。
    - 层输入、各投影、MoE 三个 GEMM 输出、`moe_routing_maps` 存 `device`（TPU yml），其余在反向里重算。
    - `remat_policy: none` 在 NNX 解码器里仍会包成整层重算，所以不用。
  - 证据：
    - 00 `phase1-remat`；03 `remat-replay`、`remat-vs-g1k`；
    - TPU 上重算路由，每层 0.3%–3% 的专家会换；
    - 01 [`t-noremat-1000`](../../docs/glossary.md#run-prefixes)（`remat_policy: save_everything`，等于不重算）loss 偏移照样出现，重算不是案例 01 偏移的来源。
- 注 34（路由是否在重算里重选）：
  - 为什么关键：不回放时 GPU 重算会重跑 top-k。
    - 首次前向在 no_grad 里跑、top-k 不排序，重算里排序。
    - 选中集合相同时顺序也不同，top-k 权重归一化的求和顺序跟着变，重算的专家输出与首次前向不逐位（03 README）。
    - 分数贴近时集合本身也可能变。
  - GPU（`rcc-*`、`rc-*`）：`moe_enable_routing_replay: true`，由 `frameworks/megatron/routing_replay.py` 按 microbatch 切换。
    - 首次前向 RECORD，反向重算 REPLAY_BACKWARD。
    - 只回放 top-k 专家下标；路由概率在重算里由重算出的 logits 重新算（sigmoid、在回放下标上取值、归一化、乘 2.826）。
    - `g1k-*` 不重算，不涉及。
  - TPU：`moe_routing_maps: "device"`。
    - top-k 专家下标与 ragged sort 的置换跨重算边界保存，反向不重跑 top-k 与排序（MaxText 代码 `moe.py` 的 `routing_tag_fn`）。
    - 路由权重不在这组名字里，推断在反向里由重算的分数重新算。
  - 证据：
    - GPU：03 `remat-replay`（重算与首次前向 57 / 57 个张量逐位，含 3 个 MoE 层的 logits、选中专家、概率）。
    - TPU：没有在编译后的程序里核实。
      - 日常回归 R20 的规则 ⑥（`moe_routing_maps` 作为保存点）暂缓：优化后的 HLO 里路由矩阵乘被融进 fusion，还没有可靠的匹配规则（`regression/README.md` R20）。
      - 写出这条规则才能关掉这一项。
    - 间接证据：L4 对 `rcc-a` 全部在噪声内。

## 为对齐改过的代码

| 改动 | 为什么 | 文件 | 状态 |
|---|---|---|---|
| 多主机 TPU 上载入检查点前调 `jax.distributed.initialize()` | `hy3-4layers.yml` 关了存档，MaxText 只在开存档时初始化分布式，多主机就没有分布式客户端 | `frameworks/maxtext/train_hy3.py` | 已合入 |
| HF → MaxText 权重转换（按 name_map 搬运，逐张量核对） | 两边从同一份 HF 权重出发，L0 逐位 | `convert_hf_to_maxtext.py`、`verify_ckpt.py`、`name_map.yaml` | 已合入 |
| 样本 → 设备 / rank / microbatch 的换序 | 原生加载器的分配约定与 Megatron 不同 | `janus/data_layout.py`、`data/native_loader_parity.py` | 已合入 |
| 运行目录由提交机取回并上传 | Pod 用 `hostNetwork`，Workload Identity 不生效，Pod 里写不进桶 | `frameworks/maxtext/run.sh` | 已合入 |
| 路由核可选先舍到 BF16（`routed_gate_bf16_rounding`），梯度直通 | 路由矩阵乘的操作数与 GPU 逐位相同 | `maxtext` 的 `src/maxtext/layers/moe.py` | 已合入 |
| RMSNorm 可选单次舍入（`rmsnorm_single_rounding`） | 对上 GPU 的 RMSNorm 只舍一次 | `maxtext` 的 `src/maxtext/layers/normalizations.py` 等 | 已合入 |
| 门控激活：稀疏路径读 `activations_in_float32`；可选按算子舍入（`mlp_activation_round_per_op`） | 对上 GPU 路由专家的激活舍入位置 | `maxtext` 的 `src/maxtext/layers/moe.py`、`layers/linears.py` | 已合入 |
| 逐层隔离回放 + L1 单算子回放 | 加钩子的训练步在 TPU 上不逐位（多出的输出改变了 XLA 的融合），逐层判定改用单独编译的回放 | `frameworks/maxtext/layer_replay.py`、`l1_probes.py` | 已合入 |
| `run.sh` 透传 `XLA_FLAGS` | excess precision 开 / 关两臂（诊断用） | `frameworks/maxtext/run.sh` | 已合入 |
| 窗口化 collective matmul 的分片部分和留在 FP32（`dense_fp32_accumulation`） | 按 BF16 累加部分积让 L1 / L2 前向 R 比超线，打开后回到线内（见注 35） | `maxtext` 的 `src/maxtext/layers/linears.py` 等；对齐配置 yml `a238883b…` | 已合入 |
| GPU 输出头 dgrad 分段 FP32 累加（`rr_arms.py` 的 `dgrad-chunk`） | 最终 golden `rcc` 唯一的数值改动：B200 上 BF16 张量核与 cuBLAS FP32 一次累加 12 万长的词表都有偏，分段累加才精确（[`goldens.md`](goldens.md)「精度」一节） | `frameworks/megatron/rr_arms.py` | 已合入 |
| 序列检查的两处测量口径（`denominator: mean`、`select.test_field` 与派生列 `param_norm_from_groups`） | grad_norm 带符号均值以 golden 为分母时不对称；TPU 的 param_norm 是整网一次 FP32 归约（[`../../docs/reference/judging.md`](../../docs/reference/judging.md)「两处测量口径」） | `janus/metrics.py`、`janus/run.py`、`janus/store.py` | 已合入 |
| TPU 诊断臂：精确轨迹 `--bf16-weights`、输出头精度 `--head-matmul-precision` / `--head-round`、回放 `--round-weights-bf16` / `--params-from` | 定位 TPU 那一份误差在输出头反向的 dlogits 舍入（只作诊断，不进对齐配置） | `frameworks/maxtext/train_hy3.py`、`layer_replay.py` | 已合入（补进 main 的情况见注 36） |
| 路由回放（把 GPU 选中的专家喂给 TPU） | L2 切断选专家的累积影响 | 要改 `maxtext` 的 `src/maxtext/models/hunyuan3.py`：`Hunyuan3MoELayer.__call__` 还没接 `forced_routed_experts`（`frameworks/maxtext/README.md`） | 未做。不阻塞判定：L2 改用逐层隔离回放，路由翻转单独计数（`L2.layer{i}.router`、ALIGNMENT-REPORT 第 4 节） |

- 注 35（`dense_fp32_accumulation`）：
  - 收缩维被 FSDP 切开的投影（q / k / v、稠密 MLP 的 wi）在窗口化 collective matmul 里按 BF16 累加部分积。
  - L1 / L2 前向 R 比超线：v5p 1.24 / 1.29，v7x 1.57 / 1.58。
  - 打开后四层前向 v5p 1.01 – 1.10、v7x 1.01 – 1.06。
- 注 36（TPU 诊断臂的提交）：
  - `cot-dgrad` / `cot-wgrad` 两档后补进 main。
  - `train_hy3.py` 与运行 [`t-v5p-bw-rd-hcd/hcw-1000`](../../docs/glossary.md#run-prefixes) 所用提交逐字节相同。
