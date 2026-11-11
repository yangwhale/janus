# GPU 侧剩余检查：TF32、kernel 回放、MoE 分阶段、poison 与因果性、特殊值存 / 恢复

**目的**：补齐第一阶段在 GPU 上还能做的五件事，每件都真跑、结果入库、能照着复现：

1. TF32 开 / 关：
   - 找到在这个镜像里真正管用的关法，确认生效；
   - 看它改了哪些算子、改了多少。
2. 专家分组 GEMM 与核心注意力：
   - 用第 1 步真实输入单独回放，先对训练逐位，再对 FP64；
   - 分组 GEMM 的反向对逐专家循环。
3. MoE 层内分阶段：
   - router → scores → top-k → permute → GEMM1 → 激活 → GEMM2 → unpermute → 加共享专家；
   - 每段对 FP64，找误差在哪一步跳上去。
4. 合成 poison 与因果性：
   - padding / 尾块填 NaN、Inf，只改未来 token；
   - 用真实模块看该不变的是否逐位不变。
5. 特殊值存 / 恢复：−0.0、次正规数走一次 Megatron 存档 / 恢复，逐位核对。

前面的单算子回放（路由 GEMM、投影、归一化、初等函数等）见 [`0003-l1-op-replay.md`](0003-l1-op-replay.md)；本文件的脚本沿用它的写法和结果格式。

## 输入与脚本

| 运行（Pod 上 `/data/janus-runs2/`） | 怎么跑出来的 | 用途 |
|---|---|---|
| `d-a` | 第一阶段的确定性基线（`/data/r.sh d-a 100 det --full --dump-steps 1,100`） | 对照 |
| `g61-obs` | 同 `d-a` 的配置 2 步，另开 `--observe attn,moe,expw,gemm`（见注 1） | 第 2–4 项的真实输入；GEMM 清单 |
| `g61-notf32` | 同 `d-a` 的配置 30 步，关 TF32 | 第 1 项 |
| `g61-sv-save` / `g61-sv-load` | 不开分布式优化器，第 2 步存档前埋特殊值；从该存档续训 | 第 5 项 |

- 注 1：`--observe` 是 `janus/dump/megatron.py` 的可选观测点，默认关。

脚本都在 `models/hunyuan3/frameworks/megatron/` 下（目录文件除外）。结果文件都在 `models/hunyuan3/alignments/00-gpu-noise-and-drills/results/` 下，表里只写文件名。

| 脚本（`models/hunyuan3/frameworks/megatron/`） | 做什么 | 结果 |
|---|---|---|
| `run_check.sh` | 与 `/data/r.sh` 同一套参数起 8 卡训练，训练脚本与 janus 取本检出；`TF32=off` 关 TF32 | 运行目录 |
| `models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/phase1-obs.yaml`、`phase1-tf32.yaml` | `janus run` 比运行 | `phase1-obs.*`、`phase1-tf32.*` |
| `l1_kernels.py` | 见注 2 | `phase1-kernels.*` |
| `moe_stages.py` | MoE 层内分阶段对 FP64 | `phase1-moe-stages.*` |
| `causal_model.py` | 整网（真实 Megatron 模块，单卡 EP1）因果性 | `phase1-causal-model.*` |
| `ckpt_special.py` | 比存档前后的特殊值 | `phase1-ckpt-special.*` |

- 注 2：`l1_kernels.py` 做四件事：
  - TF32 开关矩阵与运行内实测、GEMM 清单；
  - 专家分组 GEMM 与核心注意力回放；
  - 分组 GEMM / 注意力 poison；
  - 注意力因果性。

## 复现

- Pod `janus-gpu`，`kubectl --context <GPU 集群>`。
- GPU 是共享的，用 GPU 的命令都包 `flock /data/gpu.lock`。
- 代码打包拷到 Pod 的 `/tmp/janus-g61`（不碰共享目录 `/data/janus-src`）。

```bash
# 8 卡训练（各几分钟；g61-sv-save 另写约 160 GB 的存档）
cd /tmp/janus-g61/models/hunyuan3/frameworks/megatron
flock /data/gpu.lock ./run_check.sh g61-obs 2 det --full --dump-steps 1 --observe attn,moe,expw,gemm
TF32=off flock /data/gpu.lock ./run_check.sh g61-notf32 30 det --full --dump-steps 1
RE='^decoder\.layers\.(0\.self_attention\.q_layernorm\.weight|1\.mlp\.experts\.linear_fc2\.weight0|1\.mlp\.router\.expert_bias)$'
flock /data/gpu.lock ./run_check.sh g61-sv-save 2 det --no-dist-opt --dump-steps "" --save /data/ckpt-g61-sv --special "$RE"
flock /data/gpu.lock ./run_check.sh g61-sv-load 3 det --no-dist-opt --dump-steps "" --resume /data/ckpt-g61-sv --special "$RE"

# 比运行（CPU）
cd /tmp/janus-g61 && export PYTHONPATH=/tmp/janus-g61
python3 -m janus run models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/phase1-obs.yaml --out /data/janus-g61/obs
python3 -m janus run models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/phase1-tf32.yaml --out /data/janus-g61/tf32

# 单卡回放（每个十几分钟，大半是在 CPU 上算 ULP 等指标）
source /data/env.sh; export PYTHONPATH=/tmp/janus-g61:$PYTHONPATH
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
flock /data/gpu.lock python models/hunyuan3/frameworks/megatron/l1_kernels.py --run /data/janus-runs2/g61-obs \
    --tf32-run /data/janus-runs2/g61-notf32 --out /data/janus-g61/kernels
flock /data/gpu.lock python models/hunyuan3/frameworks/megatron/moe_stages.py --run /data/janus-runs2/g61-obs --hf /data/hy3-4l --out /data/janus-g61/moe-stages
flock /data/gpu.lock python models/hunyuan3/frameworks/megatron/causal_model.py --run /data/janus-runs2/g61-obs --hf /data/hy3-4l --out /data/janus-g61/causal-model
python models/hunyuan3/frameworks/megatron/ckpt_special.py --save-run /data/janus-runs2/g61-sv-save --load-run /data/janus-runs2/g61-sv-load \
    --out /data/janus-g61/ckpt-special
```

每个结果目录里的 `results.json` + `report.md` 原样拷进 `models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-<名>.*`。

## 检查一览

门槛的来源写在每条检查的 `source` 里；「推断」的门槛是起点。没有门槛的是诊断（`hard: false`，只报数）。

| 检查 | 结果文件 | 判据 |
|---|---|---|
| `DET.obs.*` | `phase1-obs` | 新观测点不改数值：输入指纹、前 2 步标量、第 1 步共有张量逐位 |
| `TF32.switch.matrix` / `TF32.router_bwd` | `phase1-kernels` | 诊断 |
| `TF32.switch.override_off` | `phase1-kernels` | `NVIDIA_TF32_OVERRIDE=0` 下 FP32 GEMM 对 FP64 `rel_l2 ≤ 1e-5`（见注 1） |
| `TF32.in_run` | `phase1-kernels` | 关 TF32 的运行里 `meta.json` 的两个实测值 ≤ 1e-5 |
| `TF32.gemm_inventory` | `phase1-kernels` | 诊断：第 1 步 GEMM 调用清单 |
| `TF32.step1.*`、`TF32.loss`、`TF32.grad_norm` | `phase1-tf32` | 前向逐位（硬）；反向、梯度、更新后参数、loss 对默认设置噪声（calibrated） |
| `L1.kernel.experts.{fwd,bwd}.replay`、`L1.kernel.attn.{fwd,bwd}.replay` | `phase1-kernels` | 与训练抓下的输出逐位 |
| `L1.kernel.experts.vs_loop` | `phase1-kernels` | 诊断：分组对逐专家循环 |
| `L1.kernel.experts.wgrad` | `phase1-kernels` | 分组 wgrad 对 FP64 `rel_l2 ≤ 1e-5`（推断） |
| `L1.kernel.gemm_row_invariance` | `phase1-kernels` | 诊断：GEMM 逐行结果是否随批里其他行变 |
| `L2.moe.expert_weights`、`L2.moe.movement` | `phase1-moe-stages` | 逐位（权重 sha256、搬运） |
| `L2.moe.stages.{chain,local}`、`L2.moe.topk.fp64` | `phase1-moe-stages` | 诊断 |
| `L1.poison.{grouped_gemm,attn_padding}`、`L1.causal.attn` | `phase1-kernels` | 对干净输入逐位 |
| `L1.causal.attn_future_nan` | `phase1-kernels` | 诊断 |
| `L2.causal.model.repeat` / `fwd` / `act_grad` / `param_grad.dense` | `phase1-causal-model` | 逐位（同一输入两次；改未来 token 后过去位置） |
| `L2.causal.model.param_grad.experts`、`L2.causal.model.attribution` | `phase1-causal-model` | 诊断 |
| `CKPT.special.{poked,full,values}` | `phase1-ckpt-special` | 埋值、恢复后全量、恢复后前 8 个元素的位模式逐位 |

- 注 1：门槛是推断：IEEE FP32 约 1e-6，TF32 约 3e-4。

### 先证观测不改数值

数字出自 `models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-obs.results.json`。`g61-obs` 对 `d-a`：

- 第 1 步 425 个共有张量逐位相同；
- 前 2 步 lm_loss、grad_norm、num_zeros 逐位；
- 32 个 microbatch 的输入指纹相同。

新观测点、GEMM 清单（TorchDispatchMode + 包装 TE 的 GEMM 函数）、训练开始时的 TF32 实测都不改数值。所以第 2–4 项用的输入就是 `d-a` 第 1 步的输入。

## 一、TF32 开 / 关

### 做法

- **开关矩阵**（`l1_kernels.py` 的 `TF32.switch.matrix`）：六种设置各起一个子进程。
  - `NVIDIA_TF32_OVERRIDE` 只在 cuBLAS 初始化时读，不能在同一进程里切。
  - 各算一次 FP32 输入的 `torch.mm` 和 TE `general_gemm`（1024 × 4096 · 4096 × 1024）。
  - 对 FP64 算 rel_L2，并对「IEEE FP32」那一份比逐位。
- **关 TF32 的短运行 `g61-notf32`**：`run_check.sh` 的 `TF32=off` 做三件事：
  - 去掉 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`；
  - 设 `NVIDIA_TF32_OVERRIDE=0`；
  - 训练脚本 `--no-tf32` 再把 torch 的两个开关设为 False。
  - 训练开始时在 rank 0 实测一次 FP32 GEMM 的误差，连同开关读回值写进 `meta.json` 的 `tf32`（`TF32.in_run`）。
- **GEMM 清单**：训练第 1 步每次 GEMM 调用的输入 dtype 与调用处（`g61-obs` 的 `gemm_inventory.json`，`TF32.gemm_inventory`）。
  - 回答「哪些算子的输入是 FP32」。
  - 覆盖 TE 的 `general_gemm` / `general_grouped_gemm` 与 torch 的 `mm` / `addmm` / `bmm` / `baddbmm`。
  - cuDNN 注意力、输出层的融合 wgrad 等其他自带 kernel 不在清单里（它们的输入在 `d-a` 里都是 BF16 张量，推断不受 TF32 影响）。
- **运行对比**：`g61-notf32` 对 `d-a` 比第 1 步全部张量、前 30 步 loss。
  - 目录：`models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/phase1-tf32.yaml`。
  - 尺子同 `phase1-parallel-ext`：默认设置运行两两的差。
- **路由反向单独回放**：路由反向的两个 GEMM（输入都是 FP32）用真实输入单独回放，TF32 开 / 关各算一次 dx、dW，对 FP64（`TF32.router_bwd`）。

### 结果

数字出自：

- `models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-kernels.results.json`
- `models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-tf32.results.json`

开关矩阵（对 FP64 的 rel_L2）：

| 设置 | torch.mm（FP32 入） | TE general_gemm（FP32 入） |
|---|---|---|
| 默认（容器设了 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`） | 2.94e-4（TF32） | 2.94e-4（TF32） |
| torch `allow_tf32=False`（见注 1） | 1.15e-6 | **2.94e-4（仍是 TF32）** |
| `NVIDIA_TF32_OVERRIDE=0`（torch 开关读回值仍是 True） | 1.15e-6 | 1.15e-6 |

- 注 1：或 `set_float32_matmul_precision("highest")`，或去掉那个环境变量，结果相同。

其余结果：

- **训练进程里实测**（`TF32.in_run`，探针形状与上表不同）：
  - TF32 开的 `g61-obs` 两者都是 2.94e-4；
  - `g61-notf32` 两者都是 2.30e-7——TF32 确实关掉了。
- **GEMM 清单**：第 1 步 rank 0 共 154 次 GEMM 调用。
  - **输入是 FP32 的只有 12 次，全部是路由的反向**：`moe_utils.py` 的 `RouterGatingLinearFunction.backward`，dgrad 与 wgrad 各 6 次 = 3 个 MoE 层 × 2 个 microbatch。
  - 路由前向是 BF16 入 FP32 出，其余 GEMM 都是 BF16 入。
- **运行对比**：第 1 步前向 46 个张量逐位相同；425 个张量里 234 个不同，全在反向及其下游。
  - 反向里**第一个不同的是第 3 层的 `mlp_norm.out.grad`**。
  - 逐位相同的：`logits.grad`、`final_norm.out.grad`、第 3 层的 `out.grad`、`mlp.out.grad`、专家与共享专家输出的梯度。
  - 路由的 dx 正是在 `mlp_norm.out.grad` 汇入。
  - 参数梯度里输出层、最后的 norm、第 3 层专家与共享专家逐位相同；第 3 层路由权重的梯度起就不同。
- **量级**：
  - 参数梯度最大 rel_L2 是 embedding 的 1.14e-3，是默认设置噪声上限的 1.71 倍；判 fail 靠的是 cosine 偏离（1 − cos 6.4e-7，上限的 4.95 倍，`TF32.step1.grads.tol` 的 r1 = 4.95）；
  - 激活梯度最大 5.66e-3（r1 = 1.58，inconclusive）；
  - 更新后参数 4.2e-5（r1 = 1.08，inconclusive）；
  - 30 步 loss 首步差 0、带符号均值 −1.8e-6、最大单步 1.88e-5，在噪声内（r1 = 0.61）。
- **路由反向单独回放**：dx、dW 是 BF16 出，rel_L2 主要由最后一次舍入决定。
  - rel_L2（dx / dW）：TF32 开 1.671e-3 / 1.698e-3，关 1.658e-3 / 1.685e-3。
  - 对正确舍入逐位相同率：TF32 开 89.8% / 89.7%，关 99.98% / 99.95%。

### 结论

- 在这个镜像里，**只有 `NVIDIA_TF32_OVERRIDE=0` 能关掉全部 FP32 GEMM 的 TF32**。
  - torch 的开关只管 torch 自己的 matmul，TE 的 GEMM 不看它。
  - 开了 `NVIDIA_TF32_OVERRIDE=0` 之后 torch 的开关读回来仍是 True——**开关读回值不能当证据，要实测**。
- 本模型里受 TF32 影响的只有路由反向的两个 GEMM。
  - 它改变第 1 步的梯度（cosine 偏离到 GPU 自身噪声上限的 4.95 倍、rel_L2 1.71 倍，上限取 3 对；换成 28 对的上限是 3.2 倍、1.70 倍，仍超 2 倍），但 30 步 loss 仍在噪声内。

### 意味着

- 第一阶段 `d-a` 等运行的反向里，路由那两个 GEMM 一直是 TF32。
- TPU 侧如果按「FP32 GEMM」去对路由反向，会和 GPU 参照差一个 TF32 的量级；反过来也一样。

### 该怎么做

- 对齐运行用 `run_check.sh` 的 `TF32=off`（三样一起设）。
- 以 `meta.json` 里 `tf32.probe_rel_l2_*` 的实测值为准（≤ 1e-5 才算关了）。
- TPU 侧对路由反向的 GEMM 精度单独写进 `config_diff`。现状：混元 3 验收的 `config_diff`（01 `phase2-tpu.yaml` 的 `L0.config.diff`）没有这一项；配置对照表把它记成不可避免差异，TPU 一侧「FP32 操作数单遍 BF16」是推断（[混元 3 配置对照](../../../models/hunyuan3/config-alignment.md)「TF32」一行）。

## 二、专家分组 GEMM 与核心注意力回放

### 做法

`l1_kernels.py`，输入 `g61-obs` 第 1 步 rank 0 第一个 microbatch。

- **专家**：
  - 输入：all-to-all 之后的专家输入（第 1–3 层各 27527 / 38631 / 36218 行，24 个本地专家，每个专家的行数都不是 128 的倍数）、每专家 token 数、专家侧概率、本地 24 个专家的权重。
  - 用 TE `GroupedLinear` 回放 GEMM1 → 激活 → GEMM2（训练用的同一模块类，`fuse_wgrad_accumulation` 按训练实值）。
  - 激活照 Megatron `TEGroupedMLP` 的非融合 GLU 分支写：`silu(前半) · (后半 + 0)` 在 BF16 里算，乘 FP32 概率后转回 BF16；`observe_step1.json` 记了这一路的开关实值。
  - 反向用训练抓下的 GEMM2 输出梯度。
  - 对训练抓下的每个中间量逐位比；FP64 真值逐专家算，输入取训练抓下的上一阶段值。
- **逐专家循环参照**：同一份输入逐个专家算，对分组版逐位比。
  - 前向与 dgrad 用 torch `F.linear`；
  - wgrad 用单个 TE `general_gemm`（FP32 出）。
- **注意力**：
  - 输入：核心注意力的 q / k / v（RoPE 之后）与输出梯度 dO。
  - 用 TE `DotProductAttention`（sbhd、causal、GQA 64 / 8、softmax_scale 取训练实值）回放前向与反向。
  - 对训练抓下的输出与 dq / dk / dv 逐位比。
  - FP64 真值逐头算因果 softmax 注意力及其 VJP。

### 结果

数字出自 `models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-kernels.results.json`。

- **专家（3 层）**：
  - GEMM1、激活、GEMM2 前向，GEMM2 dgrad、激活反向、GEMM1 dgrad 全部与训练 **100% 逐位**（`L1.kernel.experts.{fwd,bwd}.replay`）。
  - 对 FP64：GEMM 输出与 dgrad rel_L2 1.66e-3（BF16 出，对正确舍入逐位相同率 99.69%–99.90%，ULP p99.9 为 1）。
  - 对 FP64：激活 2.87e-3–2.91e-3（逐位相同率 65%–66%，最大 2 ULP）。
- **分组 vs 逐专家循环**：
  - 前向、dgrad 与 torch 逐专家 `F.linear` 100% 逐位；
  - wgrad（FP32 主梯度，从 0 累加）与逐专家 TE `general_gemm` 100% 逐位，对 FP64 rel_L2 1.9e-6–3.5e-6（门槛 1e-5，推断）。
- **注意力（4 层）**：
  - 前向输出、dq、dk、dv 与训练 **100% 逐位**。
  - 对 FP64：输出 rel_L2 1.66e-3–1.95e-3（逐位相同率 78%–97%）；dq / dk / dv 2.1e-3–4.0e-3（逐位相同率 53%–70%）。

### 结论

- 回放写法就是训练时的计算，两个 kernel 都能离开训练单独复现、逐位对上。
- 分组 GEMM 与逐专家各算一遍逐位相同（分组本身不引入数值差）。
- 注意力反向对 FP64 的误差是 BF16 输出舍入的 1.3–2.4 倍（推断：反向里 P、dS 先舍到 BF16 再进 GEMM）。

### 意味着 / 该怎么做

- TPU 侧这两个算子可以直接拿同一份输入（`g61-obs` 第 1 步）回放比较，GPU 这边的地板就是上面的 FP64 数。
- 分组 GEMM 的 wgrad 用 FP32 累加、对 FP64 是 1e-6 量级。
  - TPU 侧若用 BF16 出 wgrad 会差三个数量级，要在 `config_diff` 里对齐。现状：混元 3 验收的 `config_diff` 没有纳入分组 wgrad 的输出 dtype，配置对照表里也没找到对应的一行。

## 三、MoE 层内分阶段

### 做法

`moe_stages.py`：

- **输入**：观测点抓下第 1 步 rank 0 每个 MoE 层各阶段的实值。
  - scores 没抓，用训练同一算子 `torch.sigmoid` 从抓下的 logits 重算（L1 已证 top-k 与训练逐位一致）。
- **专家权重**：其余 rank 上的专家权重从 HF 原件读、转成 Megatron 布局。
  - 先按 `params_sha` 核对 1152 个专家权重与训练加载时 sha256 全部相同（`L2.moe.expert_weights`）。
- **两种 FP64 参照**：
  - **链式**——从 MoE 层输入（`mlp_norm.out`）开始全程 FP64、中间不舍入（累积误差）；
  - **本阶段**——只把这一阶段换成 FP64，输入取训练抓下的上一阶段实值（这一步自己引入的误差）。
- **搬运**：permute、all-to-all、combine 是纯搬运，另做逐位检查（`L2.moe.movement`）。
- **覆盖的行**：
  - GEMM1、激活只在「rank 0 自己发给本地专家」的行上有链式参照（3352 / 4430 / 4128 行）；
  - GEMM2 用 combine 回本 rank 的全部 32768 行。

### 结果

数字出自 `models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-moe-stages.results.json`（rel_L2，三层范围）。

| 阶段 | dtype | 链式 | 本阶段 | 本阶段对正确舍入逐位相同率 |
|---|---|---|---|---|
| router logits | FP32 | 3.76e-6–3.96e-6 | 同左 | 0%–0.04% |
| scores（sigmoid） | FP32 | 1.33e-6–3.14e-6 | 4.5e-8 | 60%–64% |
| probs（top-k 归一化 × 2.826） | FP32 | 7.9e-7–1.79e-6 | 6.4e-8–6.6e-8 | 49%–51% |
| top-k 选中集合 | — | 对 FP64 链式 100% 一致，0 行打平 | | |
| permute | BF16 | 0（逐位搬运） | 0 | 100% |
| **GEMM1** | BF16 | **1.66e-3** | 1.66e-3 | 99.69%–99.78% |
| **激活**（silu·up·prob） | BF16 | **3.73e-3–3.78e-3** | **2.84e-3–2.99e-3** | 65%–66% |
| GEMM2（combine 回来） | BF16 | 3.67e-3–3.99e-3 | 1.66e-3 | 99.89%–99.90% |
| unpermute（k 个专家相加） | BF16 | 3.42e-3–3.95e-3 | 1.66e-3 | **100%** |
| 共享专家 | BF16 | 3.03e-3–3.24e-3 | 同左 | 42%–43% |
| 输出（相加） | BF16 | 3.31e-3–4.03e-3 | 1.73e-3–1.77e-3 | **100%** |

搬运检查全部逐位：permute 的 token 与概率、all-to-all 之后 rank 0 发给本地专家的段、combine 回来的本地专家段（各 3 层）。

### 结论

误差有两个台阶：

- **第一个在 GEMM1**：从 FP32 路由的 1e-6 量级跳到 BF16 输出舍入的 1.66e-3，这是 BF16 出必然的一次舍入。
- **第二个在激活**：本阶段误差 2.8e-3–3.0e-3（三步里最大），链式从 1.66e-3 跳到 3.7e-3。
  - 激活在 BF16 里做了两次舍入（silu 输出、乘积），再乘概率后第三次舍到 BF16。

其余阶段：

- 之后 GEMM2、unpermute、加共享专家都只是再舍一次，链式基本不再长。
- unpermute 和最后的相加对正确舍入 100%（先高精度加完再舍一次）。
- 路由一路是 FP32，误差 1e-6 量级，top-k 与 FP64 完全一致。

### 意味着 / 该怎么做

- TPU 侧逐阶段对的时候，激活这一步是最可能出现「多一次 / 少一次舍入」差别的地方。
- 若 TPU 把 silu·up 融合在 FP32 里算完再舍一次，本阶段误差会降到约 1.7e-3，和 GPU 差出来的是写法不是错。
- 对齐时按 GPU 这三次舍入的位置写，或者在报告里注明。

## 四、合成 poison 与因果性

### 做法

- **分组 GEMM**（`L1.poison.grouped_gemm`，真实第 1 层专家输入与权重）：
  - ① 尾部毒值：输入与输出梯度放进更大的缓冲区，尾部 256 行填 NaN / Inf，只把有效行的视图传进去。
    - 每个专家的行数都不是 128 的倍数，最后一个 tile 越界读就会读到毒值。
  - ② 整个专家毒值：把一个专家（行数不整除 128 里最多的那个）的输入与输出梯度全换成 NaN，看其余专家的输出行、dgrad 行、wgrad。
- **注意力 padding**（`L1.poison.attn_padding`）：
  - 两条序列打包成 THD，长度 1000 / 1500 补到 1024 / 1536。
  - padding 行的 q / k / v / dO 填 NaN 或 Inf。
  - 有效行的输出与 dq / dk / dv 对干净输入（padding 填 0）逐位比。
- **注意力因果性**（`L1.causal.attn`，真实 q / k / v）：
  - 只把位置 ≥ p 的 q / k / v 换成另一层的真实值，dO 在 ≥ p 处置 0。
  - 比位置 < p 的输出与 dq / dk / dv。
  - p = 2048（整除 128）与 3001（不整除）。
  - 诊断（`L1.causal.attn_future_nan`）：未来位置填 NaN。
- **整网因果性**（`causal_model.py`）：
  - 单卡上用 Megatron-Bridge 搭出与训练同一套模块的模型（EP1，全部 192 个专家在一张卡上，确定性设置同训练）。
  - 真实输入只改位置 ≥ p 的 token、loss 只算 < p。
  - 先证同一输入跑两次逐位相同，再比位置 < p 的各层输出、逐 token loss、激活梯度与参数梯度。

### 结果

数字出自：

- `models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-kernels.results.json`
- `models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-causal-model.results.json`

隔离类：

- **分组 GEMM**：
  - 尾部填 NaN / Inf 时输出、dgrad、wgrad 100% 逐位不变；
  - 一个专家全 NaN 时其余专家的输出行、dgrad 行、wgrad 100% 逐位不变。
- **注意力 THD padding**：NaN、Inf 两种，有效行输出与 dq / dk / dv 100% 逐位不变。
- **注意力因果性**：p = 2048、3001，改未来 token 后过去位置的输出与 dq / dk / dv 100% 逐位不变。
- **未来位置填 NaN（诊断）**：
  - p = 2048 时过去位置的输出、dq 干净，**dk、dv 全部被污染**（2048 行全非有限）；
  - p = 3001 时输出、dq 从第 2816 行起到 p 前共 185 行被污染，dk、dv 全部被污染。p 在 2944–3071 这个 128 行块里，185 行装不进一个 128 行块；2816 = 11 × 256 是 p 所在 256 行块的起点，所以污染范围是 p 所在的 256 行块里 p 之前的行（分块大小没核，256 是从这组数推的；p = 2048 正好在块边界上，过去行不受污染，与此相符）。

整网（`phase1-causal-model`）：

- 同一输入跑两次 1287 个张量全部逐位（确定性前提成立）。
- 改未来 token 后，两个 p 下逐位不变的：embedding、第 0 层（注意力 + 稠密 MLP）全部、第 1 层的注意力、`mlp_norm`、共享专家输出。
- **第一个不同的是第 1 层（第一个 MoE 层）的 `mlp.out`**（逐位相同率 99.992% / 99.993%）。
  - 之后各层连带不同（逐 token loss 只剩 0.24% / 0.30% 的位置逐位）。
- 第 1 层过去位置的路由选中与未改时完全相同，但 192 个专家（p = 3001 时 189 个）收到的 token 数都变了。
- 输出不同的 61 个（p = 2048）和 83 个（p = 3001）过去 token，**全部**路由到了「token 数变了、且不超过 32」的 13 个专家之一。
  - 这些专家最少的只有 9 个 token（`L2.causal.model.attribution`）。

GEMM 逐行结果与批大小（`L1.kernel.gemm_row_invariance`，为解释上一条另做）：

- 做法：取真实专家输入的前 M 行，在后面接 d 行别的 token 重算，比前 M 行。
- GEMM1 的形状（K = 4096、N = 3072）：
  - M ≤ 32 时前 M 行会随后面接的行变（例如 M = 8 时后面接 1 行，8 行全变；M = 16 接 64 行以上全变）；
  - M ≥ 33（33 / 48 / 64 / 128 / 256 / 1024）时接 1 / 8 / 64 / 512 行都没有一行变。
- GEMM2 的形状（K = 1536、N = 4096）所有 M 都不变。
- torch `F.linear` 与 TE `GroupedLinear` 的表现完全一样。

### 结论

kernel 层面：

- 分组 GEMM 的尾块和组边界、注意力的 THD padding 都做到了真正的隔离——毒值不进有效结果。
- 因果掩码则不同：同一序列里未来位置一旦是 NaN，
  - 前向会污染 p 所在那个块里的过去行（按 p = 3001 的读数推断块是 256 行，见上）；
  - 反向会污染全部 dk / dv。
- 未来位置是有限值时则完全不影响过去（逐位）。
- 机制（推断）：
  - 被掩掉的位置在 p 所在的块里仍被读进来、乘的是 0（`0 × NaN = NaN`）；
  - 反向里未来的 q 是 NaN，dS = P ⊙ (dP − D) 在那些行上是 NaN · 0，经 dk = dSᵀ q、dv = Pᵀ dO 扩散到所有 key。

整网层面，因果性在数学上成立、在逐位上不成立：

- 未来 token 改变了每个专家收到的 token 数。
- token 数很少时，BF16 GEMM 的逐行结果会随同组的其他行变（推断是 cuBLAS 按 M 选了不同的 kernel / 分块，没有往下追）。
- 路由到这些专家的过去 token 因此差在最后一位，再经后面的注意力扩散到所有位置。
- 稠密层与注意力本身是逐位因果的。

### 意味着 / 该怎么做

- **padding**：想填什么都行的前提是用 THD（`cu_seqlens_*_padded`）把它从序列里切出去。
  - 放在同一序列尾部、只靠因果掩码挡着的 padding 必须是有限值。
  - TPU 侧做同样的 poison 用例时，结论按这两类分开判。
- **MoE 的逐位因果 / 批不变**：「整网逐位因果 / 批不变」这类检查对 MoE 不能直接要求逐位。
  - 专家 token 数少于约 32 时 GPU 自己就做不到。
  - 要么只对稠密与注意力部分要求逐位，要么对 MoE 输出改用容差、并把专家 token 数一起报。
- **本次训练的位置**：第 1 步 rank 0 的本地专家每个至少收到 183 / 334 / 233 个 token（第 1–3 层，`L1.kernel.experts.fwd.replay` 的 `m_min`），在逐行不变的区间。
  - EP 更小、batch 更小或负载更偏时会落进去（推断）。

## 五、特殊值存 / 恢复

### 做法

- **运行**：`g61-sv-save` 不开分布式优化器跑 2 步。
- **埋值时机**：第 2 步结束、框架存档之前（Megatron-Bridge 在 `on_train_step_end` 回调之后才存档）。
- **埋的值**：往三个参数的前 8 个元素写入 FP32 位模式：
  - −0.0、+0.0、±1e-40（`0x000116C2`）、±2⁻¹⁴⁹（最小次正规）、最大次正规 `0x007FFFFF`、最小正规 2⁻¹²⁶。
- **各张量怎么写**：
  - BF16 参数写入同一组值的 RNE 结果；
  - FP32 主权重、Adam 一阶动量写原值；
  - 二阶动量写绝对值；
  - FP32 buffer `expert_bias` 也写原值。
- **三个参数覆盖三种存法**：
  - 第 0 层 `q_layernorm.weight`（各 rank 副本相同）；
  - 第 1 层 `linear_fc2.weight0`（按专家并行切开，每个 rank 是不同的专家）；
  - 第 1 层路由 `expert_bias`（FP32 buffer）。
- **比对**：
  - 埋完把这些张量全量写进 `special/rank<r>.saved.safetensors`；
  - `g61-sv-load` 从该存档续训，开训前把同一批张量写进 `special/rank<r>.loaded.safetensors`；
  - `ckpt_special.py` 逐 rank 逐张量比。

### 结果

数字出自 `models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-ckpt-special.results.json`。

- 范围：8 个 rank × 9 个张量（BF16 参数、FP32 主权重、两个动量 × 2 个参数，加 1 个 buffer）。
- 埋值本身与期望位模式 100% 一致。
- 恢复后全量张量与存档前 100% 逐位相同。
- 前 8 个元素的位模式与埋入的完全一样：
  - FP32 主权重里的 1e-40 仍是 `0x000116c2`。BF16 表示不了这个值，说明主权重是按 FP32 原样存取的，不是从 BF16 参数重建。
  - BF16 参数里是 RNE 之后的 `0x8000 0x0000 0x0001 0x8001 0x0000 0x8000 0x0080 0x0080`：
    - ±1e-40 成了 BF16 次正规数 ±`0x0001`；
    - ±2⁻¹⁴⁹ 舍成 ±0；
    - 最大 FP32 次正规数舍成 BF16 最小正规数。
- 存档格式是 torch_dist（`run_config.yaml` 的 `ckpt_format`）。

### 结论 / 意味着

- Megatron 的存档 / 恢复（torch_dist 格式，不开分布式优化器）对 −0.0 与 FP32 / BF16 次正规数逐位保真。
  - 没有 flush-to-zero、没有经 BF16 中转。
- TPU 侧的存档 / 转换要过同一个用例（Orbax、MaxText ↔ Megatron 转换都可能在 CPU 上做 dtype 转换）。
- 没有测分布式优化器（主权重分片存）的路径。

## 范围与局限

- 全部只用第 1 步、rank 0 第一个 microbatch；回放与分阶段在 B200、TE 2.20.2、cuBLAS 13.8 上验证。
- GEMM 清单只截得到 TE 的 GEMM 函数和 torch 的 matmul 类算子；cuDNN 注意力、输出层的融合 wgrad 等不在清单里。
- MoE 分阶段的 scores 是重算的；GEMM1、激活的链式参照只覆盖 rank 0 自己发给本地专家的那部分行。
- 特殊值只测了不开分布式优化器的存档；只埋了有限的特殊值（没有 NaN / Inf：训练里出现它们本身就该报警）。
