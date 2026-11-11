[janus](../../README.md) › [文档](../README.md) › 参考 · 环境 ｜ 上手步骤见[第 1 章 准备环境与数据](../guide/01-environment-data.md)

# 环境：两边要钉死的版本与开关

## 一句话

先让每一边**自己可复现、开关可查**，再谈两边对齐。

没钉死的版本和开关，会让同一份代码在不同时间跑出不同的数，噪声参照也就失去意义。

## 每次运行都要记下的

写进运行目录的 `meta.json`（字段见 `dump-format.md`），每次报告照抄：

- 镜像的 `@sha256:` digest，不写 tag（tag 会被重新指向）；
- 代码仓库与 commit；
- 关键库版本：
  - TPU 侧 jax / jaxlib / libtpu；
  - GPU 侧 torch / CUDA / Transformer Engine / cuBLAS / NCCL / Megatron；
- 硬件型号、卡数、拓扑；
- 并行方式（dp / fsdp / tp / ep / cp / pp）；
- 确定性相关的环境变量与开关，**原样列出**；
- 运行时实际生效的模型、优化器、调度器配置（不是配置文件里写的值）。

`L0.config.diff` 会逐项对照这些字段；版本或并行方式不一致，直接报。

第一阶段的缺口：

- `meta.json` 的 `versions.megatron_core`、`versions.megatron_bridge` 是 `null`（训练脚本取版本的写法取不到）。
- 镜像 digest 也没记（`image` 字段是 tag）。
- 两者版本以镜像为准：Megatron-Core 0.20.0、Megatron-Bridge 0.7.0（复现审计在容器里用 `importlib.metadata` 读到）。
- TPU 侧对照前要补上。

## 两个模型实测用的版本

GPU ↔ TPU 对齐的现行运行（混元 3 的正式验收，混元 4 各尺寸的现行比较）都在下面这一套上。逐项出处在各实践 `setup.md` 的「软件版本」一节（混元 3：[01](../../models/hunyuan3/alignments/01-v5p-acceptance/setup.md#软件版本)；混元 4：[mini](../../models/hunyuan4/alignments/01-mini/setup.md#软件版本)）。

| 侧 | 组件 | 版本 | 备注 |
|---|---|---|---|
| TPU | 镜像 | MaxText 官方 `tpu_pre_training` 0.2.5，digest `b152355a…` | 钉在两个模型的 `frameworks/maxtext/run.sh` |
| TPU | jax / jaxlib | 0.11.1 / 0.11.1 | `meta.json` 的 `versions` |
| TPU | libtpu | 0.0.46，构建 `cl/963683785`（2026-08-12） | 记录：混元 4 mini 现行 `tz-off-rep`、06 rw3 的 `meta.json` 在 `versions.libtpu_pip` 记了 0.0.46；混元 3 01 / 02、混元 4 02 real-weight、03 mid 与更早的运行只记了构建号，按构建号认作 0.0.46。构建号到 pip 版本的对应已核：同一构建号的探针运行在 v5p、v7x 上都读到 `libtpu_pip` 0.0.46（[`jit-miscompile-probe-v5p.json`](../../models/hunyuan4/results/jit-miscompile-probe-v5p.json)、[`jit-miscompile-probe-v7x.json`](../../models/hunyuan4/results/jit-miscompile-probe-v7x.json)） |
| TPU | flax / optax / orbax-checkpoint | 0.12.9 / 0.2.8 / 0.12.4 | `meta.json` 的 `versions` |
| TPU | tokamax | 0.0.14（镜像自带的包） | `meta.json` 不记；读镜像层里的 `tokamax-0.0.14.dist-info` 核过（2026-11-11） |
| TPU | MaxText | 混元 3：`hunyuan` 分支 `0622774a`；混元 4：集成分支，各尺寸的提交不同 | 各实践 `setup.md` |
| GPU | 镜像 | NeMo 26.10，digest `fc75bf51…` | `meta.json` 的 `image`（混元 3 第一阶段只记了 tag） |
| GPU | torch / CUDA / NCCL | 2.14.0a0+b2c75dd062.nv26.09 / 13.4 / 2.31.2 | `meta.json` 的 `versions` |
| GPU | Megatron-Core / Megatron-Bridge | 0.20.0 / 0.7.0（容器自带，提交未登记） | 混元 4 不用 Megatron-Bridge |
| GPU | Transformer Engine | 2.20.2+44f31bbe | 实测：混元 3 的 `meta.json` 记了；混元 4 的 `meta.json` 不记，2026-11-11 在 digest `fc75bf51…` 的训练 Pod 里、训练用的库环境下读到（`import` 与已装的包都是这个版本；容器环境变量 `TRANSFORMER_ENGINE_VERSION=2.19` 是基础镜像的，不是实际装的），见混元 4 [`gpu-image-versions.json`](../../models/hunyuan4/alignments/01-mini/results/gpu-image-versions.json) |
| GPU | cuBLAS | 13.8.1（库文件 `libcublas.so.13.8.1.7`） | 实测：同上，进程实际映射的库与 `cublasGetProperty` 都是 13.8.1（cuBLASLt 130801）。混元 3 第一阶段记的 13.8.0.4 取自容器环境变量 `CUBLAS_VERSION`，不是实际加载的库 |
| GPU | cuDNN | 9.26.0 | 实测：同上（`torch.backends.cudnn.version()` 92600，`libcudnn.so.9.26.0`）；CUDA 运行时 13.4（`libcudart.so.13.4.49`），NCCL 2.31.2（`libnccl.so.2.31.2`，与 `meta.json` 相同） |

- **libtpu 以构建号和 `libtpu_pip` 为准，不以镜像 digest 为准。** 混元 4 TPU 自比的 0.0.45、0.0.47、0.0.48、0.0.52 臂都是在同一个 digest 上 `pip install --no-deps` 换装的（[自比设置「变种」](../../models/hunyuan4/alignments/04-tpu-self/setup.md#变种)），只看 digest 会把它们判成同一版本。
- **结论只对这一套版本成立。** 换 libtpu 或 jax，按 [`libtpu-052.md`](../../models/hunyuan4/alignments/04-tpu-self/libtpu-052.md) 的做法先离线编译看哪些程序变了，再重建 golden、重测。混元 4 的 TPU 自比另在 0.0.45、0.0.47、0.0.48、0.0.52 上测过，GPU ↔ TPU 对齐没有。

## GPU 侧

### 第一阶段的环境（混元 3）

- 硬件：单机 8 卡 B200（GKE Pod）。
- 镜像 `nvcr.io/nvidia/nemo:26.10`：
  - torch 2.14a、CUDA 13.4；
  - Megatron-Core 0.20.0、Megatron-Bridge 0.7.0；
  - Transformer Engine 2.20.2、cuBLAS 13.8。
  - 出处：`models/hunyuan3/frameworks/megatron/README.md`；TE 与 cuBLAS 版本取自 `phase1-l1.results.json` 的 `environment`。那里的 cuBLAS `13.8.0.4` 读的是容器环境变量 `CUBLAS_VERSION`；同一 digest 实际加载的库是 13.8.1（2026-11-11 实测，见上表）。
- 起 Pod、下权重、转格式、训练的步骤见 `models/hunyuan3/frameworks/megatron/README.md`。

### 两种设置

第一阶段用到两种设置，名字在结果里反复出现：

| 设置 | 怎么开 | 性质 |
|---|---|---|
| **确定性设置** | Megatron `deterministic_mode=True`（同时关掉 `cross_entropy_loss_fusion`），训练脚本 `--deterministic` | 同配置重跑逐步逐位相同（见注 1） |
| **默认设置** | 不加 `--deterministic`，下面三个环境变量也**都不设** | 第 1 步前向逐位相同，第 2 步起不同（见注 2）；用来量噪声 |

注 1：`phase1-determinism`：1000 步 loss、学习率逐位，第 1、1000 步各 24 个张量逐位。

注 2：反向与更新里有非确定累加。

下面三个环境变量由 `models/hunyuan3/frameworks/megatron/run.sh` 管：

- 参数里有 `--deterministic` 就设，没有就 `unset`；
- `JANUS_DET_ENV=1` / `0` 可强制设 / 不设。

| 变量 | 值 | 为什么 |
|---|---|---|
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | 不让 Transformer Engine 选非确定算法 |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | cuBLAS 确定性需要的固定工作区 |
| `NCCL_ALGO` | `Ring` | 固定集合通信算法；算法变了，归约顺序就变 |

第一阶段实际怎么设的（每次运行 `meta.json` 的 `determinism` 字段）：

- 默认设置的 18 次运行三个变量**都没设**，起它们的脚本显式 `unset` 了。
  - 这 18 次：`rd-golden-2..5`、`s-a..i`、`g1k-a..c`（第一阶段 GPU golden）、`i-nd`、`t-full`。
- 确定性设置里，只有 1000 步噪声底的 `rd-golden-1`、`rd-det-b`（以及早期的 `dump-a/b`、`gd-a/b`）设了三个变量。
  - 全覆盖与补充批次的 27 次确定性运行**没设**，只开了 `deterministic_mode`；
  - 这 27 次：`d-a`、`d-b`、`f-*`、`p-*`、`r-*`、`h-light*`、`m-opt8`、`t-2l`、`i-weight`、`g1k-det`（第一阶段 GPU golden 的确定性设置版）。
- 在确定性设置下这三个变量不改结果。
  - 设了三个变量的复现运行与没设的 `d-a`，30 步 lm_loss、grad_norm 逐步逐位相同（`models/hunyuan3/alignments/00-gpu-noise-and-drills/results/repro-audit.md`）。
  - 真正让重跑逐位相同的是 `deterministic_mode`。
- 「只设三个变量、不开 `deterministic_mode`」这种组合第一阶段没有跑过，它是否逐位、算哪一种设置，未确认。
  - `run.sh` 不加 `--deterministic` 起的就是第一阶段的默认设置（三个变量都不设）；
  - 做法见 `models/hunyuan3/frameworks/megatron/README.md`「两种设置怎么起」。

**用途分工**：

- 确定性设置做「应当逐位」的检查——任何一位不同都是信号。
  - 包括基线、故障注入、存档续训、抓数不改数值这类。
- 默认设置做噪声参照。
- 两者**不能混着当参照**（理由与数据见 `judging.md`「噪声参照只能用同一种设置的运行对」）。

### 第一阶段踩到的坑

1. **NCCL 版本冲突**。节点把 NCCL 2.27 挂到 `/usr/local/nvidia/lib64`，而 torch 需要容器里自带的 2.31。
   - 处理：`LD_LIBRARY_PATH` 前置 `/usr/lib/x86_64-linux-gnu`，让容器里的版本先被找到。
2. **NCCL 网络插件**。容器默认启用 GCP 的 NCCL 插件，它要求 RDMA 网卡，单机 Pod 没挂。
   - 处理：`NCCL_ENV_PLUGIN`、`NCCL_NET_PLUGIN`、`NCCL_PROFILER_PLUGIN`、`NCCL_TUNER_PLUGIN` 都设为 `none`。
   - 多机运行要重新评估。
3. **确定性模式要 Triton 缓存目录**。容器默认开了 `TRITON_CACHE_AUTOTUNING=1`，Megatron 确定性模式要求设 `TRITON_CACHE_DIR`。
4. **TF32 默认是开的**。镜像设了 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`。
   - 进程里 `torch.backends.cuda.matmul.allow_tf32`、`torch.backends.cudnn.allow_tf32` 默认都是 True，`float32_matmul_precision` 是 high（`phase1-l1.results.json` 的 `environment`）。
   - 影响：**FP32 输入**的矩阵乘会先降到 TF32。
     - 同一个路由 GEMM「先升 FP32 再乘」，开 TF32 时对 FP64 的 rel_L2 是 7.4e-6–7.7e-6，关掉是 7.5e-7–7.6e-7，差 10 倍（`../dev/design/0003-l1-op-replay.md`）。
   - **只有 `NVIDIA_TF32_OVERRIDE=0` 关得干净**（`models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-kernels.results.json` 的 `TF32.switch.matrix`）：
     - torch 的两个开关（或 `set_float32_matmul_precision("highest")`、或去掉 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`）只管 torch 自己的 matmul；
     - Transformer Engine 的 FP32 GEMM 照样走 TF32（rel_L2 2.94e-4 不变）；
     - 设了 `NVIDIA_TF32_OVERRIDE=0` 之后两者都是 IEEE FP32（1.15e-6），而 torch 开关读回来仍是 True——**读回值不能当证据**。
   - 第一阶段没有关。
     - 训练里输入是 FP32 的 GEMM 只有路由反向的两个，其余都是 BF16 输入。
     - 出处：第 1 步 GEMM 清单，`TF32.gemm_inventory`；清单覆盖 TE 与 torch 的 GEMM，不含 cuDNN 注意力等自带 kernel。
     - 关掉 TF32 后：第 1 步前向逐位不变，反向从路由的 dx 起不同；
     - 参数梯度最大差到默认设置噪声上限的 4.95 倍，30 步 loss 在噪声内（`models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-tf32.results.json`）。
   - 要求：对齐运行用 `models/hunyuan3/frameworks/megatron/run_check.sh` 的 `TF32=off`。
     - 它做的事：去掉 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`、设 `NVIDIA_TF32_OVERRIDE=0`、训练脚本 `--no-tf32` 再关 torch 开关。
     - 给了 `--janus-run` 时，训练脚本把开关读回值和一次实测（FP32 GEMM 对 FP64 的 rel_L2，torch 与 TE 各一个）写进 `meta.json` 的 `tf32`。
     - 以实测为准，≤ 1e-5 才算关了。
     - 做法与数字见 `../dev/design/0004-l1-gpu-kernels.md`「一、TF32 开 / 关」。
5. **抓梯度要关分布式优化器**。
   - 抓逐参数梯度、主权重、优化器状态时，需要每个 rank 上都有完整的已归约梯度，所以关掉分布式优化器。
   - 对照运行的优化器配置必须一致。
     - 第一阶段第一次拿「开着分布式优化器的轻量运行」对「关掉的全量抓数运行」，第 1 步梯度范数就不同；
     - 那是配置不同，不是钩子改了数值（`models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/phase1-hooks.yaml` 的注释）。

另外：GPU 抓数时**独占节点**，不和别的作业共享，避免资源争用改变执行。

## TPU 侧

### 已定的

各模型在 TPU 上跑到哪一步、结论是什么，看各自的报告：[混元 3](../../models/hunyuan3/ALIGNMENT-REPORT.md)、[混元 4](../../models/hunyuan4/README.md)。这里只记跨模型的约定（下面镜像、代码、配置三条的例子取混元 3）：

- 镜像 MaxText 官方 `tpu_pre_training` 0.2.5，按 digest 钉死（`models/hunyuan3/frameworks/maxtext/run.sh`）。
- 代码 `maxtext` 的 `hunyuan` 分支，提交号写进 `meta.json`。
- 配置逐项对 GPU：`models/hunyuan3/frameworks/maxtext/hy3-4layers.yml`（每行注明 GPU 一侧的值与出处）。
- 对齐阶段不加性能类 libtpu 参数（`LIBTPU_INIT_ARGS` 默认空）。
- libtpu 0.0.50 / 0.0.50.1 有已知的 scatter 正确性问题（会丢有效更新），0.0.51 已修；对齐实验不要用这两个版本。
- v5p 上实测 libtpu 0.0.46、0.0.48、0.0.51、0.0.52 都有一个 jit 静默算错的编译器 bug（0.0.47、0.0.49 没测）：含 DUS 的 kernel 和读同一缓冲的 kernel 被合进一颗多输出 fusion 时，后者的输出可能算丢。nightly 0.0.53.dev20261106 起已修，正式版截至 2026-11-11 没有。存档的 292 份训练步程序扫出 0 处，覆盖混元 3 截 4 层与混元 4 mini 存了 HLO 的运行；mini 现行比较 `tz-off-rep` 与推荐组合 `x5-prod` 没存 HLO，在 CPU 上按 `meta.json` 重编后扫，也是 0 处（离线程序与上卡程序是同一份是推断）。混元 3 正式被测不在内，生产尺寸没扫（[案例 16](../../models/hunyuan4/cases/16-jit-miscompile-sub8-row-block.md)）。换版本、改切分后用 `models/hunyuan4/analysis/hlo_mof_dus_scan.py` 扫一遍训练步程序；修法见[差异清单 F8](tpu-vs-gpu-differences.md#f8-jit-静默算错dus-和读同一缓冲的算子被合进一颗多输出-fusion)。
- 换 libtpu 版本不一定只换求和顺序：新版本可能改默认行为（混元 4 上 0.0.46 → 0.0.52 就改了 scatter 插排序的门槛，并让「RMSNorm 多舍一次」的影响变大）。换版本先重建 golden、把已知敏感的改动重测一遍，结论只对实测过的版本对成立（`models/hunyuan4/alignments/04-tpu-self/libtpu-052.md`）。libtpu 0.0.52 没有配套的官方 jax；在 jax 0.11.1 的镜像里只换 libtpu 能正常训练（v5p、v7x 实测）。
- 多主机（v5p 2x2x2 两台主机）的 Pod 必须 `hostNetwork: true` + `dnsPolicy: ClusterFirstWithHostNet`，否则 `jax.distributed` 初始化卡住。

下面的原则与待定项仍然有效。已实测的开关和取值写进对齐实践的 `setup.md`，这里只留跨模型通用的。

### 原则

- **同程序重跑必须逐位相同**。这是 TPU 侧一切比较的前提。
  - 不逐位，先查是不是有随机性没钉住（数据顺序、RNG）。
  - 测试模型上已有旁证：
    - 跨机器原样重跑 3000 步 loss 逐位相同；
    - 11 个臂各重跑 500 步逐位相同。
  - 每个模型、每个平台仍要自己实测一次。
- **开关两边语义对齐**。一边开、一边关的开关，按「配置不一致」处理，不按「精度差」处理。
- **excess precision 保持默认开，要照 GPU 舍入点的地方局部卡住**。
  - 编译器可能在某些地方用比声明更高的精度算（excess precision），这会让 TPU 和 GPU 的舍入位置不同。
  - 但不要用 `--xla_allow_excess_precision=false` 全局关掉：关掉后不是更像 GPU，而是更远。混元 4 上，v5p 那次排好序的词嵌入 scatter 从「FP32 里加完只舍一次」退回「加一次舍一次」，TPU 自比曲线 r 13.3，第 1 步词嵌入梯度离 FP64 是 golden / GPU 里较远那个的 26.1 倍（libtpu 0.0.46；0.0.52 上 r 12.6、R 26.1，[TPU 自比「关掉编译器的省舍入」](../../models/hunyuan4/alignments/04-tpu-self/impacts/02-excess-precision.md)；[差异清单 F1](tpu-vs-gpu-differences.md#f1-xla-默认允许-excess-precision)）。
  - 要固定某一处的舍入，用 `jax.lax.reduce_precision`，它不会被 excess precision 优化掉；`astype` 来回转可能会被消掉（[差异清单 A6](tpu-vs-gpu-differences.md#a6-编译器省掉了-gpu-一定会做的-bf16-往返后端省长跑归因见-u12)）。
- **只改执行方式的开关，是噪声参照的来源**。
  - remat 档位、编译器片上内存上限、调度类开关如果确实不改数学，就拿来做扰动臂；
  - 改不改数学，要先用「同程序、只改这一个开关」验证。

### 测试模型上的答案（换到被测模型要再核）

表的右边一栏是此前排查在测试模型上已经给出的答案；注里另写了混元 3 / 4 上实测到的结论，哪条在被测模型上核过就写在哪条下面。

- 此前排查的答案都是缩小模型、单次运行的结论，换到被测模型要按本仓库的检查再核一次。
- 每个模型对这些答案的采用情况写在它的配置对照里（混元 3：`models/hunyuan3/config-alignment.md`）。

| 项 | 要确认什么 | 此前排查已有的答案 |
|---|---|---|
| 矩阵乘默认精度 | 默认精度下 BF16 / FP32 输入的累加方式；与 GPU 的路由 GEMM（BF16 入、FP32 出）怎么对 | 见注 1 |
| excess precision 类开关 | 名字、默认值、关掉后是否逐位改变结果 | 见注 2 |
| RNG 相关开关 | 哪些开关影响 dropout / 初始化的随机数；对齐时 dropout 为 0，是否还有其他随机消费者 | 见注 3 |
| collective matmul 等融合通信 | 是否改变归约顺序 | 改变，见注 4 |
| 注意力 kernel 的实现选项 | 与 GPU 侧 softmax 的计算方式（如以 2 为底的指数）是否一致 | 见注 5 |
| 扰动臂候选 | 哪些开关只改执行方式，能当噪声参照 | 见注 6 |
| 版本 | 镜像 digest 只钉基础镜像（jax / jaxlib 随镜像）；libtpu 可在同一 digest 上换装，以构建号和 `libtpu_pip` 为准 | 换版本不保证逐位，见注 7 |

注 1（矩阵乘默认精度）：
- FP32 操作数在 `DEFAULT` 下只算一遍 BF16：相对误差约 2.3e-3，`HIGHEST` 约 1e-7。
- 路由 logits 因此只有 BF16 级精度，top-k 集合与 FP64 不同的 token 约 1%–2%，`HIGHEST` 为 0。
- 逐位是否等于「两个操作数先舍 BF16 再乘」还没测。

注 2（excess precision 类开关）：
- `XLA_FLAGS=--xla_allow_excess_precision=false`，默认开。
- 关掉后结果逐位改变。
- 单个算子关掉后多与逐算子执行逐位相同，但**不更准**（有些算子默认更接近正确舍入）。
- 它让前向与重算、DP 与 FSDP 的舍入位置不同。
- 改的是舍入位置，不能当扰动臂。
- 混元 4 实测：整网关掉明显变坏（v5p 曲线 r 13.3、词嵌入梯度 R 26.1），保持默认开（见上文「原则」）。

注 3（RNG 相关开关）：
- 不加 `--xla_tpu_spmd_rng_bit_generator_unsafe=false` 时，随机初始化的权重随并行方式变。
- MaxText 写死 `unsafe_rbg`，各平台随机位不同。
- 两边载入同一份检查点、dropout 为 0 时都不起作用。

注 4（collective matmul 等融合通信）：
- FSDP 下 all-gather 与 matmul 重叠，单个线性层 37.98% 元素与不重叠时不同。
- 关掉用 `--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000`。
- 另外，跨算子融合本身也会改求和（注意力后 RMSNorm 被融进输出投影的输出端），见 `levels.md`「隔离融合与逐算子执行」。

注 5（注意力 kernel 的实现选项）：
- Tokamax Splash 默认 `sa_use_base2_exp=True`。混元 4 上查明主效应是常数：内核写 `q *= LOG2E`，q 是 BF16，常数跟着舍成 1.4453125（比 log2 e 大 1.81e-3），注意力 logits（不含 sink）整体乘 1.0018，另加 q 多舍一次 BF16。TPU 一侧设 `False`（GPU 没有这个开关）；混元 4 对齐配置 2026-11-11 起也关（[差异清单 A11](tpu-vs-gpu-differences.md#a11-splash-的-base2-指数把注意力-logits-放大-10018-倍q-多舍一次-bf16生产与对齐都关)、[混元 4 根源一览第 24 行](../../models/hunyuan4/ROOT-CAUSES.md#r24)）；
- 旧 Splash 的 softmax 在 FP32 里算；
- dot-product 路径的 softmax 默认 BF16，要 `float32_logits`。

注 6（扰动臂候选）：
- 测试模型用过：
  - 多加调度类参数；
  - 改片上内存上限；
  - 换 remat 档位（路由结果保存时）；
  - 换注意力或 MoE kernel。
- 只改片上内存上限，MoE 模型也会在某一步起跳、留下固定偏移，量级与各 CP 方案相同。
- 每个臂仍要过 `regression.md` 第五节的三关。

注 7（版本）：
- v7 上换 jax 版本，梯度范数相对差最大 1.2e-4；
- 每次升级重跑验收。

## 两边都要统一的配置

这些不是精度问题，是「两边跑的是不是同一个训练」。逐项列进 `config_diff` 的映射表：

- dropout（必须等于 0，两边都断言）、随机路由类开关；
- Adam β1 / β2 / eps、权重衰减（**实际生效值**，注意调度器会覆盖）及其作用范围（一维参数通常不衰减）；
- 学习率曲线（预热、衰减方式）、梯度裁剪阈值；
- 初始化：两边载入同一份 checkpoint，不各自随机初始化；
- 数据处理约定：分词器、packing、EOS 是否计 loss、loss 的平均方式、数据顺序与种子；
- 路由：打分函数、top-k、路由计算精度、expert_bias 更新率；
- MTP、FP8 是否开启；
- 主权重、优化器状态、梯度通信的 dtype。

混元 3 两边配置的逐项对照：

- TPU 配置文件 `models/hunyuan3/frameworks/maxtext/hy3-4layers.yml` 每行注明 GPU 一侧的值与出处；
- `config_diff` 的映射表在 `models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-tpu.yaml` 的 `L0.config.diff`。
