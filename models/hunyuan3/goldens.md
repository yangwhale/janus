[janus](../../README.md) › [用户指南](../../docs/guide/README.md) ｜ golden 出处总表；对应指南[第 2 章 制作 GPU golden](../../docs/guide/02-gpu-golden.md)（[2.2 为什么是生产程序只修一处](../../docs/guide/02-gpu-golden.md#22-为什么最终-golden-是生产程序只修一处)、[2.3 要造哪几组](../../docs/guide/02-gpu-golden.md#23-要造哪几组-golden怎么跑)、[2.4 噪声带与基线](../../docs/guide/02-gpu-golden.md#24-噪声带与基线)）与[第 4 章 4.2 怎么判](../../docs/guide/04-level-by-level.md#42-怎么判)

# 混元 3 · GPU golden 是怎么产生的（出处总表）

这一页记录 GPU 侧（B200 × 8，Megatron-Bridge）每一份 golden 的出处：用了什么设置，改过哪些代码，为什么改，以及每处改动在 GPU 和 TPU 两边各是什么机制。

两边配置的逐项对照在 [`config-alignment.md`](config-alignment.md)。这一页只从 golden 的角度说明最终 golden 和上游默认差在哪里、每一处差为什么需要。
各 golden 代号怎么读见[术语表](../../docs/glossary.md#golden-codenames)。

本页先给最终 golden 一览和出处约定，然后是按类别分节的主表（长说明放在各表下面的「注」里），之后依次是 TPU 侧开关对照、比较协议，最后是从 g1k 到 rcc 的演变。

### 最终 golden 一览

- **最终 golden 是 `rcc`**（2026-11-06 定）。
  - 它是 GPU 生产程序 [`rc`](../../docs/glossary.md#rc)（BF16 混合精度、整层重算 + 路由回放）只加一处。
  - 加的这一处：输出头 dgrad（dh = dlogits · W，沿词表 120832 求和）改成分段 FP32 累加（[`rr_arms.py`](frameworks/megatron/rr_arms.py) 的 `dgrad-chunk`）。
  - 定义见术语表 [`rcc`](../../docs/glossary.md#rcc)、[最终 golden](../../docs/glossary.md#final-golden)。
- 为什么只改这一处：在 L4（1000 步）上，GPU 生产与 TPU 生产之间唯一测得出的系统偏差，就是这一次长累加在 B200 上的偏差。第 1 步还有几处测得出、但不累积的差（路由近似平局等，见对齐报告台账 #5、#20；#6 只在打开 `dense_fp32_accumulation` 之前的配置里）。BF16 张量核朝 0 截断，改成 cuBLAS FP32 一次累加也还偏。修掉它以后其余全是生产程序本身，GPU 与 TPU 的生产程序在 GPU 自己重跑的噪声内重合。整体结论见 [`ALIGNMENT-REPORT.md`](ALIGNMENT-REPORT.md)。
- 其余代号（见文末「从 g1k 到 rcc」）：
  - 迭代过程中的几步：
    - [`g1k`](../../docs/glossary.md#g1k)（第一阶段 GPU golden，不开重算，1000 步，3 份）；
    - `rc`，即上面说的 GPU 生产程序；
    - [`rcx`](../../docs/glossary.md#rcx)（生产 golden，输出头 dgrad 换 FP32 一次累加）。它是归因用的中间版，FP32 一次累加本身仍有偏。
  - 参考交叉核对：
    - [`rce`](../../docs/glossary.md#rce)（精确轨迹，BF16 权重、全程 FP32 计算）。排查时当裁判用，但输出头 dgrad 仍是有偏的 cuBLAS FP32 一次累加，不是真精确；
    - [`rcec`](../../docs/glossary.md#rcec)（真精确轨迹，精确轨迹加输出头 dgrad 分段累加）。

### 运行、meta 与验收目录

- 运行：[`rcc-a`](../../docs/glossary.md#rcc-a) 是比较[基线](../../docs/glossary.md#baseline)。
  - 噪声带是 `rcc`（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加）自己同设置重跑的各对：`rcc-a..j` 10 份两两配对，45 对。
  - GPU 生产程序本身不确定，重跑就是噪声样本，不加扰动。这是一般规矩，没有例外。
  - 此前（2026-11-05 晚到 11-06 早）的做法已取消，见下面「比较协议」：
    - 以 `rce-a`（精确轨迹，BF16 权重、全程 FP32 计算）为基线；
    - 噪声例外地取 `rcx-a..e`（生产 golden，输出头 dgrad 换 FP32 一次累加）5 份两两 10 对。
- meta：`rcc-a..j` 的 `meta.json` 都在 `gs://<POC 桶>/janus/hunyuan3-4l/golden/` 下。
  - `config.args.rr_arm: dgrad-chunk`
  - `recompute: full / uniform / 1`
  - `moe_enable_routing_replay: true`
  - `dtypes.compute: bf16`
  - `attention_backend: auto`
  - TF32 是容器默认（开）
  - `fault: null`
  - 训练里的自检（rank 0 前 4 次调用）：分段 FP32 的 dh 对 FP64 rel_L2 6.9e-7 – 7.7e-7。
- 验收目录：
  - v5p：[`phase2-acceptance-rcc-on`](alignments/01-v5p-acceptance/catalogs/phase2-acceptance-rcc-on.yaml)
  - v7x：[`phase2-v7x-acceptance-rcc`](alignments/02-v7x-acceptance/catalogs/phase2-v7x-acceptance-rcc.yaml)

### 出处与栏目约定

出处约定：

- 「meta」指运行目录里的 `meta.json`（`gs://<POC 桶>/janus/hunyuan3-4l/golden/<运行>/meta.json`），由 [`frameworks/megatron/train_hy3.py`](frameworks/megatron/train_hy3.py) 写出。
- 「Bridge 默认」指 Megatron-Bridge `v0.7.0` tag 的 `recipes/common.py::_pretrain_common` 和 `recipes/utils/optimizer_utils.py::distributed_fused_adam_with_cosine_annealing`。
  - 容器里的 Megatron-Bridge 报的版本也是 0.7.0（meta 的 `versions`），但容器里的文件没有逐字对过 tag。
- 「provider」指 AutoBridge 从 HF `config.json`（`tencent/Hy3` revision `a960ebc3…`，截 4 层）生成的 Megatron 模型配置。取值从 meta 的 `model_config` 读。
- 「案例 01」指 [`alignments/01-v5p-acceptance/cases/01-loss-offset-param-norm.md`](alignments/01-v5p-acceptance/cases/01-loss-offset-param-norm.md)。
- 推出来的结论标「推断」。查不到的标「待查」。

结论栏的取值：

- 采用（最终 golden `rcc` 用了它）
- 同原始（没改）
- 试过，无影响
- 试过，有害
- 只作诊断（改了数学，或者只记录，不能进 golden）

方向栏的取值：

- GPU 向 TPU（GPU 改成 TPU 本来的做法）
- TPU 向 GPU
- 两边向精确值（两边都朝 FP64 或 FP32 精确值靠）
- —（两边本来一样，或者跟对齐无关）

## 主表：最终 golden `rcc` 对上游默认的全部改动，以及试过的开关

各表格里只放短结论；「见注 N」的长说明紧跟在该表下面。

### 环境、版本、代码

| 类别 | 参数或代码位置 | 原始值（出处） | 最终 golden `rcc` | 结论 | 方向 | 机制（GPU / TPU）、为什么、影响与证据 |
|---|---|---|---|---|---|---|
| 环境 | 镜像 | `nvcr.io/nvidia/nemo:26.10`（[`frameworks/megatron/README.md`](frameworks/megatron/README.md)） | 同原始。digest `fc75bf516de6…`（出处见注 1） | 同原始 | — | [`g1k-*`](../../docs/glossary.md#g1k)（第一阶段 GPU golden）的 meta 只记了 tag（见注 1） |
| 环境 | 版本 | 容器自带 | 同原始，各组件版本见注 2 | 同原始 | — | Megatron 的源码一行没改，全部改动都是运行时补丁或配置（见下面的「代码补丁」） |
| 代码 | janus 训练脚本 | — | `train_hy3.py`、`rr_arms.py` 逐文件核对过 sha256（见注 3） | 采用 | — | 前几份 golden 的脚本版本见注 3 |
| 环境 | 节点 NCCL、插件、Triton 缓存 | 节点挂进来的是 NCCL 2.27；容器默认开 GCP NCCL 插件 | [`frameworks/megatron/env.sh`](frameworks/megatron/env.sh)：`LD_LIBRARY_PATH` 前置容器里的库、`NCCL_*_PLUGIN=none`、`TRITON_CACHE_DIR` | 采用 | — | 只为了让程序跑得起来，不改数值（frameworks README「环境上的三个坑」） |

- 注 1（镜像）：
  - digest 出自下面几组运行的 meta：
    - [`rc`](../../docs/glossary.md#rc)（GPU 生产 golden）；
    - [`rcx`](../../docs/glossary.md#rcx)（生产 golden，输出头 dgrad 换 FP32 一次累加）；
    - [`rce-a/b`](../../docs/glossary.md#rce)（精确轨迹，BF16 权重、全程 FP32 计算）；
    - [`rcc-a..j`](../../docs/glossary.md#rcc)（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加）。
  - `g1k-*` 的 meta 只记了 tag，Megatron 版本记成 `null`。
  - 原因是当时取法有误，已经修好（frameworks README 第 6 步）。
- 注 2（版本，出自 `rcx-a` meta）：
  - torch `2.14.0a0+b2c75dd062.nv26.09`
  - CUDA 13.4
  - Megatron-Core 0.20.0
  - Megatron-Bridge 0.7.0
  - TE `2.20.2+44f31bbe`
  - NCCL 2.31.2
- 注 3（janus 训练脚本）：
  - `train_hy3.py` sha256 `2de476cb…`，和 `main` 的 `8ecc2e9` 逐字节相同。
  - `rr_arms.py` 有两版，`dgrad-chunk` 的代码相同：
    - `rcc-a..e` 用的是 sha256 `df4647ab…`；
    - `rcc-f..j` 用的是 `3e7c53da…`（只多一个没开的诊断臂 `dlogits-bf16`）。
  - Pod 上那份 janus 不是 git 检出，meta 的 `code` 只记了容器。以上是逐文件核对的 sha256。
  - 前几份 golden 的脚本：
    - `g1k-*` 是 `4e41f0ae…`（运行时读的是一份本地拷贝），和提交 `b09982e` 里的 `models/hunyuan3/gpu/train_hy3.py`（当时的路径）逐字节相同；
      和 #143 版（`8ecc2e9`）比，后者多的是默认关的诊断参数（路由回放、观测点、关 TF32、各种诊断臂、参数导入导出等）和运行信息的记法（镜像、版本号），默认训练路径没见改动（读 diff 所得，没有上卡对比）；
    - `rc-*` 是 `09b213f9…`（`03/runs.yaml`，janus `24649e4b`）；
    - `rcx-*` 是 `ee626092…`，和 `b6c4c57` 一致。

### 模型结构

| 类别 | 参数或代码位置 | 原始值（出处） | 最终 golden `rcc` | 结论 | 方向 | 机制、为什么、影响与证据 |
|---|---|---|---|---|---|---|
| 结构 | 层数、MTP | HF 全模型，带 MTP | 截前 4 层，去掉 MTP（[`truncate_hy3.py`](model/truncate_hy3.py)：`num_nextn_predict_layers=0`；`train_hy3.py` 设 `mtp_num_layers = None`） | 采用 | — | 只为把规模缩小到能在单机上做对齐，两边截法相同。清单 sha256 `183274dbf17e…`（meta `checkpoint.manifest_sha256`） |
| 结构 | 其余结构参数 | provider | 同原始，取值见注 4 | 同原始 | — | 两边逐项对照见 [`config-alignment.md`](config-alignment.md)「模型结构」 |

- 注 4（其余结构参数，出自 [`rcx-a`](../../docs/glossary.md#rcx) meta 的 `model_config`；`rcx` 是生产 golden，输出头 dgrad 换 FP32 一次累加）：
  - `hidden_size` 4096
  - 注意力头 / KV 头 / 每头维度 64 / 8 / 128
  - 稠密 FFN 13312
  - 专家 192，top-8；专家 FFN 1536；共享专家 1536
  - 路由打分 sigmoid + expert bias（更新率 0.001），top-k 缩放 2.826
  - 辅助损失 0；dropout 0
  - RMSNorm eps 1e-5；qk norm
  - RoPE base 11158840

### 精度：最终 golden 的核心改动只有一处

代号：`rcc-a`（最终 golden 的第 1 份，验收的比较基线）、`rce`（精确轨迹，BF16 权重、全程 FP32 计算）、`rcec`（真精确轨迹，精确轨迹加输出头 dgrad 分段累加）。

| 类别 | 参数或代码位置 | 原始值（出处） | 最终 golden `rcc` | 结论 | 方向 | 机制（GPU / TPU）、为什么、影响与证据 |
|---|---|---|---|---|---|---|
| 精度 | 输出头 dgrad（dh = dlogits · W，收缩长度 = 词表 120832）：[`rr_arms.py`](frameworks/megatron/rr_arms.py) 的 `dgrad-chunk` 臂 | 生产路径：BF16 dlogits × BF16 参数，在 B200 张量核上一次乘完 | 升到 FP32，词表按 1024 一段分开乘，段间 FP32 累加，舍回 BF16（见注 5） | 采用（唯一的数值改动） | GPU 向精确值，也就是向 TPU | 见注 5 |
| 精度 | 计算精度 | `bf16_mixed`（Bridge 默认 `mixed_precision`） | 同原始（[`rcc-a`](../../docs/glossary.md#rcc-a) meta `dtypes.compute: bf16`） | 同原始 | — | 精确轨迹 [`rce`](../../docs/glossary.md#rce) / [`rcec`](../../docs/glossary.md#rcec) 用 `--fp32 --bf16-weights` 把算术换成 FP32，作参考交叉核对，见下面「参考：精确轨迹多改的」 |
| 精度 | 权重语义 | BF16 参数 + FP32 主权重 | 同原始 | 同原始 | — | — |
| 精度 | TF32 | 开：容器设了 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，FP32 GEMM 实测对 FP64 rel_L2 2.94e-4 | 同原始（rcc-a meta `tf32.probe_rel_l2_*` 2.94e-4）；`dgrad-chunk` 在它那一次乘法里临时设 `allow_tf32 = False` | 同原始 | — | 关 TF32 跑 400 步看不出影响（1000 步没跑）；`dgrad-chunk` 里为什么要关，见注 6 |
| 精度 | 注意力实现 | `AttnBackend.auto` | 同原始 | 同原始 | — | BF16 设置下换实现无影响（见下面「试过」） |
| 精度 | 梯度规约精度：`ddp.grad_reduce_in_fp32` | True（Bridge 默认） | 同原始 | 同原始 | — | 反过来改成 BF16 规约（`--grad-reduce-bf16`，模仿 TPU 上部分梯度按 BF16 规约）试过，无影响，见下面「试过」 |
| 精度 | 路由 GEMM：`moe_router_dtype` | `fp32`（provider；meta）。在这版 Megatron 里走 TE `general_gemm`，BF16 输入、FP32 输出 | 同原始 | 同原始 | — | 只在近似平局的 token 上翻转 top-k，1000 步统计在噪声内（见注 7） |
| 精度 | 流进输出头反向的 dlogits | 融合交叉熵写出 BF16 梯度 | 同原始 | 同原始 | 两边本来一样 | 两侧生产程序都舍成 BF16，不构成 GPU↔TPU 差（见注 8） |

- 注 5（输出头 dgrad）：
  - `rcc` 的写法：
    - 同一份 BF16 dlogits 和 BF16 参数升到 FP32；
    - 词表按 1024 一段分开乘（关 TF32 的 CUDA 核 FP32 乘法）；
    - 段间 FP32 累加，结果舍回 BF16；
    - rcc-a 训练自检：对 FP64 rel_L2 6.9e-7 – 7.7e-7。
  - GPU（生产）：张量核累加截断。
    - 每 16 个乘积连同累加器一起对齐到最大指数，低位向零截掉，写回 FP32 也向零舍入。
    - 这个模型逐位复现路由 logits 三层全部元素（[`../../docs/dev/design/0003-l1-op-replay.md`](../../docs/dev/design/0003-l1-op-replay.md) 第 4 条）。
    - 截掉的总是低位，在长求和里积成朝 0 的系统偏差，随 K 增长。
    - 合成输入 K = 256 / 4096 / 65536：2.2e-5 / 7.0e-5 / 3.1e-4（[`gemm_accum_bias.py`](analysis/gemm_accum_bias.py)）。
    - 输出头 dgrad 第 1 / 100 步沿 ŵ̄（所有 logits 一起升降的方向）偏 −6.51e-4 / −6.10e-4。
  - GPU（cuBLAS FP32 一次累加，即 `dgrad-exact`、`rce` 的写法）：关 TF32 也还偏。
    - 对 FP64 的 rel_L2 是 CPU FP32 的 59 – 129 倍。
    - 沿 ŵ̄ 第 1 步 −1.8e-6、θ₂₉₉（第 300 步开始时的状态）−2.6e-4。
    - 容器里能找到的开关（`fp32_precision=ieee`、去掉 TF32 override、cuBLAS 后端与 emulation 设置）都不改变结果，只有分段能去掉（[`case01-cublas-fp32-longk.json`](alignments/01-v5p-acceptance/results/case01-cublas-fp32-longk.json)）。
  - 消融：`rcc` 去掉偏差靠的是分段，不是 FP32（案例 01「输出头 dgrad 消融：分段还是 FP32」行）。
    - 同一份训练实际张量（第 1 / 100 / 300 步），沿 ŵ̄：FP32 一次乘完 −1.7e-6 / −2.6e-4 / −2.6e-4，不分段就去不掉。
    - BF16 张量核按 1024 分段、段输出 FP32、段间 FP32 累加：−6.9e-7 / +8.3e-7 / +3.1e-7，和 `rcc` 的 FP32 分段（+1e-8 / +3.5e-7 / +1.1e-7）同一量级；段输出先舍 BF16 就不行。
    - 单次乘法耗时：生产 2.45 ms，BF16 分段 7.2 ms，`rcc` 71.7 ms（[`case01-dgrad-accum-ablation.json`](alignments/01-v5p-acceptance/results/case01-dgrad-accum-ablation.json)）。
  - 便宜的等价修法 `rcb`（`rr_arms.py` 的 `dgrad-chunk-bf16`：同样按 1024 分段，但每段是 BF16 张量核乘法、输出留 FP32，段间 FP32 累加）：
    - `rcb-a` 就是 `rcc-a` 的设置只换这一个臂，1000 步对 `rcc-a`、用 `rcc` 的 45 对噪声判，L4 八项全过，最差 r 0.91（[`phase2-rcb-vs-rcc`](alignments/01-v5p-acceptance/results/phase2-rcb-vs-rcc.report.md)）；第二份 `rcb-b` 同样全过。
    - 步时 0.250 s，对生产 `rc` 0.236 s 多 6%；`rcc` 0.381 s 多 61%。
    - 正式基线不变，仍是 `rcc-a`；要更便宜地造同等的 golden，可以用 `rcb`（案例 01「便宜的等价修法 `rcb`」行）。
  - TPU：MXU 吃 BF16 输入，乘积精确，在 FP32 里累加。
    - 输出头回放沿 ŵ̄ 第 1 / 300 步 −2.6e-5 / −4.5e-6（生产）、−1.2e-6（精确）。
  - 影响：
    - 第 1000 步 param_norm 对 `rc`（GPU 生产 golden）五份均值：`rcc-a` +2.07e-2，TPU 生产 +2.09e-2。
    - 301–1000 步 loss 相对 `rc` 均值：`rcc-a` −1.71e-4，TPU 生产 −1.37e-4（v5p）/ −1.07e-4（v7x，[`v7x-tpu-vs-rc.json`](alignments/02-v7x-acceptance/results/v7x-tpu-vs-rc.json)）。
    - 原因链见案例 01「一页看完」：「所有 logits 一起升降」那股力推动 logits 整体水平，冷门词因此进入 Adam ε 区，最后体现在输出层冷门行的范数上。
- 注 6（TF32）：
  - BF16 程序里 FP32 输入的 GEMM 只有路由反向两个，关 TF32 跑 400 步看不出影响（见下面「试过」的 `g-notf32`，即 BF16 设置下关 TF32 的 GPU 诊断运行；1000 步没跑）。
  - `dgrad-chunk` 的乘法是 FP32 输入，所以在那里关掉。
  - 开着的话 FP32 乘法会进 TF32 张量核，沿 ŵ̄ 反而偏 −8.1e-4（比 BF16 还多）。
- 注 7（路由 GEMM）：
  - 路由 logits 对 FP64：GPU 3.8e-6（张量核截断），TPU 4.8e-8（`config-alignment.md`「路由 GEMM」行）。
  - 差别只在近似平局的 token 上翻转 top-k（第 1 步每层 0.44% – 0.56% 的槽位）。
  - 1000 步统计在噪声内。
- 注 8（流进输出头反向的 dlogits）：
  - 两侧生产程序都把 dlogits 舍成 BF16（TPU：logits 在 BF16 出）。
  - 它让漂移比真精确少约 6.7e-3：
    - GPU：精确轨迹多舍这一次，`g-bwcr-1k`（默认配置精确轨迹加分段 dgrad，再把 dlogits 舍 BF16）−6.74e-3；
    - TPU：只舍 dlogits，−6.84e-3，只经 dgrad 那一路。
  - 两侧一样，不构成 GPU↔TPU 差（案例 01「GPU 一侧的同一件事」「TPU 那一份再拆」两行）。
  - 想要两边都更接近精确值，两侧各自让 dlogits 留在 FP32，见 ALIGNMENT-REPORT「开关表」。

### 优化器与学习率

| 类别 | 参数或代码位置 | 原始值（出处） | 最终 golden `rcc` | 结论 | 方向 | 机制、为什么、影响与证据 |
|---|---|---|---|---|---|---|
| 优化器 | 算法 | `adam`，解耦权重衰减，TE FusedAdam（Bridge 默认） | 同原始 | 同原始 | — | 用 optax `adamw` 逐式重写的臂（`adam-optax`）试过，无影响，见下面「试过」。GPU 侧的 FP64 重放过了（00 `L1.optim.adamw.replay`） |
| 优化器 | β1 / β2 / ε | 0.9 / 0.95 / 1e-8（Bridge 默认） | 同原始（meta `optimizer_config`） | 同原始 | — | 冷门行落在 Adam ε 区，对一点点梯度偏差都很敏感（见注 9） |
| 优化器 | 权重衰减 | 配置 `weight_decay: 0.1`；调度器的设置见注 10（Bridge 默认） | 同原始：实际生效的是 0.033（meta `scheduler_config`） | 同原始 | 两边设成同一个实际值 | 0.1 只是摆设；只读配置的话会对错超参（见注 10） |
| 优化器 | 梯度裁剪 | `clip_grad: 1.0`（Bridge 默认） | 同原始 | 同原始 | — | 第 1 步就会触发裁剪（裁剪系数 0.057，`config-alignment.md`）。optax 写法的臂（`clip-optax`）试过，无影响 |
| 学习率 | 调度 | cosine，warmup 500 步，`max_lr` 3e-4，`min_lr` 3e-5（Bridge 默认） | 常数 1e-5：`lr = min_lr = 1e-5`、`lr_warmup_iters = 0`、`lr_decay_style = constant` | 采用 | 两边设成相同 | 1000 步的对齐里不让调度器成为变量；TPU 用 `wsd` 加 0 warmup 和 0 decay 得到同一个常数（30 步逐位，01 `phase2-t-loss30` 的 `L4.lr`） |
| 优化器 | 分布式优化器：`use_distributed_optimizer`、`data_parallel_sharding_strategy` | True、`optim_grads_params`（Bridge 默认） | 关、`no_shard`：`--full` 和 `--no-dist-opt` 都会强制关掉 | 采用 | — | 会改数值（见注 11） |
| 优化器 | 通信重叠：`overlap_grad_reduce`、`overlap_param_gather` | True、True（Bridge 默认） | False、False | 采用 | — | 推断：只改通信和计算的时间安排，不改规约的分组。没有单独测过它对数值的影响 |

- 注 9（β1 / β2 / ε）：冷门行为什么对一点点梯度偏差都很敏感（Adam ε 区）。
  - 输出层里没在数据中出现过的词，那些行的梯度只有 1e-8 到 4e-9，跟 ε = 1e-8 同量级或更小（案例 01 第 6 节 4a′）。在这个区间里 m̂ / (√v̂ + ε) 不再被归一成 ±1 量级，每一步的更新几乎和梯度成正比。因此梯度里每一步的系统偏差（比如输出头 dgrad 的截断）不会被 Adam 的归一化抹掉，而是原样积累进这些行的范数。
  - 敏感度实测：只把输出层梯度放大 1.25 倍，第 300 步输出层平方和就 +14.6，增量几乎全落在没出现过的词上（案例 01「敏感度」行）。
- 注 10（权重衰减）：
  - 调度器设置（Bridge 默认）：`start_weight_decay = end_weight_decay = 0.033`，`weight_decay_incr_style: constant`，`override_opt_param_scheduler=True`。
  - Megatron 的调度器每一步用 `start/end_weight_decay` 覆盖优化器的 `weight_decay`，所以 0.1 只是摆设（00 README 第 10 条）。
  - TPU 那边写的是 `adam_weight_decay: 0.033`。
  - 只读配置的话会对错超参。
- 注 11（分布式优化器）：
  - 第一阶段 9 次运行实测，「分布式优化器开 + 默认设置」这一组对其余几组有持续偏移。
  - \|均值\| / 标准误最大到 9.1（00 README 第 6 条；[`../../docs/reference/checks/multi_golden.md`](../../docs/reference/checks/multi_golden.md)）。
  - 全量抓数要每个 rank 上都有完整的已归约梯度，所以 golden 全部关掉。
  - 噪声参照也必须用同样关掉的运行。

### 数据

| 类别 | 参数或代码位置 | 原始值（出处） | 最终 golden `rcc` | 结论 | 方向 | 机制、为什么、影响与证据 |
|---|---|---|---|---|---|---|
| 数据 | 序列长度 | 4096（Bridge 默认） | 同原始 | 同原始 | — | — |
| 数据 | 全局批 / microbatch / 梯度累积 | 32 / 2（Bridge 默认） | 16 / 1；DP 8，所以每个 rank 2 个 microbatch，分两次前反向再累加 | 采用 | TPU 向 GPU（TPU 改成 GA 2） | 累加怎么分组决定了舍入发生在哪里（见注 12） |
| 数据 | 数据集 | Mock 数据，`split` 9999,8,2（Bridge 默认） | fineweb-edu `fw_hy3_text_document`（`.idx` sha256 `83d5a23b14d3…`），`split` 9999,1,0 | 采用 | TPU 向 GPU（TPU 原样回放 GPU 的 batch） | Mock 数据会被记住，不能用来量噪声；TPU 原样回放 GPU 存下的输入（见注 13） |
| 数据 | 种子 | 1234（Bridge 默认 `dataset.random_seed`、`rng.seed`） | 同原始 | 同原始 | — | 权重从检查点载入，dropout 是 0，种子只决定数据顺序 |
| 数据 | EOD 与掩码 | `reset_attention_mask`、`reset_position_ids`、`eod_mask_loss` 都是 False（Bridge 默认） | 同原始；预处理时 `--append-eod` | 同原始 | — | [`data/README.md`](data/README.md) |

- 注 12（全局批）：
  - 只换梯度累积的切分，embedding 梯度就差到噪声上限的 31 倍（00 README 第四节）。
  - TPU 那边设 `gradient_accumulation_steps: 2` 和 GPU 对齐（`config-alignment.md`「全局批」行）。
- 注 13（数据集）：
  - Mock 数据会被记住（1000 步 loss 降到 0.005 左右），不能用来量噪声（frameworks README 第 6 步）。
  - [`rcc-a`](../../docs/glossary.md#rcc-a)（最终 golden 的第 1 份，验收的比较基线）带 `--save-batches`，把每个 rank 每一步的原始输入存下来。
    - 此前的 `g1k-a`（第一阶段 GPU golden）、`rc-a`（GPU 生产 golden）、`rce-a`（精确轨迹，BF16 权重、全程 FP32 计算）也带。
  - TPU 原样回放（`config-alignment.md`「数据来源」行）。

### 并行、确定性、重算

代号：`rcc`（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加）、`rcc-a`（最终 golden 的第 1 份，验收的比较基线）、`rce`（精确轨迹，BF16 权重、全程 FP32 计算）。

| 类别 | 参数或代码位置 | 原始值（出处） | 最终 golden `rcc` | 结论 | 方向 | 机制、为什么、影响与证据 |
|---|---|---|---|---|---|---|
| 并行 | EP / TP / PP / CP / DP；token 分发 | 待查（见注 14） | EP 8，TP / PP / CP 1，DP 8；`moe_token_dispatcher_type = alltoall`（`train_hy3.py`） | 采用 | — | 只换 EP 度，反向就已经不逐位，所以 golden 和噪声参照固定同一种切分；TPU 用 FSDP 8（见注 14） |
| 确定性 | `deterministic_mode` 和三个环境变量（`NVTE_ALLOW_NONDETERMINISTIC_ALGO`、`CUBLAS_WORKSPACE_CONFIG`、`NCCL_ALGO`） | False，三个都不设（Bridge 默认） | 同原始（`run.sh` 没收到 `--deterministic`，就把三个变量 unset 掉） | 同原始 | — | 默认设置下重跑自然散开，这就是噪声来源（见注 15） |
| 噪声带 | `rcc-b..j`：与 `rcc-a` 同设置原样重跑，不加扰动（GPU 生产程序的反向有不确定的规约顺序，重跑本身就不同） | — | `rcc-b..e` 2026-11-06 凌晨、`rcc-f..j` 同日早上跑完；meta `fault: null` | 采用 | — | 和 golden 同一种设置，是一般规矩；45 对的最大值见注 16 |
| 参考附录 | rce-b..e 的扰动：`--fault "weight_scale:decoder\.layers\.0\.self_attention\.linear_proj\.weight:<ε>"`（[`../../janus/dump/megatron.py`](../../janus/dump/megatron.py) 的故障注入，记在 meta 的 `fault`） | —（golden 不扰动） | 只用于 `rce-b..e`（ε 取值见注 17）。`rce-a` 不扰动（meta `fault: null`） | 只作参考附录（不进门槛，也不是 golden 本身的设置） | — | 原来打算用它造噪声带，后来没用（见注 17 与下面「比较协议」） |
| 重算 | `recompute_granularity` / `recompute_method` / `recompute_num_layers` | 不重算（Bridge 默认） | `full` / `uniform` / `1`（`--recompute`）：每层单独整层重算，只存层输入 | 采用 | GPU 向 TPU（生产两边都开重算） | 生产训练都开重算；重算设置是假设的（见注 18） |
| 重算 | 路由回放：`moe_enable_routing_replay` 加 [`frameworks/megatron/routing_replay.py`](frameworks/megatron/routing_replay.py)（`--routing-replay`） | 关（Megatron 有这个功能，但训练循环不会切换它的动作，只有推理引擎会设 RECORD） | 开（做法见注 19） | 采用 | GPU 向 TPU | 回放以后，重算和首次前向逐位相同；1000 步的统计看不出变化（见注 19） |

- 注 14（并行）：
  - 原始值待查：Bridge recipe 默认没设；provider 默认没核对。
    - 不影响 golden：golden 实际用的值在「最终 golden」栏，出自 `train_hy3.py` 与 meta `parallelism`。
  - 只换 EP 度，反向就已经不逐位（00 `phase1-parallel`）。所以所有 golden 和噪声参照都固定用同一种切分。
  - TPU 用 FSDP 8，这个差别不可避免。
  - 这个差别占多少（2026-11-05，排查期对参考轨迹 `rce-a` 的读数）：TPU 生产配置只把专家并行度改成和 GPU 相同的 EP = 8（词嵌入表沿 expert 轴切 8 份（读 HLO 核过），其余非专家参数按逻辑轴规则推断同样切，≈ FSDP 8，见注 50）。
    - 对 `rce-a` 的 param_norm 带符号均值，和同平台当前 FSDP 配置比：v5p 少 18%（4.27e-7 降到 3.52e-7），v7x 多 17%（3.19e-7 升到 3.74e-7）；差值 7.5e-8、5.5e-8 都在 GPU 精确五份两两的上限 1.07e-7 以内（案例 01「GPU 精确 + 分段 dgrad：1000 步」行）。这把 1.07e-7 是 `rce-a..e` 的 1e-6 扰动散布，按注 17 比真实执行噪声严，只作参考。
    - 第 1000 步的 param_norm 差（另列，不是上面「少 18%」的口径）：v7x 不变（+2.35e-3），v5p +2.35e-3 → +1.86e-3。
    - EP = 8 本身对 `rce-a` 的 param_norm 带符号均值仍是噪声上限的 2.8–3.0 倍，判不过（案例 01 第 0 节「并行方式」行）：切分方式只解释其中一小部分。
  - 对最终 golden `rcc-a`：EP = 8 的 TPU 在 v5p、v7x 上 L4 各项都在噪声内（`phase2-acceptance-rcc-ep8`、`phase2-v7x-acceptance-rcc-ep8`）。
    - 当前的 FSDP 8 配置见 ALIGNMENT-REPORT。
  - 已知限制：GPU 一侧换成 FSDP 还没做成：
    - Megatron-FSDP 载入检查点时报 `MegatronFSDP` 没有 `sharded_state_dict`（Bridge 要求 FSDP 用 `fsdp_dtensor` 格式的检查点，我们的是 `torch_dist`）；
    - torch FSDP2 在专家权重上报 `Invalid sharding pattern`（EP 与 FSDP 的切分叠在一起，分片模式校验不过）。
  - 后续方案：
    - 非 FSDP 下用 `param_swap.py` 导出整套参数；
    - FSDP 模型建好后按 DTensor 分片灌进去，灌完逐位核对再跑；
    - 因为 TPU 侧已经说明切分方式只占一小部分，这项降为确认性实验。
- 注 15（确定性）：
  - BF16 golden 用默认设置，反向的原子加不确定，所以重跑会自然散开，这就是噪声来源。
  - 全程 FP32 的程序同一设置跑两次逐位相同（根因调查实测，没有单独入库的结果文件）。
    - 所以精确轨迹 `rce` 自己量不出执行噪声。
    - 它的参考附录只能靠 1e-6 扰动造散布（下面「参考附录」行）。
  - 正式噪声带取最终 golden `rcc` 自己的 10 份重跑（`rcc-a..j`，两两配对 45 对，见「比较协议」）。
  - `g1k-det`（第一阶段 GPU golden 的确定性设置版）只用来归因，不和默认设置混在一起当参照。
- 注 16（噪声带）：10 份两两配对 45 对的最大值：
  - param_norm 带符号均值 2.27e-7
  - loss 全程 9.46e-5
  - loss 301–1000 步 1.21e-4
  - 只有 a..e 五份、两两 10 对时是 7.42e-8 / 6.72e-5 / 7.92e-5（案例 01 附录「rcc 的散布」）。
- 注 17（参考附录的扰动）：
  - ε = +1e-6 / −1e-6 / +3e-6 / −3e-6（rce-b meta `fault.eps: 1e-6`；c..e 来自发起命令）。
  - 原来打算用它造噪声带，后来没用（下面「比较协议」）。
    - 精确轨迹是确定性的，不扰动的话几份运行逐位相同，量不出散布。
  - 做法：把第 0 层注意力输出投影的权重乘 (1 + ε)。
    - ε 是 1e-6 量级，比 BF16 半格（2⁻⁹）小三个数量级。
    - 推断：第 1 步舍到 BF16 以后，大部分元素不变，只有贴着舍入边界的少数元素翻转；之后靠 FP32 主权重上的差慢慢把轨迹分开。
  - 这样量出来的是「轨迹对极小扰动有多敏感」，不是「只改执行方式会差多少」。
  - 实测在默认配置下两者同量级：
    - `g-bw-1k`（默认配置、不重算的 GPU 精确轨迹，1000 步）对 `g-bw-1k-p`（同上加 +1e-6 扰动）：第 1000 步 param_norm 差 3e-4，301–1000 步 loss 差 3.7e-5；
    - GPU 默认运行之间 param_norm 最大差 6.6e-4。
  - 但放在 rc（GPU 生产 golden）配置下跑 1000 步、按带符号均值量时就差得多：
    - rce-a 对 rce-b 的 param_norm 只有 6.9e-9；
    - `rcx-a..e`（生产 golden，输出头 dgrad 换 FP32 一次累加）五份两两 10 对的最大值是 1.24e-7，小约 18 倍；
    - 1e-6 的扰动要几百步才分开，前面大半段几乎重合，拿它当尺子会比真实执行噪声严一个数量级。
  - 五份的散布只记在案例 01 当参考（附录「rce 五份的散布」）。
- 注 18（重算）：
  - 生产训练都开重算，TPU 那边是 `remat_policy: custom`。
  - 重算设置是假设的，真实的配置还没拿到（03 `setup.md`）。
  - 只开重算（不回放）时，重算的专家输出和首次前向不逐位（00 `phase1-remat`），原因见下一行。
- 注 19（路由回放）：
  - `rcc` 的做法：每个 microbatch 的首次前向 RECORD，反向重算 REPLAY_BACKWARD。
    - 只回放 top-k 下标（连同顺序）。
    - 路由概率在重算里由重算出的 logits 重新算。
  - GPU：整层重算时，首次前向在 no_grad 里跑，反向之前再建图重跑一遍。
    - Megatron 调的是 `torch.topk(..., sorted=torch.is_grad_enabled())`：首次前向不排序，重算排序。
    - 即使选中的集合相同，顺序也不同，top-k 权重归一化的求和顺序跟着变。
    - probs 差 1 ULP 量级（0.84% 的元素，rel_L2 4.8e-8）。
    - 分数贴得很近时，选中的集合本身也可能变。
    - 回放以后，重算和首次前向逐位相同：57 / 57 个张量，03 [`remat-replay`](alignments/03-gpu-remat-golden/results/remat-replay.report.md)、[`remat-router`](alignments/03-gpu-remat-golden/results/remat-router.rv-det-vs-d-a.json) 的 `topk_order_repro`。
  - TPU：`moe_routing_maps: device` 把 top-k 下标和排序置换跨过重算边界存下来，反向不再重跑 top-k。
  - 影响：改变首次前向（第 1 步 loss 相对差 7.5e-7），1000 步的统计看不出变化（03 README 结论 4）。

### 抓数与观测代码（janus 侧补丁，是否改数值）

| 类别 | 参数或代码位置 | 原始值（出处） | 最终 golden `rcc` | 结论 | 方向 | 机制、为什么、影响与证据 |
|---|---|---|---|---|---|---|
| 抓数 | [`../../janus/dump/megatron.py`](../../janus/dump/megatron.py) 的 `JanusDump` 回调和 `wrap_forward_step`（`--janus-run`、`--dump-steps`） | 无 | 各运行的抓数参数见注 20 | 采用 | — | 只读张量、写盘。GPU 上不改数值（见注 20） |
| 观测 | `--group-norms`、`--row-stats-every`（`JanusDump` 里读状态） | 无 | 不开 | 同原始 | — | 只读主权重和 Adam 状态；GPU 上开和不开的逐位核对没有做（见注 21） |
| 观测 | [`frameworks/megatron/resid_probe.py`](frameworks/megatron/resid_probe.py)（`--resid-probe-every`） | 无 | 不开 | 同原始 | — | 另跑一次前向，不动路由偏置；GPU 上逐位核对没有做（见注 22） |
| 观测 | `rr_arms.py` 的 `head-log-<步>` 臂 | 无 | 不开 | 只作诊断（只记录） | — | 记下输出层实际用到的张量。确定性设置下开和不开，5 步里所有逐步量逐位相同（案例 01「输出头反向实际用到的张量」行） |
| 诊断 | [`param_swap.py`](frameworks/megatron/param_swap.py) 的 `ParamSwap`（`--import-params`、`--export-params-at`、`--import-round-bf16`）、`--skip-steps` | 无 | 不开 | 只作诊断 | — | 把同一个状态换到另一种精度或另一侧，重算某一步的梯度（案例 01「逐步拆分」）。不进 golden |
| 诊断 | 故障注入 `--fault`（其余种类） | 无 | 不开 | 只作诊断 | — | 只用来验证检查会不会报警（00 `faults/`）。golden 里唯一用到的是上面「噪声带的扰动」那一种 |

- 注 20（抓数）：
  - 各运行的参数：
    - [`rcc-a`](../../docs/glossary.md#rcc-a)（最终 golden 的第 1 份，验收的比较基线）：`--dump-steps 1000 --save-batches`；
    - `rcc-b..j`：`--dump-steps 100000`（等于不抓）；
    - 参考 [`rce-a`](../../docs/glossary.md#rce)（精确轨迹，BF16 权重、全程 FP32 计算）：`--dump-steps 1,1000 --full --save-batches`；
    - `rce-b..e`：`--dump-steps 1 --full`。
  - GPU 上不改数值（frameworks README「已验证」）：
    - 确定性设置下开回调和不开回调，逐步逐位相同；
    - 两次开回调的确定性运行，第 1、20 步 24 个张量逐位相同。
  - 唯一会改数值的副作用是 `--full` 强制关掉分布式优化器，见上面的「分布式优化器」行。
  - TPU 侧加钩子会改 XLA 的融合，那是另一回事，见 01 README 的 D 类。
- 注 21（`--group-norms`、`--row-stats-every`）：
  - 只读主权重和 Adam 状态（案例 01 第 6 节「噪声带」行）。
    - `g-gn`（多记分组范数的 GPU 噪声运行）、`g-rs`（多记输出层逐行统计的 GPU 噪声运行）多记了这些统计。
  - GPU 上开和不开的逐位核对没有做；`rcc` 不开它，不影响 golden。
  - 要核对：确定性设置下开、不开各跑 20 步，比逐步标量逐位。
- 注 22（`resid_probe.py`）：
  - 用 no_grad 加 eval 在固定序列上另跑一次前向。
  - 路由偏置的计数只在开梯度时才累加，所以探针不动它。
  - GPU 上开和不开的逐位核对没有做（`rcc` 不开它，不影响 golden；做法同上一行）。
  - TPU 侧的 `t-probe-1000`（TPU 开残差流探针的 1000 步运行）和 `t-a`（TPU 旧配置、第 1 步全量抓数的 1000 步运行）逐位相同。

### 试过、没有采用的开关

一条一条换成别的写法，看 GPU 会不会因此出现 TPU 那样的漂移，或者更接近精确值。
下面的运行都是 BF16 golden 的设置（多数不重算），对照用同设置的 GPU 运行。

代号：`rce`（精确轨迹，BF16 权重、全程 FP32 计算）、`rcx`（生产 golden，输出头 dgrad 换 FP32 一次累加）、`rcc-a`（最终 golden 的第 1 份，验收的比较基线）。

| 类别 | 参数或代码位置 | 原始值（出处） | 最终 golden `rcc` | 结论 | 方向 | 机制、为什么、影响与证据 |
|---|---|---|---|---|---|---|
| 精度 | `rr_arms.py` 的 `dgrad-f32`（输出头 dh 改成 FP32 矩阵乘，但没关 TF32） | BF16 张量核 | 不用 | 试过，有害（实际跑的是 TF32） | 本意是 GPU 向精确值 | 实际跑在 TF32 上，比 BF16 还偏；一度误判（见注 23） |
| 精度 | `rr_arms.py` 的 `wgrad-exact`（输出层权重梯度换成关 TF32 的 FP32 乘法） | BF16 张量核，FP32 累加进 main_grad | 不用（`rce` 里由全程 FP32 覆盖） | 试过，无影响 | GPU 向精确值 | 1000 步和只开 `dgrad-exact` 一样（见注 24） |
| 精度 | 交叉熵融合：`--ce-fusion off`（`cross_entropy_loss_fusion`） | True（provider；`rcx-a` meta） | 同原始 | 试过，无影响 | — | 见注 25 |
| 精度 | 注意力实现：`--attention-backend unfused` / `flash`（BF16 设置下） | `auto` | 同原始（`auto`）；`rce` 为了 FP32 用 unfused，见「参考」 | 试过，无影响（BF16 设置下） | — | 落在 10 次参照的范围内；TPU 换注意力照样漂（见注 26） |
| 精度 | 梯度 BF16 规约：`--grad-reduce-bf16` | FP32 规约 | 不用 | 试过，无影响 | GPU 向 TPU | 见注 27 |
| 精度 | 门控激活融合：`--bias-act-fusion` | False（provider） | 同原始 | 试过，无影响 | GPU 向 TPU | 见注 28 |
| 精度 | 关 TF32（BF16 设置下）：`--no-tf32` 加 `NVIDIA_TF32_OVERRIDE=0` | 开 | 同原始（开）；`dgrad-chunk` 在它那一次乘法里临时关 | 试过，无影响（BF16 设置下） | GPU 向 TPU | 见注 29 |
| 精度 | `rr_arms.py` 的 `ce-mt`（交叉熵按 MaxText 自定义 VJP 的写法） | Megatron 融合交叉熵 | 不用 | 试过，无影响 | GPU 向 TPU | 差在 7 次对照的散布内（见注 30）。下面七臂同一组实验、同一个结论 |
| 精度 | `rr_arms.py` 的 `adam-optax`（Adam 按 optax `adamw` 逐式重写） | TE FusedAdam | 不用 | 试过，无影响 | GPU 向 TPU | 同上 |
| 精度 | `rr_arms.py` 的 `clip-optax`（先除后乘，不加 1e-6） | `g · max_norm / (norm + 1e-6)` | 不用 | 试过，无影响 | GPU 向 TPU | 同上 |
| 精度 | `rr_arms.py` 的 `fnorm-mt` / `fnorm-f32`（最后一层 RMSNorm 按 MaxText 舍两次 / 舍一次） | TE RMSNorm 舍一次 | 不用 | 试过，无影响 | GPU 向 TPU | 同上 |
| 精度 | `rr_arms.py` 的 `wgrad-bf16`（输出层权重梯度每个 microbatch 先舍 BF16） | FP32 累加进 main_grad | 不用 | 试过，无影响 | GPU 向 TPU | 同上 |
| 精度 | `rr_arms.py` 的 `attn-scale-bf16` / `attn-qscale-mt`（注意力缩放系数用 BF16 舍过的 0.08837890625） | 1/√128 精确值 | 不用 | 试过，无影响 | GPU 向 TPU | 同上。MaxText 在 BF16 的 query 上乘 Python 标量，JAX 按弱类型把它转成 BF16 常量（`rr_arms.py` 文件头） |
| 精度 | `rr_arms.py` 的 `resid-fp32`（残差流用 FP32） | BF16 残差 | 不用（`rce` 里由全程 FP32 覆盖） | 试过，无影响（1000 步） | GPU 向 TPU | 单步能去掉一半剩余偏差，1000 步不动（见注 31） |
| 精度 | `rr_arms.py` 的 `moe-combine-f32`（路由权重在 fc2 之后乘，k 个专家的输出在 FP32 里加权求和） | 先把 act·p 舍成 BF16 再进 fc2 | 不用 | 试过，无影响 | GPU 向 TPU | 300 步同上；单步 `g-s3-mc`（θ₂₉₉ 上算第 300 步的 batch，单步开这一臂）−0.22%（和只开 `dgrad-exact` 一样）；1000 步 `g-dxrm-1k`（`dgrad-exact` 加残差 FP32 和 MoE 合并 FP32）和只开 `dgrad-exact` 一样 |
| 精度 | `rr_arms.py` 的 `router-bwd-bf16`（路由 logits 的梯度先舍 BF16 再进路由 GEMM 反向） | TF32 | 不用 | 试过，无影响 | GPU 向 TPU | 第 300 步输出层 −0.08；第 3 层路由偏置标准差 z = +0.5，TPU 是 −11.1（案例 01「反向复现（续）」行） |
| 精度 | 上面八处一起开，再加 `dgrad-exact`（`g-dxall-1k`） | — | 不用 | 试过，无影响 | GPU 向 TPU | 第 1000 步 param_norm +1.28e-2、输出层平方和 +78，和只开 `dgrad-exact` 一样（案例 01 第 0 节「剩下 1/3」行） |
| 诊断 | [`frameworks/megatron/router_fp32.py`](frameworks/megatron/router_fp32.py)（`--router-true-fp32`：路由权重留成 FP32 参数，路由 GEMM 前向、反向都是 FP32 × FP32） | TE `general_gemm`，BF16 入、FP32 出 | 不用 | 只作诊断 | GPU 向 TPU | 用来给路由 GEMM 的精度定责（见注 32） |
| 诊断 | `rr_arms.py` 的 `dhproj-<a>`、`dhscale-<k>`、`dh-center`（改 dh 沿 ŵ̄ 的分量）、`ogauge-<k>`（改输出层梯度的跨行均值） | — | 不用 | 只作诊断（改了数学） | — | 人为加减一股力，验证「同一股力」的假设（见注 33） |
| 诊断 | `rr_arms.py` 的 `dgrad-exact`（输出头 dgrad 关 TF32、cuBLAS FP32 一次乘完；`rcx` 用它） | BF16 张量核 | 不用（`rcc` 改用 `dgrad-chunk`） | 试过，部分有效（只去掉 BF16 那一部分） | GPU 向精确值 | 训练中途 cuBLAS FP32 长累加有偏（见注 34） |
| 诊断 | `rr_arms.py` 的 `dlogits-bf16`（全程 FP32 程序里把流进输出层反向的 dlogits 舍 BF16） | — | 不用 | 只作诊断 | — | 证实从 `rcc` 到真精确之间的那一份差是 dlogits 舍入：`g-bwcr-1k`（`g-bwc-1k` 再把 dlogits 舍 BF16）比 `g-bwc-1k`（默认配置精确轨迹加分段 dgrad）少漂 6.74e-3，落到 TPU 生产与 `rcc-a` 上（案例 01「GPU 一侧的同一件事」行） |
| 诊断 | `rr_arms.py` 的 `gscale-<x>`、`ogscale-<x>`、`egscale-<x>`（裁剪之后再缩放梯度） | — | 不用 | 只作诊断（敏感度） | — | 不是 TPU 的行为，只用来看敏感度（见上面「β1 / β2 / ε」行的 Adam ε 区） |

- 注 23（`dgrad-f32`）：
  - 容器里 `allow_tf32 = True`，这个「FP32」乘法实际在 TF32 张量核上算，输入尾数截到 10 位。
  - 沿 ŵ̄ 是 −8.08e-4，比 BF16 的 −6.51e-4 还差。
  - 300 步训练看不出影响（`rr-dgradf32-a/b`），一度被误判成「输出头 dgrad 不是原因」，后来更正（案例 01「输出头反向数值」「输出头反向实际用到的张量」行）。
  - 教训：FP32 臂必须实测 TF32 真的关了。
- 注 24（`wgrad-exact`）：
  - 收缩长度是 token 数，不是词表。
  - 每个 microbatch 的 BF16 贡献对精确值的 rel_L2 只有 7e-6；只有沿词表的均值（规范分量，本身只有行尺度的 1e-6）大 47%–63%。
  - 和 `dgrad-exact` 一起跑 1000 步（`g-dwx-1k-a`）：param_norm +1.25e-2、输出层平方和 +78.8。
  - 只开 `dgrad-exact` 是 +1.27e-2 – +1.31e-2、+78 – +79，一样（案例 01 第 0 节「输出层 wgrad」行）。
- 注 25（交叉熵融合）：
  - 交叉熵反向对精确值的误差，第 1 步只有 +2.7e-6，全部来自最后舍到 BF16 的那一步；关掉融合是 +2.66e-6（`p2-hl-cefoff`）。
  - 400 步的 `g-cefuse-off`（关交叉熵融合的 GPU 运行），输出层贡献在 GPU 对之间的量级（案例 01「反向复现」「输出头反向实际用到的张量」行）。
  - 确定性设置会自动关掉融合（`train_hy3.py`），所以 `g1k-det`（第一阶段 GPU golden 的确定性设置版）是关的。
- 注 26（注意力实现）：
  - `g-attn-unfused` 和 `g-attn-flash` ×2（BF16 设置下 GPU 把注意力换成 unfused / flash），301–1000 步均值都落在 10 次参照的范围内，param_norm 不漂。
  - TPU 换成 dot-product 注意力，照样漂（案例 01 第 6 节 4f）。
- 注 27（梯度 BF16 规约）：
  - TPU 上专家、embedding、归一化层的梯度在 8 个设备之间按 BF16 规约。
  - GPU 反过来这样做（`g-bf16grad`，1000 步）：param_norm 对 g1k-a（第一阶段 GPU golden 的第 1 份）+6.6e-4。
  - 同设置的 `g-ctl`（扩充噪声带用的 GPU 对照运行）是 +8.2e-4（案例 01 第 6 节 4c，[`case01-arms-1000.json`](alignments/01-v5p-acceptance/results/case01-arms-1000.json)）。
- 注 28（门控激活融合）：
  - silu·up 走融合核一次算完，接近 TPU 默认的「FP32 算完舍一次」。
  - `g-actfuse` 400 步，输出层贡献在 GPU 对之间的量级（案例 01「反向复现」行）。
  - 对齐时 TPU 反过来改成按算子舍入（见下面 TPU 表）。
- 注 29（关 TF32）：
  - BF16 golden 里只有路由反向的两个 GEMM 是 FP32 输入。
  - `g-notf32` 400 步看不出影响（案例 01「反向复现」行）。
  - 在同一个状态上单步：`g-s3-ntf` −0.21% / −0.25%，只开 `dgrad-exact` 是 −0.22% / −0.28%（案例 01 第 0 节「GPU 精确 dgrad 之外再换一处」行）。
  - 第一阶段 30 步 loss 在噪声内（00 `phase1-tf32`）。
- 注 30（`ce-mt`）：
  - 300 步对 7 次同设置运行的均值：第 300 步输出层平方和差在 −0.24 ～ +0.09 之间。
  - 7 次对照之间约 ±0.3；TPU `t-gn-300`（TPU 记分组范数的 300 步运行）是 +23.3。
  - 出处：案例 01「反向复现（续）」行，[`case01-rr-arms.json`](alignments/01-v5p-acceptance/results/case01-rr-arms.json)。
- 注 31（`resid-fp32`）：
  - 在同一个状态上单步，它能把 `dgrad-exact` 之后剩下的每步偏差去掉一半（`g-s3-rs` −0.10% / −0.12%）。
  - 但 1000 步（`g-dxr-1k`）param_norm、输出层平方和、lse 都不动。
  - 出处：案例 01 第 0 节「剩下 1/3」行和「GPU 精确 dgrad 之外再换一处」行。
- 注 32（`router_fp32.py`）：
  - 用来定责：开和不开对照一次运行，看某个差里有多少来自路由 GEMM 的精度。
  - 第 1 步每次路由 GEMM 对 FP64 的误差写进 meta 的 `router.probe`。
  - 对训练轨迹的影响没有入库的结果（`rcc` 不开它，不影响 golden）。
- 注 33（`dhproj` 等）：
  - 这几个臂人为加上或去掉一股力，用来验证「同一股力」的假设。
  - 它们改的是数学本身，不是写法，不能进 golden。
  - 结果：GPU 反号加力（`dhscale-−8`）能定量复现 TPU 的 loss 偏移和 param_norm 漂移。
  - 规范分量不是根因（案例 01「dh 里的 logit 水平分量」「规范分量（GPU 侧）」行）。
- 注 34（`dgrad-exact`）：
  - 第 1 步对 FP64 沿 ŵ̄ −1.7e-6，看着精确。
  - 训练中途 cuBLAS FP32 长累加有偏（θ₂₉₉ −2.6e-4）。
  - 第 1000 步 param_norm：[`rcx`](../../docs/glossary.md#rcx) − `rc` +1.24e-2，[`rcc`](../../docs/glossary.md#rcc) − `rc` +2.07e-2（案例 01「GPU 精确的输出头 dgrad」行）。

### 参考：精确轨迹 `rce` / `rcec` 在 rcc 之外多改的（交叉核对用，不是最终 golden）

- [`rce`](../../docs/glossary.md#rce) = `rc`（GPU 生产 golden） + `--fp32 --bf16-weights`（真关 TF32、unfused 注意力），是 BF16 训练的精确轨迹。
  - 但它的输出头 dgrad 是 cuBLAS FP32 一次累加，仍带偏差。
- [`rcec`](../../docs/glossary.md#rcec) = `rce` + `dgrad-chunk`，才是真精确。
- 两者只作参考：
  - TPU 精确与 `rcec` 在噪声内重合；
  - TPU 生产对它们的差就是两侧共有的 BF16 dlogits（见上面主表最后一行）。
- 下表是它们的设置与当初的理由，原样保留。

代号：`rcx`（生产 golden，输出头 dgrad 换 FP32 一次累加）。

| 类别 | 参数或代码位置 | 原始值（出处） | 参考 `rce` / `rcec` | 结论 | 方向 | 机制（GPU / TPU）、为什么、影响与证据 |
|---|---|---|---|---|---|---|
| 精度 | 计算精度：`train_hy3.py --fp32`，对应 `MixedPrecisionConfig(fp32=True, params_dtype=float32, autocast_enabled=False, grad_reduce_in_fp32=True)` | `bf16_mixed`（Bridge 默认 `mixed_precision`） | 前向和反向的算术全部 FP32（rce-a meta `dtypes.compute: fp32`）；权重仍是 BF16 语义，见下一行 | 参考 golden 采用 | 两边向精确值 | 离精确值远的是 GPU BF16 那一侧，只有精确轨迹能当裁判（见注 35） |
| 精度 | 权重语义：`--bf16-weights`，对应 [`param_swap.py`](frameworks/megatron/param_swap.py) 的 `Bf16WeightsFp32Compute` | BF16 参数 + FP32 主权重（Bridge `bf16_mixed`） | FP32 主权重，每步前向前舍到 BF16 再算（见注 36） | 参考 golden 采用 | 两边向精确值 | 保留生产的权重语义，只把算术换成 FP32（见注 36） |
| 精度 | TF32：环境变量 `NVIDIA_TF32_OVERRIDE`；`train_hy3.py --no-tf32`（把 torch 的 `allow_tf32` 两个开关设为 False） | 开（rel_L2 2.94e-4，见注 37） | 关：`NVIDIA_TF32_OVERRIDE=0` 加 `--no-tf32`（rel_L2 2.30e-7，见注 37） | 参考 golden 采用 | 两边向精确值 | 不关的话「精确」不成立；torch 的开关读回来不能信（见注 37） |
| 精度 | 注意力实现：`--attention-backend unfused` | `AttnBackend.auto`（provider；`rcx-a` meta） | `unfused`（rce-a meta `model_config.attention_backend: AttnBackend.unfused`；PyTorch 写法：QKᵀ、softmax、乘 V 分开算） | 参考 golden 采用（为了 FP32） | 两边向精确值 | `--fp32` 必须配 unfused（见注 38） |
| 精度 | 输出头 dgrad（dh = dlogits·W，沿词表 120832 求和）：[`rr_arms.py`](frameworks/megatron/rr_arms.py) 的 `dgrad-exact` 臂 | 生产路径：BF16 dlogits × BF16 参数，在张量核上乘 | `rce` 里由全程 FP32 覆盖，但不等于精确；`rcx` 里单独开着（见注 39） | `rcx` 单独开；`rce` 被覆盖；`rcec` 改用 `dgrad-chunk` | GPU 向精确值，也就是向 TPU | 机制、影响与 2026-11-06 的更正见注 39 |
| 精度 | 梯度规约精度：`ddp.grad_reduce_in_fp32` | True（Bridge 默认） | 同原始；`--fp32` 下梯度本来就是 FP32 | 同原始 | — | 反过来改成 BF16 规约（`--grad-reduce-bf16`，模仿 TPU 上部分梯度按 BF16 规约）试过，无影响，见下面「试过」 |
| 精度 | 路由 GEMM：`moe_router_dtype` | `fp32`（provider；meta）。在这版 Megatron 里走 TE `general_gemm`，BF16 输入、FP32 输出 | 全程 FP32 下没有区别；meta 的 `router.mode` 字符串写错（见注 40） | 同原始 | — | GPU 3.8e-6、TPU 4.8e-8；meta 字段已修（见注 40） |

- 注 35（计算精度）：为什么要改。
  - 案例 01 查到，GPU 的 BF16 golden 自己就系统地偏离精确值。
  - 默认配置下的精确轨迹 `g-bw-1k` 跑 1000 步：
    - param_norm 比 GPU 默认运行的均值多出 +1.82e-2（第 975 步），TPU 是 +1.99e-2；
    - lse 走了九成以上（案例 01 第 0 节「精确轨迹」行，原始数字在 [`case01-exact-traj.json`](alignments/01-v5p-acceptance/results/case01-exact-traj.json)）。
  - 所以离精确值远的是 GPU BF16 那一侧。
  - 精确轨迹不偏向任何一边的实现，只有它能当裁判。
- 注 36（权重语义）：
  - 做法：
    - 主权重是 FP32。
    - 每步前向之前把参数按 RNE 舍到 BF16，前向和反向都用这份舍过的权重。
    - 优化器更新之前换回 FP32 主权重，再在主权重上更新。
    - rce-a meta `dtypes.weight`：`fp32 main params, rounded to bf16 for fwd/bwd each step`。
  - 为什么不直接用全程 FP32 训练：生产里 GEMM 实际用的是主权重舍到 BF16 的那一份，权重本身每个元素就差半个 BF16 格。纯 FP32 训练（`g-f32-1k`）前 100 步 loss 就比 BF16 低 1.75%，走的是另一条轨迹，不能当参照（案例 01 第 6 节「逐步拆分」行的附带说明）。这里保留生产的权重语义，只把算术换成 FP32，每一步的梯度就是「同一份权重下 BF16 程序应该算出来的精确值」。
  - 3 步自检，第 2 / 3 步 loss（案例 01「精确轨迹」行）：
    - 精确轨迹 11.23569 / 10.58785
    - TPU 11.23569 / 10.58784
    - GPU BF16 11.23543 / 10.58775
- 注 37（TF32）：
  - 原始值：开。
    - 容器设了 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，`allow_tf32_matmul` 和 `allow_tf32_cudnn` 都是 True，`float32_matmul_precision: high`（`rcx-a` meta 的 `tf32`）。
    - 启动实测 FP32 GEMM 对 FP64 的 rel_L2 是 2.94e-4，torch 和 TE 两条路一样。
  - 参考 `rce`：关，`NVIDIA_TF32_OVERRIDE=0` 加 `--no-tf32`。
    - rce-a meta：`env.NVIDIA_TF32_OVERRIDE: 0`，`allow_tf32_matmul` / `allow_tf32_cudnn` 都是 False，`float32_matmul_precision: highest`。
    - 启动实测 rel_L2 2.30e-7（torch 与 TE 两条路一样）。
    - `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` 还留在容器里。
  - GPU：开 TF32 时，FP32 矩阵乘一进 cuBLAS，就悄悄换到 TF32 张量核上算，输入尾数被截到 10 位。
    - 所以名义上的「FP32」比 BF16 张量核还不准：输出头 dgrad 第 1 步沿 ŵ̄，FP32 + TF32 是 −8.08e-4，BF16 是 −6.51e-4。
    - 只有关掉 TF32，才会走 CUDA 核上真正的 FP32 FMA，这时是 −1.7e-6（案例 01 第 6 节「输出头反向实际用到的张量」行）。
  - torch 的开关读回来不能信：
    - 只有 `NVIDIA_TF32_OVERRIDE=0` 能把 TE 的 FP32 GEMM 也关掉（rel_L2 2.94e-4 → 1.15e-6）。
    - 关掉以后 torch 的开关读回来仍是 True（[`../../docs/reference/environment.md`](../../docs/reference/environment.md) 第 4 条；00 `phase1-kernels` 的 `TF32.switch.matrix`）。
    - 所以每次启动都实测一遍，记进 meta 的 `tf32.probe_rel_l2_*`，≤ 1e-5 才算关了。
  - TPU：没有 TF32；FP32 操作数按 `matmul_precision` 处理（见下面 TPU 表的「路由核 BF16 舍入」行）。
  - 在 BF16 golden 里，FP32 输入的 GEMM 只有路由反向的两个，关 TF32 跑 400 步看不出影响（1000 步没跑；见下面「试过」的 `g-notf32`、`g-s3-ntf`：关 TF32 跑 400 步、关 TF32 在同一状态上单步）。
  - 到了全程 FP32 的 `rce`，每个 GEMM 都是 FP32 输入，TF32 不关的话「精确」就不成立。
- 注 38（注意力实现）：
  - `--fp32` 必须配 unfused（`train_hy3.py` 的帮助、frameworks README 参数表）。
  - 推断：cuDNN fused / flash 内核只吃 BF16 / FP16 输入，FP32 要整条路径保持 FP32，只能走 unfused。
  - 在 BF16 设置下单独换注意力实现，1000 步没有影响（见下面「试过」）。
- 注 39（输出头 dgrad 的 `dgrad-exact` 臂）：
  - 参考 `rce` / `rcec` 栏：
    - `rce` 里由全程 FP32 覆盖（输出层的 dgrad 是 FP32、关 TF32），不再单独开这个臂。
    - 但这不等于精确：B200 上 cuBLAS FP32 一次累加整个词表也有偏，训练中途沿 ŵ̄ 到 −2.6e-4，分段累加才精确（见本注末「更正」与 `rcec`）。
    - `rcx` 里单独开着。
  - GPU 张量核的累加截断：两个 BF16 的乘积在 FP32 里是精确的，问题出在把许多乘积加起来的那一步。
    - 实测的 B200 张量核模型：每 16 个乘积连同累加器一起对齐到最大指数，低于 2^(emax−25) 的位向零截掉，写回 FP32 时也向零舍入，而不是就近舍入。
    - 这个模型逐位复现路由 logits 三层全部 4096 × 192 个元素（100%）。
    - 同一个模型只要改一个参数（8 个一组、24 位、就近舍入），命中率就掉到 1% 以下（[`../../docs/dev/design/0003-l1-op-replay.md`](../../docs/dev/design/0003-l1-op-replay.md) 第 4 条）。
    - 截掉的总是低位，所以在很长的求和里会积成一个朝 0 的系统偏差，而且随收缩长度 K 增长。
    - 合成正数输入 K = 256 / 4096 / 65536 时是 2.2e-5 / 7.0e-5 / 3.1e-4，负数输入对称（[`gemm_accum_bias.py`](analysis/gemm_accum_bias.py)）。
    - 输出头 dgrad 的 K = 120832，第 1 / 100 步沿 ŵ̄ 偏 −6.51e-4 / −6.10e-4（案例 01「输出头反向实际用到的张量」行）。
    - 这个方向是「所有 logits 一起升降」，每一步同向少推一点，几百步积成 lse、输出层冷门行范数和 loss 的偏移。
  - TPU：MXU 吃 BF16 输入，乘积精确，在 FP32 里累加。
    - 输出头回放第 1 / 100 步对 FP64 是 −2.4e-5 / −7.4e-5（案例 01「输出头反向数值」行）。
    - MXU 累加用的是就近舍入，这一点是推断：TPU 路由 logits 对 FP64 4.8e-8，GPU 3.8e-6（[`config-alignment.md`](config-alignment.md)「路由 GEMM」行）。
  - 影响：`rcx` − `rc`：301–1000 步 loss −9.8e-5，第 1000 步 param_norm +1.24e-2（案例 01 第 6 节「新协议下的验收」）。
  - **更正（2026-11-06）：`rce` 和 `rcx` 的这一次乘法都还有偏**。
    - 两者都用 cuBLAS 的 FP32 乘法（关 TF32）一次累加整个词表。
    - 在 B200 上它对 FP64 的 rel_L2 是 CPU FP32 的 59 – 129 倍（ALIGNMENT-REPORT 台账「cuBLAS FP32 长累加」行）。
    - 沿 ŵ̄ 第 1 步 −1.8e-6，到 θ₂₉₉（第 300 步开始时的状态）长到 −2.6e-4（BF16 的六成）。
    - 容器里能找到的开关（`fp32_precision=ieee`、去掉 TF32 override、cuBLAS 后端与 emulation 设置）都不改变结果。
    - 只有把词表分段、段间 FP32 累加（`rr_arms.py` 的 `dgrad-chunk` 臂）能去掉。
    - 默认配置下精确轨迹加这个臂（`g-bwc-1k`），1000 步 param_norm 比 `rce-a` 多 +8.55e-3，和 TPU 精确重合（案例 01「GPU 精确的输出头 dgrad」「GPU 精确 + 分段 dgrad」两行）。
    - 所以 `rce` 不是真精确。
  - 两个候选：
    - rc 配置下的真精确候选 `rcec-a`（`rce` + `dgrad-chunk`）：1000 步比 `rce-a` 多 +8.87e-3，与 TPU 精确的带符号均值差 8.3e-9（拿它当基线时 TPU 各配置的读数见案例 01「rc 配置下的真精确轨迹 `rcec-a`」行）。
    - 另一个候选是 `rcc`（GPU 生产程序 + `dgrad-chunk`，即 `rcx` 本来想做的「只修输出头 dgrad」）：
      - `rcc-a` 与 TPU 生产在噪声内重合，v5p 当前配置对它 L4 五项全过（`phase2-acceptance-rcc-on`，只报）；
      - `rcc-a..e` 五份重跑两两最大 param_norm 带符号均值 7.4e-8、loss 尾段 7.9e-5；
      - TPU 生产对每一份都在这个范围附近（案例 01 附录「rcc 的散布」）。
  - 2026-11-06 定：最终 golden 是 `rcc`，第一层基线 `rcc-a`（本页开头）。
- 注 40（路由 GEMM）：
  - 全程 FP32 下没有区别。
  - 注意：`rce` 的 meta 里 `router.mode` 写的是 `te_general_gemm bf16-in fp32-out`。
    - 这是 `train_hy3.py` 里写死的字符串，不反映 `--fp32`。
    - FP32 模式下路由权重也是 FP32 参数（根因调查：FP32 前向的路由 logits 和 TPU FP32 差 6e-6，路由专家对 FP64 1.2e-6）。
  - BF16 golden 里，路由 logits 对 FP64 是 3.8e-6，来自上面说的张量核截断；TPU 是 4.8e-8（`config-alignment.md`「路由 GEMM」行）。
  - meta 的 `router` 字段在 `--fp32` 下写错了：
    - 脚本已修，之后的 `--fp32` 运行写 `fp32-in fp32-out`、`weight_param_dtype: fp32`；
    - `rce-a..e` 的 meta 是修之前写的，仍是旧字符串。

## 对照：TPU 侧为对齐配的开关

这些开关不是 golden 的设置，列在这里是为了讲清楚两边各自在做什么。
逐项对照和证据都在 [`config-alignment.md`](config-alignment.md)。

代号：`rcc-a`（最终 golden 的第 1 份，验收的比较基线）、`rcec-a`（真精确轨迹，精确轨迹加输出头 dgrad 分段累加）。

| 类别 | 参数或代码位置 | 原始值（出处） | 对齐配置 | 结论 | 方向 | 机制（GPU / TPU）与证据 |
|---|---|---|---|---|---|---|
| TPU 精度 | `dense_fp32_accumulation` | 关（MaxText 默认） | 开（yml `a238883b7f91…` 起） | 采用（L1 / L2 必需；L4 打开前后对 [`rcc-a`](../../docs/glossary.md#rcc-a) 都过） | TPU 向 GPU（也离精确值更近） | 窗口化 collective matmul 的部分积从 BF16 改成 FP32 累加（见注 41） |
| TPU 精度 | `LIBTPU_INIT_ARGS=--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000`（关掉窗口化） | 窗口化开着 | 不用 | 试过，有害（多舍和少舍不对称） | — | 比 GPU 少舍一次，MLP 前归一化 R 1.64；所以用开关，不用这个 flag（见注 42） |
| TPU 精度 | `rmsnorm_single_rounding` | 关：MaxText 的写法是归一化结果、scale 各转一次 BF16，舍两次 | 开 | 采用 | TPU 向 GPU | 前向逐位不变，改的是反向（见注 43） |
| TPU 精度 | `mlp_activation_round_per_op` | 关：默认编出来是「FP32 算完舍一次」 | 开：`silu` 的输出先舍 BF16，乘完再舍一次 | 采用 | TPU 向 GPU | 按算子舍入后与 GPU 100% 逐位（见注 44） |
| TPU 精度 | `routed_gate_bf16_rounding` | 关 | 开：路由核先 `reduce_precision` 到 BF16 再乘，梯度直通 | 采用（显式保证） | TPU 向 GPU | 开和不开逐位相同，留着作显式保证（见注 45） |
| TPU 精度 | excess precision（`--xla_allow_excess_precision`） | 开（XLA 默认） | 同原始 | 试过关，不采用：param_norm 漂移不变、loss 偏移减半（排查期对 `rc` 的读数） | — | 关掉只是让舍入位置确定，不是更准（见注 46） |
| TPU 精度 | `logits_dot_in_fp32`（单开） | False | 同原始 | 试过，无影响 | — | 单开等于没开；1000 步对 `rcc-a` 四项都过（见注 47） |
| TPU 精度 | `logits_dot_in_fp32` + `matmul_precision: highest`（logmm） | False + default | 同原始 | 只作诊断：对最终 golden 有害，对真精确有效 | 两边向精确值（只动 TPU 一侧） | 对 `rcc-a` 不过，对 [`rcec-a`](../../docs/glossary.md#rcec) 过（见注 48） |
| TPU 精度 | `matmul_precision: highest`（单开） | default | 同原始 | 试过，无影响 | — | `t-v7x-mmhi-1000`（v7x 当前配置只加这一项，1000 步）对 `rcc-a` 四项都过（最差 r 0.93；见注 49） |
| TPU 精度 | 注意力 FP32 点积：`attention: dot_product` + `float32_qk_product: true` | splash（BF16 输入） | 同原始 | 单开无影响；与 `matmul_precision: highest` 同开待定（只测了 v7x） | — | `t-v7x-attn32-1000`（v7x 当前配置只加这一项，1000 步）对 `rcc-a` 四项都过（最差 r 0.78）；和 highest 一起开（`t-v7x-attnmm-1000`）loss 全程 1.11、grad_norm 1.03 倍，待定；和 logit32 一起开都过 |
| TPU 精度 | `float32_gate_logits: true` | False | 同原始 | 试过，无影响（编出同一个程序） | — | `t-v7x-gate32-1000b`（v7x 当前配置只加这一项）1000 步与当前配置逐位相同（案例 01「路由 logits 用 FP32」行） |
| TPU 并行 | `ici_expert_parallelism: 8`（EP = 8；词嵌入表沿 expert 轴切 8 份，读 HLO 核过；其余非专家参数按逻辑轴规则推断同样切，不是整份复制） | FSDP 8，专家不切 | 同原始（FSDP 8） | 可选：两种都对得上（只在 libtpu 构建 `cl/963683785`（按构建号认作 0.0.46）、截 4 层、L4 四个量上验过） | 专家的切法和 GPU 相同 | 对 `rcc-a` 在 v5p、v7x 上 L4 都过（`phase2-acceptance-rcc-ep8`、`phase2-v7x-acceptance-rcc-ep8`）；词嵌入的切法读 HLO 核过，见注 50 |
| TPU 诊断 | `train_hy3.py --bf16-weights`（TPU 精确轨迹，全程 FP32、BF16 权重语义）、`--head-matmul-precision`、`--head-round input / cotangent / cot-dgrad / cot-wgrad`、`--jax-default-matmul-precision` | — | 不用 | 只作诊断 | — | 定位 TPU 那一份误差用的臂（见注 51） |

- 注 41（`dense_fp32_accumulation`）：
  - TPU：收缩维被 FSDP 切开的投影（注意力的 q / k / v、稠密 MLP 的 wi），XLA 会把它编成窗口化的 collective matmul。
    - v5p T1 配置的 HLO 里有 41 个 `windowed_dot_general` 循环。
    - 每一轮的分片部分积先以 BF16 输出，再在 BF16 累加器里相加，比 GPU 多舍了几次。
    - 开关打开后用 FP32 累加器，最后只舍一次。
  - GPU：TE `LayerNormLinear` 做 GEMM 时 FP32 累加、输出只舍一次。
  - 证据（`config-alignment.md`「收缩维被 FSDP 切开的投影」行）：
    - qkv.out 对 GPU 的 rel_L2 从 2.36e-3 降到 1.8e-4；
    - 单算子 FP64 R 比 qkv 1.235 → 1.000。
  - 对 param_norm 漂移没有作用（`t-wcoff-1000`：TPU 关窗口化的 1000 步运行，案例 01）。
- 注 42（关掉窗口化）：
  - 关掉窗口化以后，残差加直接用上了没舍过的投影输出，反而比 GPU 少舍一次。
  - MLP 前归一化的 R 到了 1.64（`phase2-l1-ops-wcoff`）。
  - 另外它前向、反向、输出层全都关，还会改性能。
  - 所以用开关，不用这个 flag。
- 注 43（`rmsnorm_single_rounding`）：
  - GPU：TE RMSNorm 在 FP32 里算完只舍一次，对 FP64 正确舍入 99.9988%–99.9993%。
  - TPU：在 v5p 上，默认写法的前向已经被 XLA 省掉了中间那次舍入。
    - 打开开关后前向逐位不变，改的是反向（显式 FP32 路径，梯度不经过 BF16 转换）。
    - 各层 `in.grad` 对 golden 略降（`config-alignment.md`「RMSNorm 舍入次数」行）。
- 注 44（`mlp_activation_round_per_op`）：
  - GPU：路由专家用非融合 GLU，`silu` 在 BF16 里舍一次、乘积再舍一次（00 `L1.kernel.experts.fwd.replay` 逐位）。
  - TPU 默认写法和 GPU 只有 73.7% 逐位，按算子舍入以后 100% 逐位（`config-alignment.md`「门控激活舍入」行）。
- 注 45（`routed_gate_bf16_rounding`）：
  - TPU：`matmul_precision: default` 下，FP32 操作数本来就是先舍 BF16、单遍乘、FP32 累加。
    - 实测开和不开逐位相同（回归试跑 20 步，整条训练逐位相同）。
    - 留着是为了不依赖后端，作为显式的保证。
  - GPU：路由核本来就是 BF16 参数（`config-alignment.md`「路由核 BF16 舍入」「路由 GEMM」行）。
- 注 46（excess precision）：
  - TPU：XLA 在融合内部跳过中间的 BF16 舍入。
  - GPU：没有这个机制，每个算子各舍一次。
  - 关掉只是让舍入位置确定，不是更准；需要和 GPU 一样逐个舍的地方，用上面的开关显式写。
  - `t-noexc-1000`（TPU 关 excess precision 的 1000 步运行）param_norm 照样漂，loss 偏移减半（案例 01「剩下约四成：TPU excess precision」行）。
  - 混元 4 查明，省掉的舍入能用 `reduce_precision` 显式补回（[根源一览第 16 行](../hunyuan4/ROOT-CAUSES.md#r16)），而且关掉这个开关会让排好序的词嵌入 scatter 退回每加一次舍一次（[案例 14](../hunyuan4/cases/14-ddp-fsdp-embedding.md#回头看-10-月的建议)）。混元 3 保持默认开。
- 注 47（`logits_dot_in_fp32` 单开）：
  - 单开时 DEFAULT 精度在 MXU 上仍按单遍 BF16 乘，等于没开。
  - `t-gn-l32-300`（TPU 单开 `logits_dot_in_fp32`，300 步）第 300 步输出层贡献 +3.85e-3，和 `t-gn-300`（不开，300 步）一样（案例 01 第 6 节 4b）。
  - 1000 步 `t-v7x-logit32-1000` 对 `rcc-a` 四项都过（最差 r 0.88）。
- 注 48（logmm）：
  - 两者一起开，流进输出头反向的 dlogits 才留在 FP32；同时输出头改用 FP32 主权重（权重语义和 GPU 不同）。
  - 对 `rcc-a`：param_norm 4.80 倍、loss 全程 3.16 倍，不过。
    - GPU 生产程序也把 dlogits 舍成 BF16，只改 TPU 一侧就离开了 GPU。
  - 对真精确 `rcec-a`：param_norm、grad_norm、loss 尾段都过。
  - 步时 +9.6%。
  - 出处：[`case01-tpu-arms-vs-rcc.json`](alignments/01-v5p-acceptance/results/case01-tpu-arms-vs-rcc.json)；对 `rcec-a` 的读数见案例 01「rc 配置下的真精确轨迹 `rcec-a`」行。
- 注 49（`matmul_precision: highest` 单开）：
  - 改的是 FP32 操作数的 dot：反向里 FP32 梯度 28 个、路由 4 个、MoE 加权求和 3 个。
  - 不改权重语义（案例 01「`matmul_precision=highest` 改了哪些算子」行）。
- 注 50（EP = 8）：
  - 步时（各运行 `metrics.jsonl` 第 50–1000 步中位数）：v7x 0.980 → 0.710 s，v5p 1.607 → 0.999 s。
  - 切法：MaxText 的逻辑轴规则把 `embed` / `embed_vocab` 也映射到 expert 轴。读 HLO 核过的只有词嵌入表（每卡 [120832, 512]，[`ep8-embed-scatter-hlo.json`](alignments/01-v5p-acceptance/results/ep8-embed-scatter-hlo.json)），对它 EP = 8 等于 FSDP 8；其余非专家参数没读 HLO，按同一组规则推断同样沿 expert 轴切。
  - 词嵌入（2026-11-11 读优化后 HLO 核实，[`ep8-embed-scatter-hlo.json`](alignments/01-v5p-acceptance/results/ep8-embed-scatter-hlo.json)）：表每卡 [120832, 512]，反向 scatter-add 每卡 32768 个下标，超过 15104（表行数的 1/8）。v5p 上排了序；v7x 上交给 SparseCore，按什么精度累加没有实测（推断）；没有整表 all-reduce。混元 4 [案例 14](../hunyuan4/cases/14-ddp-fsdp-embedding.md) 的纯数据并行问题在这里不出现。
  - 只在 libtpu 构建 `cl/963683785`（按构建号认作 0.0.46）上核过。换版本要重看 HLO（[`libtpu-052.md`](../hunyuan4/alignments/04-tpu-self/libtpu-052.md)）。改成真正的纯数据并行（每卡一个微批 4096 个下标）会落到门槛以下，要开 `embedding_grad_fp32`。
- 注 51（TPU 诊断臂）：
  - 定位 TPU 那一份（只舍输出头反向的 dlogits、只经 dgrad 一路）用的臂（案例 01「TPU 自己的 BF16 那一份」「TPU 那一份再拆」两行）。
  - `--jax-default-matmul-precision highest` 会让 splash / megablox 在 BF16 操作数上出错，只配全 FP32 用。

## 比较协议：几份 golden 只造噪声带，正式比较只对一份

2026-11-06 定。通用写法见 [`../../docs/reference/judging.md`](../../docs/reference/judging.md) 的三节：

- 「golden 怎么用」
- 「golden 本身有系统偏差时」
- 「两处测量口径」

### 协议要点

- **正式比较只对一份事先指定的 golden**：第一层是 [`rcc-a`](../../docs/glossary.md#rcc-a)（[基线](../../docs/glossary.md#baseline)）。
  - 不对几份 golden 取平均，也不取对各份里最差的那个。
- 噪声带：`rcc-a..j`（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加）10 份同设置重跑，两两配对得到 45 对。
  - 每一对算一遍判定量，取所有对里的最大值当噪声上限 N。
  - 和被测同一种设置（GPU 生产程序），是一般规矩。
  - 起初只有 `a..e` 5 份（两两 10 对），几项落在 1.02–1.24 倍之间；再跑 `f..j` 定下 45 对。
  - 两种读数都记在案例 01 附录「rcc 的散布」。
- 两处测量口径（写进目录，原口径的结果作为只报项并排列出）：
  - grad_norm 用两边均值作分母（`params: {denominator: mean}`）；
  - param_norm 的被测读分组合成的 `param_norm_from_groups`（`select.test_field`）；
  - 理由与数字见 ALIGNMENT-REPORT 第 8 节。
- 取消的例外：2026-11-05 晚到 11-06 早曾以精确轨迹 [`rce-a`](../../docs/glossary.md#rce)（BF16 权重、全程 FP32 计算）为基线。
  - 噪声例外地取 GPU 生产程序重跑 [`rcx-a..e`](../../docs/glossary.md#rcx)（生产 golden，输出头 dgrad 换 FP32 一次累加）5 份两两的 10 对（方案 A，已退役）。
  - 查清 `rce` 的输出头 dgrad 也带 cuBLAS FP32 长累加的偏差之后，这个例外已经取消。
  - 对 `rce-a` / `rcec-a`（真精确轨迹，精确轨迹加输出头 dgrad 分段累加）的目录只作参考交叉核对（下面「参考：对精确轨迹」）。
- 实现：
  - `golden` 只列 `rcc-a`、噪声对写进 `runs.noise`；
  - loss 用 `multi_golden` 的 `params.baseline: rcc-a`。
- 选一份当基线的代价：基线自己也是噪声带里的一个样本。
  - `rcc-a` 第 2 步 loss 就比其余九份都高约 8e-5（绝对值），之后回到散布里。
  - 正因为这样，N 取所有对里的最大值。

### 第一层（对 `rcc-a`，噪声：`rcc` 10 份重跑、两两 45 对）的结果

噪声是 [`rcc-a..j`](../../docs/glossary.md#rcc)（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加）10 份重跑两两配对得到的 45 对。
每格是三个量（带符号均值、MARE、最大单步）里对 N 最差的那个比值 r；判定按它（≤ 1 过）。
数字出自各目录的 `results/<目录名>.results.json`。

代号：`t-rmap-on-1000`（v5p 当前对齐配置 1000 步）、`t-v7x-on-1000`（同配置在 v7x）、`t-ep8-1000`（v5p 改成 EP = 8，1000 步）、`t-v7x-ep8-1000`（v7x 改成 EP = 8，1000 步）。

| 目录 | 被测 | `L4.loss` | `L4.loss.tail` | `L4.grad_norm` | `L4.param_norm` | `L4.lr` |
|---|---|---|---|---|---|---|
| [`phase2-acceptance-rcc-on`](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-on.report.md)（v5p 当前配置） | `t-rmap-on-1000` | 过（0.80） | 过（0.80） | 过（0.85） | 过（0.50） | 过 |
| [`phase2-v7x-acceptance-rcc`](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc.report.md)（v7x 当前配置） | `t-v7x-on-1000` | 过（0.82） | 过（0.83） | 过（0.85） | 过（0.53） | 过 |
| [`phase2-acceptance-rcc-ep8`](alignments/01-v5p-acceptance/results/phase2-acceptance-rcc-ep8.report.md)（v5p，EP = 8） | `t-ep8-1000` | 过（0.76） | 过（0.78） | 过（0.74） | 过（0.48） | 过 |
| [`phase2-v7x-acceptance-rcc-ep8`](alignments/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc-ep8.report.md)（v7x，EP = 8） | `t-v7x-ep8-1000` | 过（0.77） | 过（0.81） | 过（0.68） | 过（0.35） | 过 |

被测运行名的前缀见术语表[「运行名前缀」](../../docs/glossary.md#run-prefixes)。

只报项（原口径）：grad_norm 以 golden 为分母、param_norm 读原始列，四个目录也全部过（最差 0.84 / 0.73）。

### 参考：对精确轨迹 `rce-a` / `rcec-a` 的读数（交叉核对，只报）

[`rce`](../../docs/glossary.md#rce)、[`rcec`](../../docs/glossary.md#rcec) 是 BF16 训练的精确轨迹（见主表「参考：精确轨迹多改的」）。
TPU 生产对它们不过，原因已经查清，不是 GPU↔TPU 差：

- 对 `rce-a`（噪声 `rcx-a..e` 5 份两两 10 对；`rcx` 是生产 golden，输出头 dgrad 换 FP32 一次累加）：v5p / v7x param_norm 3.45 / 3.53 倍。
  - `rce` 自己的输出头 dgrad 带 cuBLAS FP32 长累加偏差（少漂 +8.7e-3）。
  - TPU 生产的 dlogits 舍 BF16（少漂 −6.8e-3）。
  - 两份相减剩 +2.35e-3。
- 对 `rcec-a`（真精确）：TPU 生产 param_norm 少漂 6.5e-3，6 – 7 倍不过。
  - 这是两边生产程序共有的 dlogits 舍 BF16。
  - GPU 生产 + 分段 dgrad（`rcc`）对 `rcec` 也差同样的 +6.7e-3。
  - TPU 让 dlogits 留在 FP32 的诊断臂 logmm 对 `rcec-a` 的 param_norm、grad_norm、loss 尾段都过。

逐项读数与当时的判定见案例 01 第 0 节「rc 配置下的真精确轨迹 `rcec-a`」「并行方式」两行与附录「rce 五份的散布」。
对应的检查项目录与结果已删除（git 历史里有）。

## 从 g1k 到 rcc：每一步加了什么、带来了什么

| golden | 相对上一步多了什么 | 这一步带来的变化（证据） | 现在的角色 |
|---|---|---|---|
| `g1k-a/b/c`（00；另有确定性的 `g1k-det`） | 起点：Bridge 默认配置加主表里「采用」的改动（见注 52） | 第 1 步整网梯度：TPU 对 `g1k-a` 整网 rel_L2 1.28e-3（见注 52） | 归因参照（不重算）；L1–L3 第 1 步比较的参照；日常回归 R25 的单一基线 `g1k-a`（见注 52） |
| `rc-a..e`（03） | 整层重算 + 路由回放（`routing_replay.py`） | 改变首次前向；1000 步统计在噪声内（03 README 结论 4） | GPU 生产程序本身；对它的差写成已解释差异（ALIGNMENT-REPORT） |
| `rcx-a..e`（01 第 6 节） | 输出头 dgrad 换成关 TF32 的 FP32 乘法（`rr_arms.py` 的 `dgrad-exact` 补丁） | `rcx` − `rc`：301–1000 步 loss −9.8e-5，第 1000 步 param_norm +1.24e-2；当时算的比例已作废（见注 53） | 归因证据（比例作废） |
| `rce-a`（`b..e` 只作参考附录） | BF16 权重、全部计算用 FP32（`param_swap.py` 的 `Bf16WeightsFp32Compute`），真关 TF32，unfused 注意力 | 漂移走到 TPU 的约九成，但输出头 dgrad 仍有偏（见注 54） | 参考交叉核对（2026-11-05 晚到 11-06 早曾是第一层基线，已取消） |
| `rcec-a` | `rce` + `dgrad-chunk`：真精确 | 比 `rce-a` 多漂 +8.87e-3，与 TPU 精确带符号均值差 8.3e-9 | 参考交叉核对：真精确裁判 |
| `rcc-a..j` | `rc` + `dgrad-chunk`（GPU 生产程序只改输出头 dgrad） | `rcc-a` 第 1000 步 param_norm 对 `rc` 均值 +2.07e-2，TPU 生产 +2.09e-2；v5p / v7x 当前配置对 `rcc-a` 第一层全过（上面「第一层的结果」） | 最终 golden（第一层基线 `rcc-a`，噪声 `rcc-a..j` 10 份两两 45 对） |

各代号的定义见术语表：
[`g1k`](../../docs/glossary.md#g1k)、
[`rc`](../../docs/glossary.md#rc)、
[`rcx`](../../docs/glossary.md#rcx)、
[`rce`](../../docs/glossary.md#rce)、
[`rcec`](../../docs/glossary.md#rcec)、
[`rcc`](../../docs/glossary.md#rcc)、
[`rcc-a`](../../docs/glossary.md#rcc-a)。

- 注 52（`g1k`）：
  - 起点：Bridge 默认配置，只改了上面主表里「采用」的超参、数据、并行，加上 `--full`（关分布式优化器）。
    - 不重算，BF16，TF32 开。
    - 代码和上游相同（只挂了抓数回调）。
  - 第 1 步整网梯度（案例 01「第 1 步整网梯度」一节）：
    - TPU 对 `g1k-a` 整网 rel_L2 1.28e-3。
    - 总览里「最差张量 0.102」是分到 token 少的路由专家（梯度能量占比 1e-9 – 1e-8）。
    - 它来自 0.5% 路由槽位在近似平局处翻转；GPU 重跑路由不变，所以噪声为 0。
  - 日常回归 R25 的单一基线 `g1k-a`：新 LKG `reg-t0-fa-a`、`reg-t1-fa-a` 于 2026-11-05 换代。
- 注 53（`rcx`）：
  - 当时算出解释了 TPU 对 `rc` 的 loss 偏移的 73%、param_norm 差的 59%（案例 01 第 6 节「新协议下的验收」）。
  - ~~59%~~ 已被阶梯取代：`rcx` 的 dgrad 也不精确，按 `rc`、`rcc`、`rcec` 三级阶梯重拆见案例 01「一页看完」。
- 注 54（`rce-a`）：
  - 默认配置下的同款 `g-bw-1k`：param_norm 漂移走到 TPU 的约九成。
  - 但它的输出头 dgrad 是 cuBLAS FP32 一次累加，仍有偏（TPU 精确比它多漂 +8.7e-3）。

### 各 golden 用了哪些补丁

- `g1k`（第一阶段 GPU golden）、`rc`（GPU 生产 golden）的代码和上游 Megatron 相同，只挂了不改数值的抓数回调。
  - `rc` 另加了路由回放，那是运行时补丁。
- `rcx`、`rce`、`rcec`、`rcc` 都用了 janus 侧的数值补丁：
  - `rcx`（归因用的中间版）是 `rr_arms.py` 的 `dgrad-exact`；
  - `rce`（精确轨迹，BF16 权重、全程 FP32 计算）是 `param_swap.py` 的 `Bf16WeightsFp32Compute`；
  - `rcec`（真精确轨迹）是两者再加 `dgrad-chunk`；
  - `rcc`（最终 golden）只有 `dgrad-chunk`。
- 都没改 Megatron 源码。

### 噪声和诊断用的 GPU 运行：设置和对应的 golden 差在哪

| 运行 | 和 golden 一样的地方 | 不一样的地方（meta 或发起命令） | 用途 |
|---|---|---|---|
| `g-ctl`、`g-ctl-b`、`g-ctl-c` | 和 `g1k` 同一个数值设置：默认、不重算、关分布式优化器 | `--no-dist-opt`，不带 `--full`（梯度钩子不挂，优化器不包）；抓数参数见注 55 | 把 GPU 噪声带从 3 份扩到 10 份（案例 01 第 6 节「噪声带」行） |
| `g-gn-a/b` | 同上 | `--group-norms`、`--dump-steps 1000` | 噪声带；分组范数漂移定位 |
| `g-rs-a/b` | 同上 | `--group-norms`、`--row-stats-every 50`、`--dump-steps 100000` | 噪声带；输出层逐行统计 |
| `g-probe-a..f` | 同上 | `--no-dist-opt --group-norms --resid-probe-every 25`；`c`、`d` 跑了 300 步。没有 meta（见注 56） | 精确轨迹和 `dgrad-exact` 对比时的默认参照（案例 01 用的是 `a/b/e/f`） |
| `g-bw-1k`、`g-bw-1k-p` | 默认配置（不重算）下的精确轨迹 | `--fp32 --bf16-weights`，关 TF32，unfused 注意力；`-p` 多一个 +1e-6 扰动。桶里没有 meta，设置以案例 01「精确轨迹」行为准 | 证明精确轨迹站在 TPU 一边，是 `rce` 的前身 |

`g-` 前缀的含义见术语表[「运行名前缀」](../../docs/glossary.md#run-prefixes)。

- 注 55（`g-ctl` 的抓数参数）：
  - `g-ctl` 用 `--dump-steps 1000`；
  - `-b/-c` 是 `--dump-steps 100000`（等于不抓）加 `--group-norms`。
- 注 56（`g-probe`）：桶里只上传了 `metrics.jsonl`，没有 meta，设置以案例 01「残差流探针」行为准。
