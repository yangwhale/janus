[janus](../../README.md) › [用户指南](README.md) › 第 1 章

[回目录](README.md) ｜ [下一章：第 2 章 制作 GPU golden →](02-gpu-golden.md)

# 第 1 章 · 准备环境与数据

这一章在 GPU 一侧把「被比较的东西」备齐：

- 一台 8 卡 B200 的 Pod；
- 一份截成 4 层的混元 3 权重（同时有 HF 与 Megatron 两种格式）；
- 一份分好词的训练数据。

TPU 一侧用的是**同一份** HF 权重和**同一批**数据（第 3 章）。所以这里每一步都要留下能核对的锚点：张量个数、sha256、参数总数。

- 完整命令：GPU 侧操作手册 [`frameworks/megatron/README.md`](../../models/hunyuan3/frameworks/megatron/README.md)。
- 本章讲做什么、为什么、做完该看到什么。

| 这一步 | |
|---|---|
| 做什么 | 在 GPU 一侧备齐被比较的东西：8 卡 B200 Pod、截 4 层的混元 3 权重（HF 与 Megatron 两种格式）、分好词的训练数据 |
| 怎么做 | 照下面「步骤」0–5 步；命令都在 Pod 里跑，先 `source /data/env.sh` |
| 做完看到什么算对 | 下表的锚点全对上：<br>`tensors 1781`<br>清单 sha256 `183274db…7014`<br>Megatron 参数数 12,386,079,744 且最后一行 `IMPORT_DONE`<br>`.idx` sha256 `83d5a23b…fdad2` |
| 下一步 | [第 2 章](02-gpu-golden.md)：在这份权重和数据上造 GPU golden |

## 本章做完手里有什么

| 产物 | 默认位置（Pod 里） | 锚点：对上了才往下走 |
|---|---|---|
| janus 仓库检出、环境脚本 | `/data/janus-src`、`/data/env.sh` | `source /data/env.sh` 之后 `import torch` 不报错 |
| 原始权重（只下前 4 层用到的分片） | `/data/hy3-full` | 10 个分片，约 27 GB |
| 截 4 层的 HF 目录 | `/data/hy3-4l` | `tensors 1781`；`tensor_sha256.json` 的 sha256 = `183274db…7014` |
| Megatron 检查点 | `/data/hy3-4l-mcore` | 参数数 12,386,079,744，最后一行 `IMPORT_DONE` |
| 分词后的数据 | `/data/ds/fw_hy3_text_document.{bin,idx}` | 726000 篇文档；`.idx` 的 sha256 = `83d5a23b…fdad2` |

锚点的完整取值与出处见各步的链接。

## 为什么截 4 层

- **两种层类都在**：截 4 层两种都覆盖到。
  - 混元 3 第 0 层是稠密层；
  - 第 1–3 层是 MoE 层（192 个专家、top-8、1 个共享专家）；
  - MTP 截层时关掉。出处：
    - [`frameworks/maxtext/README.md`](../../models/hunyuan3/frameworks/maxtext/README.md) 开头
    - [`../../models/hunyuan3/config-alignment.md`](../../models/hunyuan3/config-alignment.md#模型结构)「模型结构」
- **单机能跑完整套**：4 层的参数约 124 亿。
  - B200×8 一台、TPU v5p-8 / v7x-4 一个切片都装得下。
  - L0–L4 全部检查都能在这个规模上做。
- **截层本身也要验证**：截 2 层与截 4 层用同一份权重，第 1 步 embedding、前 2 层前向、路由逐位相同。
  - 出处：`phase1-truncate`，[00 结果一览](../../models/hunyuan3/alignments/00-gpu-noise-and-drills/README.md)。

80 层的生产形状不在前几章里，见第 7 章日常回归的「档位」。

## 步骤

所有命令在 Pod 里执行。

- 截层、转格式、数据预处理、训练之前都要先 `source /data/env.sh`。
- 否则镜像里 NCCL 版本冲突，`import torch` 直接报错。

```bash
source /data/env.sh
```

### 0. 准备之前

- 看一遍默认路径、磁盘与「GPU 独占」的要求。
- 详细：[「0. 准备之前要知道的」](../../models/hunyuan3/frameworks/megatron/README.md#0-准备之前要知道的)

### 1. 起 Pod、放仓库

- 起 Pod，把仓库打成 bundle 拷进去。
- `pod.yaml` 复制一份到仓库外，把 `nodeSelector` 改成你自己 8×B200 节点池的名字。
  - 仓库工作树保持干净，第 3 章打包会检查。

```bash
kubectl apply -f <改好的 pod.yaml>
```

做完看到什么算对：

- 仓库检出在 `/data/janus-src`，环境脚本在 `/data/env.sh`。
- `source /data/env.sh` 之后 `import torch` 不报错。

详细：[「1. 起 Pod、把仓库放进去」](../../models/hunyuan3/frameworks/megatron/README.md#1-起-pod把仓库放进去)

### 2. 下权重

- `tencent/Hy3` 固定 revision。
- 只下 embed、输出层与前 4 层所在的分片。
- 关键命令：一段 `curl` + Python 选分片（见下面「详细」）。

做完看到什么算对：

- `/data/hy3-full` 下 10 个分片，约 27 GB。

详细：[「2. 下权重」](../../models/hunyuan3/frameworks/megatron/README.md#2-下权重只下需要的分片)

### 3. 截 4 层、去 MTP

- 截 4 层、去 MTP，写逐张量 sha256 清单。

```bash
python3 /data/jw/truncate_hy3.py /data/hy3-full /data/hy3-4l 4
```

做完看到什么算对：

- 输出 `/data/hy3-4l`，`tensors 1781`。
- `tensor_sha256.json` 的 sha256 = `183274db…7014`。

详细：[「3. 截 4 层、去 MTP」](../../models/hunyuan3/frameworks/megatron/README.md#3-截-4-层去-mtp)

### 4. 转 Megatron

- HF → Megatron（单卡，几分钟）。

```bash
CUDA_VISIBLE_DEVICES=0 python3 /data/jw/import_hy3.py /data/hy3-4l /data/hy3-4l-mcore
```

做完看到什么算对：

- 输出 `/data/hy3-4l-mcore`，参数数 12,386,079,744。
- 最后一行 `IMPORT_DONE`。

详细：[「4. 转 Megatron」](../../models/hunyuan3/frameworks/megatron/README.md#4-转-megatron)

### 5. 准备数据

- fineweb-edu 一个 parquet → jsonl → Megatron `preprocess_data.py`。

```bash
bash /data/janus-src/models/hunyuan3/frameworks/megatron/prep_data.sh
```

做完看到什么算对：

- 输出 `/data/ds/fw_hy3_text_document.{bin,idx}`，726000 篇文档。
- `.idx` 的 sha256 = `83d5a23b…fdad2`。

详细：[「5. 准备数据」](../../models/hunyuan3/frameworks/megatron/README.md#5-准备数据)

### 容易踩的点

- **权重 revision 要钉死**：第一阶段用 `a960ebc3…`。
  - `model.safetensors.index.json` 的 sha256 写在第 2 步里。
  - 换了 revision，后面所有 sha256 锚点都对不上。
- **数据脚本按 `main` 分支下载 parquet，没有钉版本**。
  - 所以先核 `fw.parquet` 的 sha256 再往下做。
  - sha256 只能发现上游变了，不能恢复：HF 数据集当时的 revision 没记，原件的备份位置也没登记（待补：把 `prep_data.sh` 的 `resolve/main` 换成提交号，或存一份原件并写明位置）。
- **参数数差 576 是正常的**：
  - HF 侧 1781 个张量共 12,386,080,320 个元素；
  - 多出的 576 是 3 个 MoE 层 × 192 个专家的 `expert_bias`；
  - Megatron 把它当 buffer，不计入参数数。
- **`/data` 是本地 SSD，删 Pod 就没了**。
  - 一次 30 步全量抓数的运行目录约 41 GB。
  - 要留的先用 `upload_run.py` 传到 `gs://<POC 桶>/…`。
- **抓数时 GPU 独占**：别的作业抢资源可能改变 kernel 选择和通信时序。
  - 多人共用一台时用 `flock /data/gpu.lock <命令>` 排队。
  - 原因见[方法详解：抓数](../reference/dump.md#三条规矩以及为什么)「三条规矩，以及为什么」。

## 版本要记下来

### 记进 `meta.json` 的

每次运行都写进运行目录的 `meta.json`，第 4 章的 `L0.config.diff` 逐项比它们：

- 镜像 digest、代码提交、关键库版本；
- 硬件与并行方式、确定性开关；
- **运行时实际生效**的配置。

容器里拿不到镜像 digest，`pod.yaml` 通过环境变量 `JANUS_IMAGE` 把它设好。混元 3 第一阶段的运行还没有这一步，`meta.json` 的 `image` 只记了 tag（[环境「第一阶段的缺口」](../reference/environment.md#每次运行都要记下的)）。

要钉死哪些、为什么，见[方法详解：环境](../reference/environment.md#每次运行都要记下的)「每次运行都要记下的」。

### GPU 侧镜像

- `nvcr.io/nvidia/nemo:26.10`
  - torch 2.14、CUDA 13.4
  - Megatron-Core 0.20.0、Megatron-Bridge 0.7.0
  - Transformer Engine 2.20.2、NCCL 2.31.2

### `env.sh` 已经处理的三个坑

- NCCL 版本冲突；
- NCCL 网络插件；
- 确定性模式要 Triton 缓存目录。

原因见[方法详解：环境](../reference/environment.md#第一阶段踩到的坑)「第一阶段踩到的坑」。

---

下一步：环境与数据就绪。第 2 章先让 GPU 自己和自己比，证明工具可信、量出噪声，再跑最终的 golden。

[回目录](README.md) ｜ [下一章：第 2 章 制作 GPU golden →](02-gpu-golden.md)
