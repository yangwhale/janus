# 01 · B200×8 → TPU v5p-8（截 4 层）· 设置

- 配置对照见 [`../../config-alignment.md`](../../config-alignment.md)（模型一层，所有实践共用）。这里只写本实践特有的东西。
- 数字出自 golden `g1k-a`（第一阶段 GPU golden 的第 1 份，不开重算，1000 步）与被测 `t-loss30`（TPU 30 步冒烟，只看 loss）两个运行目录的 `meta.json`。
- 本实践的结果都是对 `g1k`（第一阶段 GPU golden，不开重算，1000 步，3 份）算的。
  - 开整层重算 + 路由回放的生产近似 golden `rc-*`（GPU 生产 golden，5 份）在 [`03`](../03-gpu-remat-golden/setup.md)；
  - `t-a`（TPU 早期 1000 步运行，第 1 步全量抓数）对它的比较见那里的结论 6。
- 运行目录在 POC 桶里，不对外；清单见 `runs.yaml`。
- GPU golden（`g1k` / `rc` / `rcx` / `rce` / `rcec` 和最终的 `rcc`）每一份是怎么产生的、和上游默认差在哪，见 [`../../goldens.md`](../../goldens.md)。
  - 代号：`g1k` 第一阶段 GPU golden；`rc` GPU 生产 golden；`rcx` 生产 golden，输出头 dgrad 换 FP32 一次累加；`rce` 精确轨迹，BF16 权重、全程 FP32 计算；`rcec` 真精确轨迹，精确轨迹加输出头 dgrad 分段累加；`rcc` 最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加。
  - 代号的意思见[术语表](../../../../docs/glossary.md#golden-codenames)。

## 两边

代号：`g1k-a/b/c` 第一阶段 GPU golden（3 份）；`t-loss30` TPU 30 步冒烟，只看 loss。

| | 源（golden：`g1k-a` / `g1k-b` / `g1k-c`） | 目标（被测：`t-loss30`） |
|---|---|---|
| 硬件与规模 | NVIDIA B200 × 8，单机 | TPU v5p，8 片 |
| 拓扑 / 进程数 | 1 node，8 进程 | `2x2x2`，2 个主机进程 |
| 并行切分 | DP 8 + EP 8，TP / PP / CP 1；不开分布式优化器（全量抓数要求关） | FSDP 8，其余 1 |
| 镜像 | `nvcr.io/nvidia/nemo:26.10`（meta 里没有 digest；`janus/dump/runinfo.py` 对这种情况会告警） | MaxText 官方 `tpu_pre_training`，digest `sha256:b152355a1425c28e99841c014703495059705de1733df8ac837ba43b4fe52fcf` |
| 框架代码 | Megatron-Bridge（容器自带）+ `train_hy3.py`（sha256 `4e41f0ae455c…`） | `maxtext` @ `64cd9a640b8ea65aaf856b6f24755f61ee97e057` |
| janus | 未记录（第一阶段 golden 的 meta 不记 janus commit） | `16af8f7b116b337fae03265c26bf0ba7256c6fc7` |
| 关键版本 | torch 2.14.0a0、CUDA 13.4、Transformer Engine 2.20.2、NCCL 2.31.2 | jax / jaxlib 0.11.1、flax 0.12.9、optax 0.2.8、orbax-checkpoint 0.12.4、libtpu 0.0.46（2026-08-12 构建 `cl/963683785`；版本号按构建号认，见「软件版本」） |
| 训练配置 | `train_hy3.py` 的参数（见表下），默认设置（非确定性） | `frameworks/maxtext/hy3-4layers.yml`，命令行另加 `mtp_num_layers=0`（见表下） |
| 权重 | HF 截 4 层 → Megatron（`import_hy3.py`）；清单 `tensor_sha256.json` 的 sha256 `183274dbf17e…` | 同一份 HF → MaxText（`convert_hf_to_maxtext.py`），FP32；清单 sha256 相同 |
| 数据 | `fw_hy3_text_document`（Megatron `GPTDataset`） | 回放 `g1k-a` 的 `data/batches`（清单 sha256 `e4db6392d89a…`），`replay_of` 记着 GPU 数据清单 sha256 `83d5a23b14d3…` |
| TF32 | 开（容器默认），影响路由反向的 12 次 FP32 GEMM | 无 |

**「训练配置」行的细节**

- GPU：`--gbs 16 --mbs 1 --seq 4096 --ep 8 --lr 1e-05 --seed 1234 --dump-steps 1,1000 --full`。
- TPU：
  - `t-loss30` 用的是 sha256 `66a03bf31d35…`，改名前叫 `hy3-4l.yml`，内容相同；
  - yml 后来改过两次，各运行用的版本见下面「配置文件版本」。

## 软件版本

以正式验收第一层为准：TPU `t-rmap-on-1000` 对最终 golden `rcc-a`，噪声 `rcc-a … rcc-j` 10 份（`phase2-acceptance-rcc-on`）。

| 侧 | 组件 | 版本 | 证据出处 |
|---|---|---|---|
| TPU | 镜像 | MaxText 官方 `tpu_pre_training`，digest `b152355a…` | `t-rmap-on-1000` 的 `meta.json` 的 `image`；[`frameworks/maxtext/run.sh`](../../frameworks/maxtext/run.sh) 的 `IMAGE` |
| TPU | MaxText | `maxtext` `hunyuan` 分支 `0622774a` | `meta.json` 的 `code` |
| TPU | janus（打进 Pod 的程序） | `958d173e` | `meta.json` 的 `code` |
| TPU | jax / jaxlib | 0.11.1 / 0.11.1 | `meta.json` 的 `versions` |
| TPU | libtpu | 0.0.46（推断；构建 `cl/963683785`，2026-08-12） | `meta.json` 只记了构建号，没记 pip 版本号。推断：混元 4 同一构建号的运行记了 `libtpu==0.0.46`（混元 4 [mini 设置「软件版本」](../../../hunyuan4/alignments/01-mini/setup.md#软件版本)），jax 0.11.1 的 `tpu` 依赖也钉 0.0.46.* |
| TPU | flax / optax / orbax-checkpoint | 0.12.9 / 0.2.8 / 0.12.4 | `meta.json` 的 `versions` |
| TPU | tokamax | 0.0.14（镜像自带的包） | `meta.json` 不记；读镜像层里的 `tokamax-0.0.14.dist-info` 核过（2026-11-11） |
| GPU | 镜像 | NeMo 26.10，digest `fc75bf51…` | `rcc-*` 的 `meta.json` 的 `image`（10 份相同） |
| GPU | torch / CUDA / NCCL | 2.14.0a0+b2c75dd062.nv26.09 / 13.4 / 2.31.2 | `meta.json` 的 `versions` |
| GPU | Megatron-Core / Megatron-Bridge | 0.20.0 / 0.7.0（容器自带，提交未登记） | `meta.json` 的 `versions` 与 `code` |
| GPU | Transformer Engine | 2.20.2+44f31bbe | `meta.json` 的 `versions` |
| GPU | cuBLAS | 13.8.0.4 | `rcc` 的 `meta.json` 不记；第一阶段在同一镜像里读到（[`phase1-l1.results.json`](../00-gpu-noise-and-drills/results/phase1-l1.results.json) 的 `environment.cublas`；第一阶段的 digest 见 [`pod.yaml`](../../frameworks/megatron/pod.yaml) 的注释） |

- 运行目录的 `meta.json` 在 POC 桶里，不入库；结果文件 [`phase2-acceptance-rcc-on.results.json`](results/phase2-acceptance-rcc-on.results.json) 的 `runs.*.meta_sha256` 记着每份的 sha256。
- 本页上面「两边」表是早期运行 `t-loss30` 的，版本与这里相同，MaxText 提交不同。
- **结论在这些版本上测得。** 换 libtpu 或 jax 不能假定只换了求和顺序，要按混元 4 [`libtpu-052.md`](../../../hunyuan4/alignments/04-tpu-self/libtpu-052.md) 的做法重建 golden、重测。

## 配置文件版本

- `hy3-4layers.yml` 改过几次。
- 已经跑完的运行保留当时的 sha256，结论按当时的配置读。
- 运行目录 `meta.json` 的 `config.overrides` 记着命令行覆盖。
- 表里的运行（都是 TPU）：
  - `t-loss30` 30 步冒烟；`t-a` 早期 1000 步、第 1 步全量抓数；`t-nohooks` 同 `t-a` 但不加钩子；`t-hlo`、`t-hlo2` 钩子自检运行（带 XLA dump）；
  - `t-ga2-30` 30 步，只把累加改成每设备 1 行 + GA 2；`t-fix-*-30` 30 步，GA 2 加精度开关；`t-ga2-noexcess-30` 在 `t-ga2-30` 上关 excess precision；
  - `r-l2-*` 逐层隔离回放；`r-l2-noexcess` 关 excess precision 的逐层回放；`r-l2-on` 当前对齐配置的逐层回放；
  - `t-vmem24-1000` 只改片上内存上限的 1000 步对照；`t-rmap-1000` TPU 生产配置 1000 步（整层重算、只存路由）；
  - `t-st-on-100` 当前对齐配置 100 步（量步时与显存）；`t-rmap-on-1000` v5p 当前对齐配置 1000 步。

| yml sha256 | janus 提交 | 内容 | 用它的运行 |
|---|---|---|---|
| `66a03bf31d35…` | 改名后到 `6549396` | `per_device_batch_size: 2`、GA 1；不开精度开关 | `t-loss30`、`t-a`、`t-nohooks`、`t-hlo`、`t-hlo2`；`t-ga2-30`（命令行改成 1 + 2） |
| `7704115ba7a7…` | `67208dd` | 改成 `per_device_batch_size: 1`、GA 2 | `r-l2-*`（逐层回放）、`t-fix-*-30`（精度开关走命令行）、`r-l2-noexcess`、`t-ga2-noexcess-30`；`t-vmem24-1000`（命令行改回 2 + 1，与 `t-a` 相同） |
| `2a76c7590866…` | `3a5f03f` | 打开三个精度开关（见表下）；要 maxtext hunyuan@4d4ffac 或更新 | `t-rmap-1000`、`r-l2-*` 之后到 2026-11-05 的运行 |
| `a238883b7f91…` | 本 PR | 再打开 `dense_fp32_accumulation`（见表下）；要 maxtext hunyuan@0622774 或更新 | `t-st-on-100`、`r-l2-on`、`t-rmap-on-1000` 及之后 |

- **`2a76c7590866…`**：
  - 打开 `routed_gate_bf16_rounding`（路由核先舍 BF16 再乘）、`rmsnorm_single_rounding`（RMSNorm 在 FP32 里算完只舍一次）、`mlp_activation_round_per_op`（门控激活按 GPU 的两个算子分别舍入）；
  - 注释里不存在的文档引用改成 `config-alignment.md` 与 issue。
- **`a238883b7f91…`**：
  - `dense_fp32_accumulation`：注意力与稠密 MLP 投影 FP32 累加、只舍一次；
  - 2026-11-05 决定：对齐优先。

## 和模型对照表不一样的地方

- 本实践没有偏离对照表。
- 对照表里标「有差异」的项在本实践都存在，跑结论时要一起看：

| 差异 | 对本实践的影响 | 怎么定责 |
|---|---|---|
| 切分：GPU DP 8 + EP 8，TPU FSDP 8 | 规约顺序不同。第一阶段只换 EP 度，反向就不逐位 | L2 逐层比，看差从哪层、哪个张量开始 |
| 累加分组：GPU 每卡 2 个 microbatch 分两次；`t-loss30`（30 步冒烟）/ `t-a`（早期 1000 步）时 TPU 每设备一次 2 行 | 第一阶段改累加分组后，embedding 梯度差到噪声上限的 31 倍 | 已改成和 GPU 一样的 1 + 2（`t-ga2-30`（30 步，只改累加分组）起）；30 步只改这一项没有变好（`results/phase2-precision-fixes.md`） |
| TF32：GPU 路由反向是 TF32，TPU 没有 | 影响路由反向的梯度，30 步 loss 看不出来 | GPU 侧补一组 `TF32=off` 的 golden 再比 |
| 注意力 kernel：cuDNN 与 splash | 末位不同 | L1 注意力回放 |

## config_diff

- **目前用的**：`catalogs/phase2-t-loss30.yaml`（`t-loss30` 30 步冒烟的检查项目录）的 `L0.config.diff.dropout`。
  - 只查了 6 项：dropout 为 0、学习率、实际权重衰减、路由偏置更新率、检查点清单、数据清单。
- **全量映射**：`catalogs/phase2-tpu.yaml` 的 `L0.config.*`。
  - 草稿，没有按它跑；
  - L0 正式检查见 `phase2-l0-state`，配置项见 `phase2-t-loss30` 的 `L0.config.*`。
