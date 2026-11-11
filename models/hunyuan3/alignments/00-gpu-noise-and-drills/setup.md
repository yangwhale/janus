# 00 · B200×8 自比（截 4 层）· 设置

- **GPU 自己比自己**：同配置的两次 GPU 运行，一次当被测、一次当 golden。
- 配置对照见 [`../../config-alignment.md`](../../config-alignment.md) 的 GPU 一栏。
- 每次运行的完整参数见 [`../../frameworks/megatron/README.md`](../../frameworks/megatron/README.md)「第一阶段每次运行的参数」，这里不重复。
- GPU golden（`g1k` / `rc` / `rcx` / `rce` / `rcec` 和最终的 `rcc`）每一份是怎么产生的、和上游默认差在哪，见 [`../../goldens.md`](../../goldens.md)。
  - `g1k`（第一阶段 GPU golden，不开重算，1000 步，3 份）；
  - `rc`（GPU 生产 golden，整层重算加路由回放，即原版生产程序，5 份）；
  - `rcx`（生产 golden，输出头 dgrad 换 FP32 一次累加；归因用的中间版，FP32 一次累加本身仍有偏）；
  - `rce`（精确轨迹，BF16 权重、全程 FP32 计算；排查时当裁判用，输出头 dgrad 仍是 cuBLAS FP32 一次累加，有偏，不是真精确）；
  - `rcec`（真精确轨迹，精确轨迹加输出头 dgrad 分段累加）；
  - `rcc`（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加；其余设置与 rc 相同，10 份）。
  - 代号的意思见[术语表](../../../../docs/glossary.md#golden-codenames)。

| | 两边相同 |
|---|---|
| 硬件与规模 | NVIDIA B200 × 8，单机 GKE Pod |
| 并行切分 | DP 8 + EP 8，TP / PP / CP 1（并行等价类运行另改，见运行参数表） |
| 镜像 | `nvcr.io/nvidia/nemo:26.10`（torch 2.14.0a0、CUDA 13.4、Megatron-Core 0.20.0、Megatron-Bridge 0.7.0、Transformer Engine 2.20.2、NCCL 2.31.2） |
| 模型 | 混元 3 截前 4 层、去 MTP（`model/truncate_hy3.py`），真权重；清单 sha256 `183274dbf17e…` |
| 数据 | fineweb-edu，`prep_data.sh` 预处理成 `fw_hy3_text_document` |
| 训练 | 全局 batch 16、microbatch 1、序列 4096、学习率 1e-5 常数、种子 1234 |
| TF32 | 开（容器默认）；只有 `g61-notf32` 关（`NVIDIA_TF32_OVERRIDE=0`） |

## 软件版本

本实践只有 GPU。镜像那一行的版本逐项出处：

| 组件 | 版本 | 证据出处 |
|---|---|---|
| 镜像 | NeMo 26.10，digest `fc75bf51…` | 第一阶段的 `meta.json` 只记 tag；digest 见 [`pod.yaml`](../../frameworks/megatron/pod.yaml) 的注释（第一阶段用的就是它） |
| torch / CUDA | 2.14.0a0+b2c75dd062.nv26.09 / 13.4 | [`phase1-l1.results.json`](results/phase1-l1.results.json) 的 `environment` |
| Transformer Engine | 2.20.2+44f31bbe | 同上 |
| cuBLAS | 13.8.0.4 | 同上 |
| Megatron-Core / Megatron-Bridge | 0.20.0 / 0.7.0（容器自带，提交未登记） | 复现审计在容器里读到（[`repro-audit.md`](results/repro-audit.md)）；第一阶段 `meta.json` 的这两项是 `null`（[环境](../../../../docs/reference/environment.md)） |
| NCCL | 2.31.2 | 容器自带（[环境](../../../../docs/reference/environment.md)「第一阶段踩到的坑」第 1 条） |

噪声宽度在这套版本上量得；换镜像要重量。

**两种设置**（确定性 / 默认）

- 含义见 [`../../../../docs/reference/environment.md`](../../../../docs/reference/environment.md)。
- 噪声参照只用同一种设置的运行对。
- 分布式优化器开关也算设置（README 第 6 条）。
