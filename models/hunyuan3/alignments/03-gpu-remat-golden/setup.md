# 03 · B200×8 自比，整层重算 + 路由回放（截 4 层）· 设置

- **GPU 自己比自己**：和 [`00`](../00-gpu-noise-and-drills/setup.md) 的硬件、镜像、模型、数据、训练超参完全相同，**只多两项：整层激活重算、路由回放**。
- 配置对照见 [`../../config-alignment.md`](../../config-alignment.md)「重算」两行。
- 每次运行的完整参数见 [`../../frameworks/megatron/README.md`](../../frameworks/megatron/README.md)「第一阶段每次运行的参数」末尾的重算 golden 表。
- GPU golden（`g1k` / `rc` / `rcx` / `rce` / `rcec` 和最终的 `rcc`）每一份是怎么产生的、和上游默认差在哪，见 [`../../goldens.md`](../../goldens.md)。
  - `g1k`（第一阶段 GPU golden，不开重算，1000 步，3 份）、`rc`（GPU 生产 golden，整层重算加路由回放，即原版生产程序，5 份）；
  - `rcx`（生产 golden，输出头 dgrad 换 FP32 一次累加）；
  - `rce`（精确轨迹，BF16 权重、全程 FP32 计算）、`rcec`（真精确轨迹，精确轨迹加输出头 dgrad 分段累加）；
  - `rcc`（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加，其余设置与 rc 相同，10 份）。
  - 代号的意思见[术语表](../../../../docs/glossary.md#golden-codenames)。

## 两份 golden 各管什么

| golden | 设置 | 用途 |
|---|---|---|
| **`rc-a`..`rc-e`**（本实践） | `g1k-a` 的设置 + 整层重算 + 路由回放 | **生产近似 golden**；正式迁移验收对最终 golden `rcc-a`，`rc` 只做第二层对照（01 [README「golden 的分工」](../01-v5p-acceptance/README.md#golden-的分工)） |
| `g1k-a`/`b`/`c`（`00`） | 不重算 | **归因参照**（见表下） |

- **`rc` 为什么当生产近似 golden**：
  - 生产训练都开重算；
  - TPU 侧也是重算且保存路由结果（`remat_policy=custom` + `moe_routing_maps=device`），两边在「路由不重选」上一致。
- **`g1k` 怎么用来归因**：不重算的轨迹少一个变量。
  - TPU 对 `rc` 有差、对 `g1k` 也有同样的差，就和重算无关；
  - 只对其中一份有差，差在重算上。

**为什么不直接把 `g1k` 换掉**

- `g1k` 已经是 `01` 案例 01 与日常回归 T1 档里引用的参照，换掉会让旧结论失去可比对象。
- 「重算 + 回放」本身改不改训练轨迹，要靠 `rc` 对 `g1k` 才看得出来（结论见 README）。

## 重算设置（假设的，可能重做）

| 项 | 取值 | 出处 |
|---|---|---|
| `recompute_granularity` | `full`（整层） | `train_hy3.py --recompute`，与 00 的 `r-remat`（整层重算、不回放，确定性，3 步）相同；`meta.json` 的 `recompute` |
| `recompute_method` / `recompute_num_layers` | `uniform` / `1`：每层单独一个重算块，只存层输入 | 同上 |
| `moe_enable_routing_replay` | `true`；`routing_replay.py` 首次前向 RECORD、反向重算 REPLAY_BACKWARD，每个 microbatch 清空 | `train_hy3.py --routing-replay` |
| 回放什么 | 只有 top-k 专家下标（连顺序）；路由概率在重算里由重算的 logits 重新算 | Megatron-Core `moe_utils.topk_routing_with_score_function` 的 sigmoid + expert_bias 分支 |

- **真实的生产重算配置还不知道**（整层还是选择性、每块几层、存哪些张量）。
- 这里取的是 Megatron 最常见的整层、逐层重算。
- 拿到真实配置后，如果不同，按真实配置重跑 `rc-*`（GPU 生产 golden），旧的这批改名保留，不覆盖。

## 和 `g1k` 一样的地方

| | 取值 |
|---|---|
| 硬件与规模 | NVIDIA B200 × 8，单机 GKE Pod |
| 并行切分 | DP 8 + EP 8，TP / PP / CP 1 |
| 镜像 | `nvcr.io/nvidia/nemo@sha256:fc75bf516de6845fd85fe627c7c18e69e7568428fe56438e31ff3b1917044b6c`（torch 2.14.0a0、CUDA 13.4、Megatron-Core 0.20.0、Megatron-Bridge 0.7.0、Transformer Engine 2.20.2、NCCL 2.31.2） |
| 模型 | 混元 3 截前 4 层、去 MTP，真权重；清单 sha256 `183274dbf17e…` |
| 数据 | fineweb-edu `fw_hy3_text_document`（`.idx` sha256 `83d5a23b14d3…`），种子 1234 |
| 训练 | 全局 batch 16、microbatch 1、序列 4096、学习率 1e-5 常数 |
| 设置 | 默认设置（不设三个确定性变量，`deterministic_mode: false`）；分布式优化器关（`--full`）；TF32 开 |

`rv-det`（回放验证）是确定性设置，只用来做逐位核对，不进噪声参照。

## 软件版本

本实践只有 GPU，版本同 [00 的「软件版本」](../00-gpu-noise-and-drills/setup.md#软件版本)。`rc-*` 的 `meta.json` 直接记了镜像 digest `fc75bf51…` 与 torch、CUDA 13.4、Megatron-Core 0.20.0、Megatron-Bridge 0.7.0、Transformer Engine 2.20.2+44f31bbe、NCCL 2.31.2（结果文件 [`remat-noise.results.json`](results/remat-noise.results.json) 记着它们的 `meta_sha256`）；cuBLAS 不记；同一 digest 的镜像在混元 4 的 Pod 里实测加载的是 13.8.1（`rc` 运行自己没读过加载的库；容器环境变量写 13.8.0.4，见 01 setup「软件版本」的 cuBLAS 行与 00 的「软件版本」）。
