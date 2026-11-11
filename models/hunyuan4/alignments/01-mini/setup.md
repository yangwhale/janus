[janus](../../../../README.md) › [混元 4](../../README.md) › [mini](README.md) ｜ 设置

# 混元 4 · mini（hy4-mini）· 设置

只写本实践和[配置对照表](config-alignment.md)不一样的地方。每一份运行的来历见 [`goldens.md`](goldens.md)。

## T1（hy4-mini）

| 项 | TPU（被测；语义以它为准） | GPU（golden） |
|---|---|---|
| 硬件 | v5p 8 片（2x2x2，2 台主机，8 个 JAX 设备） | B200 × 8，单机 |
| 程序 | MaxText，`maxtext` 混元 4 集成分支（提交见各运行 `meta.json` 的 `code`） | Megatron-Core 0.20 + TE（NeMo 26.10 镜像所带）之上的 [`frameworks/megatron/`](../../frameworks/megatron/README.md) |
| 镜像与库版本 | 见下面「软件版本」 | 同左 |
| 配置 | [`frameworks/maxtext/hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml) + `--stage mini` | `PROTOCOL.md` 第 1.2 节；各运行的 `meta.json` |
| 初始权重 | hy4-mini 随机初始化的 HF 目录转换的 MaxText 检查点 | 同一 HF 目录 |
| 数据 | `replay.py` 原样回放 GPU 写下的 batch 与 loss_mask | 写下每个 rank、每步、每个 microbatch 的 batch（`--save-batches`） |

命令：`YML=hy4-mini-t1.yml STAGE=mini GOLDEN=<GPU 基线运行目录> bash run.sh <运行名> <步数>`（300 步回放 `g2-a`，1000 步回放 `g3-a`；`GOLDEN` 指定从哪一份 GPU 运行读 batch 与 loss_mask）（[`frameworks/maxtext/README.md`](../../frameworks/maxtext/README.md)）。这条命令用的是现行 yml 与现行 MaxText。复现现行比较 `tz-off-rep` 用现行 yml（2026-11-11 起 `sa_use_base2_exp: false`）与 MaxText `b87a55aa`，1000 步回放 `g3-a`；它用的 janus 提交与覆盖参数以 `tz-off-rep` 的 `meta.json` 为准。上一代 `t1zg-a`（base2 开，MaxText `47d31984`）与再上一代 `t1zw-a`（base2 开，MaxText `2c28a4f4`，janus `9c0f19a` 起 yml 里打开了 M42b）的提交与覆盖参数各以 `meta.json` 为准；现行 yml 已关 base2，照它跑出来的不是这两代。复现更早的 `t1zu-a` 用 janus `2679a837` 的 yml 与 MaxText `3741e818`（覆盖参数里没有改开关的，见 `t1zu-a` 的 `meta.json`）。复现更早一代 `t1zr-a` 要用 janus `dec9315c` 的 yml、MaxText `c902c147`，再加覆盖参数 `ihc_force_bf16_rounding=true`（那一版 yml 里还没有 M36，见 `t1zr-a` 的 `meta.json` 里的 `overrides`）；现行 yml 另开了 M34、M40、M41，MaxText 之后又有 M37（`transcendental_accuracy_highest` 的代码改动）与 M39，照现行配置跑出来的不是 `t1zr-a`。各运行用的提交以 `meta.json` 为准。

## 软件版本

以现行比较 `tz-off-rep` 为准：TPU `tz-off-rep` 对 GPU `g3-a`，噪声是 `g3-a … g3-p` 16 份。上一代运行的 MaxText 提交见上一段与各运行的 `meta.json`；它们的镜像、jax、libtpu 与下表相同（运行记录里构建号都是 `cl/963683785`）。

| 侧 | 组件 | 版本 | 证据出处 |
|---|---|---|---|
| TPU | 镜像 | MaxText `tpu_pre_training`，digest `b152355a…` | `tz-off-rep` 的 `meta.json` 的 `image`；[`frameworks/maxtext/run.sh`](../../frameworks/maxtext/run.sh) 的 `IMAGE` |
| TPU | MaxText | 集成分支 `b87a55aa` | `meta.json` 的 `code`；[`runs.yaml`](runs.yaml) |
| TPU | janus（打进 Pod 的程序） | `b2bc099f` | `meta.json` 的 `code` |
| TPU | jax / jaxlib | 0.11.1 / 0.11.1 | `meta.json` 的 `versions` |
| TPU | libtpu | 0.0.46（构建 `cl/963683785`，2026-08-12） | `meta.json` 的 `versions.libtpu`（构建号）与 `versions.libtpu_pip`（`libtpu==0.0.46`） |
| TPU | flax / optax / orbax-checkpoint | 0.12.9 / 0.2.8 / 0.12.4 | `meta.json` 的 `versions` |
| TPU | tokamax | 0.0.14（镜像自带的包） | `meta.json` 不记；读镜像层里的 `tokamax-0.0.14.dist-info` 核过（2026-11-11）。MaxText `b87a55aa` 的 splash 从这个包导入（`layers/attention_op.py`）。CPU 上核 base2 语义用的是 0.0.17，不是这个版本（[知识库](../../knowledge.md)） |
| GPU | 镜像 | NeMo 26.10，digest `fc75bf51…` | 16 份的 `meta.json` 的 `image`，都相同 |
| GPU | torch / CUDA / NCCL | 2.14.0a0+b2c75dd062.nv26.09 / 13.4 / 2.31.2 | `meta.json` 的 `versions` |
| GPU | Megatron-Core | 0.20.0；提交未登记 | `meta.json` 的 `versions` |
| GPU | Transformer Engine | 2.20.2（推断） | `meta.json` 不记。推断：同一 digest 的镜像，混元 3 写明带 TE 2.20.2（[`03-gpu-remat-golden/setup.md`](../../../hunyuan3/alignments/03-gpu-remat-golden/setup.md)），混元 3 第一阶段在容器里读到 `2.20.2+44f31bbe`（`phase1-l1.results.json` 的 `environment`） |
| GPU | cuBLAS | 13.8（推断） | `meta.json` 不记。推断：混元 3 第一阶段在容器里读到 `13.8.0.4`（同上 `environment.cublas`），第一阶段用的就是这个 digest（混元 3 [`pod.yaml`](../../../hunyuan3/frameworks/megatron/pod.yaml) 的注释） |
| GPU | janus 训练程序 | `hy4_mcore.py` sha256 `4dab75aa…` | `meta.json` 的 `code` |

- 运行目录的 `meta.json` 在 POC 桶里，不入库。结果文件（如 [`t1zo-l4.results.json`](results/t1zo-l4.results.json)）的 `runs.*.meta_sha256` 记着每一份的 sha256，可以对着核。
- libtpu 版本以构建号和 `libtpu_pip` 为准，不以镜像 digest 为准（[环境](../../../../docs/reference/environment.md#两个模型实测用的版本)）。
- **结论在这些版本上测得。** 换 libtpu 或 jax 不能假定只换了求和顺序，要按 [`libtpu-052.md`](../04-tpu-self/libtpu-052.md) 的做法先离线编译看哪些程序变了，再重建 golden、重测。

## 真实权重截层（L0 与阶段 A 单算子）

| 项 | 值 |
|---|---|
| 检查点 | hy4-preview 截第 0–3 层 + MTP 的 HF 目录（135 个张量，清单 sha256 `8ec763b5…`），按阶段 A–D 转换（`model/convert_hf_to_maxtext.py --stage`） |
| TPU 配置 | [`frameworks/maxtext/hy4-4layers.yml`](../../frameworks/maxtext/hy4-4layers.yml)（不设 `HY4_YML` 时的默认） |
| 单算子回放的 GPU 一侧 | `ga1-a` 第 1 步抓到的算子输入；FP64 真值由 [`analysis/fp64_ops.py`](../../analysis/fp64_ops.py) 算 |
