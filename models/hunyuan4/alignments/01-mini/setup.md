[janus](../../../../README.md) › [混元 4](../../README.md) › [mini](README.md) ｜ 设置

# 混元 4 · mini（hy4-mini）· 设置

只写本实践和[配置对照表](config-alignment.md)不一样的地方。每一份运行的来历见 [`goldens.md`](goldens.md)。

## T1（hy4-mini）

| 项 | TPU（被测；语义以它为准） | GPU（golden） |
|---|---|---|
| 硬件 | v5p 8 片（2x2x2，2 台主机，8 个 JAX 设备） | B200 × 8，单机 |
| 程序 | MaxText，`maxtext` 混元 4 集成分支（提交见各运行 `meta.json` 的 `code`） | Megatron-Core 0.20 + TE（NeMo 26.10 镜像所带；`meta.json` 没有记 TE 版本，未核） 之上的 [`frameworks/megatron/`](../../frameworks/megatron/README.md) |
| 镜像 | MaxText `tpu_pre_training`（digest 见 `frameworks/maxtext/run.sh`） | NeMo 26.10（digest 见各运行的 `meta.json`） |
| 配置 | [`frameworks/maxtext/hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml) + `--stage mini` | `PROTOCOL.md` 第 1.2 节；各运行的 `meta.json` |
| 初始权重 | hy4-mini 随机初始化的 HF 目录转换的 MaxText 检查点 | 同一 HF 目录 |
| 数据 | `replay.py` 原样回放 GPU 写下的 batch 与 loss_mask | 写下每个 rank、每步、每个 microbatch 的 batch（`--save-batches`） |

命令：`YML=hy4-mini-t1.yml STAGE=mini GOLDEN=<GPU 基线运行目录> bash run.sh <运行名> <步数>`（300 步回放 `g2-a`，1000 步回放 `g3-a`；`GOLDEN` 指定从哪一份 GPU 运行读 batch 与 loss_mask）（[`frameworks/maxtext/README.md`](../../frameworks/maxtext/README.md)）。这条命令用的是现行 yml 与现行 MaxText。复现现行比较 `t1zw-a` 用现行 yml（janus `9c0f19a` 起 yml 里打开了 M42b）与 MaxText `2c28a4f4`；它当时用的 janus 提交与覆盖参数以 `t1zw-a` 的 `meta.json` 为准。复现上一代 `t1zu-a` 用 janus `2679a837` 的 yml 与 MaxText `3741e818`（覆盖参数里没有改开关的，见 `t1zu-a` 的 `meta.json`）。复现更早一代 `t1zr-a` 要用 janus `dec9315c` 的 yml、MaxText `c902c147`，再加覆盖参数 `ihc_force_bf16_rounding=true`（那一版 yml 里还没有 M36，见 `t1zr-a` 的 `meta.json` 里的 `overrides`）；现行 yml 另开了 M34、M40、M41，MaxText 之后又有 M37（`transcendental_accuracy_highest` 的代码改动）与 M39，照现行配置跑出来的不是 `t1zr-a`。各运行用的提交以 `meta.json` 为准。

## 真实权重截层（L0 与阶段 A 单算子）

| 项 | 值 |
|---|---|
| 检查点 | hy4-preview 截第 0–3 层 + MTP 的 HF 目录（135 个张量，清单 sha256 `8ec763b5…`），按阶段 A–D 转换（`model/convert_hf_to_maxtext.py --stage`） |
| TPU 配置 | [`frameworks/maxtext/hy4-4layers.yml`](../../frameworks/maxtext/hy4-4layers.yml)（不设 `HY4_YML` 时的默认） |
| 单算子回放的 GPU 一侧 | `ga1-a` 第 1 步抓到的算子输入；FP64 真值由 [`analysis/fp64_ops.py`](../../analysis/fp64_ops.py) 算 |
