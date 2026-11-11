[janus](../../../../README.md) › [混元 4](../../README.md) › [real-weight](README.md) ｜ 配置对照

# 混元 4 · real-weight（T2）· 配置对照

**同 mini，只列不同的地方。** 两边每一项数值与优化器开关的逐项对照见 [mini 的配置对照](../01-mini/config-alignment.md)；T2 的两份 TPU 配置 [`hy4-t2-b.yml`](../../frameworks/maxtext/hy4-t2-b.yml)、[`hy4-t2-d1.yml`](../../frameworks/maxtext/hy4-t2-d1.yml) 与 mini 的 [`hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml) 除 `model_name` 外逐键相同（单测 `tests/models/hunyuan4/test_t2_configs.py`）。GPU 一侧的参数与 g2 / g3 相同，只改层数与 MTP（[`setup.md`](setup.md#gpu-golden-的作业规格)「GPU golden 的作业规格」）。

| 项 | mini（T1） | B | D1 |
|---|---|---|---|
| TPU `model_name` | `hy4-mini` | `hy4-preview`（发布配置的维度） | `hy4-preview` |
| 层数 | 7 层 + 1 层 MTP | 第 0 层稠密 + 第 1 层 MoE，无 MTP（`STAGE=B`：`base_num_decoder_layers=2 mtp_num_layers=0`） | 第 0 层稠密 + 1 层 MTP（`STAGE=D1`） |
| 路由专家 | 16 个，top-4，宽 256 | 256 个，top-8，宽 2048，另 1 个共享 | MTP 层同 B |
| 初始权重 | 随机初始化（`model/make_tiny_hf.py --config mini --layers 7 --mtp 1 --seed 0`） | hy4-preview 真实权重截层（`hf/hy4-4l`，MaxText 检查点 `hy4-b-maxtext`） | 同左（`hy4-d1-maxtext`） |
| 参数量 | 427,783,489 | 12,078,550,957 | 12,153,673,625 |
| GPU 层数（`RUNS` 的第二栏）与 MTP | 7 层，带 `--mtp` | 2 层，不带 `--mtp` | 1 层，带 `--mtp`、`--qk-clip-mtp=on` |
| GPU 第 1 步全量抓数的专家范围 | 带 `--dump-all-experts`（见 [`setup.md`](setup.md)「GPU golden 的作业规格」） | 不带（256 个专家全抓每份要上百 GB；按 janus 默认每层只留两个专家的全量张量） | 同左 |

各代 TPU 运行实际打开了哪些开关（与 mini 现行比较的差别）：

| 代 | 运行 | 开关 |
|---|---|---|
| 第一代 | `td2-a`、`tb2-a` | t1v 等价：有 M28，没有 M31、M33（`meta.json` 已核） |
| 第二代 | `td3-a`、`tb3-b`（`tb3-a` 中断） | t1z 的开关（M28、M31、M33），M25 暂关；没有 M34、M36、M37 |
| 第三代 | `td4-a`、`tb4-a` | t1zr（已含 M36）再加 M34、M37，M25 重新打开（相对第二代即加 M34、M36、M37）；MaxText `469dbb85`、janus `fd796f6`（`meta.json` 已核）：M34、M36、M37 与 M25 的开关都开；不带 M39–M41（这个提交早于 M39 的 `1d9cc8c7` 与 M41 的 `3741e818`，完整配置里也没有 M40 的 `moe_mcore_prob_before_fc2`） |
| 1000 步 | `td5-a`、`tb5-b` | 当时 T2 yml 的全部开关，含 M39–M41；MaxText `3741e818`、janus `2679a837`；base2 开，没有 M42b、M50（`meta.json` 已核） |
| 1000 步加 M42b | `td5b-b`、`tb5b-a` | 再加 M42b（`splash_dsinks_gpu_order` 开，M42 关）；MaxText `2c28a4f4`、janus `9c0f19a`；base2 开，没有 M50（`meta.json` 已核） |

2026-11-11 起的现行对齐配置（关 base2、含 M50）在 B、D1 上尚无运行。
