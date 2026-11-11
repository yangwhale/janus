[janus](../../../../README.md) › [混元 4](../../README.md) › [real-weight](README.md) ｜ 配置对照

# 混元 4 · real-weight（T2）· 配置对照

**同 mini，只列不同的地方。** 两边每一项数值与优化器开关的逐项对照见 [mini 的配置对照](../01-mini/config-alignment.md)；T2 的两份 TPU 配置 [`hy4-t2-b.yml`](../../frameworks/maxtext/hy4-t2-b.yml)、[`hy4-t2-d1.yml`](../../frameworks/maxtext/hy4-t2-d1.yml) 与 mini 的 [`hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml) 除 `model_name` 外逐键相同（单测 `tests/models/hunyuan4/test_t2_configs.py`）。GPU 一侧的参数与 g2 / g3 相同，只改层数与 MTP（[README](README.md)「GPU golden 的作业规格」）。

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
| 第三代 | `td4-a`、`tb4-a` | 登记时同 T1 的诊断代 `t1zt`（t1zr 再加 M34、M37，M25 重新打开）；之后打包的还会带 M39–M41，以 `meta.json` 为准 |
