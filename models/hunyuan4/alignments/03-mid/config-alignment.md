[janus](../../../../README.md) › [混元 4](../../README.md) › [mid](README.md) ｜ 配置对照

# 混元 4 · mid（hy4-mid）· 配置对照

**同 mini，只列不同的地方。** 两边每一项数值与优化器开关的逐项对照见 [mini 的配置对照](../01-mini/config-alignment.md)。TPU 配置 [`hy4-mid-t1.yml`](../../frameworks/maxtext/hy4-mid-t1.yml) 与 mini 的 [`hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml) 除 `model_name` 外逐键相同（单测 `tests/models/hunyuan4/test_t2_configs.py`）；GPU 参数同 g2 / g3，只换 HF 目录（[`setup.md`](setup.md)「GPU golden 的作业规格」）。

| 项 | mini（T1） | mid |
|---|---|---|
| TPU `model_name` | `hy4-mini` | `hy4-mid`（MaxText 集成分支的 `hy4_mid_dict`） |
| 宽 / 头数 / q_lora | 1024 / 8 / 512 | 6144 / 64 / 2048 |
| 稠密 FFN | 3072 | 18432 |
| 路由专家 | 16 个，top-4，宽 256 | 16 个，top-4，宽 512 |
| 层 | 1 稠密 + 6 MoE + 1 MTP | 同 mini |
| 初始权重 | `make_tiny_hf.py --config mini --layers 7 --mtp 1 --seed 0` | `make_tiny_hf.py --config mid --layers 7 --mtp 1 --seed 0`（权重矩阵标准差 0.006，同 mini） |
| 参数量 | 427,783,489 | 5,189,678,849 |
| 已跑的 TPU 运行 | 现行比较 `t1zg-a` 等（见 mini） | `tm-a`：MaxText `469dbb85`、janus `fd796f6`，与 T1 诊断代 `t1zt` 同一个 MaxText 提交；之后又加了 M39（MaxText 代码）与 M40、M41、M42b（yml 开关），`tm-a` 都没有 |
