[janus](../../../../README.md) › [混元 4](../../README.md) › [tiny](README.md) ｜ 配置

# 混元 4 · tiny（hy4-tiny）· 配置

hy4-tiny 只在 TPU 一侧的 CPU 冒烟里用，没有 GPU 对 TPU 的比较运行，所以没有逐项的两边配置对照。

| 项 | 取值 | 出处 |
|---|---|---|
| 模型 | 宽 256、4 头、8 个专家、7 层 + 1 层 MTP、词表 1024 | MaxText 分支 `hf_model_configs.hy4_tiny_dict`（[`make_tiny_hf.py`](../../model/make_tiny_hf.py) `--config tiny` 读它；按 `b87a55aa` 核过）；MaxText 模型 `hy4-tiny`；参数量见 [`PROTOCOL.md`](../../PROTOCOL.md) 第 1.1 节 |
| 检查点读法 | vLLM 读法：q_pe / k_pe 交错存放、转换时去交错，latent norm eps 1e-5，索引器 k_norm eps 1e-6（同各尺寸） | 台账 M7 |
| 冒烟设置 | 8 个假 CPU 设备，四个阶段（A–D）各几步 | `frameworks/maxtext/cpu_smoke.py`、`train_hy4.py --cpu-small` |

hy4-mini（T1）的两边逐项配置见 [mini 的配置对照](../01-mini/config-alignment.md)。
