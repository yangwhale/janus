[janus](../../../../../README.md) › [混元 4](../../../README.md) › [mini](../README.md) › [GPU 自比与演练](README.md) ｜ 设置

# 混元 4 · mini · B200 × 8 自比 · 设置

- **GPU 自己比自己**：同配置的两份 GPU 运行，一份当被测、一份当参照。判定时 GPU 是 golden 一侧（语义以 TPU 一侧 MaxText 为准，[`PROTOCOL.md`](../../../PROTOCOL.md) 第 1.2 节），这里量的是 golden 一侧自己的散布、确定性，以及工具能不能抓到 GPU 一侧的错。
- 两边的配置对照见 [`../config-alignment.md`](../config-alignment.md) 的 GPU 一栏；GPU 程序与跑法见 [`../../../frameworks/megatron/README.md`](../../../frameworks/megatron/README.md)。
- T1 的训练设置登记在 [`../../../PROTOCOL.md`](../../../PROTOCOL.md) 第 1.2 节，这里不重复。

| | 设置 | 出处 |
|---|---|---|
| 硬件与规模 | NVIDIA B200 × 8，单机 | `PROTOCOL.md` 第 1.2 节「序列 / 批」行 |
| 程序 | Megatron-Core 0.20 + TE（NeMo 26.10 镜像所带；`meta.json` 不记 TE 版本，2026-11-11 在同一镜像的训练 Pod 里实测是 TE 2.20.2，见[软件版本](../setup.md#软件版本)）之上的 janus 程序（`hy4_mcore.py`、`train_hy4mg.py`） | `PROTOCOL.md` 第 1.2 节「GPU 一侧的舍入位置」行 |
| 镜像 | NeMo 26.10（digest 见各运行的 `meta.json`） | `CHANGES-LEDGER.md` GPU 一侧表头 |
| 并行切分 | 数据并行 8、专家并行 4，TP / CP 1 | `PROTOCOL.md` 第 1.2 节 |
| 模型 | hy4-mini（427,783,489 参数，7 层 + 1 层 MTP），从共同的随机初始化 HF 目录载入（清单 sha256 见 `PROTOCOL.md` 第 1.1 节）；载入后逐参数自检 | 台账 G7 |
| 数据 | fineweb-edu `sample/10BT` 第一个分片，混元 4 分词器；eod 位置不算 loss | `PROTOCOL.md` 第 1.2 节、台账 G9 |
| 训练 | 全局批 16 = 8 × microbatch 1 × 累积 2、序列 4096、常数学习率 1e-5、300 步（`g3-*` 1000 步）；Muon + QK-Clip + 索引器 KD + MTP | `PROTOCOL.md` 第 1.2 节 |
| 注意力 logits | `g1-*`：q·k 在 BF16 里算；`g2-*`、`g3-*`：q·k 在 FP32 里算（`--attn-scores-fp32`） | 台账 G17 |
| 确定性 | 默认设置（`deterministic_mode: False`，写在每个运行的 `meta.json`） | `train_hy4mg.py` 写的 meta |
| TF32 | 关（`allow_tf32 = False`），启动时对 FP64 实测写进 meta | `train_hy4mg.py` |

**两种设置**（确定性 / 默认）

- 含义见 [`../../../../../docs/reference/environment.md`](../../../../../docs/reference/environment.md)。
- 本实践现有的 GPU 运行都是默认设置；确定性设置的两份 300 步是计划（hy4-mini 确定性模式两次 20 步逐位已在 05 做过，`05-gpu-scale/results/ckpt-rerun-dp8.json`）（[`catalogs/t1-gpu-determinism.yaml`](catalogs/t1-gpu-determinism.yaml)）。
- 噪声参照只用同一种设置的运行对。
