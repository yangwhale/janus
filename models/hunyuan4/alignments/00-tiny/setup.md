[janus](../../../../README.md) › [混元 4](../../README.md) › [tiny](README.md) ｜ 设置

# 混元 4 · 00 · tiny · 设置

| 项 | 取值 |
|---|---|
| 硬件 | TPU 一侧：CPU 上 8 个假 JAX 设备（`JAX_PLATFORMS=cpu`）；GPU 一侧冒烟：B200 × 2 |
| 程序 | MaxText 混元 4 集成分支（模型 `hy4-tiny`）；GPU 一侧 [`frameworks/megatron/`](../../frameworks/megatron/README.md) |
| 命令 | `JAX_PLATFORMS=cpu python models/hunyuan4/frameworks/maxtext/cpu_smoke.py --work <工作目录> [--stages A,B,C,D] [--steps 3]` |
| HF 目录 | `model/make_tiny_hf.py` 生成的假目录（小维度） |
