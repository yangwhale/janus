[janus](../../../../README.md) › [GLM-5.2](../../README.md) ｜ 01 · B200 × 8 → TPU v5p 8 片

# 01 · B200 × 8 → TPU v5p 8 片

和[混元 3 的 01](../../../hunyuan3/alignments/01-v5p-acceptance/README.md) 一一对应：L0 到 L4 的迁移验收。

**现状**：TPU 训练还没开始，GPU 参照跑通之后再开始。现在只做编译期显存分析（阶段 1、阶段 1 + MTP、阶段 1 + 第 7 层；v5p 2x2x2、FSDP 8、整层重算、Muon + QK-Clip），结果放 `results/`。
