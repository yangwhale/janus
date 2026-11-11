# 01 · 设置

- TPU：v5p 8 片，拓扑 2x2x2，8 个 JAX 设备；MaxText 的 DeepSeek 解码器（MLA + DSA）+ GLM-5.2 模型配置。
- 并行与精度：FSDP 8，整层重算，`per_device_batch_size: 1`、`gradient_accumulation_steps: 2`（与 GPU 的 8 卡 × microbatch 1 × 累积 2 一一对应），BF16 计算、FP32 主权重；Muon + QK-Clip。
- 被测的具体配置在跑 L4 之前写进这里，跑完不改（[`PROTOCOL.md`](../../PROTOCOL.md) 第 2 节）。
