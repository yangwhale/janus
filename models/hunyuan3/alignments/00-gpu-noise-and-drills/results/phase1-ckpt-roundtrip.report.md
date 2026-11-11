# 权重转换往返（HF → Megatron → HF）

原件 `/data/hy3-4l`，往返件 `/data/hy3-4l-rt`；张量 1781 个（原件）/ 1781 个（往返）。

| 检查 | 状态 | 说明 |
|---|---|---|
| `L0.ckpt.roundtrip.bitwise` | pass | 全部参数逐位相同 |
| `L0.ckpt.roundtrip.coverage` | pass | 全部比到 |
| `L0.ckpt.dtype_manifest` | pass | dtype 与原件一致 |

往返逐位相同只证明两个方向互逆；映射表本身靠逐层比（加载后的权重对原件）。
