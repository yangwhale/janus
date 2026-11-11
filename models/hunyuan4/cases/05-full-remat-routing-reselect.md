[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 05

# 案例 05：`full` 重算下反向会重选路由——改用 `custom` 重算并存下路由索引

> **状态：已修（台账 M26，T1 开，`t1t-*` 起）。**

## 现象

- [`PROTOCOL.md`](../PROTOCOL.md) 第 1.2 节「重算」行登记的是「整层重算、路由索引存下不重选」。
- TPU 的 T1 配置在 `t1t-*` 之前用 `remat_policy: full`，反向重算时会重新选路由（台账 M26）。

## 机制

MaxText 只在 `remat_policy: custom` 下给 `moe_routing_maps` 打标记；`full` 下不读这个设置，反向重算时重跑路由选择。

- 路由索引是整数：重算的输入与前向逐位相同时，选出来的也相同；不同时反向用的专家与前向不同，改变梯度（[`../alignments/01-mini/config-alignment.md`](../alignments/01-mini/config-alignment.md)「重算」两行，标「推断」）。
- GPU 一侧 Megatron-Core 的 full 重算不单独存路由（推断，同上）：「存下不重选」是协议的写法，不一定是 GPU 的实际行为。

## 证据

- 按 T1 配置在本机解析 `remat_policy: custom`：`remat_policy=custom`、`moe_routing_maps=device`、其余张量 remat，与对齐 yml 无不一致（台账 M26「测试」栏）。
- 只差这一个开关的 300 步运行见 [`tq-m26-l4.report.md`](../alignments/01-mini/results/tq-m26-l4.report.md)；反向重选槽位的读数见[根源一览第 5 行](../ROOT-CAUSES.md#r5)。
- `t1t-a` 与 `t1t-b` 除 `step_time` 外逐位相同（提交 `850ecb3`）。`t1t-*` 同时改了 M13、M25、M26、M27，GPU 参照也换成 `g2-*`（G17），合在一起的前后数字见[案例 06](06-gpu-attn-scores-bf16.md)。

## 结论

- **修法**：台账 M26，T1 改用 `remat_policy: custom`（除层输入外全部重算，`moe_routing_maps: device` 存下路由索引），不改代码（[`hy4-mini-t1.yml`](../frameworks/maxtext/hy4-mini-t1.yml)「重算」那段注释；[`CHANGES-LEDGER.md`](../CHANGES-LEDGER.md)）。
- **定责依据**：读代码，对照的语义标准是协议第 1.2 节「整层重算、路由索引存下不重选」：`full` 下 MaxText 不读 `moe_routing_maps`。没有对 FP64 的 R；GPU 一侧是否真的不重选是推断。
- **生产建议**：见[根源一览第 5 行](../ROOT-CAUSES.md#r5)；对齐报告 [D8](../alignments/01-mini/DETAILS.md#d8)。
