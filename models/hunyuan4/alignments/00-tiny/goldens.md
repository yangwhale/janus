[janus](../../../../README.md) › [混元 4](../../README.md) › [tiny](README.md) ｜ 运行登记

# 混元 4 · tiny（hy4-tiny）· 运行登记

| 运行 | 侧 | 做什么 | 结果 |
|---|---|---|---|
| CPU 冒烟 | TPU 一侧（CPU，8 个假设备） | `cpu_smoke.py` 四个阶段整条链路 | [`results/hy4-cpu-smoke.json`](results/hy4-cpu-smoke.json) |
| Muon 分区表、QK-Clip 扫描层检查 | TPU 一侧（CPU） | `muon_partition.py`、`qk_clip_scan_check.py` | [`hy4-tiny-muon-partition.md`](../../results/hy4-tiny-muon-partition.md)、修前 [`hy4-tiny11-qk-clip-scan.json`](../../results/hy4-tiny11-qk-clip-scan.json)；修后复跑（MaxText `b87a55aa`）[`hy4-tiny11-qk-clip-scan-postfix.json`](../../results/hy4-tiny11-qk-clip-scan-postfix.json) |
| GPU 冒烟（B200 × 2，20 步，两份） | GPU | Megatron 训练入口的冒烟 | 两份逐位相同（未存证：只记在台账 G7，没有结果文件，不作判定依据） |

没有 GPU golden，没有噪声参照。
