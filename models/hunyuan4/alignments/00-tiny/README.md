[janus](../../../../README.md) › [混元 4](../../README.md) ｜ 00 · tiny（hy4-tiny）

# 混元 4 · 00 · tiny：hy4-tiny（约 5.4M 参数）

**这个尺寸干什么**：最小的混元 4（宽 256、4 头、8 个专家、7 层 + 1 层 MTP、词表 1024，[`PROTOCOL.md`](../../PROTOCOL.md) 第 1.1 节 T0 一行），用来在 CPU 上证明 TPU 一侧的实现链路通、语义对（GPU 一侧只做了冒烟），不用来比数值。参数量约 5.4M 是 Muon 分区表里两项相加（Muon 4,851,712 + Adam 530,609，[`hy4-tiny-muon-partition.md`](../../results/hy4-tiny-muon-partition.md)）。

**现状（2026-11-08）**：只做了 TPU 一侧的语义检查与 GPU 一侧的冒烟；**GPU 对 TPU 的比较一项都没做**。判定见[检查清单](CHECKS.md)与[对齐报告](ALIGNMENT-REPORT.md)。

## 适用的根源（[`ROOT-CAUSES.md`](../../ROOT-CAUSES.md)）

- **本尺寸覆盖的**：第 8 行（QK-Clip 在扫描层下各层共用缩放，M1），由本尺寸的扫描层检查覆盖。
- **其余各行没有在本尺寸上验证**：tiny 没做 GPU 对 TPU 的比较，表里的数都不是在 tiny 上量的。

## 做了什么

| 检查 | 怎么做的 | 结果 | 出处 |
|---|---|---|---|
| CPU 冒烟 | `frameworks/maxtext/cpu_smoke.py`：hy4-tiny 维度、8 个假 CPU 设备，四个阶段（A–D）各走一遍：生成 HF 目录、转换、往返逐位、RoPE 去交错（含不去交错必须判不一致的负对照）、训练几步、初始权重等于 HF、索引器冻结、数据指纹、学习率常数 | 全部通过（`ok`） | [`results/hy4-cpu-smoke.json`](results/hy4-cpu-smoke.json) |
| Muon 分区表 | `frameworks/maxtext/muon_partition.py` 写出每个参数走 Muon 还是 Adam、Newton–Schulz 作用的矩阵（台账 J1） | Muon 142 个张量 / 4,851,712 个参数；Adam 112 个张量 / 530,609 个参数 | [`hy4-tiny-muon-partition.md`](../../results/hy4-tiny-muon-partition.md) |
| QK-Clip 扫描层检查 | `qk_clip_scan_check.py`（台账 J1）在 hy4-tiny 的 11 层变体 hy4-tiny11（1 稠密 + 10 MoE：MoE 层按 IndexCache 每 4 层一组扫描，扫描长度 2，另有 2 层余数块；扫描长度至少 2 才看得出各层是不是共用一个缩放，见脚本开头的说明）上实测每层每头的 √scale | 修前：扫描块里各层共用一个缩放（MaxText 的 bug，台账 M1）；修后同一脚本复跑（MaxText `b87a55aa`），每层每头与按该层最大 logit 算的期望逐位相同 | 修前 [`hy4-tiny11-qk-clip-scan.json`](../../results/hy4-tiny11-qk-clip-scan.json)、修后 [`hy4-tiny11-qk-clip-scan-postfix.json`](../../results/hy4-tiny11-qk-clip-scan-postfix.json)、[案例 07](../../cases/07-qk-clip-bugs.md) |
| GPU 一侧冒烟 | Megatron 训练入口（台账 G7）在 hy4-tiny 上 B200 × 2 跑 20 步，两份 | 两份逐位相同 | 台账 G7（没有结果文件） |
| 检查点读法 | hy4-tiny 改用 vLLM 读法（转换时去交错、latent eps 1e-5、索引器 k_norm eps 1e-6），与各尺寸一致（台账 M7） | 单测通过 | 台账 M7 |

## 没做什么

- **T0：两边逐层前向 / 反向对 FP64**（待做）：待做。
- **GPU 对 TPU 的多步比较**：没有做。L4 在 [mini](../01-mini/README.md) 上做。
- GPU 一侧确定性设置下的同配置两份逐位：只有上面 B200 × 2、20 步的冒烟，hy4-mini 的确定性计划见 [mini 的 GPU 自比](../01-mini/gpu-noise/README.md)。

文件：[`setup.md`](setup.md)、[`runs.yaml`](runs.yaml)、[`config-alignment.md`](config-alignment.md)、[`goldens.md`](goldens.md)、[`dashboard/`](../../dashboard/dashboard.md)。这个尺寸没有检查项目录（`catalogs/`）：冒烟脚本自己写结果文件。
