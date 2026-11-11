[janus](../../../../README.md) › [混元 4](../../README.md) › [real-weight](README.md) ｜ 运行登记

# 混元 4 · real-weight（T2）· 运行登记

完整清单在 [`runs.yaml`](runs.yaml)，作业规格在 [README](README.md)。数据位置都在 `gs://<POC 桶>/janus/hunyuan4/` 下。

## B（第 0 层稠密 + 第 1 层 MoE）

| 运行 | 侧 | 步数 | 用途 | 结果 |
|---|---|---|---|---|
| `gb2-a` | GPU | 300 | 基线；写下全部 batch 给 TPU 回放 | `t2bz-l4`、`t2b4-l4`、`t2b-l0-settings`、`t2b4-l0-settings` 的基线 |
| `gb2-b … gb2-h` | GPU | 300 | 同设置重跑：与 `gb2-a` 两两 28 对是 B 的 L4 噪声尺子 | — |
| `gb2s-a … gb2s-c` | GPU | 1 | 第 1 步全量抓数参照（L3 噪声） | — |
| `gb3-a` | GPU | 1000 | B 1000 步的基线；写下全部 batch | `t2b5-l4`、`t2b5-l0-settings` 的基线 |
| `gb3-b … gb3-h` | GPU | 1000 | 同设置重跑：B 1000 步的噪声尺子 | — |
| `tb5-a` / `tb5-b` | TPU（1000 步，同 `td5-a` 的设置） | 1000 | `tb5-a` 未完成；`tb5-b` 同设置补跑，已判：过 3、不过 0、待定 4 | [`t2b5-l4`](results/t2b5-l4.report.md)、[`t2b5-l0-settings`](results/t2b5-l0-settings.report.md) |
| `tb5b-a` | TPU（1000 步，加 M42b；base2 开、早于 M50） | 1000 | 最近一代，MaxText `2c28a4f4`、janus `9c0f19a`；已判：过 4、不过 0、待定 3 | [`t2b5b-l4`](results/t2b5b-l4.report.md) |
| `tb2-a` | TPU（第一代） | 300 | 第 1 步更新后出 NaN（M35 已修），不是判定 | [`t2b-l0-settings`](results/t2b-l0-settings.report.md) |
| `tb3-a`、`tb3-b` | TPU（第二代） | 300 | `tb3-a` 约第 200 步因节点重建中断；`tb3-b` 跑完 | [`t2bz-l4`](results/t2bz-l4.report.md) |
| `tb4-a` | TPU（第三代） | 300 | t1zr 再加 M34、M37，M25 重新打开 | [`t2b4-l4`](results/t2b4-l4.report.md)（过 3、不过 1、待定 3）、[`t2b4-l0-settings`](results/t2b4-l0-settings.report.md) |

## D1（第 0 层稠密 + 1 层 MTP）

| 运行 | 侧 | 步数 | 用途 | 结果 |
|---|---|---|---|---|
| `gd2-a` | GPU | 300 | 基线；写下全部 batch 给 TPU 回放 | `t2d1z-l4`、`t2d1-l4`、`t2d14-l4`、`t2d1-l0-settings`、`t2d14-l0-settings` 的基线 |
| `gd2-b … gd2-h` | GPU | 300 | 同设置重跑：D1 的 L4 噪声尺子 | — |
| `gd2s-a … gd2s-c` | GPU | 1 | 第 1 步全量抓数参照（L3 噪声） | — |
| `gd3-a` | GPU | 1000 | D1 1000 步的基线；写下全部 batch | `t2d15-l4`、`t2d15-l0-settings` 的基线 |
| `gd3-b … gd3-h` | GPU | 1000 | 同设置重跑：D1 1000 步的噪声尺子 | — |
| `td5-a` | TPU（1000 步，T2 yml 全部开关，含 M39–M41） | 1000 | 已判：过 5、不过 1（索引器 KD）、待定 3 | [`t2d15-l4`](results/t2d15-l4.report.md)、[`t2d15-l0-settings`](results/t2d15-l0-settings.report.md) |
| `td5b-b` | TPU（1000 步，加 M42b；base2 开、早于 M50） | 1000 | 最近一代，MaxText `2c28a4f4`、janus `9c0f19a`；已判：过 1、不过 1（索引器 KD）、待定 7 | [`t2d15b-l4`](results/t2d15b-l4.report.md) |
| `td2-a` | TPU（第一代） | 300 | t1v 等价设置的首次判定 | [`t2d1-l4`](results/t2d1-l4.report.md)、[`t2d1-l0-settings`](results/t2d1-l0-settings.report.md) |
| `td3-a` | TPU（第二代） | 300 | t1z 的开关、M25 暂关 | [`t2d1z-l4`](results/t2d1z-l4.report.md) |
| `td4-a` | TPU（第三代） | 300 | t1zr 再加 M34、M37，M25 重新打开 | [`t2d14-l4`](results/t2d14-l4.report.md)（过 1、不过 4、待定 4）、[`t2d14-l0-settings`](results/t2d14-l0-settings.report.md) |

噪声只用 GPU 同设置重跑；TPU 同设置重跑只核逐位，不进尺子。全部 TPU 运行 base2 开、早于 M50；2026-11-11 起的现行对齐配置在 B、D1 上尚无运行。
