[janus](../../../../README.md) › [混元 4](../../README.md) › [real-weight](README.md) ｜ 运行登记

# 混元 4 · real-weight（T2）· 运行登记

完整清单在 [`runs.yaml`](runs.yaml)，GPU golden 的作业规格在 [`setup.md`](setup.md#gpu-golden-的作业规格)。数据位置都在 `gs://<POC 桶>/janus/hunyuan4/` 下。

## B（第 0 层稠密 + 第 1 层 MoE）

| 运行 | 侧 | 步数 | 用途 | 结果 |
|---|---|---|---|---|
| `gb2-a` | GPU | 300 | 基线；写下全部 batch 给 TPU 回放 | `t2bz-l4`、`t2b4-l4`、`t2b-l0-settings`、`t2b4-l0-settings` 的基线 |
| `gb2-b … gb2-h` | GPU | 300 | 同设置重跑：与 `gb2-a` 两两 28 对是 B 的 L4 噪声尺子 | — |
| `gb2s-a … gb2s-c` | GPU | 1 | 第 1 步全量抓数参照（L3 噪声；只有 3 份，2026-11-12 起第 1 步参照 6 份起步，参照不足、待补） | — |
| `gb2s-d … gb2s-f` | GPU | 1 | 第 1 步全量抓数参照再三份（PROTOCOL 2026-11-12 前瞻登记，登记合入后提交），同一代码包与参数；与 `-a … c` 合成 6 份 15 对 | [`t2b8c-step1-train-grad-vs-fp64-n6`](results/t2b8c-step1-train-grad-vs-fp64-n6.json) 的尺子 |
| `tb8s-c` → `tb8s-c-r1` | TPU | 1 | 前瞻登记的运行：现行 C 口径 yml、MaxText `8d3d7561`，第 1 步全量抓数，动量反推训练步梯度做 L3。`tb8s-c` 节点盘满被驱逐（基础设施失败），按登记补跑成 `tb8s-c-r1` | [`t2b8c-step1-train-grad-vs-fp64-n6`](results/t2b8c-step1-train-grad-vs-fp64-n6.json)（❌）、[`t2b8c-step1-train-grad-file-check`](results/t2b8c-step1-train-grad-file-check.json)；3 份对照 [`-n3`](results/t2b8c-step1-train-grad-vs-fp64-n3.json) |
| `gb3-a` | GPU | 1000 | B 1000 步的基线；写下全部 batch | `t2b5-l4`、`t2b5-l0-settings` 的基线 |
| `gb3-b … gb3-h` | GPU | 1000 | 同设置重跑：B 1000 步的噪声尺子 | — |
| `tb5-a` / `tb5-b` | TPU（1000 步，同 `td5-a` 的设置） | 1000 | `tb5-a` 未完成；`tb5-b` 同设置补跑，已判：过 3、不过 0、待定 4 | [`t2b5-l4`](results/t2b5-l4.report.md)、[`t2b5-l0-settings`](results/t2b5-l0-settings.report.md) |
| `tb5b-a` | TPU（1000 步，加 M42b；base2 开、早于 M50） | 1000 | 上一代，MaxText `2c28a4f4`、janus `9c0f19a`；已判：过 4、不过 0、待定 3 | [`t2b5b-l4`](results/t2b5b-l4.report.md) |
| `tbc-a` | TPU（1000 步，现行 C 口径：M36、M52、M58 都开，base2 关，含 M50） | 1000 | 现行，MaxText `8d3d7561`、janus `7314865d`；前瞻登记（PROTOCOL 2026-11-12）后判：过 6、不过 0、待定 1 | [`t2bc-l4`](results/t2bc-l4.report.md)、[`t2bc-l0-settings`](results/t2bc-l0-settings.report.md) |
| `tb2-a` | TPU（第一代） | 300 | 第 1 步更新后出 NaN（M35 已修），不是判定 | [`t2b-l0-settings`](results/t2b-l0-settings.report.md) |
| `tb3-a`、`tb3-b` | TPU（第二代） | 300 | `tb3-a` 约第 200 步因节点重建中断；`tb3-b` 跑完 | [`t2bz-l4`](results/t2bz-l4.report.md) |
| `tb4-a` | TPU（第三代） | 300 | t1zr（已含 M36）再加 M34、M37，M25 重新打开（相对第二代即加 M34、M36、M37）；MaxText `469dbb85`，不带 M39–M41（`meta.json` 已核） | [`t2b4-l4`](results/t2b4-l4.report.md)（过 3、不过 1、待定 3）、[`t2b4-l0-settings`](results/t2b4-l0-settings.report.md) |
| `tb6diag` | TPU 第 1 步诊断（MaxText `3741e818`，T2 B yml 当时的全部开关，含 M39–M41；base2 开、早于 M50） | 1 | L3 判定证据；回放 `gb2s-a` | [`t2b-step1-grad-vs-fp64`](results/t2b-step1-grad-vs-fp64.json)、[`t2b-step1-attn-along-ref`](results/t2b-step1-attn-along-ref.json) |
| `tnorm-base` / `tnorm-sg` | TPU 第 1 步诊断（MaxText `2306c7a5`、janus `effa41e`；`tnorm-sg` 再开 `rmsnorm_scale_grad_fp32`，只改反向；base2 开） | 1 | D22 RMSNorm 权重梯度的开关对照；`tnorm-sg` 也是 D20 的 base2 开臂 | [`t2b-step1-norm-scale-grad-fp32`](results/t2b-step1-norm-scale-grad-fp32.json) |
| `tqa-nb2` | TPU 第 1 步诊断（同 `tnorm-sg`，只差 `sa_use_base2_exp` 关） | 1 | D20 的 base2 关臂；D22 q latent norm 的对照 | [`t2b-step1-qpath-base2-exp`](results/t2b-step1-qpath-base2-exp.json)、06 [`step1-base2-observed`](../06-rw3/results/step1-base2-observed.json) |

## D1（第 0 层稠密 + 1 层 MTP）

| 运行 | 侧 | 步数 | 用途 | 结果 |
|---|---|---|---|---|
| `gd2-a` | GPU | 300 | 基线；写下全部 batch 给 TPU 回放 | `t2d1z-l4`、`t2d1-l4`、`t2d14-l4`、`t2d1-l0-settings`、`t2d14-l0-settings` 的基线 |
| `gd2-b … gd2-h` | GPU | 300 | 同设置重跑：D1 的 L4 噪声尺子 | — |
| `gd2s-a … gd2s-c` | GPU | 1 | 第 1 步全量抓数参照（L3 噪声；只有 3 份，2026-11-12 起第 1 步参照 6 份起步，参照不足、待补） | — |
| `gd2s-d … gd2s-f` | GPU | 1 | 第 1 步全量抓数参照再三份（PROTOCOL 2026-11-12 前瞻登记，登记合入后提交），同一代码包与参数；与 `-a … c` 合成 6 份 15 对 | [`t2d18c-step1-train-grad-vs-fp64-n6`](results/t2d18c-step1-train-grad-vs-fp64-n6.json) 的尺子 |
| `td8s-c` → `td8s-c-r2` | TPU | 1 | 前瞻登记的运行：现行 C 口径 yml、MaxText `8d3d7561`，第 1 步全量抓数，动量反推训练步梯度做 L3。`td8s-c` 节点盘满被驱逐；`td8s-c-r1` spot 节点被收回（基础设施失败），按登记补跑成 `td8s-c-r2` | [`t2d18c-step1-train-grad-vs-fp64-n6`](results/t2d18c-step1-train-grad-vs-fp64-n6.json)（❌）、[`t2d18c-step1-train-grad-file-check`](results/t2d18c-step1-train-grad-file-check.json)；3 份对照 [`-n3`](results/t2d18c-step1-train-grad-vs-fp64-n3.json) |
| `gd3-a` | GPU | 1000 | D1 1000 步的基线；写下全部 batch | `t2d15-l4`、`t2d15-l0-settings` 的基线 |
| `gd3-b … gd3-h` | GPU | 1000 | 同设置重跑：D1 1000 步的噪声尺子 | — |
| `td5-a` | TPU（1000 步，T2 yml 全部开关，含 M39–M41） | 1000 | 已判：过 5、不过 1（索引器 KD）、待定 3 | [`t2d15-l4`](results/t2d15-l4.report.md)、[`t2d15-l0-settings`](results/t2d15-l0-settings.report.md) |
| `td5b-b` | TPU（1000 步，加 M42b；base2 开、早于 M50） | 1000 | 上一代，MaxText `2c28a4f4`、janus `9c0f19a`；已判：过 1、不过 1（索引器 KD）、待定 7 | [`t2d15b-l4`](results/t2d15b-l4.report.md) |
| `tdc-a` | TPU（1000 步，现行 C 口径） | 1000 | 现行，MaxText `8d3d7561`、janus `7314865d`；前瞻登记后判：过 7、不过 1（索引器 KD）、待定 1 | [`t2d1c-l4`](results/t2d1c-l4.report.md)、[`t2d1c-l0-settings`](results/t2d1c-l0-settings.report.md) |
| `td2-a` | TPU（第一代） | 300 | t1v 等价设置的首次判定 | [`t2d1-l4`](results/t2d1-l4.report.md)、[`t2d1-l0-settings`](results/t2d1-l0-settings.report.md) |
| `td3-a` | TPU（第二代） | 300 | t1z 的开关、M25 暂关 | [`t2d1z-l4`](results/t2d1z-l4.report.md) |
| `td4-a` | TPU（第三代） | 300 | t1zr（已含 M36）再加 M34、M37，M25 重新打开（相对第二代即加 M34、M36、M37）；MaxText `469dbb85`，不带 M39–M41（`meta.json` 已核） | [`t2d14-l4`](results/t2d14-l4.report.md)（过 1、不过 4、待定 4）、[`t2d14-l0-settings`](results/t2d14-l0-settings.report.md) |
| `td6diag` | TPU 第 1 步诊断（同 `tb6diag` 的设置，MaxText `3741e818`） | 1 | L3 判定证据；回放 `gd2s-a`；5 个张量的梯度文件取回损坏、不判 | [`t2d1-step1-grad-vs-fp64`](results/t2d1-step1-grad-vs-fp64.json)、[`td6diag-grad-file-holes`](results/td6diag-grad-file-holes.json) |
| `td7diag` | TPU 第 1 步诊断（MaxText `2306c7a5`、janus `3ed31f8`，含 M42：`splash_dsinks_fp32` 开，之后从对齐配置撤下） | 1 | 只补看 `td6diag` 损坏的 5 个张量；不是同配置重跑，sink 梯度不能与 `td6diag` 互换 | [`t2d1-td7-step1-grad-vs-fp64`](results/t2d1-td7-step1-grad-vs-fp64.json) |

噪声只用 GPU 同设置重跑；TPU 同设置重跑只核逐位，不进尺子。全部 TPU 运行 base2 开、早于 M50；2026-11-11 起的现行对齐配置在 B、D1 上尚无运行。
