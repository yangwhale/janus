[janus](../../../../README.md) › [混元 4](../../README.md) › [mid](README.md) ｜ 运行登记

# 混元 4 · mid（hy4-mid）· 运行登记

完整清单在 [`runs.yaml`](runs.yaml)，作业规格在 [README](README.md)。数据位置都在 `gs://<POC 桶>/janus/hunyuan4/` 下。

| 运行 | 侧 | 步数 | 用途 | 结果 |
|---|---|---|---|---|
| `gm-a` | GPU | 300 | 基线；`--save-batches` 写下全部 batch | `tm-l4`、`tm-l0-settings` 的基线 |
| `gm-b … gm-h` | GPU | 300 | 同设置重跑：与 `gm-a` 两两 28 对是 L4 噪声尺子 | GPU 8 份载入自检：[`hy4-mid-gpu-load-check.json`](results/hy4-mid-gpu-load-check.json) |
| `gms-a … gms-c` | GPU | 1 | 第 1 步全量抓数参照（L3 噪声） | — |
| `tm-a` | TPU | 300 | 被测，回放 `gm-a`；MaxText `469dbb85`，每步约 2.8 s，峰值每设备 9.57 GB | [`tm-l4`](results/tm-l4.report.md)、[`tm-l0-settings`](results/tm-l0-settings.report.md) |
| `tm-b` | TPU | 300 | 现行被测，回放 `gm-a`，开关含 M39–M41 与 M42b、没有 M50；MaxText `2c28a4f4` | [`tmb-l4`](results/tmb-l4.report.md) |
| `tmdiag` | TPU | 1 | 第 1 步诊断（`--diag-step1`，不更新），回放 `gms-a` | [`mid-step1-grad-vs-fp64.json`](results/mid-step1-grad-vs-fp64.json) |
| `tm2diag` | TPU | 1 | 第 1 步诊断，现行 yml（MaxText `3741e818`，含 M39–M41），回放 `gms-a`；梯度文件 9 个张量取回时一段被写成 0（设备上的平方和与 grad_norm 不受影响），这 9 个不可判 | [`mid-tm2-step1-grad-vs-fp64.json`](results/mid-tm2-step1-grad-vs-fp64.json) |

TPU 一侧检查点的 L0（离线）：[`hy4-mid-l0-ckpt-convert`](results/hy4-mid-l0-ckpt-convert.report.md)、`results/hy4-mid-l0-rope.results.json`。

第 1 步的 CPU FP64 参照（L3 定责用，不是任何一侧的运行）：GPU golden 同一套 Megatron-Core 代码在 CPU 上全程 FP64 跑第 1 步（`frameworks/megatron/step1_ref.py --recompute --expert-store`），同一份 HF 目录（`tensor_sha256.json` 的 sha256 `c4444d77…`）、tokens.npy 第 0–15 行；逐算子核过 316,291 个算子输出没有 FP64 以外的浮点类型。产物不入库，比较结果见 [`mid-tm2-step1-grad-vs-fp64.json`](results/mid-tm2-step1-grad-vs-fp64.json)（现行开关）与 [`mid-step1-grad-vs-fp64.json`](results/mid-step1-grad-vs-fp64.json)（上一代）。
