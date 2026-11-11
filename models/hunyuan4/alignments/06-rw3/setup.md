[janus](../../../../README.md) › [混元 4](../../README.md) › [真实宽度三层](README.md) ｜ 设置

# 混元 4 · 06 · 真实宽度三层（多机 GPU ↔ TPU）· 设置

只写和 T2（[`../02-real-weight/setup.md`](../02-real-weight/setup.md)）不一样的地方：层数多一层 shared 索引器层、GPU 一侧是 2 台。训练设置、两边每个开关同 T1 / T2（[`../01-mini/config-alignment.md`](../01-mini/config-alignment.md)）。

## 模型与初始权重

| 项 | 值 |
|---|---|
| 层 | hy4-preview 第 0 层稠密 + 第 1、2 层 MoE（256 个路由专家 top-8 + 1 个共享）+ 1 层 MTP；索引器第 0、1 层 full、第 2 层 shared（复用第 1 层的 top-k），MTP 自带 full |
| 参数量（HF 张量形状数出，与 MaxText 抽象模型一致） | 32,101,224,257 |
| 索引器产出层（KD loss 平均的分母） | 3（第 0、1 层、MTP 层） |
| HF 目录 | `hf/hy4-4l`（同 T2），两边只读第 0–2 层与 MTP 的 111 个张量 |
| MaxText 检查点 | `maxtext/hy4-d3-maxtext/0/items`（阶段 D3，MaxText `b87a55aa` 转换）；反向取回 111 个张量对 HF 原件逐位（[`hy4-d3-l0-ckpt-convert`](results/hy4-d3-l0-ckpt-convert.report.md)） |
| GPU 初始权重 | 开训前载入自检每个参数 == 检查点张量（`rw1k-a` 的 `load_check.json`），载入用的清单与 TPU 一侧是同一份 `tensor_sha256.json`（sha256 `8ec763b5…`）。两边初始权重逐位相同（传递） |

路径都在 `gs://<POC 桶>/janus/hunyuan4/` 下。

## 两边的程序与配置

| 项 | TPU（被测；语义以它为准） | GPU（golden） |
|---|---|---|
| 硬件 | v5p 2x2x4（16 个 JAX 设备，每设备 95 GiB），spot 节点池 | B200 × 16（2 台，机间 GPUDirect RDMA） |
| 程序 | MaxText 混元 4 集成分支 `b87a55aa`（TPU 自比 golden 的同一提交）；1000 步两臂 `rfx-on`、`rfx-off` 的 `meta.json` 记 janus `f46f6981`、jax 0.11.1、libtpu 0.0.46 | [`frameworks/megatron/`](../../frameworks/megatron/README.md)，多机见 [05](../05-gpu-scale/README.md) |
| 配置 | [`hy4-t2-d1.yml`](../../frameworks/maxtext/hy4-t2-d1.yml)（与 mini 现行配置除 `model_name` 外逐键相同；层数与 MTP 由 `STAGE=D3` 给），命令行另加 `gradient_accumulation_steps=1 ici_fsdp_parallelism=16`。`rfx-off` 除这两项外与现行 yml 逐键相同（含 M42b、M50，base2 关）；`rfx-on` 只多 `sa_use_base2_exp: true`（两份 `meta.json` 的 `maxtext_config_full` 已核）。更早的 `trw-*`、`b2r-*` 等运行的提交以各自 `meta.json` 为准 | 附加参数见下 |
| 切分 | FSDP 16，每设备微批 1、不累积 | DP 16 + EP 16，每卡微批 1、不累积 |
| 重算 | `custom`（同 T1） | 整层重算 |
| 数据 | `replay.py` 原样回放 `rw1k-a` 写下的 1000 步 batch 与 loss_mask：设备 d 拿 GPU rank d 的那一条 | `--save-batches`；`tokens.npy`，第 s 步第 (s−1)·16 + rank 条 |

GPU 附加参数（与 g3、T2 同一套训练设置；没写的参数取 `train_hy4mg.py` 的默认值，与 T2 的 `C` 里写明的值相同）：

```text
--mtp,--optimizer=muon,--opt-kv=muon_scale_mode=shape_scaling,--qk-clip,--qk-clip-threshold=100,--wd=0,--lr=1e-5,--adam-eps=1e-10,--adam-beta2=0.95,--attn-scores-fp32=on,--qk-clip-mtp=on,--recompute,--ep=16
```

TPU 命令（[MaxText 一侧](../../frameworks/maxtext/README.md)）：

```bash
STAGE=D3 YML=hy4-t2-d1.yml PLATFORM=v5p-16 GOLDEN=gs://<POC 桶>/janus/hunyuan4/golden-rw/rw1k-a \
  bash run.sh trw-a 1000 --dump-steps none gradient_accumulation_steps=1 ici_fsdp_parallelism=16
```

## 软件版本

以现行比较为准：TPU `rfx-on` 对 GPU `rw1k-a`，噪声是同设置 8 份。其余各行同 [T1 的「软件版本」](../01-mini/setup.md#软件版本)。

| 侧 | 组件 | 版本 | 证据出处 |
|---|---|---|---|
| TPU | 镜像 | MaxText `tpu_pre_training`，digest `b152355a…`（同 T1） | `rfx-on` 的 `meta.json` 的 `image` |
| TPU | MaxText | 集成分支 `b87a55aa` | `meta.json` 的 `code`；[`runs.yaml`](runs.yaml) |
| TPU | janus（打进 Pod 的程序） | `f46f6981` | `meta.json` 的 `code` |
| TPU | jax / jaxlib | 0.11.1 / 0.11.1 | `meta.json` 的 `versions` |
| TPU | libtpu | 0.0.46（构建 `cl/963683785`） | `meta.json` 的 `versions.libtpu` 与 `versions.libtpu_pip` |
| TPU | tokamax | 0.0.14（镜像自带的包） | 同 T1 |
| GPU | 镜像、torch、CUDA、NCCL、Megatron-Core、TE、cuBLAS | 同 T1 | `rw1k-*` 8 份的 `meta.json` 的 `image`、`versions` 与 T1 的 `g3-*` 逐字相同 |
| GPU | janus 训练程序 | 多机的代码包，`hy4_mcore.py` sha256 `8acdc54a…`（单机 T1 / T2 是 `4dab75aa…`） | `meta.json` 的 `code` |

`meta.json` 不入库，结果文件 [`rw3-l4.results.json`](results/rw3-l4.results.json) 记着每份的 sha256。**结论在这些版本上测得**；换 libtpu 或 jax 要按 [`libtpu-052.md`](../04-tpu-self/libtpu-052.md) 的做法重测。

## 显存：要几片

16 片（v5p 2x2x4）跑得起来；训练步程序每设备要预留 72.8 GB 临时量（`trw-e` 装载程序时的日志：要预留 72.80G，当时空着 68.41G，之后装上开跑）。1000 步跑完的 `rfx-on`、`rfx-off` 在 `meta.json` 里记下：训练状态每设备 1.64e10 字节（装载训练步之前活着的数组），跑完时在用 1.74e10、`peak_bytes_in_use` 3.03e10。装载时既然预留了 72.8G 并开跑，峰值却只有 3.03e10，所以这个峰值不含训练步程序预留的临时区（由这两条读数推出；分配器为什么这样记没查）。因此装不装得下以实机装载训练步成功为准，`peak_bytes_in_use` 不能单独当判据；02 的设置页按同一判据写。从存档续跑时旧状态曾多留一份，见 [README](README.md#从存档续跑为什么装不下已修)。

事先按 T2 两份 1000 步运行的实测外推过（常驻每参数 9.01 字节 ÷ 设备数，峰值再加 FP32 梯度与约 7–14 GB），得出「8 片约 66 GB、16 片约 40 GB，8 片就放得下」。这个外推不成立：16 片上光临时量就要 72.8 GB，8 片再加一倍的常驻量，超过每设备 102.8 GB 的上限。外推错在把 T2 的「峰值 − 常驻」（12.9 GB）当成随层数线性涨的小量，D3 多出的两层 MoE（每层 256 个专家、整层聚出来）让临时量大得多。

## 怎么判

判法同 01-mini；尺子是 2 台同设置 8 份两两 28 对（mini 现行 1000 步用单机 16 份 120 对；多机重跑的波动比单机略大，见 [05](../05-gpu-scale/README.md#多机自己的尺子)）：

| 级 | 检查 | 目录 |
|---|---|---|
| L0 | 检查点转换逐位（111 个张量）、dtype 清单 | [`l0-ckpt-convert-d3.yaml`](catalogs/l0-ckpt-convert-d3.yaml) |
| L0 | 两边每一步学习率 = 1e-5（只核了 `rfx-on`，`rfx-off` 没核） | [`rw3-l0-settings.yaml`](catalogs/rw3-l0-settings.yaml) |
| L4 | 1000 步 9 项（主 loss 两个窗口、MTP loss 两个窗口、索引器 KD、grad_norm、param_norm、路由偏置均值与标准差），GPU `rw1k-a` 为基线，尺子 `rw1k-a … rw1k-h`（2 台同设置 8 份）两两 28 对 | [`rw3-l4.yaml`](catalogs/rw3-l4.yaml) |
