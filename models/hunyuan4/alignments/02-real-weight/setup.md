[janus](../../../../README.md) › [混元 4](../../README.md) › [real-weight](README.md) ｜ 设置

# 混元 4 · 02 · real-weight（T2：真实权重截层 B、D1）· 设置

只写和 T1（[`../01-mini/setup.md`](../01-mini/setup.md)）不一样的地方。训练设置登记在 [`../../PROTOCOL.md`](../../PROTOCOL.md) 第 1.2 节，两边每个开关见 [`../01-mini/config-alignment.md`](../01-mini/config-alignment.md)；T2 的事先登记见 `PROTOCOL.md` 变更记录。

## 模型与初始权重

| 项 | B | D1 |
|---|---|---|
| 层 | 第 0 层稠密 + 第 1 层 MoE（256 个路由专家 top-8 + 1 个共享），无 MTP | 第 0 层稠密 + 1 层 MTP（MTP 层是 MoE，自带 full 索引器） |
| 参数量（HF 张量形状数出，与 MaxText 抽象模型一致） | 12,078,550,957 | 12,153,673,625 |
| 索引器产出层（KD loss 平均的分母） | 2（第 0、1 层） | 2（第 0 层、MTP 层） |
| HF 目录 | hy4-preview 截第 0–3 层 + MTP 的目录 `hf/hy4-4l`（135 个张量；两边按阶段只读需要的层，`hy4_map.stage_hf_config` / GPU `--layers`） | 同左 |
| MaxText 检查点 | `maxtext/hy4-b-maxtext/0/items`（AdamW 时期转换，60 个张量反向取回逐位：[`hy4-b-l0-ckpt-convert`](../01-mini/results/hy4-b-l0-ckpt-convert.report.md)；T2 用 MaxText `5eaa12f3` 重做的见 [`hy4-b-l0-ckpt-convert-t2`](results/hy4-b-l0-ckpt-convert-t2.report.md)） | `maxtext/hy4-d1-maxtext/0/items`（T2 新转换，58 个张量） |

路径都在 `gs://<POC 桶>/janus/hunyuan4/` 下（与 TPU 同区域的那份）。

## 两边的程序与配置

| 项 | TPU（被测；语义以它为准） | GPU（golden） |
|---|---|---|
| 硬件 | v5p 2x2x2（8 个 JAX 设备，每设备 95 GiB） | B200 × 8（每卡 183,359 MiB，`nvidia-smi` 实测） |
| 程序 | MaxText 混元 4 集成分支（版本见「软件版本」） | Megatron-Core 0.20 + TE（NeMo 26.10 镜像所带，版本见「软件版本」）上的 [`frameworks/megatron/`](../../frameworks/megatron/README.md)；`hy4_mcore.py` 与 g2 / g3 同一份（sha256 `4dab75aa…`） |
| 配置 | [`hy4-t2-b.yml`](../../frameworks/maxtext/hy4-t2-b.yml) / [`hy4-t2-d1.yml`](../../frameworks/maxtext/hy4-t2-d1.yml)：[`hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml) 现在的全部数值与优化器开关（含 M31、M33；已跑的 `tb2-a`、`td2-a` 是之前打包的 t1v 等价设置，没有这两个开关），只换 `model_name: hy4-preview`；层数由 `STAGE` 给 | `train_hy4mg.py`，参数同 g2 / g3（下面 README 的作业规格） |
| 切分 | FSDP 8（同 T1） | 数据并行 8、专家并行 4（同 T1） |
| 重算 | `custom`：除层输入外全部重算，路由索引存下（同 T1） | 整层重算（`--recompute`，同 g2）；第 1 步参照不重算（同 g2s） |
| 数据 | `replay.py` 原样回放 GPU 基线运行写下的 batch 与 loss_mask | `--save-batches`；token 文件与 T1 同一份（`tokens.npy`），样本排法 `(step−1)·16 + micro·8 + rank` |

TPU 命令：`YML=hy4-t2-b.yml STAGE=B GOLDEN=<gb2-a 运行目录> HF_DIR=<…/hf/hy4-4l> bash run.sh tb2-a 300 --dump-steps none`（D1：`YML=hy4-t2-d1.yml STAGE=D1`，回放 `gd2-a`）。这条命令用的是现行 yml 与现行 MaxText，复现不了已判的运行：复现 `td2-a` 要用 janus `83e705cd` 的 yml 与 MaxText `5eaa12f3`（t1v 等价设置）；复现 `td3-a` 要用 janus `eef558af` 的 yml、MaxText `64652880`（M25 关）；复现 `td4-a` 要用 janus `fd796f6` 的 yml 与 MaxText `469dbb85`；B 的 `tb3-b`、`tb4-a` 分别同 `td3-a`、`td4-a` 的提交。1000 步：`td5-a`、`tb5-b` 用 janus `2679a837` 的 yml 与 MaxText `3741e818`；`td5b-b`、`tb5b-a` 用 janus `9c0f19a` 的 yml 与 MaxText `2c28a4f4`（四份都 base2 开、没有 M50，`meta.json` 已核）；步数写 1000，回放 `gb3-a` / `gd3-a`。之后加的开关都会改变数值，以各运行的 `meta.json` 为准。2026-11-11 起的现行 yml（关 base2、含 M50）在 B、D1 上还没跑过。

## 软件版本

以现行 1000 步比较为准：TPU `tb5b-a` / `td5b-b` 对 GPU `gb3-a` / `gd3-a`，噪声各是同设置 8 份。和 T1 不同的只有 TPU 一侧的提交与 libtpu 的证据；其余各行同 [T1 的「软件版本」](../01-mini/setup.md#软件版本)。

| 侧 | 组件 | 版本 | 证据出处 |
|---|---|---|---|
| TPU | 镜像 | MaxText `tpu_pre_training`，digest `b152355a…`（同 T1） | `tb5b-a`、`td5b-b` 的 `meta.json` 的 `image` |
| TPU | MaxText | 集成分支 `2c28a4f4` | `meta.json` 的 `code`（[`runs.yaml`](runs.yaml) 原先写「未登记」，现按它补上） |
| TPU | janus（打进 Pod 的程序） | `9c0f19af` | `meta.json` 的 `code` |
| TPU | jax / jaxlib | 0.11.1 / 0.11.1 | `meta.json` 的 `versions` |
| TPU | libtpu | 0.0.46（推断） | `meta.json` 只记了构建号 `cl/963683785`，没记 pip 版本号。推断：T1 现行 `tz-off-rep` 是同一构建号，记的是 `libtpu==0.0.46` |
| TPU | tokamax | 0.0.14（镜像自带的包） | 同 T1 |
| GPU | 镜像、torch、CUDA、NCCL、Megatron-Core、TE、cuBLAS | 同 T1 | `gb3-*`、`gd3-*` 16 份的 `meta.json` 的 `image`、`versions` 与 T1 的 `g3-*` 逐字相同 |
| GPU | janus 训练程序 | `hy4_mcore.py` sha256 `4dab75aa…`（与 g2 / g3 同一份） | `meta.json` 的 `code` |

`meta.json` 不入库，结果文件 [`t2b5b-l4.results.json`](results/t2b5b-l4.results.json)、[`t2d15b-l4.results.json`](results/t2d15b-l4.results.json) 记着每份的 sha256。**结论在这些版本上测得**；换 libtpu 或 jax 要按 [`libtpu-052.md`](../04-tpu-self/libtpu-052.md) 的做法重测。

## 显存（v5p 编译期分析是实测；B200 是估算）

结果文件 [`results/hy4-t2-aot-memory.json`](results/hy4-t2-aot-memory.json)。

| | B | D1 |
|---|---|---|
| TPU 每设备 参数 + 临时量（编译期，`aot_memory.py`，v5p 2x2x2） | 73.13 GB（68.1 GiB） | 70.75 GB（65.9 GiB） |
| GPU 每卡（估算，专家并行 4、重算开） | 约 92.3 GiB | 约 94.6 GiB |
| GPU 每卡（估算，第 1 步参照：不重算，上界） | 约 117.5 GiB | 约 121.1 GiB |

编译期分析只作初筛，TPU 装不装得下以实机装载训练步成功为准。`meta.json` 的 `memory.peak_bytes_in_use` 不能单独当判据：06 的 D3 装载日志写着训练步要预留 72.80G 并且装上开跑，跑完的峰值却只记 3.03e10，所以这个峰值不含那块临时区（由这两条读数推出，[06 设置「显存」](../06-rw3/setup.md#显存要几片)）。B 的 `tb5b-a` 记 2.65e10 字节，也远小于编译期的 73.13 GB，与此一致。GPU 估算的推导写在结果文件的 `gpu_estimate` 里：常驻量按每参数字节数逐项乘出，激活取 AdamW 之后同一 GPU 程序在真实权重 B / D1 上的冒烟实测（专家并行 8、Muon、重算、q·k 在 BF16 里）减去按同一公式算出的常驻量，再加上 q·k 改成 FP32 的增量。

## 读这份设置要注意

- 真实权重下 QK-Clip 的最大 logit 比 T1 大得多（AdamW 之后同一 GPU 程序的冒烟 300 步：B 13.06–16.38，D1 35.25–59.75），仍低于阈值 100，预计不触发（推断，以运行的记录为准）。真实宽度 D3 的 1000 步 GPU golden 全程最大 76.9，也没触发（[05](../05-gpu-scale/README.md#真实宽度截层)）；裁剪真正生效时两边是否一致，只在 mini 上用低阈值查过。
- B 检查点是 AdamW 时期转换的；已用 MaxText `5eaa12f3`（第一代 `td2-a` 的提交）重做 L0（[`../../model/t2_l0.sh`](../../model/t2_l0.sh)）：60 个张量反向取回逐位、RoPE 12 / 12，这份检查点和现在的参数树对得上。之后各代的提交（`64652880`、`469dbb85`、`3741e818`、`2c28a4f4`）上没有重跑这一步。

## GPU golden 的作业规格

程序 [`frameworks/megatron/pod_run.sh`](../../frameworks/megatron/pod_run.sh) + [`train_hy4mg.py`](../../frameworks/megatron/train_hy4mg.py)，与 g2 / g3 同一套参数，只改层数与 MTP。代码包取本分支当前的提交；`hy4_mcore.py` 的 sha256 要等于 g2 / g3 的 `4dab75aa22e3aff58428fd85d31b11eb708232058a01d67dff53dae474c12800`（不同就先记一行台账再跑）。

Pod 环境变量（`BKT` / `PFX` 由站点配置给，同 T1）：

| 变量 | B | D1 |
|---|---|---|
| `HFDIR` | `hy4-4l` | `hy4-4l` |
| `HF_FILES` | `config.json model.safetensors.index.json tensor_sha256.json model-layer0.safetensors model-layer1.safetensors model-other.safetensors` | `config.json model.safetensors.index.json tensor_sha256.json model-layer0.safetensors model-mtp.safetensors model-other.safetensors` |
| `NPROC` | `8` | `8` |
| `TOKFILE` | `tokens.npy`（与 T1 同一份） | 同左 |

`RUNS` 每项是「运行名:层数:步数:附加参数（逗号分隔）」。下面 `C` 是两边共用的附加参数（与 g2 / g3 的 `RUN_START` 行逐项相同，另把几个取默认值的参数写明，防止默认值变了不知道）：

```text
C=--ep=4,--optimizer=muon,--opt-kv=muon_scale_mode=shape_scaling,--qk-clip,--qk-clip-threshold=100,--wd=0,--lr=1e-5,--adam-eps=1e-10,--adam-beta2=0.95,--attn-scores-fp32=on,--head-chunk=1024,--indexer-loss=0.1,--eod-mask=on,--per-token-loss=on,--muon-exclude-ihc=ihc+head,--muon-gate=per_head,--bias-rate=1e-3,--clip=1.0,--seq=4096
```

| 组 | `RUNS` |
|---|---|
| B，L4 | `gb2-a:2:300:$C,--recompute,--save-batches,--side=golden-gb2-a`，`gb2-b:2:300:$C,--recompute,--side=golden-gb2-b`，… 到 `gb2-h` |
| B，第 1 步参照 | `gb2s-a:2:1:$C,--dump-steps=1,--full,--save-batches,--side=golden-gb2s-a`，`gb2s-b:2:1:$C,--dump-steps=1,--full,--side=golden-gb2s-b`，`gb2s-c` 同 |
| D1，L4 | `gd2-a:1:300:--mtp,$C,--qk-clip-mtp=on,--recompute,--save-batches,--side=golden-gd2-a`，`gd2-b … gd2-h` 同（不带 `--save-batches`） |
| D1，第 1 步参照 | `gd2s-a:1:1:--mtp,$C,--qk-clip-mtp=on,--dump-steps=1,--full,--save-batches,--side=golden-gd2s-a`，`gd2s-b`、`gd2s-c` 同（不带 `--save-batches`） |

- 和 T1 的两处不同：没有 `--dump-all-experts`（256 个专家全抓每份要上百 GB；按 janus 默认每层只留两个专家的全量张量，L3 的路由专家只比这两个）；B 不带 `--mtp`。
- 一个作业里串几份由 GPU 一侧按截止时间排。参考：AdamW 之后同一程序在真实权重上的冒烟（专家并行 8、q·k 在 BF16 里），B 300 步 589 s、D1 772 s（含启动）。
- 跑完把每个运行目录（至少 `meta.json`、`metrics.jsonl`、`load_check.json`、`optimizer_partition.json`、`params_sha/`、`data/`，第 1 步参照另有 `step_1/`）放到 `gs://<POC 桶>/janus/hunyuan4/golden-t2/<运行名>/`，写 `DONE`。
- 必须存下的数据：`gb2-a`、`gd2-a`、`gb2s-a`、`gd2s-a` 的 `data/batches/rank<r>.{bin,jsonl}`（tokens、labels、loss_mask，`--save-batches`），全部运行的 `data/rank<r>.jsonl`（每步每 rank 每个 microbatch 的输入指纹）。TPU 只从 `meta.json` 与 `data/` 回放。
- 显存（估算，见 [`setup.md`](setup.md)）：专家并行 4 时每卡约 92.3 / 94.6 GiB，第 1 步参照不重算上界约 117.5 / 121.1 GiB，都在 179.1 GiB 之内。真要 OOM，第 1 步参照可改 `--ep=8`（只改执行方式），三份一起改并记在 `meta.json`。

## 1000 步 GPU golden 的作业规格（gb3 / gd3）

已跑完，16 份入库。规格与上面 300 步相同（同一 `C`、同一 Pod 环境变量、同一代码要求），只改步数与运行名；数据沿同一条数据流往后取，第 1–300 步与 `gb2-*` / `gd2-*` 逐条相同（样本排法 `(step−1)·16 + micro·8 + rank`，`tokens.npy` 要有 16,000 条以上）。

| 组 | `RUNS` |
|---|---|
| B，1000 步 | `gb3-a:2:1000:$C,--recompute,--save-batches,--side=golden-gb3-a`，`gb3-b:2:1000:$C,--recompute,--side=golden-gb3-b`，… 到 `gb3-h` |
| D1，1000 步 | `gd3-a:1:1000:--mtp,$C,--qk-clip-mtp=on,--recompute,--save-batches,--side=golden-gd3-a`，`gd3-b … gd3-h` 同（不带 `--save-batches`） |

- 跑完放到 `gs://<POC 桶>/janus/hunyuan4/golden-t2/<运行名>/`，写 `DONE`；`gb3-a`、`gd3-a` 必须带 `data/batches/`（TPU 回放）。
- 时长（推断）：300 步的 B 约 10 分钟、D1 约 13 分钟（冒烟实测，含启动），1000 步约 3.3 倍；每份 B 约 30 分钟、D1 约 40 分钟，8 份各约 4–5.5 小时机时。
- TPU 一侧对应 `tb5-b`（`tb5-a` 未完成后补跑）/ `td5-a`（1000 步，回放 `gb3-a` / `gd3-a`），目录 `t2b5-l4` / `t2d15-l4`（窗口第 1–1000 步与第 101–1000 步，噪声 `gb3-*` / `gd3-*` 两两 28 对）。
