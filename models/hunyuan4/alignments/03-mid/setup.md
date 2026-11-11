[janus](../../../../README.md) › [混元 4](../../README.md) › [mid](README.md) ｜ 设置

# 混元 4 · 03 · mid（hy4-mid）· 设置

只写和 T1（[`../01-mini/setup.md`](../01-mini/setup.md)）不一样的地方。

| 项 | TPU（被测） | GPU（golden） |
|---|---|---|
| 模型 | MaxText `hy4-mid`（集成分支；与 `hy4-mini` 只差宽度、头数、q_lora、稠密 FFN、专家宽度） | 同一 HF 目录 |
| 初始权重 | `hf/hy4-mid-seed0`（`model/make_tiny_hf.py --config mid --layers 7 --mtp 1 --seed 0`，权重矩阵标准差 0.006）转换的 MaxText 检查点 `maxtext/hy4-mid-maxtext/0/items` | `hf/hy4-mid-seed0` |
| 配置 | [`hy4-mid-t1.yml`](../../frameworks/maxtext/hy4-mid-t1.yml)：`hy4-mini-t1.yml` 的全部开关，只换 `model_name`；`STAGE=mid`。`tm-b` 的 `meta.json` 记 jax 0.11.1，libtpu 构建号 `cl/963683785`，即镜像自带的 0.0.46（见下面「软件版本」） | 参数同 g2 / g3，只换 HF 目录 |
| 数据 | 回放 `gm-a` 的 batch 与 loss_mask | `tokens.npy`（与 T1 同一份） |

路径都在 `gs://<POC 桶>/janus/hunyuan4/` 下。TPU 命令：`YML=hy4-mid-t1.yml STAGE=mid GOLDEN=<gm-a 运行目录> HF_DIR=<…/hf/hy4-mid-seed0> bash run.sh tm-a 300 --dump-steps none`。这条命令用现行 yml，复现不了已判的运行：复现 `tm-a` 要用 janus `fd796f6` 的 yml 与 MaxText `469dbb85`；复现 `tm-b` 要用 janus `9c0f19a` 的 yml 与 MaxText `2c28a4f4`（base2 开、没有 M50，`meta.json` 已核），运行名换成 `tm-b`。之后加的开关都会改变数值，以各运行的 `meta.json` 为准。现行 yml（关 base2、含 M50、C 口径的 M52 与 M58）在 mid 上还没跑过；照上面的命令用现行 yml 跑出来的就是 C 口径。

## 软件版本

以最近一代判定运行为准（`tm-b`，base2 开、早于 M50；现行对齐配置在 mid 上没有运行）：TPU `tm-b` 对 GPU `gm-a`，噪声是同设置 8 份。其余各行同 [T1 的「软件版本」](../01-mini/setup.md#软件版本)。

| 侧 | 组件 | 版本 | 证据出处 |
|---|---|---|---|
| TPU | 镜像 | MaxText `tpu_pre_training`，digest `b152355a…`（同 T1） | `tm-b` 的 `meta.json` 的 `image` |
| TPU | MaxText | 集成分支 `2c28a4f4` | `meta.json` 的 `code`；[`runs.yaml`](runs.yaml) |
| TPU | janus（打进 Pod 的程序） | `9c0f19af` | `meta.json` 的 `code` |
| TPU | jax / jaxlib | 0.11.1 / 0.11.1 | `meta.json` 的 `versions` |
| TPU | libtpu | 0.0.46（构建 `cl/963683785`） | `meta.json` 只记了构建号；同一构建（`Built on Aug 12 2026 16:43:34`，`cl/963683785`）在 [`jit-miscompile-probe-v5p.json`](../../results/jit-miscompile-probe-v5p.json) 里读到 pip 版本 0.0.46（`runs[0]`，镜像默认 libtpu；[`jit-miscompile-mof-off-steptime.json`](../../results/jit-miscompile-mof-off-steptime.json) 记同一镜像 digest `b152355a…`） |
| TPU | tokamax | 0.0.14（镜像自带的包，实测） | `meta.json` 不记；同 T1，读镜像层里的 `tokamax-0.0.14.dist-info` 核过（2026-11-11，[mini 的「软件版本」](../01-mini/setup.md#软件版本)） |
| GPU | 镜像、torch、CUDA、NCCL、Megatron-Core、TE、cuBLAS | 同 T1 | `gm-*` 8 份的 `meta.json` 的 `image`、`versions` 与 T1 的 `g3-*` 逐字相同（比对输出 [`gpu-meta-versions-vs-g3.json`](../../results/gpu-meta-versions-vs-g3.json)，脚本 `analysis/gpu_meta_versions_check.py`，各份 sha256 与已提交结果文件里记的相同） |
| GPU | janus 训练程序 | `hy4_mcore.py` sha256 `4dab75aa…` | `meta.json` 的 `code` |

`meta.json` 不入库，结果文件 [`tmb-l4.results.json`](results/tmb-l4.results.json) 记着每份的 sha256。**结论在这些版本上测得**；换 libtpu 或 jax 要按 [`libtpu-052.md`](../04-tpu-self/libtpu-052.md) 的做法重测。

## GPU golden 的作业规格

程序与 Pod 环境同 T2（[`../02-real-weight/setup.md`](../02-real-weight/setup.md)「GPU golden 的作业规格」），两边共用的附加参数 `C` 是：

```text
C=--ep=4,--optimizer=muon,--opt-kv=muon_scale_mode=shape_scaling,--qk-clip,--qk-clip-threshold=100,--wd=0,--lr=1e-5,--adam-eps=1e-10,--adam-beta2=0.95,--attn-scores-fp32=on,--head-chunk=1024,--indexer-loss=0.1,--eod-mask=on,--per-token-loss=on,--muon-exclude-ihc=ihc+head,--muon-gate=per_head,--bias-rate=1e-3,--clip=1.0,--seq=4096
```

相对 T2 只换：`HFDIR=hy4-mid-seed0`、`HF_FILES="config.json model.safetensors.index.json tensor_sha256.json model.safetensors"`，层数 7、带 `--mtp`、`--qk-clip-mtp=on`；可以带 `--dump-all-experts`（16 个专家）。

```text
gm-a:7:300:--mtp,$C,--qk-clip-mtp=on,--recompute,--save-batches,--side=golden-gm-a      （gm-b … gm-h 同，不带 --save-batches）
gms-a:7:1:--mtp,$C,--qk-clip-mtp=on,--dump-steps=1,--full,--dump-all-experts,--save-batches,--side=golden-gms-a   （gms-b、gms-c 同，不带 --save-batches）
```

跑完放到 `gs://<POC 桶>/janus/hunyuan4/golden-mid/<运行名>/`，写 `DONE`。

显存：v5p 编译期每设备参数 + 临时量 63.16 GB；B200 每卡约 86.7 GiB（估算）。见 [`../02-real-weight/results/hy4-t2-aot-memory.json`](../02-real-weight/results/hy4-t2-aot-memory.json)。
