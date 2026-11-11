[janus](../../../../README.md) › [混元 4](../../README.md) › [GPU 扩展](README.md) ｜ 设置

# 混元 4 · 05 · GPU 扩展 · 设置

## 参照与被测

| 项 | 值 |
|---|---|
| 模型 | hy4-mini（7 层 + 1 层 MTP，随机初始化；同 [mini](../01-mini/setup.md)） |
| 参照 | 单机 `g3-a`：B200 × 8，DP 8 + EP 4，每卡微批 1 × 累积 2，全局批 16 × 4096（[mini 运行登记](../01-mini/goldens.md)） |
| 尺子 | `g3-a … g3-h`（1000 步 golden 16 份里的前 8 份）两两 28 对的波动上限，前 300 步窗口 |
| 被测 | 只改执行方式：机器数、DP / EP / PP 的切法；训练设置、权重、数据与 `g3-a` 相同 |
| 程序 | [`frameworks/megatron/train_hy4mg.py`](../../frameworks/megatron/train_hy4mg.py)，Megatron-Core 0.20.0，NeMo 26.10 镜像（与 `g3-a` 同一镜像） |
| 机间网络 | GPUDirect RDMA，每台 8 张 400 Gb/s 网卡；镜像自带的 NCCL 2.31.2 与 GCP 网络插件（gIB） |

训练设置（附加参数，与 `g3-a` 的 `RUN_START` 行逐项相同；没写的参数取 `train_hy4mg.py` 的默认值，与 T2 的 `C` 里写明的值相同：`--seq=4096`、`--clip=1.0`、`--head-chunk=1024`、`--bias-rate=1e-3`、`--indexer-loss=0.1`、`--muon-exclude-ihc=ihc+head`、`--muon-gate=per_head`、`--eod-mask=on`、`--per-token-loss=on`）：

```text
G3=--mtp,--ep=4,--optimizer=muon,--opt-kv=muon_scale_mode=shape_scaling,--qk-clip,--qk-clip-threshold=100,--wd=0,--lr=1e-5,--adam-eps=1e-10,--adam-beta2=0.95,--attn-scores-fp32=on,--recompute
```

被测在 `$G3` 后面加并行参数（后写的 `--ep` 覆盖前面的）：

| 被测 | 机器 | 加的参数 |
|---|---|---|
| `mn-a` … `mn-h` | 2 | 无（DP 16 + EP 4，8 份同设置重跑；两两 28 对是多机自己的尺子） |
| `mn-ep16` | 2 | `--ep=16` |
| `mn-pp2` | 2 | `--pp=2,--pp-first=5`（段 0：层 0–4；段 1：层 5–6、MTP、输出头） |
| `mn4-pp2` | 4 | `--ep=16,--pp=2,--pp-first=5`（PP 2 × DP 16） |
| `mn-lr`（正对照） | 2 | `--lr=1.1e-5`（学习率大 10%，应当被判出来） |
| `sn-a` | 1 | 无（新代码在单机上重跑 `g3-a` 的设置：代码改动不改单机数值） |
| `sn-ep8` | 1 | `--ep=8` |

真实宽度截层（hy4-preview 第 0–2 层 + MTP，HF 目录 `hf/hy4-4l`，`层数` 写 3）：附加参数（没写的同样取默认值）

```text
C2=--mtp,--optimizer=muon,--opt-kv=muon_scale_mode=shape_scaling,--qk-clip,--qk-clip-threshold=100,--wd=0,--lr=1e-5,--adam-eps=1e-10,--adam-beta2=0.95,--attn-scores-fp32=on,--qk-clip-mtp=on,--recompute
```

| 运行 | 机器 | 加的参数 | 步数 |
|---|---|---|---|
| `rws-a` … `rws-f` | 1 | `--ep=8`（6 份同设置重跑，两两 15 对是这个尺寸的尺子） | 30 |
| `rw-a` | 2 | `--ep=16` | 50 |
| `rw-b` … `rw-e` | 2 | `--ep=16`（4 份重跑） | 30 |
| `rw-pp2` | 2 | `--ep=8,--pp=2,--pp-first=1`（段 0：层 0；段 1：层 1–2、MTP、输出头） | 50 |
| `rw1k-a` | 2 | `--ep=16,--save-batches`（1000 步，存下每步的 batch 给以后回放） | 1000 |
| `rw4-pp2` | 4 | 层数写 4（第 0–3 层 + MTP，要多拉 `model-layer3.safetensors`），`--ep=16,--pp=2,--pp-first=1` | 50 |

多机作业的写法见 [Megatron 一侧「多机、并行与存档」](../../frameworks/megatron/README.md#多机并行与存档)。

## 怎么判

| 项 | 怎么判 | 工具 |
|---|---|---|
| 吃的是同一批数据 | 每份运行的逐 rank 输入指纹还原到样本号，与 `g3-a` 逐样本比 | [`analysis/gpu_scale_compare.py`](../../analysis/gpu_scale_compare.py) `data` |
| 多机自己的尺子 | 8 份多机重跑两两 28 对另做一把尺子，和单机的尺子比；单机 `g3-a` 反过来对多机 `mn-a` 判；两组（7 份单机、8 份多机）对 `g3-a` 的带符号均值比 Welch t | [`catalogs/multi-node-self.yaml`](catalogs/multi-node-self.yaml)；`gpu_scale_compare.py` `pairs`、`groups` |
| 曲线 | 前 300 步 8 项检查（主 loss 两个窗口、MTP loss、索引器 KD、grad_norm、param_norm、路由偏置均值与标准差），每项三个判定量（带符号均值、MARE、最大单步）都 ≤ 尺子算在噪声内，最大的一个在 1–2 倍之间算「1–2 倍」，> 2 倍算超出 | `janus run` + [`catalogs/vs-single-node.yaml`](catalogs/vs-single-node.yaml) |
| 存档续跑 | 确定性模式下「跑 20 步」与「跑 10 步、存、续跑 10 步」：第 11–20 步每个记录量、最终参数与 buffer 的 sha256 逐位相同；2 台的两组另比全部状态（FP32 主权重、Muon 动量、Adam 矩），单机 DP 8 那组没比状态；两次 20 步重跑也要逐位相同（先证明确定性） | `gpu_scale_compare.py` `bitwise` |

命令：

```bash
janus run models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml --root <运行根目录> --test <被测> --out <输出>
python models/hunyuan4/analysis/gpu_scale_compare.py data --root <运行根目录> --runs g3-a mn-a mn-ep16 … --json results/data-same-batches.json
janus run models/hunyuan4/alignments/05-gpu-scale/catalogs/multi-node-self.yaml --root <运行根目录> --test g3-a --out <输出>
janus run models/hunyuan4/alignments/05-gpu-scale/catalogs/rw-vs-single-node.yaml --root <运行根目录> --test rw-a --out <输出>
python models/hunyuan4/analysis/gpu_scale_compare.py groups --root <运行根目录> --ref g3-a --a g3-b … g3-h --b mn-a … mn-h --steps 1:300 --json results/groups-single-vs-multi.json
python models/hunyuan4/analysis/gpu_scale_compare.py bitwise --root <运行根目录> --a ck-a --b ck-d --from-step 11 --json results/ckpt-resume-dp8.json
```
