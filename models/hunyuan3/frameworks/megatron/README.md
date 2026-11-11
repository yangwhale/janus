[janus](../../../../README.md) › [用户指南](../../../../docs/guide/README.md) › [混元 3](../../README.md) ｜ GPU 侧操作手册；对应指南[第 1 章 准备环境与数据](../../../../docs/guide/01-environment-data.md)

# 混元 3 · GPU 侧（Megatron）

- 硬件：单机 8 卡 B200，GKE Pod。
- 镜像：`nvcr.io/nvidia/nemo:26.10`，已带 `hy_v3` 模型桥。
  - torch 2.14.0a0、CUDA 13.4；
  - Megatron-Core 0.20.0、Megatron-Bridge 0.7.0；
  - Transformer Engine 2.20.2、NCCL 2.31.2。

本页从零走到「复现第一阶段的代表性结果」（第 0–7 节），后面是训练脚本参数、L1 与其余 GPU 侧检查、环境坑、已验证项。

- 每一步都写了完成后该看到什么（锚点）；对不上就停下查，不要往下走。
- 下面的命令都在 Pod 里执行，除非另有说明。

## 0. 准备之前要知道的

- **默认路径**。[`run.sh`](run.sh)、[`env.sh`](env.sh)、[`prep_data.sh`](prep_data.sh) 默认用下面这套布局，照着放就能直接用。
  - 想换位置，设对应的环境变量即可（见第 1 步），不用改脚本：

  | 默认路径 | 放什么 | 环境变量 | 谁用 |
  |---|---|---|---|
  | `/data/env.sh` | `frameworks/megatron/env.sh` 的副本 | `JANUS_ENV` | `run.sh`、`prep_data.sh` 先 source 它 |
  | `/data/janus-src` | janus 仓库检出（`janus` 包从这里 import） | `JANUS_SRC` | `env.sh` 加进 `PYTHONPATH` |
  | `/data/jw` | `frameworks/megatron/` 与 `model/` 下的 Python 脚本（训练从这里起） | `JANUS_WORK` | `run.sh` |
  | `/data/hy3-4l` | 截好的 4 层 HF 目录 | `JANUS_HF` | `run.sh` 的 `--hf`（见注 1）、`prep_data.sh` 的分词器 |
  | `/data/ds` | 数据（不存在就建） | `JANUS_DS` | `prep_data.sh` |
  | `/data/runs/<运行名>`、`/data/tlogs/<运行名>` | TensorBoard 与 torchrun 日志 | `JANUS_RUNS`、`JANUS_TLOGS` | `run.sh`，**每次启动先 `rm -rf` 同名目录** |
  | `/data/triton_cache` | Triton 缓存 | `JANUS_TRITON_CACHE` | `env.sh` 设成 `TRITON_CACHE_DIR` |
  | `/opt/Megatron-Bridge/3rdparty/Megatron-LM` | Megatron-LM 源码（`tools/preprocess_data.py`） | `MEGATRON_LM` | `prep_data.sh` |

  - 注 1：参数里再给 `--hf` 也会覆盖。

- **磁盘**：
  - 一次 30 步、第 1 步全量抓数（`--full`）的运行目录约 41 GB；
  - 权重原件 10 个分片约 27 GB，4 层 HF 与 Megatron 各约 24 GB；
  - [`pod.yaml`](pod.yaml) 里 `/data` 是 `emptyDir`（本地 SSD），**删 Pod 就全没了**，要留的先传走（[`upload_run.py`](upload_run.py)）。
- **GPU 独占**：抓数时不和别的作业共享节点（[`docs/reference/dump.md`](../../../../docs/reference/dump.md)）。
  - 多人共用一台时，用 GPU 的命令都包在 `flock /data/gpu.lock <命令>` 里排队。
- **先 source env.sh**：这个镜像里不 source 它，`import torch` 就报 `libtorch_cuda.so: undefined symbol: ncclCommResume`。
  - 原因是 NCCL 版本冲突，见下文「环境上的三个坑」。
  - 截层、转格式、数据预处理、训练都要先 `source /data/env.sh`。

## 1. 起 Pod、把仓库放进去

```bash
# 在能连集群的机器上。pod.yaml 的 nodeSelector 是占位，换成你自己 8×B200 节点池的名字
kubectl apply -f models/hunyuan3/frameworks/megatron/pod.yaml
# 仓库是私有的，Pod 里没有凭据：打成 bundle 拷进去
git bundle create /tmp/janus.bundle main
kubectl cp /tmp/janus.bundle janus-gpu:/data/janus.bundle
```

```bash
# Pod 里
git clone -b main /data/janus.bundle /data/janus-src
G=/data/janus-src/models/hunyuan3/frameworks/megatron
cp $G/env.sh /data/env.sh
mkdir -p /data/jw /data/ds /data/hy3-full && cp $G/*.py /data/janus-src/models/hunyuan3/model/*.py /data/jw/
```

不想用 `/data/jw`、`/data/runs` 这套位置（例如在已有别人运行的 Pod 上另起一套），设环境变量即可，脚本直接从仓库检出里跑：

```bash
R=/tmp/repro                              # 仓库检出在 $R/janus
export JANUS_SRC=$R/janus JANUS_ENV=$R/janus/models/hunyuan3/frameworks/megatron/env.sh JANUS_WORK=$R/janus/models/hunyuan3/frameworks/megatron \
       JANUS_RUNS=$R/tb JANUS_TLOGS=$R/tlogs JANUS_TRITON_CACHE=$R/triton_cache
$R/janus/models/hunyuan3/frameworks/megatron/run.sh <运行名> <步数> ...
```

## 2. 下权重（只下需要的分片）

`tencent/Hy3` 公开、不需要登录。第一阶段用的版本：

- revision `a960ebc3da325ba167f069f76c41eb62c9280d22`；
- `model.safetensors.index.json` 的 sha256 是 `9594f1a9419e62ca7afca51bb644f38ef19039374f7812449381ccf42f0ef79b`。

```bash
cd /data/hy3-full
REV=a960ebc3da325ba167f069f76c41eb62c9280d22
B=https://huggingface.co/tencent/Hy3/resolve/$REV
for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
  curl -sfL --retry 5 -o $f $B/$f
done
python3 - > shards.txt <<'PY'
import json
wm = json.load(open("model.safetensors.index.json"))["weight_map"]
# embed / final norm / lm_head（不以 model.layers. 开头）+ 前 4 层所在的分片
print("\n".join(sorted({v for k, v in wm.items() if not k.startswith("model.layers.") or int(k.split(".")[2]) < 4})))
PY
xargs -P 5 -I{} sh -c "curl -sfL --retry 5 -o {}.part $B/{} && mv {}.part {}" < shards.txt
```

**锚点**：`shards.txt` 是 10 个分片，合计约 27 GB。

- 分片：`model-00001`、`00007`、`00012`、`00013`、`00024`、`00036`、`00060`、`00090`、`00091`、`00098`，均为 `-of-00099`。

## 3. 截 4 层、去 MTP

```bash
source /data/env.sh
python3 /data/jw/truncate_hy3.py /data/hy3-full /data/hy3-4l 4
```

脚本做三件事，CPU 上几十秒：

- 把配置改成 4 层、`num_nextn_predict_layers=0`；
- 只拷 embed / final norm / lm_head 和前 4 层的张量；
- 另写逐张量 sha256 清单 `tensor_sha256.json`。

**锚点**：

- 最后一行 `tensors 1781`，输出 10 个分片；
- `sha256sum /data/hy3-4l/tensor_sha256.json` 应为 `183274dbf17e54560812f0b91d3dc8f3ffbf959e94e43905588ea913f2837014`。
  - 这个值也会写进每个运行的 `meta.json` 的 `checkpoint.manifest_sha256`，`L0.config.diff` 比它。

## 4. 转 Megatron

```bash
source /data/env.sh
CUDA_VISIBLE_DEVICES=0 python3 /data/jw/import_hy3.py /data/hy3-4l /data/hy3-4l-mcore
```

单进程，用一张卡，几分钟。产出 torch_dist 检查点（`iter_0000000/`、`latest_checkpointed_iteration.txt` 内容为 `0`）。

**锚点**：日志 `number of parameters ... 12386079744` 与最后一行 `IMPORT_DONE`。

- HF 侧 1781 个张量合计 12,386,080,320 个元素。
- 多出的 576 是 3 个 MoE 层 × 192 个专家的 `expert_bias`：Megatron 把它当 buffer，不计入参数数。

## 5. 准备数据

```bash
bash /data/janus-src/models/hunyuan3/frameworks/megatron/prep_data.sh      # 数据目录不存在会自己建；任一步失败即停
```

脚本做三步：

1. 下载 fineweb-edu 的 `sample/10BT/000_00000.parquet`；
2. 抽 `text` 列写 jsonl；
3. 用 `/data/hy3-4l` 的分词器（所以必须先做第 3 步）跑 Megatron 的 `preprocess_data.py`（`--append-eod`）。

脚本按 `main` 分支下载 parquet、没有钉版本，所以先核对 sha256。

**锚点**：

| 产物 | 期望 |
|---|---|
| `fw.parquet` 的 sha256（脚本写在 `fw.parquet.sha256`） | `b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871` |
| 文档数（脚本打印 `docs …`） | 726000 |
| `fw_hy3_text_document.idx` 的 sha256 | `83d5a23b14d38944dbd7323ca7e794fbac6018bf3e6263398bd0b04db98fdad2` |
| `fw_hy3_text_document.bin` 大小 | 2,922,796,144 字节 |

- 只想先验证流程，可以拿前 200 篇试：`head -200 fw.jsonl > small.jsonl`，对它跑同样的 `preprocess_data.py` 命令。
  - 得到的 200 条 token 序列应与完整数据的前 200 条逐位相同（复现审计实测如此）。
- 训练第一次读数据时，Megatron 会在 `/data/ds/fw_hy3_text_document/` 下建样本索引缓存。

## 6. 训练并抓数

```bash
run.sh <运行名> <步数> --load /data/hy3-4l-mcore --data /data/ds/fw_hy3_text_document \
    [--deterministic] [--janus-run <运行目录> --side golden-<n> --dump-steps 1 [--full]] [其他参数]
```

### `run.sh` 做了什么

1. 先 `source /data/env.sh`（`JANUS_ENV`）。
2. 按设置处理三个环境变量（见下面「两种设置怎么起」）：
   - `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`；
   - `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
   - `NCCL_ALGO=Ring`。
3. 然后 `torchrun --nproc-per-node 8` 起 [`train_hy3.py`](train_hy3.py)，`--hf` 默认 `/data/hy3-4l`（`JANUS_HF`）。

跑完怎么判成败：

- 最后一行打印 `RUN_DONE <运行名> rc=<退出码>`，`run.sh` 以同一个退出码退出。
- `rc=0` 才算成功；非 0 去看 `/data/tlogs/<运行名>/` 下各 rank 的日志。
- 另核对运行目录里有没有 `DONE`。

### 注意事项

- 不给 `--data` 时用 MockGPTDataset，只能用来冒烟，不能量噪声。
  - 1000 步 loss 降到约 0.005，数据被记住。
- `--side` 只接受 `golden-<n>`、`tpu`、`tpu-perturb-<n>`、`gpu-perturb-<n>` 这几种前缀（[`docs/reference/dump-format.md`](../../../../docs/reference/dump-format.md)）。
  - 写成别的（比如 `test`），给了 `--janus-run` 时训练脚本启动就报错退出（退出码 2，提示合法前缀），不占 GPU 时间。
  - 第一阶段写的是 `golden-<n>`（`rd-*`）或 `golden-<运行名>`（其余）。
- `meta.json` 的 `image` 取环境变量 `JANUS_IMAGE`。
  - 容器里拿不到镜像 digest，[`pod.yaml`](pod.yaml) 已把它设成第一阶段的 digest（`nvcr.io/nvidia/nemo@sha256:fc75bf516de6845fd85fe627c7c18e69e7568428fe56438e31ff3b1917044b6c`）。
  - 没设时记成 tag `nvcr.io/nvidia/nemo:26.10` 并打印一条警告。
  - 第一阶段的运行没设这个变量，`meta.json` 里记的是 tag。
  - 第一阶段 Megatron 两个版本也记成 `null`：取法有误，已修，实际是 Megatron-Core 0.20.0、Megatron-Bridge 0.7.0。
- `--janus-run` 每次用新目录：`run.sh` 不清理它，重名会把新旧文件混在一起（旧的 `DONE` 还在）。
- 耗时：8 卡 B200 上，30 步、第 1 步全量抓数一次约 3–4 分钟（含启动和写盘）。

### 两种设置怎么起

- [`run.sh`](run.sh) 按参数里有没有 `--deterministic` 决定三个环境变量：有就设，没有就 `unset`（调用方环境里设了也清掉）。
- 环境变量 `JANUS_DET_ENV=1` / `0` 可以强制设 / 不设，用来复现「确定性但没设三个变量」的那批运行。

| 设置 | 第一阶段实际怎么起的 | 用 `run.sh` 怎么起 |
|---|---|---|
| 确定性 | `--deterministic`；三个环境变量见注 1 | `run.sh … --deterministic`（设三个变量）；要和 `d-a` 这类完全一样见注 2 |
| 默认 | 三个环境变量**都不设**，不加 `--deterministic` | `run.sh …`（不加 `--deterministic`） |

- 注 1：1000 步噪声底的 `rd-golden-1`、`rd-det-b` 另设了上面三个环境变量；其余确定性运行（`d-a`、`f-*`、`p-*` 等）**没设**。
- 注 2：`JANUS_DET_ENV=0 run.sh … --deterministic`。

三个环境变量的影响：

- 每次运行实际生效的值记在 `meta.json` 的 `determinism` 里，以它为准。
- 确定性设置下，设不设不改结果。
  - 复现审计设了三个变量的 `det-a` 与第一阶段没设的 `d-a`，30 步 lm_loss、grad_norm 逐步逐位相同。
- 默认设置下设不设有没有影响，第一阶段没有测。

### 第一阶段每次运行的参数

公共参数（[`train_hy3.py`](train_hy3.py) 的默认值就是这些，写出来是为了和 `meta.json` 对照）：

```text
--load /data/hy3-4l-mcore --data /data/ds/fw_hy3_text_document --gbs 16 --mbs 1 --seq 4096 --ep 8 --lr 1e-05 --seed 1234 --janus-run <根>/<运行名> --side golden-<运行名>
```

- 「确定性」= 加 `--deterministic`；
- 「默认」= 上表的默认设置。

#### 第一阶段 golden 与故障注入

| 运行 | 步数 | 设置 | 另加的参数 |
|---|---|---|---|
| `rd-golden-1`、`rd-det-b` | 1000 | 确定性（设了三个变量） | `--dump-steps 1,1000` |
| `rd-golden-2..5` | 1000 | 默认 | `--dump-steps 1,1000` |
| `s-a`..`s-i` | 100 | 默认 | `--dump-steps 1,100 --full` |
| `d-a`、`d-b` | 100 | 确定性 | `--dump-steps 1,100 --full` |
| `f-weight` | 30 | 确定性 | `--dump-steps 1 --full --fault "weight_scale:decoder\.layers\.2\.self_attention\.linear_proj\.weight:1e-3"` |
| `f-weight2` | 30 | 确定性 | 同上，最后的 `1e-3` 换成 `1e-2` |
| `f-data` | 30 | 确定性 | `--dump-steps 1 --full --fault data_shift:1` |
| `f-router` | 30 | 确定性 | `--dump-steps 1 --full --fault router_bf16` |
| `f-lr` | 30 | 确定性 | `--dump-steps 1 --full --fault lr_ulp` |
| `f-drop` | 30 | 确定性 | `--dump-steps 1 --full --fault dropout:0.1` |
| `f-grad` | 30 | 确定性 | `--dump-steps 1 --full --fault "grad_scale:decoder\.layers\.1\.self_attention\.linear_qkv\.weight:0.125"` |
| `f-gscale` | **3** | 确定性 | `--dump-steps 1 --full --fault "grad_scale_seen:decoder\.layers\.1\.self_attention\.linear_qkv\.weight:0.125"` |
| `f-bias` | 30 | 确定性 | `--dump-steps 1 --full --fault bias_freeze` |
| `f-nan` | 12 | 确定性 | `--dump-steps 4,5,6 --full --full-steps 4,5,6 --full-only "decoder\.layers\.0\.self_attention\." --fault "grad_nan:decoder\.layers\.0\.self_attention\.linear_proj\.weight:5"` |
| `f-spike` | 30 | 确定性 | `--fault lr_spike:15:300` |
| `i-weight` | 3 | 确定性 | `--dump-steps 1 --fault "weight_scale:decoder\.layers\.2\.self_attention\.linear_proj\.weight:1e-2" --inject-from <根>/d-a` |
| `h-light2` | 30 | 确定性 | `--no-dist-opt`（见注 1） |
| `m-opt8` | 9 | 确定性 | `--dump-steps 1,2,3,4,5,6,7,8 --full --full-steps 1,2,3,4,5,6,7,8 --full-only "decoder\.layers\.[01]\.(self_attention\|mlp\.router\|mlp\.shared_experts\|pre_mlp)"` |
| `p-ep4` / `p-ep4f` | 3 | 确定性 | `--dump-steps 1 --ep 4` / 另加 `--full` |
| `p-tp2` / `p-cp2` | 3 | 确定性 | `--dump-steps 1 --full --ep 4 --tp 2` / `--dump-steps 1 --ep 4 --cp 2` |
| `p-mbs2` | 3 | 确定性 | `--dump-steps 1 --full --mbs 2` |
| `r-remat` | 3 | 确定性 | `--dump-steps 1 --recompute` |
| `r-cont` | 20 | 确定性 | `--dump-steps 11 --save /data/ckpt/r-cont --save-interval 10` |
| `r-resume` | 20 | 确定性 | `--dump-steps 11 --resume /data/ckpt/r10`（见注 2） |
| `t-2l` | 3 | 确定性 | `--dump-steps 1`，另把 `--hf`、`--load` 换成截 2 层的目录（见注 3） |

- 注 1（`h-light2`）：不给 `--full`；`--dump-steps` 取默认 `1`。
  - 所以第 1 步仍抓前向张量，只是不挂梯度钩子、不包优化器。
- 注 2（`r-resume`）：`r10` 是只含 `r-cont` 第 10 步存档的目录，做法：

  ```bash
  mkdir -p /data/ckpt/r10 && ln -s /data/ckpt/r-cont/iter_0000010 /data/ckpt/r10/ && echo 10 > /data/ckpt/r10/latest_checkpointed_iteration.txt
  ```

- 注 3（`t-2l`）：第 3、4 步换成 2 层。
  - `--hf` 用 `JANUS_HF=<2 层目录>` 或在参数里再给一次。
- 故障种类的完整写法见 `train_hy3.py` 的 `--fault` 帮助。
  - `weight_scale` 改的是 FP32 主权重再拷回 BF16。
  - 所以 `1e-3`（小于 BF16 半格）BF16 权重不变、`1e-2` 会变。

#### 重算 golden

`alignments/03-gpu-remat-golden/`：公共参数同上，默认设置（同 `g1k-a`，第一阶段 GPU golden 的第 1 份），另加 `--recompute --routing-replay`：

代号：`rc`（GPU 生产 golden，整层重算加路由回放，即原版生产程序，5 份）、`rcc`（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加；其余设置与 rc 相同，10 份）、`rcb`（`rcc` 的设置只把输出头 dgrad 换成 BF16 张量核分段，2 份；与 `rcc` 在 L4 上不可分，见案例 01「便宜的等价修法 `rcb`」行）。

| 运行 | 步数 | 设置 | 另加的参数 |
|---|---|---|---|
| `rv-det` | 2 | 确定性 | `--recompute --routing-replay --dump-steps 1 --full`（回放验证） |
| `rc-a` | 1000 | 默认 | `--recompute --routing-replay --dump-steps 1,1000 --full --save-batches` |
| `rc-b`..`rc-e` | 1000 | 默认 | `--recompute --routing-replay --dump-steps 1 --full` |
| `rcc-a` | 1000 | 默认 | **最终 golden 的基线**：`--recompute --routing-replay --rr-arm dgrad-chunk --no-dist-opt --dump-steps 1000 --save-batches`（见注 1） |
| `rcc-b`..`rcc-j` | 1000 | 默认 | 同 `rcc-a`，`--dump-steps 100000`、不带 `--save-batches`（见注 2） |
| `rcb-a` | 1000 | 默认 | 便宜的等价修法（不是基线）：同 `rcc-a`，只把 `--rr-arm dgrad-chunk` 换成 `--rr-arm dgrad-chunk-bf16` |
| `rcb-b` | 1000 | 默认 | 同 `rcb-a`，`--dump-steps 100000`、不带 `--save-batches` |

- 注 1（`rcc-a`）：和 `rc` 相比只多输出头 dgrad 分段 FP32 累加。
  - 不带 `--full`，分布式优化器由 `--no-dist-opt` 关掉。
- 注 2（`rcc-b`..`rcc-j`）：`--dump-steps 100000` 即不抓张量，只记逐步指标。
  - 和 `rcc-a` 一起共 10 份，两两配对得到第一层噪声带的 45 对。
- 完整命令与理由见[指南第 2 章 2.3](../../../../docs/guide/02-gpu-golden.md#23-要造哪几组-golden怎么跑)；复现以运行目录 `meta.json` 的实际配置为准。
- 排查时用过的下面几种设置见 [`goldens.md`](../../goldens.md)，不是 golden：
  - `rce`（精确轨迹，BF16 权重、全程 FP32 计算）；
  - `rcec`（真精确轨迹，精确轨迹加输出头 dgrad 分段累加）；
  - 中间一步 `rcx`（生产 golden，输出头 dgrad 换 FP32 一次累加）。

## 7. 复现第一阶段的代表性结果（约 30 分钟）

最小的一条路：

- 两次确定性运行做确定性自证；
- 再加一两个故障注入运行，确认报警项和 [`models/hunyuan3/alignments/00-gpu-noise-and-drills/faults/phase1-faults.md`](../../alignments/00-gpu-noise-and-drills/faults/phase1-faults.md) 一致。

第一阶段的目录写死了它自己的运行目录根（`runs.root`）和运行名。`--test` 只能换被测，根和 golden 要复制一份目录改掉。

```bash
cd /data/janus-src
R=/data/janus-runs-repro                 # 你的运行目录根
C="--load /data/hy3-4l-mcore --data /data/ds/fw_hy3_text_document --deterministic --dump-steps 1 --full"
flock /data/gpu.lock /data/janus-src/models/hunyuan3/frameworks/megatron/run.sh det-a 30 $C --janus-run $R/det-a --side golden-1
flock /data/gpu.lock /data/janus-src/models/hunyuan3/frameworks/megatron/run.sh det-b 30 $C --janus-run $R/det-b --side golden-2
flock /data/gpu.lock /data/janus-src/models/hunyuan3/frameworks/megatron/run.sh f-w2 30 $C --janus-run $R/f-w2 --side golden-3 \
    --fault "weight_scale:decoder\.layers\.2\.self_attention\.linear_proj\.weight:1e-2"
flock /data/gpu.lock /data/janus-src/models/hunyuan3/frameworks/megatron/run.sh f-lr 30 $C --janus-run $R/f-lr --side golden-4 --fault lr_ulp

# 确定性自证：原目录是 1000 步、比第 1 与 1000 步；这里只有 30 步，第 1 步全量抓数
mkdir -p /data/cat
sed -e "s#test: rd-det-b#test: det-b#; s#golden: \[rd-golden-1\]#golden: [det-a]#; s#root: /data/janus-runs#root: $R#; s#1:1001#1:31#g" \
    models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/phase1-determinism.yaml | sed '/L4.step1000.bitwise/,$d' > /data/cat/determinism.yaml
python3 -m janus run /data/cat/determinism.yaml --out /data/results/determinism --html --verbose

# 故障注入：golden 换成 det-a；窗口本来就是 1–30 步，不用改
sed -e "s#golden: \[d-a\]#golden: [det-a]#; s#root: /data/janus-runs2#root: $R#" \
    models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/phase1-faults.yaml > /data/cat/faults.yaml
for t in det-b f-w2 f-lr; do
  python3 -m janus run /data/cat/faults.yaml --test $t --out /data/results/faults/$t --html --verbose
done
```

- `janus run` 只用 CPU，可以和训练并行。
- 故障目录一次约 5 分钟（主要是 FP64 优化器重放），加 `--html` 再多一两分钟。

**期望**（复现审计实测，[`models/hunyuan3/alignments/00-gpu-noise-and-drills/results/repro-audit.md`](../../alignments/00-gpu-noise-and-drills/results/repro-audit.md)）：

| 项 | 期望 |
|---|---|
| 第 1 步 `lm_loss`（`metrics.jsonl`） | `11.4657564163208`（确定性、默认设置都一样；第 1 步前向逐位确定） |
| 第 1 步 `grad_norm`（确定性设置） | `17.534862518310547`（关分布式优化器，见注 1） |
| 确定性自证 | 4 pass，退出码 0；`L3.step1.bitwise` 比 425 个张量全部逐位 |
| 故障目录，被测 `det-b`（对照） | 32 / 32 pass |
| 故障目录，被测 `f-w2` | 25 / 32 fail，与 `phase1-faults.md` 的 `f-weight2` 列逐项相同 |
| 故障目录，被测 `f-lr` | 7 / 32 fail（`L0.config.diff`、`L3.step1.updated`、`L4.loss`、`L4.lr`、`L4.grad_norm`、`L4.router.bias_std`、`L4.determinism`），与 `f-lr` 列相同 |

- 注 1：
  - `17.534862518310547` 是关分布式优化器，即带 `--full` / `--no-dist-opt`；
  - 开着分布式优化器是 `17.534860610961914`（`rd-golden-1`）；
  - 默认设置下第 1 步 grad_norm 就不逐位（反向非确定），不能当锚点。
- 故障目录有 fail 时退出码是 1，这是期望的。

## 训练脚本的常用参数（`train_hy3.py`）

### 基本、抓数、并行

| 参数 | 作用 |
|---|---|
| `--deterministic` | Megatron `deterministic_mode=True`，同时关 `cross_entropy_loss_fusion`；同配置重跑逐位相同 |
| `--janus-run <目录>` `--side <名>` `--dump-steps 1,100` | 挂抓数回调，写 janus 运行目录；`--dump-steps` 是迭代号（从 1 开始）；不给 `--dump-steps` 的值时默认 `1` |
| `--full` | 全量抓数（会关分布式优化器，见注 1） |
| `--dump-grads` | 只另抓逐参数梯度（会关分布式优化器） |
| `--no-dist-opt` | 关分布式优化器但不挂钩子：给「和全量抓数运行比」的对照用 |
| `--inject-from <参照运行目录>` | 逐层注入 |
| `--ep` `--tp` `--cp` `--mbs` `--recompute` | 并行方式、microbatch 大小、整层激活重算 |
| `--routing-replay` | 路由回放（要求 `--recompute`），见注 2 |
| `--save` `--save-interval` `--resume` | 存档、续训 |
| `--save-batches` | 每 rank 每步的原始输入整批存下（另一侧回放同一批数据用） |
| `--fault <种类>` | 故障注入（只用于验证检查会报警，种类见上表与脚本） |
| `--observe attn,moe,expw,gemm` | 可选观测点（默认不开），见注 3 |
| `--special <参数名正则>` | 特殊值存 / 恢复，见注 4 |
| `--no-tf32` | 把 torch 的 TF32 开关设为 False，见注 5 |

- 注 1（`--full`）：另抓子模块梯度、逐参数梯度、更新前后主权重与 Adam 动量。
  - 默认只在 dump 步里最小的一步做，`--full-steps` 可改。
- 注 2（`--routing-replay`）：反向重算沿用首次前向选中的专家，不重新选。
  - 只回放 top-k 下标，概率由重算的 logits 重新算。
  - 开 Megatron 的 `moe_enable_routing_replay`，由 [`routing_replay.py`](routing_replay.py) 按 microbatch 切换 RECORD / REPLAY_BACKWARD。
  - 每个 microbatch 开始、每步结束自检回放队列已取空，不空就报错。
  - 重算 golden `rc-*`（GPU 生产 golden）用它（`alignments/03-gpu-remat-golden/`）。
- 注 3（`--observe`）：核心注意力 q/k/v、MoE 层内各阶段、本地全部专家权重、第 1 个抓数步的 GEMM 清单。
  - 名字见 [`docs/reference/dump-format.md`](../../../../docs/reference/dump-format.md)。
- 注 4（`--special`）：
  - 不带 `--resume` 时，最后一步存档前往命中的张量埋 −0.0 与次正规数；
  - 带 `--resume` 时开训前读出；
  - 见 [`docs/dev/design/0004-l1-gpu-kernels.md`](../../../../docs/dev/design/0004-l1-gpu-kernels.md)。
- 注 5（`--no-tf32`）：TE 的 FP32 GEMM 还要靠环境变量 `NVIDIA_TF32_OVERRIDE=0`。
  - [`run_check.sh`](run_check.sh) 的 `TF32=off` 两样一起设。

### 诊断用（不用于 golden）

| 参数 | 作用 |
|---|---|
| `--router-true-fp32` | **只作诊断，不用于 golden**：路由权重留成 FP32 参数，路由 GEMM 前向、反向都是 FP32 × FP32（见注 1） |
| `--rr-arm <臂>[,<臂>…]` | **只作诊断，不用于 golden**（唯一例外：`dgrad-chunk` 臂用来造最终 golden `rcc`）：反向复现臂（见注 2） |
| `--resid-probe-every N --resid-probe-seq <npz>` | 每 N 步（和第 1 步）在参数更新后另跑一次前向，记残差流统计（见注 3） |
| `--fp32` | **诊断参照**：参数、激活、梯度全程 FP32（`MixedPrecisionConfig(fp32=True)`），见注 4 |
| `--bf16-weights` | 配 `--fp32`：BF16 权重、全程 FP32 计算，即精确轨迹（见注 5） |

- 注 1（`--router-true-fp32`）：
  - 隐含 `--no-tf32`，要求 `NVIDIA_TF32_OVERRIDE=0`（用 `TF32=off run_check.sh …`）；
  - 第 1 步每次路由 GEMM 对 FP64 的误差写进 `meta.json` 的 `router.probe`；
  - 做法与取舍见 [`router_fp32.py`](router_fp32.py)。
- 注 2（`--rr-arm`）：把一处数值行为换成 TPU（MaxText）的写法。
  - 涉及交叉熵、Adam、裁剪、最后 RMSNorm、注意力缩放系数、MoE 加权求和等；
  - 另有输出层梯度尺度与规范分量的敏感度探针；
  - 全部是运行时补丁，臂的清单与每臂的 MaxText 出处见 [`rr_arms.py`](rr_arms.py)；
  - 结果见案例 01 第 6 节「反向复现（续）」。
- 注 3（`--resid-probe-every`）：用当前参数在固定序列上前向，no_grad、eval，不改训练。
  - 记各层输入输出与最后归一化输出的 token 均值、logits 与 lse 均值、最后归一化的 scale。
  - 字段与 TPU 侧 `--resid-probe-every` 同名：`resid_mean.*`、`tok_rms.*`（每个 token 沿隐藏维的 RMS）、`probe_logit_mean`、`probe_lse_mean`、`final_norm_scale`。
  - 固定序列是 golden 第 1 步 rank 0 第一个 microbatch 的 tokens 与 position_ids（`.npz`，tokens 的 sha256 记进 `meta.json` 的 `resid_probe`）。
  - 见 [`resid_probe.py`](resid_probe.py)。
- 注 4（`--fp32`）：要配 `--no-tf32`、环境变量 `NVIDIA_TF32_OVERRIDE=0`、`--attention-backend unfused`。
  - 案例 01「逐步拆分」用它当梯度的精确参照。
  - 配 `--bf16-weights` 造精确轨迹 `rce`（排查时的参考裁判，不是 golden）。
- 注 5（`--bf16-weights`）：每步前向前把参数舍到 BF16，优化器更新前换回 FP32 主权重（`param_swap.py` 的 `Bf16WeightsFp32Compute`）。
  - 整条轨迹是 BF16 训练的精确版本，即精确轨迹（排查时的裁判，不是 golden）。

### 导出、导入参数

| 参数 | 作用 |
|---|---|
| `--export-params-at N[,…] --export-dir <目录>` | 第 N 步更新之后把整套 FP32 主权重（含路由偏置）按 rank 导出到 `<目录>/step_N/`（见注 1） |
| `--import-params <目录>` | 开训前把整套参数换成目录里的，换完读回逐张量核 sha256（见注 2） |
| `--import-round-bf16` | `--import-params` 导入时先把 FP32 主权重舍到 BF16 再放回（见注 3） |
| `--skip-steps K` | 开训时每个 rank 先丢掉 K 步的 microbatch（见注 4） |

- 注 1（`--export-params-at`）：
  - 文件：`rank<r>.safetensors` + `manifest.json`，Megatron 名与布局；
  - 非专家参数只在 rank 0 的文件里；
  - 与 TPU 侧 `--export-params-at` / `--params-from` 同一种格式，见 [`param_swap.py`](param_swap.py)。
- 注 2（`--import-params`）：换的是 FP32 主权重、BF16 参数（= FP32 按 RNE 舍入）、路由偏置。
  - 优化器动量不动，只用于重算一步梯度。
- 注 3（`--import-round-bf16`）：全程 FP32 参照用 BF16 程序实际参与计算的那份权重，只比算术误差。
  - 不混进「主权重 vs 舍入后权重」那半个 BF16 格的差。
- 注 4（`--skip-steps`）：本侧第 1 步吃原数据第 K+1 步的 batch（`meta.steps.convention` 记成 `golden_step = step + K`）。
  - 配 `--import-params` 在同一状态上重算第 K+1 步的梯度。

## L1 单算子回放

```bash
source /data/env.sh; export PYTHONPATH=/data/janus-src:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python3 /data/janus-src/models/hunyuan3/frameworks/megatron/l1_ops.py --run <一次带 --full 的确定性运行目录，第一阶段用 d-a> --out <目录>
```

一张卡，产出 results.json + details/ + report.md，见 [`docs/dev/design/0003-l1-op-replay.md`](../../../../docs/dev/design/0003-l1-op-replay.md)。

## kernel 回放、MoE 分阶段、poison / 因果性、TF32、特殊值存恢复

GPU 侧剩余的五项检查。

- 训练用 [`run_check.sh`](run_check.sh)：
  - 与第一阶段 `d-a` 用的 `/data/r.sh` 同一套参数；
  - 确定性设置下也不设那三个环境变量，和 `d-a` 一样；
  - 训练脚本与 janus 取本检出，不碰共享目录；
  - `TF32=off` 关 TF32。
- 回放与比较（一张卡）：
  - [`l1_kernels.py`](l1_kernels.py)
  - [`moe_stages.py`](moe_stages.py)
  - [`causal_model.py`](causal_model.py)
  - [`ckpt_special.py`](ckpt_special.py)
- 完整命令、做法与结论见 [`docs/dev/design/0004-l1-gpu-kernels.md`](../../../../docs/dev/design/0004-l1-gpu-kernels.md)。
- 结果在 `results/phase1-{obs,tf32,kernels,moe-stages,causal-model,ckpt-special}.*`。

## 环境上的三个坑（`env.sh` 已处理）

1. 节点把 NCCL 2.27 挂到 `/usr/local/nvidia/lib64`，torch 需要容器里的 2.31 → `LD_LIBRARY_PATH` 前置 `/usr/lib/x86_64-linux-gnu`。
2. 容器默认启用 GCP 的 NCCL 插件，需要 RDMA 网卡，单机 Pod 没挂 → `NCCL_*_PLUGIN=none`。
3. Megatron 确定性模式要求 `TRITON_CACHE_DIR`（容器默认开了 `TRITON_CACHE_AUTOTUNING=1`）。

另：容器里 TF32 默认是开的，[`run.sh`](run.sh) 没有关。

- 训练里受影响的是路由反向的两个 GEMM。
- 关法与影响见 [`docs/reference/environment.md`](../../../../docs/reference/environment.md) 第 4 条。
- 给了 `--janus-run` 时，`meta.json` 的 `tf32` 字段记开关读回值与实测。

## 已验证（2026-11-03）

- 确定性设置下同配置两次 20 步：lm_loss、grad_norm 逐步逐位相同。
- 默认设置两次：第 1 步逐位相同，第 2 步起不同（前向确定，反向 / 更新引入差）。
- 梯度抓取（关分布式优化器）：
  - 开 / 不开 lm_loss 逐位相同；
  - 两次确定性运行第 1 步 182 个梯度张量逐位相同；
  - 梯度名暂用 Megatron 参数名（`grad.decoder.layers.<i>....`），专家参数只含 rank 0 持有的那 24 个专家。
- 抓数回调不改数值：
  - 开回调的确定性运行与不开的逐步逐位相同；
  - 两次开回调的确定性运行，第 1、20 步 24 个张量逐位相同、覆盖率完整。
- 从零复现（2026-11-04）：照本页在另一个目录重跑，第 1 步 loss、确定性自证、两个故障注入运行的报警项都与第一阶段一致。
  - 记录见 [`models/hunyuan3/alignments/00-gpu-noise-and-drills/results/repro-audit.md`](../../alignments/00-gpu-noise-and-drills/results/repro-audit.md)。
