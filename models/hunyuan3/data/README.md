[janus](../../../README.md) › [用户指南](../../../docs/guide/README.md) › [混元 3](../README.md) ｜ TPU 原生数据加载器；对应指南[第 3 章 3.3](../../../docs/guide/03-tpu-setup.md#33-数据回放同一批输入)

# 混元 3 · TPU 原生数据加载器对拍（MaxText Grain 读 Megatron 索引格式）

TPU 侧用 MaxText 自己的数据加载器，直接读 GPU 训练用的那份 Megatron `.bin/.idx` 数据。

- 加载器：Grain 输入管线，`grain_file_type: mmap_npy`，上游 AI-Hypercomputer/maxtext#4625。
- 对拍对象：GPU（Megatron-Bridge `GPTDataset`）第一阶段 1000 步 golden 实际吃进去的输入，逐样本比。
- 全程只用 CPU。

相关材料：

- 结论和数字见 [`../alignments/01-v5p-acceptance/results/phase2-native-loader.report.md`](../alignments/01-v5p-acceptance/results/phase2-native-loader.report.md)（从同名 `results.json` 渲染）。
- 每项检查的设计与状态见 [`docs/dev/design/0002-l0-data-weights.md`](../../../docs/dev/design/0002-l0-data-weights.md)。

> **和迁移验收的关系**：
> - TPU v5p / v7x 的对齐运行不用这个加载器，而是原样回放 GPU 每一步每个 rank 的 batch。
>   - 「每 rank 每步输入指纹」0 / 1950 个单元不同，见[对齐报告](../ALIGNMENT-REPORT.md)第 4 节。
> - 原因：原样的设备分配在梯度累积下与 Megatron 不同，换序后才逐元素对上。
>   - `L0.native.device_map.native`：原样布局只有 2000 / 16000 个（设备 · microbatch）单元与 GPU 同号 rank 相同。
> - 改用原生加载器训练时，按本页重跑对拍，并按「设备与 rank 的对应」一节换序。

## 这里有什么

| 文件 | 跑在哪 | 做什么 |
|---|---|---|
| [`run_parity.sh`](run_parity.sh) | GPU Pod（只用 CPU） | 一条命令从头跑完下面全部步骤 |
| [`maxtext_dump.py`](maxtext_dump.py) | MaxText venv | 走 MaxText 训练入口，8 个 CPU 假设备，逐步存全局批；`--emulate-hosts H` 模拟多 host（注） |
| [`megatron_reference.py`](megatron_reference.py) | Pod 的 `/opt/venv`（Megatron-Bridge） | 用 Megatron 自己的 `GPTDataset` 现建索引、取样本：跨 epoch（`epochs`）、合成边界用例（`synthetic`） |
| [`maxtext_reference.py`](maxtext_reference.py) | MaxText venv | 与上面一一对应，用 MaxText 同样的数据源与变换取样本 |
| [`native_loader_parity.py`](native_loader_parity.py) | MaxText venv（只要 numpy） | 逐项对拍，写 `results.json` + `report.md` |
| [`janus/data_layout.py`](../../../janus/data_layout.py) | — | 两边「样本 → 设备 / rank / microbatch」的分法和换序（见下文「设备与 rank 的对应」） |

- 注：训练入口指 `pyconfig` → `get_mesh_from_config` → `create_data_iterator`。

## 环境

### GPU Pod

- GPU Pod `janus-gpu`（`frameworks/megatron/pod.yaml` 起的那个；下面的 `kubectl` 按自己的集群加 `--context`）。
- 数据 `/data/ds/fw_hy3_text_document.{bin,idx}`。
  - fineweb-edu 一个分片，混元 3 分词器、`--append-eod` 预处理；文档数、token 数见报告。
- 分词器 `/data/hy3-4l`。
- GPU golden `/data/janus-runs2/g1k-a`（第一阶段 GPU golden 的第 1 份）。
  - 与 `gs://<POC 桶>/janus/hunyuan3-4l/golden/g1k-a/` 上的副本相同。
  - 数据指纹与整批输入索引的 sha256、`rank0.bin` 的 md5 都核对过。

### MaxText

- MaxText 固定在上游 `ce608b918c4fdfbfdb62f5a7976d2e15e4fdd720`。
  - `uv pip install -e .[tpu] --resolution=lowest` 装进单独的 venv，`JAX_PLATFORMS=cpu`。
  - 不碰 Pod 的系统 Python、`/opt/venv` 和 `/data/janus-src`。
- 训练用的 `maxtext` `hunyuan` 分支已包含该加载器（相对 `AI-Hypercomputer/maxtext#4625` 的合入提交 behind 0）。
  - `_mmap_datasource.py`、`_mmap_index_utils.py` 与上面的上游提交逐字节相同。
  - `grain_data_processing.py`、`input_pipeline_utils.py`、`multihost_dataloading.py`、`gradient_accumulation.py` 有差异。
  - 按关键词（mmap、megatron、eod、segment、position、reshape、microbatch、host 下标）核对差异行：只命中 colocated python 远程迭代器取 host 下标的一处，不在这条路径上。
- 换分支或升级 MaxText 后要重跑本对拍。

## 一条命令复现

在本地 janus 仓库根目录：

```bash
K="kubectl"          # 需要时加 --context <集群>
tar czf /tmp/janus.tgz --exclude=.git . && $K cp /tmp/janus.tgz janus-gpu:/tmp/janus.tgz
$K exec janus-gpu -- bash -c 'rm -rf /tmp/janus-src-g62 && mkdir /tmp/janus-src-g62 && tar xzf /tmp/janus.tgz -C /tmp/janus-src-g62'
$K exec janus-gpu -- bash -c 'setsid nohup bash /tmp/janus-src-g62/models/hunyuan3/data/run_parity.sh /tmp/g62 > /tmp/g62.log 2>&1 &'
# 要跑十几到二十几分钟，最慢的是最后对拍那步（单线程逐单元比）
$K exec janus-gpu -- tail -3 /tmp/g62.log
$K cp janus-gpu:/tmp/g62/parity/results.json models/hunyuan3/alignments/01-v5p-acceptance/results/phase2-native-loader.results.json
$K cp janus-gpu:/tmp/g62/parity/report.md models/hunyuan3/alignments/01-v5p-acceptance/results/phase2-native-loader.report.md
```

退出码按 [`docs/reference/result-schema.md`](../../../docs/reference/result-schema.md)：硬检查有 fail / error 时非 0。

## MaxText 怎么配才和 GPU 一致

GPU 侧配置来自两处：

- [`models/hunyuan3/frameworks/megatron/train_hy3.py`](../frameworks/megatron/train_hy3.py)（`_pretrain_common()` 的默认 + 覆盖）；
- 运行的 `meta.json`。

对应的 MaxText 配置：

```yaml
dataset_type: grain
grain_file_type: mmap_npy
grain_train_files: "/tmp/g62/npy|/data/ds/fw_hy3_text_document"   # 索引缓存目录|数据前缀；目录里没有就现建
max_target_length: 4096          # = seq_length；MaxText 读 4097 个 token，inputs 取前 4096、targets 取后 4096
mmap_eod_id: 120025              # = 预处理 --append-eod 写进去的 EOD（混元 3 分词器的 eos）；MaxText 默认 0，必须改
data_shuffle_seed: 1234          # = dataset.random_seed；MaxText 默认 0，必须改
mmap_npy_split: "9999,1,0"       # = dataset.split；不设就用全部文档，文档索引整个不同
reset_attention_mask: false      # GPU：reset_attention_mask = reset_position_ids = false；MaxText 默认 true，必须改
eod_mask_loss: false             # GPU：eod_mask_loss = false
packing_max_segments_per_sample: 0
mmap_split_sentences: false
per_device_batch_size: 1         # = micro_batch_size
gradient_accumulation_steps: 2   # 全局 batch 16 = 8 个设备 × 1 × 2，对应 GPU 的 DP 8 × mbs 1 × 每 rank 2 个 microbatch
steps: 1000                      # 和全局 batch 一起决定索引的样本数（steps × 16 = 16000 = GPU 的 train_iters × gbs）
```

### 几条容易配错的

几条容易配错的（都有反例，见报告的「反例」一节）：

- `reset_attention_mask` 在 MaxText 里**同时**管注意力分段和位置重置。
  - Megatron 是 `reset_attention_mask`、`reset_position_ids` 两个独立开关。
  - 两者取值相同时能对上；不同时 MaxText 表达不了。
  - MaxText 默认 `true`、PR 里的示例也是 `true`，而 Megatron-Bridge 默认两个都是 `false`。
- `steps` 不只是训练步数：`mmap_npy` 用 `steps × 全局 batch` 算 epoch 数，epoch 数不同索引就不同。
  - 只想少读几步时不要改 `steps`（[`maxtext_dump.py`](maxtext_dump.py) 用 `--dump-steps`）。
- `mmap_npy_split` 要照抄 GPU 的 `split`。
  - Megatron-Bridge `_pretrain_common()` 的默认是 `9999,8,2`，混元 3 的脚本改成了 `9999,1,0`。

## 设备与 rank 的对应

两边每步取的是同一组 16 个样本、顺序也相同（`L0.native.stream`），但一步之内分给谁不同：

- Megatron（`MegatronPretrainingSampler`）：每个 microbatch 取连续 8 个。
  - rank d 的第 m 个 microbatch 是步内第 `m×8 + d` 个。
- MaxText：一次读整个全局批，设备 d 持有第 `2d`、`2d+1` 行。
  - 梯度累积按 `reshape(gbs//ga, ga)` 再 `swapaxes` 切 microbatch。
  - 设备 d 的第 m 个 microbatch 是第 `d×2 + m` 行。

MaxText 没有配置项能改这个分法，所以换序放在 janus（[`janus/data_layout.py`](../../../janus/data_layout.py)）：

```python
from janus.data_layout import maxtext_perm_for_megatron, per_device
perm = maxtext_perm_for_megatron(dp=8, mbs=1, ga=2)          # [0, 8, 1, 9, 2, 10, ..., 7, 15]
batch = {k: v[perm] for k, v in batch.items()}               # 对每步的全局批（[16, 4096]）做行置换
dev = per_device(batch["inputs"], dp=8, mbs=1, ga=2)         # {(设备, microbatch): [1, 4096]}，= GPU rank / microbatch
```

### 多 host

- 多 host 时 MaxText 先按 host 切样本流（host h 读第 h、h+H、… 个），全局批里的行序又变一次。
- `maxtext_perm_for_megatron(..., hosts=H)` 一并换回来（`L0.native.hosts` 在单进程里模拟了 2、4 个 host）。

### 什么时候要换序

- 这个换序只影响「哪个设备拿哪个样本」，不影响全局批的内容。
- 按 rank 逐个对比（L0 指纹、逐 rank 的抓数）时要用；只看全局 loss、梯度时不需要。
- 要在 TPU 训练里让设备 d 真的吃 rank d 的样本，需要在训练循环取到全局批后做同样的行置换。
  - 改训练循环，不在本次范围内。

## 分步跑

[`run_parity.sh`](run_parity.sh) 就是下面这些命令按顺序跑（`W=/tmp/g62`，`S=<janus>/models/hunyuan3/data`，`PY=$W/venv/bin/python`）：

```bash
# MaxText 读 1000 步；索引缓存目录里没有就现建（host 0 写盘）
$PY $S/maxtext_dump.py --data /data/ds/fw_hy3_text_document --steps 1000 --npy-dir $W/npy --out $W/mt-w1 --workers 1
#   读取进程数、模拟多 host、反例：--workers 0|8；--emulate-hosts 2|4；--reset-attention-mask --packing-divisor 25；--split ""；--seed 0
# Megatron 参照（跨 epoch：16000 / 356736 / 400000 个样本 = 1 / 2 / 3 个 epoch；合成边界用例）
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu CUDA_VISIBLE_DEVICES= /opt/venv/bin/python $S/megatron_reference.py epochs \
  --data /data/ds/fw_hy3_text_document --hf /data/hy3-4l --num-samples 400000 --cache $W/meg-cache --out $W/meg-ep-400000
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu CUDA_VISIBLE_DEVICES= /opt/venv/bin/python $S/megatron_reference.py synthetic \
  --hf /data/hy3-4l --work $W/syn --out $W/meg-syn
# MaxText 参照
$PY $S/maxtext_reference.py epochs --ref $W/meg-ep-400000 --data /data/ds/fw_hy3_text_document --npy-dir $W/npy-ref --out $W/mt-ep-400000
$PY $S/maxtext_reference.py synthetic --ref $W/meg-syn --npy-dir $W/npy-ref --out $W/mt-syn
# 对拍（完整参数见 run_parity.sh）
$PY $S/native_loader_parity.py --mt $W/mt-w1 --gpu-run /data/janus-runs2/g1k-a \
  --megatron-index-dir /data/ds/fw_hy3_text_document/cache/GPTDataset_indices --out $W/parity --pad-id 120002 ...
```

## 比了什么、怎么定位

| 检查 | 比什么 |
|---|---|
| `L0.native.gpu_integrity` | GPU 整批输入每个数组重算 sha256，对它自己的索引和数据指纹（参照先可信） |
| `L0.native.index` | doc / sample / shuffle 三个索引：GPU 训练时 Megatron 缓存的 `.npy` 对 MaxText 运行时现建的，文件 sha256 |
| `L0.native.stream` | 每步全局批按行对 Megatron 步内样本流（与设备分配无关） |
| `L0.native.device_map.native` | MaxText 原样的设备 / microbatch 对 GPU 同号 rank / microbatch（不换序） |
| `L0.native.fingerprint` | 换序后每步 × 8 设备 × 2 microbatch × 4 字段，逐元素 + sha256 对 GPU 数据指纹（硬检查） |
| `L0.native.nonpad_total` | 每步全局非 padding token 数（GPU：loss_mask 之和；MaxText：`targets_segmentation != 0` 之和） |
| `L0.native.fields` | 字段对应与 dtype / shape、注意力分段与位置约定、EOD 与 pad id 计数 |
| `L0.native.readers` | `grain_worker_count` 0 / 8 对 1 逐步逐位 |
| `L0.native.hosts` | 模拟 2 / 4 个 host |
| `L0.native.epochs` | 跨 epoch、最后一个 epoch 单独打乱：两边现建的三个索引与边界样本 |
| `L0.native.eod_cases` / `L0.native.pad_id` | 合成边界用例在 Megatron 8 种开关组合下的全部字段与注意力 mask |
| `L0.native.negative_controls` | 故意配错 MaxText，对拍必须报出来并定位对 |
| `L0.native.coverage` | 1000 步 × 8 × 2 × 4 个单元都比到 |

### 字段与注意力的对应

- 字段对应：
  - `inputs → tokens`
  - `targets → labels`
  - `targets_segmentation != 0 → loss_mask`
  - `inputs_position → position_ids`
- 注意力：MaxText `inputs_segmentation` 全 1 = 整条序列一个因果段 = GPU `reset_attention_mask=false`。
  - GPU 没存注意力 mask，按配置判。

### 对不上时怎么定位

对不上时 `native_loader_parity.localize` 按顺序判：

1. 样本在这一步但分给了别的设备（分配约定）
2. 这一步没有这个样本（索引重建 / 打乱）
3. 只有最后一个 label 不同（seq+1 取 label 的约定）
4. loss_mask 在 EOD / pad id 处不同（EOD / mask 约定）
5. position 不同（位置重置约定）

[`tests/models/hunyuan3/test_native_loader_parity.py`](../../../tests/models/hunyuan3/test_native_loader_parity.py) 对每一种都故意改坏一次，确认能抓到、定位对。
