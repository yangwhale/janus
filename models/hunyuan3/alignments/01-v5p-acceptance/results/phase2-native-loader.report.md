# TPU 原生数据加载器对拍（MaxText Grain mmap_npy ↔ Megatron GPTDataset）

GPU 参照 `/data/janus-runs2/g1k-a`（DP 8、mbs 1、全局 batch 16、每 rank 2 个 microbatch、序列 4096、种子 1234）；MaxText `ce608b918c4f`，CPU 上 8 个设备；第 1–1000 步，共 1000 步。

数据 `/data/ds/fw_hy3_text_document`：`.idx` sha256 两边都是 `83d5a23b14d38944…`。

| 检查 | 状态 | 说明 |
|---|---|---|
| `L0.native.gpu_integrity` | pass | GPU 整批输入与其 sha256 索引、数据指纹全部一致 |
| `L0.native.index` | pass | 三个索引两边重建结果逐字节相同（文件 sha256 也相同） |
| `L0.native.stream` | pass | 每步 16 个样本按顺序与 Megatron 的样本流逐位相同 |
| `L0.native.device_map.native` | fail | 原样布局只有 2000 / 16000 个（设备 · microbatch）单元与 GPU 同号 rank 相同：MaxText 梯度累积按 reshape(gbs//ga, ga) 交错取行，Megatron 每个 microbatch 取连续 dp×mbs 个；样本本身没错（见 L0.native.stream），差在分配约定 |
| `L0.native.fingerprint` | pass | 换序后 1000 步 × 8 个设备 × 2 个 microbatch 的 tokens / labels / loss_mask / position_ids 全部逐元素相同，sha256 与 GPU 数据指纹相同 |
| `L0.native.nonpad_total` | pass | 每步全局非 padding token 数逐步相等 |
| `L0.native.fields` | pass | 字段约定一致：MaxText 用 int32、按全局批 [gbs, L] 存，GPU 用 int64 / float32、按 [mbs, L] 存；按 inputs→tokens、targets→labels、targets_segmentation≠0→loss_mask、inputs_position→position_ids 换算后数值逐位相同；inputs_segmentation 全 1（整条序列一个因果段）、位置 0..L-1，与 GPU 配置一致 |
| `L0.native.readers` | pass | grain_worker_count 0 / 8 与 1 逐步逐位相同 |
| `L0.native.hosts` | pass | 2 host：读进来的行按 host 交错（16000/16000 行符合 host 切分公式），用单 host 换序只对上 36000/64000，按 host 数换序后 64000/64000；4 host：读进来的行按 host 交错（16000/16000 行符合 host 切分公式），用单 host 换序只对上 40000/64000，按 host 数换序后 64000/64000 |
| `L0.native.epochs` | pass | 16000 个样本（1 个 epoch）：三个索引相同，8 个样本（含跨 epoch 边界的 0 个）逐位相同；356736 个样本（2 个 epoch）：三个索引相同，11 个样本（含跨 epoch 边界的 1 个）逐位相同；400000 个样本（3 个 epoch，最后一个 epoch 单独打乱）：三个索引相同，14 个样本（含跨 epoch 边界的 2 个）逐位相同 |
| `L0.native.eod_cases` | pass | 合成数据（15 篇文档，含 1 token 空文档、恰好 seq / seq+1 长、跨两个样本的长文档，序列 16，2 个 epoch）：MaxText 能表达的 4 种组合（reset_attention_mask = reset_position_ids）下，不含 pad id 的样本 tokens / labels / loss_mask / position_ids / 注意力 mask 全部逐位相同；reset_attention_mask ≠ reset_position_ids 的 4 种组合 MaxText 表达不了（位置重置跟着注意力走） |
| `L0.native.pad_id` | fail | 含 pad id 的样本在 4 种组合下共 8 处两边不同（字段 ['labels', 'loss_mask', 'tokens']）：Megatron GPTDataset 把 tokens / labels 里的 pad id 换成 0、label 为 pad id 的位置 loss_mask 置 0，MaxText mmap_npy 原样保留、照常算 loss。只在数据里出现 pad id 时触发（见 L0.native.fields 的计数） |
| `L0.native.negative_controls` | pass | MaxText 默认的 reset_attention_mask 与 packing：68/80 个单元不同（position）；不设 mmap_npy_split：80/80 个单元不同，索引不同（index/shuffle）；MaxText 默认种子 0：80/80 个单元不同，索引不同（index/shuffle） |
| `L0.native.coverage` | pass | 全部比到 |

## 索引 sha256（文件）

| 索引 | dtype / shape | Megatron | MaxText | 相同 |
|---|---|---|---|---|
| document_index | int32 [725927] | `bcf1a5ecb2cb387a…` | `bcf1a5ecb2cb387a…` | 是 |
| sample_index | int32 [178375, 2] | `d61e4bc3542304f4…` | `d61e4bc3542304f4…` | 是 |
| shuffle_index | uint32 [178374] | `8f61dfcec9ce2f7e…` | `8f61dfcec9ce2f7e…` | 是 |

## 设备与 rank 的对应

MaxText 原样：设备 d 的第 m 个 microbatch 是全局批第 d×ga+m 行；Megatron：rank d 的第 m 个 microbatch 是步内第 m×dp+d 个样本。换序 `batch[perm]`，perm = `[0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]`（`janus.data_layout.maxtext_perm_for_megatron`）。

## 边界与约定计数

- 比到的输入里 EOD 共 65444 个（跨文档边界确实被覆盖）；GPU loss_mask 为 0 的位置 0 个
- pad id 120002 在比到的样本里出现：GPU 0 次、MaxText 0 次；整份 .bin（726000 篇文档、730699036 个 token、EOD 726000 个）里 0 次（两边对 pad id 的处理不同，见 `L0.native.pad_id`；出现次数为 0 时这条差异不触发）

## 多 host（模拟）

| host 数 | 行符合 host 切分公式 | 单 host 换序对上 | 按 host 数换序对上 |
|---|---|---|---|
| 2 | 16000 / 16000 | 36000 / 64000 | 64000 / 64000 |
| 4 | 16000 / 16000 | 40000 / 64000 | 64000 / 64000 |

单进程里对每个 host 调 MaxText 的 get_datasets(host_index=h, host_count=H) + _mmap_pretrain_pipeline，本地批按 host 拼成全局批（假定 host h 的设备在 mesh 批维上连续）；没有跑真的多进程 / MultiHostDataLoadIterator。

## 跨 epoch（两边各自现建索引）

| 样本数 | epoch | 最后一个 epoch 单独打乱 | 样本总数 | 三个索引相同 | 比的样本 | 跨 epoch 边界的流下标 |
|---|---|---|---|---|---|---|
| 16000 | 1 | 否 | 178374 | 是 | 8 | — |
| 356736 | 2 | 否 | 356748 | 是 | 11 | 186901 |
| 400000 | 3 | 是 | 535122 | 是 | 14 | 216201, 512075 |

## 合成边界用例（EOD / 位置 / 注意力 / loss mask）

文档长度 `[3, 1, 17, 16, 2, 7, 1, 5, 15, 4, 9, 35, 6, 1, 3]`（每篇末尾是 EOD），序列 16。

| Megatron（ram / rpi / eml） | MaxText（ram / eml） | 不含 pad 的样本逐位相同 | 含 pad 的样本 | 说明 |
|---|---|---|---|---|
| ram0_rpi0_eml0 | ram0_eml0 | 13 / 13 | 2 | |
| ram0_rpi0_eml1 | ram0_eml1 | 13 / 13 | 2 | |
| ram0_rpi1_eml0 | — | — | — | MaxText 表达不了：reset_attention_mask 同时决定注意力分段与位置重置（不含 pad 的样本对 ram0_eml0 不同：['position_ids']；不含 pad 的样本对 ram1_eml0 不同：['attention_mask']） |
| ram0_rpi1_eml1 | — | — | — | MaxText 表达不了：reset_attention_mask 同时决定注意力分段与位置重置（不含 pad 的样本对 ram0_eml1 不同：['position_ids']；不含 pad 的样本对 ram1_eml1 不同：['attention_mask']） |
| ram1_rpi0_eml0 | — | — | — | MaxText 表达不了：reset_attention_mask 同时决定注意力分段与位置重置（不含 pad 的样本对 ram1_eml0 不同：['position_ids']；不含 pad 的样本对 ram0_eml0 不同：['attention_mask']） |
| ram1_rpi0_eml1 | — | — | — | MaxText 表达不了：reset_attention_mask 同时决定注意力分段与位置重置（不含 pad 的样本对 ram1_eml1 不同：['position_ids']；不含 pad 的样本对 ram0_eml1 不同：['attention_mask']） |
| ram1_rpi1_eml0 | ram1_eml0 | 13 / 13 | 2 | |
| ram1_rpi1_eml1 | ram1_eml1 | 13 / 13 | 2 | |

## 反例（故意配错，对拍必须报出来）

| 反例 | 配置 | 前几步单元相同 | 索引相同 | 定位 |
|---|---|---|---|---|
| MaxText 默认的 reset_attention_mask 与 packing | reset_attention_mask=True, packing_max_segments_per_sample=25, mmap_npy_split='9999,1,0', data_shuffle_seed=1234 | 12 / 80 | 是 | position：position_ids 不同（EOD 后是否重置，reset_position_ids / reset_attention_mask）（68） |
| 不设 mmap_npy_split | reset_attention_mask=False, packing_max_segments_per_sample=0, mmap_npy_split='', data_shuffle_seed=1234 | 0 / 80 | 否 | index/shuffle：这一步根本没有这个样本（索引重建或打乱不同）（80） |
| MaxText 默认种子 0 | reset_attention_mask=False, packing_max_segments_per_sample=0, mmap_npy_split='9999,1,0', data_shuffle_seed=0 | 0 / 80 | 否 | index/shuffle：这一步根本没有这个样本（索引重建或打乱不同）（80） |
