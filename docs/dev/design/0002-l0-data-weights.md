# L0 设计：数据加载一致与权重直接转换

L0 要求两边拿到的数据和权重**逐位相同**。这份文档讲两件迟早要做成生产能力的事：

- TPU 侧的数据加载器与 GPU 逐样本一致；
- GPU（Megatron）与 TPU（MaxText）权重的直接互转。

每一项标明状态：

- **已落地**：代码已合入，并在混元 3 上跑出了结论
- **部分**：做了一部分，缺什么写在说明里
- **未落地**：还没做，写清要做什么

## 一、数据：两条路，用途不同

| 路线 | 做法 | 用途 | 状态 |
|---|---|---|---|
| **回放** | GPU 训练时把实际吃进去的数据原样存下，TPU 按同样顺序回放（见注 1） | **数值对齐**：把数据管线从变量里拿掉，只比算法与数值 | 已落地（GPU 侧导出）；TPU 侧回放见第二阶段（见注 1） |
| **原生加载一致** | TPU 用自己的数据加载器读同一份原始数据，要求与 GPU **逐元素相同**（见注 2） | **生产训练**：TPU 上正式训练时用的就是它，必须证明和 GPU 读出来的是同一批样本 | 部分（见注 2） |

- 注 1（回放）：
  - 存的是每个 rank、每步、每个 microbatch 实际吃进去的 tokens / labels / loss_mask / position_ids（`--save-batches`，带 sha256）。
  - TPU 不读原始数据，按同样顺序回放。
  - 状态：GPU 侧导出已落地，第一阶段 1000 步 golden 已带。
- 注 2（原生加载一致）：
  - 原始数据是 Megatron `.bin/.idx` 索引格式；要求每个设备拿到的样本与 GPU 对应 rank **逐元素相同**。
  - 已做：MaxText `mmap_npy` 读混元 3 第一阶段数据，1000 步全程逐元素对上 GPU（设备分配要换序）。
  - 未做：TPU 上的首步 loss、续训位置、真多机。
  - 结果：`models/hunyuan3/alignments/01-v5p-acceptance/results/phase2-native-loader.*`

为什么两条都要：

- 回放让对齐结论不受数据管线干扰。
- 但生产训练不可能永远回放 GPU 的数据。
- 上生产前，原生加载器必须单独过下面的检查。

### 原生加载器要满足的检查

| 检查 | 比什么 / 判据 | 状态 |
|---|---|---|
| 设备与 rank 的对应 | TPU 设备 d 拿 GPU rank d 的数据；先把两边的顺序对上，再谈逐位 | 已落地（对拍），见注 1 |
| 每个设备每步的样本逐位相同 | tokens、labels、loss_mask、position_ids 的 sha256，按「步 · rank · microbatch」配对 | 已落地，见注 2 |
| 每步全局非 padding token 数 | 逐步相等 | 已落地，见注 3 |
| 同一份权重、同一批数据，首步 loss 逐位相同 | 两次运行首步 loss | 已落地（GPU 自比）；TPU 侧未做，见注 4 |
| 不只比前几步 | 窗口要覆盖续训点、epoch 边界、混合数据集的切换点 | 部分，见注 5 |
| 混合配比 / 打乱索引表逐位 | 两边重建的 doc / sample / shuffle 索引与混合表的 sha256 | 部分，见注 6 |
| 字段级 dtype / shape / sha256 | 每个样本的全部字段 | 已落地，见注 7 |
| mask 与 EOD 边界用例 | 合成边界用例（文档边界、packing、EOD 后 position 是否重置）两边输出一致 | 已落地，见注 8 |
| 续训后的数据位置 | 第 k 步存档恢复后，每个 rank 的样本与连续训练逐位相同 | 已落地（GPU 侧）；TPU 侧回放数据已核（[`phase2-resume`](../../../models/hunyuan3/alignments/01-v5p-acceptance/results/phase2-resume.report.md) 的 `L0.resume.data`），原生加载器（Grain 迭代器）续训未做，见注 9 |
| 多机、多读取进程 | 不同 host 数、不同读取进程数下样本顺序不变 | 部分，见注 10 |
| 分词器与预处理钉版本 | 分词器文件 sha256、是否在线重新分词，写进运行元信息并对照 | 部分，见注 11 |

1. **设备与 rank 的对应**
   - 为什么：顺序没对上，后面全是假失败。
   - 状态：两边每步取同一组样本、顺序相同，但梯度累积时一步之内的分法不同。
     - MaxText 原样只有 2000 / 16000 个（设备 · microbatch）单元对上同号 rank。
     - 用 `janus.data_layout` 换序后全部对上。
     - MaxText 没有配置项能改分法，训练里要按 rank 对齐需在训练循环里做同样的行置换。
2. **每个设备每步的样本逐位相同**
   - 为什么：最直接的证据。
   - 状态：1000 步 × 8 × 2 × 4 个字段 = 64000 个单元逐元素相同，sha256 与 GPU 数据指纹相同。
3. **每步全局非 padding token 数**
   - 为什么：一根绊线。不等先查数据竞态（预取线程、epoch 切换），不要先查数值。
   - 状态：1000 步逐步相等。
4. **首步 loss 逐位相同**
   - 为什么：首步就不同 = 数据或权重不同，与数值无关。
   - 状态：GPU 自比已落地；TPU 侧未做（要 TPU）。
5. **不只比前几步**
   - 为什么：前几步全对、后面才错的情况真实存在。
     - 混合配比表按浮点算样本归属，一个末位不同就把样本分到另一个数据集。
   - 状态：
     - GPU 训练的 1000 步全程比。
     - 跨 epoch（2、3 个 epoch，含最后一个 epoch 单独打乱）：两边各自现建索引，跨 epoch 边界的样本逐位相同。
     - 未做：混合数据集切换点（只有一个数据集）、TPU 续训点。
6. **混合配比 / 打乱索引表逐位**
   - 为什么：索引表是浮点算出来的（归一化、累加）。
     - 编译器的乘加融合（FMA）或累加方式不同，就会差一个末位。
     - 要求两边用同样的精度（FP64）和同样的计算顺序。
   - 状态：
     - doc / sample / shuffle 三个索引：GPU 训练时 Megatron 缓存的与 MaxText 运行时现建的文件 sha256 相同（1、2、3 个 epoch 都相同）。
     - 混合表没测（这份数据只有一个前缀，Megatron 不建混合表）。
7. **字段级 dtype / shape / sha256**
   - 为什么：只比 tokens 会漏掉 mask、位置、loss 权重。
   - 状态：
     - 四个字段逐元素比。
     - MaxText 另两个字段（`inputs_segmentation`、`targets_position`）按 GPU 配置核对。
     - dtype / shape 约定不同（MaxText int32、按 [全局批, L]；GPU int64 / float32、按 [mbs, L]），换算后数值逐位相同。
     - GPU 没存注意力 mask，按配置判。
8. **mask 与 EOD 边界用例**
   - 为什么：边界处的约定最容易两边写得不一样。
   - 状态：合成数据在 Megatron 8 种开关组合下比全部字段与注意力 mask。
     - MaxText 能表达的 4 种（`reset_attention_mask` = `reset_position_ids`）下，不含 pad id 的样本全部一致。
     - 发现两处差异：
       - MaxText 一个开关同时管注意力和位置，另 4 种组合表达不了；
       - 数据里出现 pad id 时 Megatron 把它换成 0 并屏蔽 loss，MaxText 不处理（混元 3 这份数据里 pad id 出现 0 次）。
9. **续训后的数据位置**
   - 为什么：存档没存对数据位置，续训会吃到错的样本。
   - 状态：GPU 侧已落地（`phase1-resume`）；TPU 侧用 janus 回放数据的存档续训已核（`phase2-resume` 的 `L0.resume.data`：token 与首步 loss 逐位）；MaxText 原生加载器存 Grain 迭代器状态的续训未做。
10. **多机、多读取进程**
    - 为什么：读取并行度变了顺序就变，是常见的隐性差异。
    - 状态：
      - 读取进程数（`grain_worker_count` 0 / 1 / 8）1000 步逐位相同。
      - 多 host 在单进程里模拟（2、4 个 host）：MaxText 先按 host 交错切样本流，全局批行序随 host 数变；按 host 数换序后仍全部对上。
      - 真的多进程 / 多机未跑。
11. **分词器与预处理钉版本**
    - 为什么：分词器差一个版本，token 就不同。
    - 状态：
      - 两边读同一份 `.idx`（sha256 对照相同），MaxText 这条路不在线分词。
      - 分词器文件 sha256 还没写进两边的运行元信息。

### 实测结论

条件：MaxText 上游 `ce608b9`，混元 3 第一阶段数据与 1000 步 golden，全程 CPU。

- 数字见 `models/hunyuan3/alignments/01-v5p-acceptance/results/phase2-native-loader.results.json`
- 复现见 `models/hunyuan3/data/README.md`

要点：

- **配对了就逐位**：`grain_file_type: mmap_npy` 按 GPU 配置设好后，结果与 Megatron 一致。
  - 要设的配置：EOD id、种子、split、`reset_attention_mask: false`、`steps × 全局 batch` = GPU 的样本数。
  - 三个索引与 Megatron 缓存逐字节相同。
  - 1000 步每步 16 个样本，按顺序与 Megatron 样本流相同。
- **唯一的结构性差异在「一步之内分给谁」**：
  - GA > 1 时 MaxText 按 `reshape(gbs//ga, ga)` 交错切 microbatch，Megatron 每个 microbatch 取连续 dp×mbs 个；
  - 多 host 时 MaxText 还按 host 交错切样本流。
  - 不改 MaxText 源码，换序放在 janus（`janus/data_layout.py`）。
- **MaxText 的默认值对不上 Megatron-Bridge 的默认**（PR 文档里的示例也开着 `reset_attention_mask`）。都要显式改：
  - `reset_attention_mask`（默认 true）
  - `data_shuffle_seed`（默认 0）
  - `mmap_eod_id`（默认 0）
  - `mmap_npy_split`（默认不切）
  - 三个反例（默认 reset、不设 split、默认种子）对拍都报了不同，并定位到对的环节。
- **约定差异**：
  - pad id 的处理（只在数据含 pad id 时触发）；
  - `reset_attention_mask` ≠ `reset_position_ids` 的 Megatron 配置，MaxText 表达不了。

## 二、权重：从「经 HF 中转」到「直接互转」

### 现状：经 HF 中转（已落地）

```
Megatron 检查点 ──（Megatron-Bridge 导出）──▶ HF safetensors ──（MaxText 转换工具）──▶ MaxText 检查点
```

第一阶段已验证 HF → Megatron → HF 往返：

- 1781 / 1781 个张量逐位相同、dtype 不变。
- 结果：`models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-ckpt-roundtrip.results.json`

问题：

- **只有模型权重**：HF 格式不带 FP32 主权重和优化器状态。
  - 从 GPU 训到一半的检查点迁到 TPU 续训，优化器状态会丢，只能从头积累动量。
- **多写两遍**：大模型每次转换都要把完整权重落盘两次，耗时、占空间。
- **多一个出错环节**：每经过一次格式就多一次 dtype、转置、拆分出错的机会。

### 目标：直接互转，含完整训练状态（未落地）

```
Megatron torch_dist 检查点（参数 + FP32 主权重 + 优化器状态）  ◀──────▶  MaxText / Orbax 完整训练状态
```

设计要点：

| 要点 | 说明 |
|---|---|
| 映射表驱动 | 用同一份 `models/<模型>/model/name_map.yaml` 描述参数名、拼接（qkv、gate/up）、拆分（专家）、转置；两个方向共用，避免各写一份 |
| 完整训练状态 | 参数、FP32 主权重、两个动量、步数计数一起转；浮点 buffer（如 expert_bias）单独列出 |
| 按模板写目标格式 | 以目标框架自己存出的一份检查点为模板（张量清单、分片、元数据），只替换数值，保证目标框架能直接加载 |
| 内存有界、流式 | 按张量分批读写，不把整个模型读进内存，大模型也能在一台机器上转 |
| 先离线核对映射 | 正式转换前，先证明源检查点的每个张量都在映射表里找到了去处、目标的每个张量都有来源（零未解析） |

### 权重转换要满足的检查

| 检查 | 比什么 / 判据 | 为什么 | 状态 |
|---|---|---|---|
| 全量逐张量逐位 | 每个张量按原始字节比（NaN 的位模式、−0.0 都算），**不抽样** | 抽样比，没抽到的坏了查不出 | 已落地（HF 往返） |
| 数清张量数 | 源、目标、往返后的张量清单一致，零缺失、零多余 | 漏转一个张量，比对工具可能静默跳过 | 已落地（HF 往返） |
| 往返逐位 | A → B → A 逐位相同 | 证明两个方向互逆，但证明不了映射表本身对（见注 1） | 已落地（HF 往返）；直接互转未做 |
| FP32 小张量三方核对 | 同一个张量在 GPU 检查点、中间格式、TPU 检查点三处逐位相同，重点是 expert_bias | 这类张量常被顺手降成 BF16；1 个比特就能改变路由选中的专家 | 已落地：TPU 加载后的 buffer（expert_bias）对 GPU 逐位（[`phase2-l0-state`](../../../models/hunyuan3/alignments/01-v5p-acceptance/results/phase2-l0-state.report.md) 的 `L0.buffers.loaded`），HF 中间格式对 MaxText 检查点逐位、expert_bias 为 FP32（[`phase2-ckpt-convert`](../../../models/hunyuan3/alignments/01-v5p-acceptance/results/phase2-ckpt-convert.report.md)） |
| dtype 清单 | 导出、转换后每个张量的 dtype 符合规则（expert_bias 必须 FP32） | 转换工具可能一次性把一批张量降精度 | 已落地（GPU 侧 `L0.dtype.manifest`、HF 往返；TPU 侧 [`phase2-l0-state`](../../../models/hunyuan3/alignments/01-v5p-acceptance/results/phase2-l0-state.report.md) 的 `L0.dtype.manifest`、`L0.dtype.state`） |
| 新参数 / 改名参数确实被加载 | 加载后的参数对源检查点逐张量比；加载日志里没有 missing / unexpected key | 名字没对上的参数会被随机初始化，第 1 步 loss 离谱时先查这条 | 部分（见注 2） |
| 优化器状态命中计数 | 动量、主权重等每类路径实际匹配到的张量数，与参数数一致 | 路径写错时会「零命中」却不报错（见注 3） | 未落地（随直接互转一起做） |
| 恢复后特殊值保留 | 存档前埋 −0.0、次正规数，存 / 恢复后逐位不变 | 恢复路径里若有浮点合并，会把它们抹掉 | 已落地（GPU 侧，[`phase1-ckpt-special`](../../../models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-ckpt-special.report.md) 3 条全过，详见 [0004](0004-l1-gpu-kernels.md) 第五节）；TPU 侧未做 |
| 恢复训练对得上 | 转换后续训，与原框架连续训练的 loss / 统计量在门槛内 | 前几档只证明「写出来的字节对」，这一档证明「能接着训」 | 部分：GPU 自身存档续训已逐位（`phase1-resume`）；跨框架未做 |

- 注 1：**证明不了映射表本身对**——两边错得互逆时往返照样对上。映射表靠 L2 逐层比。
- 注 2：GPU 侧加载后参数已比；对 HF 原件逐张量比、TPU 侧未做。
- 注 3：零命中时续训后靠优化器自己慢慢恢复，看起来还挺平滑。

**验收分四档，报告里分开写，不能混用**：写入成功 → 抽样通过 → 全量逐字节相同 → 恢复训练对得上。
