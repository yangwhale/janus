[janus](../../../../README.md) › [混元 4](../../README.md) ｜ TPU 侧操作手册

# 混元 4 · TPU 侧（MaxText）

流程（转换 → L0 逐位 → 回放 GPU 的数据 → 抓数 → janus 比对），
这一页只写混元 4 不一样的地方。比较协议见 [`../../PROTOCOL.md`](../../PROTOCOL.md)；T1 的配置是 [`hy4-mini-t1.yml`](hy4-mini-t1.yml)（`HY4_YML` / `YML=` 传入）。

- 平台：v5p 8 片（2x2x2，2 台主机，8 个 JAX 设备），对 GPU B200 × 8；设备 d 第 m 个累积步吃 GPU rank d 的第 m 个 microbatch。
- 代码：`maxtext` 的混元 4 集成分支（`stage.sh` 用环境变量 `MAXTEXT_BRANCH` 指定分支名；含 `0b7dead5` 及之后的全部改动）。
  具体提交由 [`stage.sh`](stage.sh) 记进每个运行的 `meta.json` 的 `code`。
- 镜像：与混元 3 相同的 MaxText `tpu_pre_training` 0.2.5（[`run.sh`](run.sh) 里的 `IMAGE`，按 digest 钉死）。TPU 上实跑的库版本以镜像为准：jax / jaxlib 0.11.1、libtpu 0.0.46（构建 `cl/963683785`）、flax 0.12.9、optax 0.2.8、orbax-checkpoint 0.12.4，每个运行记在 `meta.json` 的 `versions`；tokamax 0.0.14（`meta.json` 不记，读镜像层里的 `tokamax-0.0.14.dist-info` 核过）。逐项与出处见 [mini 的「软件版本」](../../alignments/01-mini/setup.md#软件版本)。
- CPU 验证环境：jax / jaxlib 0.11.1、flax 0.12.10、optax 0.2.8、orbax-checkpoint 0.12.7（分支 `tpu-requirements.txt` 的下限，去掉 libtpu）；
  编译期显存分析另装 libtpu 0.0.46（只用编译器）。

## 这些开关在哪个分支

本仓库混元 4 的结论都在 `maxtext` 的混元 4 集成分支上测得。读者文档「怎么修」里的开关只在这条分支上，上游 `AI-Hypercomputer/maxtext` 没有（2026-11-11 核：上游 main `e66c28c2` 的 `src/maxtext/configs/base.yml` 里没有这些键；按开关名搜上游的 PR，没有搜到）。下表的 PR 都合进了集成分支，`maxtext` 自己的 main 也没有它们。

先知道三件事：

- **上游有另一份混元 4 实现。** 上游 2026-10-29 起有自己的 `decoder_block: hy4`（`models/hy4.py`）；集成分支的混元 4 建在 DeepSeek 解码器上（`decoder_block: deepseek`）。两份不是同一份代码，本仓库的结论没在上游实现上测过。
- **分叉点。** 集成分支与上游 main 的公共祖先是上游 `493d5928`（2026-10-01）。移植时以它为底看差异。
- **上游没有的键会报错，不会静默。** 上游 `configs/pyconfig.py` 对配置里不认识的键抛 `ValueError`。所以对齐配置 [`hy4-mini-t1.yml`](hy4-mini-t1.yml) 原样拿到上游加载不起来。上游**有**这个键、只是代码路径没接上时，才会静默不生效（下表 `float32_logits` 那行）。

| 开关或修复 | 做什么 | 集成分支的提交（PR） | 上游 | 用上游要怎么办 |
|---|---|---|---|---|
| `embedding_grad_fp32`（台账 M12） | 词嵌入梯度在 FP32 里累加 | `23f71c66`（从 `hunyuan` 分支移植） | 没有这个键：写了报错 | 移植 `layers/embeddings.py` 的取行写法，`configs/base.yml`、`configs/types.py` 加键。纯数据并行 + 梯度累积下还要带上下一行的修复 |
| `d5c9b113` 的修复（台账 M55） | 纯数据并行 + 梯度累积时，`embedding_grad_fp32` 丢掉前面微批的词嵌入梯度（r 611，[impacts/01](../../alignments/04-tpu-self/impacts/01-data-parallel.md)） | `d5c9b113` | 不涉及：上游没有 `embedding_grad_fp32` | 移植上一行时一起移植 `utils/gradient_accumulation.py` 的改动。集成分支上也要 `d5c9b113` 之后的提交：TPU 自比 golden 与 mini 现行比较用的 `b87a55aa` 在它之前（这两份是 FSDP，不走这条路径） |
| `embedding_grad_fp32_local_bf16_comm`（台账 M56） | 词嵌入梯度本地 FP32 累加、BF16 跨卡求和 | 合并提交 `f70d74cb` | 没有这个键：写了报错 | 移植 `layers/embeddings.py`、`utils/gradient_accumulation.py` 与两处配置 |
| `embedding_shard_over_data`（台账 M54） | 只把词嵌入表沿 `data` 轴切开，纯数据并行下按 FSDP 的方式更新词嵌入 | `9099c8ce` | 没有这个键：写了报错 | 移植 `layers/embeddings.py` 与两处配置（`types.py` 里追加逻辑轴规则）。上游已有的 `shard_optimizer_over_data` 是另一件事，和它互斥 |
| `rmsnorm_single_rounding`（台账 M9） | RMSNorm 在 FP32 里算完只舍一次 | `2f745f37`（`hunyuan` 分支的 MaxText 上的一处修改），接到混元 4 路径是另一次修改 | 没有这个键：写了报错 | 移植 `layers/normalizations.py` 与两处配置，再把开关接到用到 RMSNorm 的模型文件（一次 MaxText 修改接了 `attentions.py`、`nnx_decoders.py`、`qwen3.py`，一次 MaxText 修改接了 `deepseek.py`、`attention_mla.py`、`multi_token_prediction.py`） |
| MLA 接上 `float32_qk_product`、`float32_logits` | 普通注意力（`attention=dot_product`）的打分与 softmax 走 FP32；上下文并行的修法要它（[impacts/06](../../alignments/04-tpu-self/impacts/06-context-parallel.md)） | `56492817` | 两个键都有。上游 `models/deepseek.py` 建 MLA 时没把它们传下去，和集成分支修之前一样：**写了静默不生效**（TPU 自比 G3c：打开后训练步程序不变）。上游自己的 `models/hy4.py` 传了 | 用 `decoder_block: deepseek` 时移植 `models/deepseek.py` 的三行；用上游的 `hy4` 实现不用改，但那份实现本仓库没测过 |
| `hc_head_force_bf16_rounding`（台账 M52） | hc_head 收成一路后先舍成 BF16 再进末层 RMSNorm（诊断开关） | `0e33d77c`（与 M53 的五个诊断开关同一提交） | 没有这个键：写了报错 | 只在复现混元 4 主 loss 排查时需要 |
| 其余对齐开关 | [`CHANGES-LEDGER.md`](../../CHANGES-LEDGER.md) MaxText 表里有开关名的各行 | 见台账「提交」列 | 都没有，只有 `sa_use_base2_exp`、`logits_dot_in_fp32`、`matmul_precision`、`shard_optimizer_over_data`、`float32_logits` 这几个已有的键上游也有 | 要逐个移植；台账「提交」列是起点 |

## 文件

| 文件 | 作用 |
|---|---|
| [`../../model/hy4_map.py`](../../model/hy4_map.py) | 分阶段截层、名字映射（复用分支的 Hy4 映射，补上 MoE 不足一组的情形）、反向变换、HF 的 dtype 表、GPU golden 的参数名与分片写法 |
| [`../../model/convert_hf_to_maxtext.py`](../../model/convert_hf_to_maxtext.py) | HF → MaxText 参数检查点，`--stage A/B/C/D/4L` |
| [`../../model/verify_ckpt.py`](../../model/verify_ckpt.py) | L0：`names`（名字、形状、dtype、参数量）、`hf-manifest`、`maxtext`（反向逐位）、`rope`（去交错）、`gpu-manifest`（对 GPU 初始参数清单） |
| [`../../model/t2_l0.sh`](../../model/t2_l0.sh) | T2 的 L0 一条龙：hf-manifest、反向取回、RoPE、janus 目录、GPU 载入自检汇总（阶段 B / D1） |
| [`../../model/make_tiny_hf.py`](../../model/make_tiny_hf.py) | 小维度的假 HF 目录（CPU 冒烟用）；也是 HF 形状表 `hf_shapes` 的出处 |
| [`hy4-mini-t1.yml`](hy4-mini-t1.yml) | T1 的 MaxText 配置；逐项对照见 [`../../alignments/01-mini/config-alignment.md`](../../alignments/01-mini/config-alignment.md) |
| [`hy4-t2-b.yml`](hy4-t2-b.yml) / [`hy4-t2-d1.yml`](hy4-t2-d1.yml) | T2（真实权重截层 B、D1）的配置：`hy4-mini-t1.yml` 现在的全部数值与优化器开关（含 M31、M33；`qk_clip_max_logit_exclude_sink` 也开着：阶段 B 上打开时的 NaN 由 M35 修好，2026-11-08 起重新打开，real-weight 的对齐细节 D20；`tb2-a`、`td2-a` 跑的是之前的 t1v 等价设置），换 `model_name: hy4-preview`（单测逐键核对）；见[实践 02](../../alignments/02-real-weight/README.md) |
| [`hy4-4layers.yml`](hy4-4layers.yml) | 不设 `HY4_YML` 时的默认配置：真实权重截层的转换、L0 核对与单算子回放用它 |
| [`mt_common.py`](mt_common.py) | 按阶段建配置；和模型 yml 冲突的键自动补到命令行（`forced_overrides`），并核对 yml 每个键实际生效（`effective_mismatch`） |
| [`replay.py`](replay.py) | 数据回放；T1 / T2 的 golden 开 `--eod-mask on` 与 `--save-batches`，loss_mask 一起存下（台账 G9）；更早只存 tokens / labels 的运行，缺的 loss_mask 补 1、position_ids 补 0…S−1 |
| [`train_hy4.py`](train_hy4.py) | 训练入口：回放 golden 每步的 batch，写 metrics、输入指纹、参数清单、全量状态抓数 |
| [`hf_view.py`](hf_view.py) | 训练状态 → HF 布局与 GPU 名字的张量（参数、两个 Adam 动量） |
| [`run.sh`](run.sh) / [`stage.sh`](stage.sh) / [`profile.sh`](profile.sh) / [`site-profile.example.sh`](site-profile.example.sh) | 提交 JobSet、打包代码、站点配置钩子（多一个 `STAGE`） |
| [`make_fake_golden.py`](make_fake_golden.py) | 合成 golden（随机 token），只给冒烟与显存实测，不和 GPU 比 |
| [`cpu_smoke.py`](cpu_smoke.py) | CPU 冒烟：hy4-tiny、8 个假设备，四个阶段整条链路 |
| [`muon_partition.py`](muon_partition.py) | Muon 分区表：每个参数走 Muon 还是 Adam、Newton-Schulz 作用在哪个二维矩阵上（用 MaxText 自己的 `muon_utils` 判） |
| [`qk_clip_scan_check.py`](qk_clip_scan_check.py) | QK-Clip 在层扫描下是否每层各自一个缩放：前向一次，调训练里同一个 `apply_qk_clip_nnx`，逐层逐头核 |
| [`aot_memory.py`](aot_memory.py) | 编译期显存分析（CPU 上为 v5p 2x2x2 编译训练步），只作初筛 |
| [`probe_forward.py`](probe_forward.py) | 接线核对：整段前向，模块边界逐个比 GPU 第 1 步的抓数（诊断，不当精度结论） |
| [`op_replay.py`](op_replay.py) | L1 逐算子隔离回放：每个算子吃 GPU 的输入，对 FP64 算 R；`SCRIPT=op_replay.py bash run.sh …` 提交 |
| [`op_replay_mini.py`](op_replay_mini.py) | 同上，阶段 mini（hy4-mini 7 层 + MTP，GPU 分母是 Megatron-Core + TE 的 `g7-l1-a`）；TPU 上抓输出，`--compare` 在 CPU 上对 FP64 算 R |
| [`elementwise_ulp.py`](elementwise_ulp.py) | 逐元素函数对 FP64 的 ULP（FP32 / BF16、合成与真实输入、LSE 拆解）；`SCRIPT=elementwise_ulp.py` 用 run.sh 提交 |
| [`optim_replay.py`](optim_replay.py) | 优化器回放：GPU 的梯度与主权重喂 MaxText 优化器（T1 配置），逐张量对 FP64 参照算 R；起先只回放第 1 步，台账 J14 起连续 8 步（`tw-optim8`）；`SCRIPT=optim_replay.py` 用 run.sh 提交 |
| [`layer_replay.py`](layer_replay.py) | L2 逐层隔离回放：GPU 第 1 步每层输入进 TPU 单层程序，比输出与输入梯度；判定用的 R 对 `frameworks/megatron/layer_ref.py` 的 CPU FP64 单层重算；`SCRIPT=layer_replay.py` 用 run.sh 提交 |

## 步骤

| 步 | 命令（在 `models/hunyuan4/model/` 下，CPU） | 产物 |
|---|---|---|
| 1. 核对 HF 目录 | `python verify_ckpt.py names --hf <HF 目录> --out <结果>` | 135 个张量名、形状、dtype 逐个对上；参数量 42,070,313,621 |
| 2. 转换 | `JAX_PLATFORMS=cpu python convert_hf_to_maxtext.py --hf <HF 目录> --stage A --out <输出>` | `<输出>/0/items`（FP32） |
| 3. L0 本侧 | `verify_ckpt.py hf-manifest` / `maxtext` / `rope`（见文件开头），再 `python -m janus run …/catalogs/l0-ckpt-convert-a.yaml` | 逐张量逐位、dtype 清单、去交错 |
| 4. L0 跨侧 | `verify_ckpt.py gpu-manifest …`，再 `python -m janus run …/catalogs/l0-init-vs-gpu-a.yaml` | TPU 初始权重对 GPU golden 初始参数清单逐位 |
| 5. 上传 | 检查点传到 Pod 读的桶 `…/janus/hunyuan4/maxtext/hy4-a-maxtext/`，HF 目录的 `config.json`、`tensor_sha256.json` 传到 `…/hf/hy4-4l/` | |
| 6. 训练（T1） | `source <(bash stage.sh <maxtext 克隆>)`，`YML=hy4-mini-t1.yml STAGE=mini GOLDEN=<GPU 运行目录> bash run.sh <运行名> 300` | `…/janus/hunyuan4/tpu/<运行名>/` |

### 和混元 3 不一样的地方

- **转换复用分支的映射**：分支的 `HY4_MAXTEXT_TO_HF_PARAM_MAPPING` 要求 MoE 层至少一整组（`indexer_share_interval` = 4），
  对齐用的截层（1–3 层 MoE）都不足一组；模型本身把它们建成一个「余数块」`moe_layers_remainder`。`hy4_map.mapping_and_hooks`
  把 HF 配置按 Hy4 的规律补齐到一组，取分支的映射与变换，再改名到余数块、去掉补出来的层。键名和变换函数仍全部来自分支。
- **RoPE 去交错在转换时做**（`checkpoint_rope_interleave: true`）：MLA 的 `wq_b` / `wkv_a` 与 indexer 的 `wq_b` / `wk`、`k_norm`
  的 rope 段从交错 (2i, 2i+1) 改成 [偶, 奇]，TPU 跑拼接式 RoPE；GPU 在运行时按交错式算。两者是同一个函数，L0 的 `rope`
  子命令逐位核置换、逆置换；负对照：不去交错转换的检查点必须被判不一致。
- **参数类张量写 GPU golden 的名字**（AdamW 时期的写法，那时 GPU golden 是 HF 实现；T1 / T2 的 golden 是 Megatron 程序，名字照同一套 `hf.…`）：参数名是 transformers 加载后的 `hf.…`（`attn_hc` / `ffn_hc` / `hc_head.hc_*` /
  `self_attn.gate_proj` / `self_attn.sinks` 有改名，`hy4_map.gpu_name`）。参数清单记 FP32 主权重，并按 GPU 的写法分 rank：
  GPU 每个 rank 记本地 FSDP2 分片（沿第 0 维、每片 ceil(N/8) 行）的 sha256，形状一栏写完整形状；没被切分的参数
  （iHC、hc_head、sink、indexer 的 weights_proj / k_norm）记完整张量（`hy4_map.gpu_rank_pieces`）。
- **分组参数平方和**：与 GPU 同名的组（attn、attn_gate、dense_mlp、embedding、final_norm、hc_head、ihc、indexer、norms、output、sinks），
  记第 s 步更新之前的参数（GPU 的第 1 步值等于初始权重逐组复算的值）。
- **梯度累积 2 时不加钩子**：MaxText 的加钩子训练步只支持不累积，所以图内激活与梯度不抓；
  第 1 步逐层比要另做逐层回放（`layer_replay.py`）。

### 显存

编译期显存分析（`aot_memory.py`，v5p 2x2x2）只作初筛。装不装得下以实机装载训练步成功为准。`meta.json` 的 `memory.peak_bytes_in_use` 不能单独当判据：它不含训练步程序预留的临时区（D3 在 v5p 16 片上预留 72.80 GiB 并开跑，峰值只记 3.03e10 字节；推导见 [06 设置「显存：要几片」](../../alignments/06-rw3/setup.md#显存要几片)）。
hy4-mini 的结果见 [`hy4-mini-aot-memory.json`](../../alignments/01-mini/results/hy4-mini-aot-memory.json)。

- 重算策略：本想用 `save_everything`（反向不重算任何前向算子），但它在 maxtext `0b7dead5` 上对 Hy4 一 trace 就报错
  （YaRN RoPE 的复数中间量被 `jax.checkpoint` 加了 `reduce_precision`，后者不接受 complex64；已修）。
  T1 在 `t1t-*` 之前用 `full`，`t1t-*` 起用 `custom`（除层输入外全部 remat，等价于整层重算，路由索引存下不重选；台账 M26，`PROTOCOL.md` 第 1.2 节「重算」行）；`custom` 并把全部命名张量存 HBM 是 TPU 扰动 `-4` 的设置。

## 已知限制

- L2 逐层回放：`layer_replay.py` 现在吃 GPU `g7-l1-a` 的逐层输入，含 MTP 层（台账 J19；原来用 `g2s-a`，没有 MTP 的输入）；之前两次 TPU 运行在多主机上失败（J17、J22），`tw-l2-f` 跑通、全过（`results/t1-l2-layer-tpu.json`）。
