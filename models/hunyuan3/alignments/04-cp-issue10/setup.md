# 04 · 上下文并行的 loss 偏移 · 设置

## 阶梯实验（README 正文）

| 项 | 取值 |
|---|---|
| 模型 | 16 层随机初始化混元 3：`hunyuan3-smoke` 只改 6 项尺寸（层数 16、隐藏维 1024、稠密前馈 4096、查询头 / 键值头 16 / 4、专家前馈 256；192 专家选 8 + 1 共享专家，词表 120832，同 原问题 10-29），2.624e9 参数，关 MTP。配置 [`ladder/hy3-smoke16.yml`](ladder/hy3-smoke16.yml)，名字映射 [`ladder/name_map_smoke16.yaml`](ladder/name_map_smoke16.yaml) |
| 初始权重 | 所有臂同一份：[`tools/make_ladder_inputs.py hf`](tools/make_ladder_inputs.py)（种子 0；矩阵 N(0, 0.006²) 舍成 BF16，RMSNorm 权重 1，路由偏置 0）→ [`convert_hf_to_maxtext.py`](../../model/convert_hf_to_maxtext.py)（每个元素恰好写一次）。CPU FP64 参照读同一份 HF 权重 |
| 数据 | 所有臂同一份、同一顺序：[`tools/make_ladder_inputs.py data`](tools/make_ladder_inputs.py)，FineWeb-Edu sample-10BT 第一个 parquet 分片按行号顺序取 97000 篇，混元 3 分词器切词、文档间插 EOS、首尾相接、每 4097 个词元切一条；3000 步 × 8 条 × 4096，写成 golden 同格式的回放目录（manifest sha256 `5152129a9210…`），[`train_hy3.py`](../../frameworks/maxtext/train_hy3.py) 原样喂（[`replay.py check`](../../frameworks/maxtext/replay.py) 通过） |
| 学习率 | 恒定（`lr_schedule_type=wsd`，不预热、不衰减）。长跑主组 1e-4（运行名 `lbd-*`）；另一组 3e-4（`lad-*`）第 55–56 步有一次 loss 尖峰（README 第 5 节），长跑只做了 L0–L2，L3 只有 7 个臂的第 1 步可用。第 1 步的梯度与学习率无关，两组的第 1 步结果一起用 |
| 其余优化器 | `base.yml` 默认：AdamW β 0.9 / 0.95、ε 1e-8、权重衰减 0.1、裁剪 1.0 |
| 精度 | L0 = `base.yml` 默认（BF16 计算、FP32 主权重与梯度、`matmul_precision=default`）+ tokamax splash（`sa_use_base2_exp` 默认开）；不加 `XLA_FLAGS`、不加 `LIBTPU_INIT_ARGS`（excess precision 默认开）。各级在此之上加的见下表 |
| 代码 | `maxtext` hunyuan@`5d1b0e95`（MaxText 上的一处修改 合入后：`embedding_shard_over_data` 等三个开关）；镜像 `tpu_pre_training` 0.2.5（jax 0.11.1、libtpu 0.0.46）+ tokamax 0.0.16 |
| 硬件 | v5p-8（2×2×2，2 台主机，8 个设备）；每步 0.43–0.47 s，3000 步约 23 分钟 |
| 每次运行 | `train_hy3.py --steps 3000 --dump-steps 1 --full --no-hooks --no-param-sha`：第 1 步只读状态（零扰动）抓一阶动量，`JANUS_FULL_KINDS=exp_avg` 只写这一类；FSDP 参照只跑 2 步 |

臂（全部 8 个设备，`per_device_batch_size=1`、不做梯度累积，每步全局 8 条）：

| 臂 | 并行 | 额外参数 |
|---|---|---|
| `dp8`（基线） | 纯数据并行 8，每卡整份权重 | — |
| `fsdp8`（参照，只跑第 1 步） | FSDP 8 | `ici_fsdp_parallelism=8 ici_data_parallelism=1` |
| `ag2` / `ag4` | 数据并行 4 × CP2 / 数据并行 2 × CP4，All-Gather | `ici_context_parallelism=2|4 context_parallel_strategy=all_gather` |
| `rg2` / `rg4` | 同上，Ring | `… context_parallel_strategy=ring` |
| `ul2` / `ul4` | 同上，Ulysses（MaxText 要求关负载均衡） | `… context_parallel_strategy=ulysses context_parallel_load_balance=False` |
| `usp4` | 数据并行 2 × Ring 2 × Ulysses 2 | `ici_context_parallelism=2 ici_context_usp_ulysses_parallelism=2 context_parallel_strategy=usp` |
| 尺子 `v64` / `v56` / `v40` / `v24` | 同基线，只改片上内存上限 | `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536|57344|40960|24576` |

级（每级在上一级上只加一项；L0′ 是 L0 的变体，不往下累加）：

| 级 | 加了什么 |
|---|---|
| L0 | 原始默认 |
| L0′ | L0 + `XLA_FLAGS=--xla_allow_excess_precision=false` |
| L1 | L0 + `sa_use_base2_exp=False` |
| L2 | L1 + `embedding_shard_over_data=True` |
| L3 | L2 + `qkv_proj_wgrad_fp32=True` |
| L4 | L3 + `routed_gate_bf16_rounding=True rmsnorm_single_rounding=True mlp_activation_round_per_op=True dense_fp32_accumulation=True`（混元 3 对齐配置里其余四个开关） |

离线编译与上一级同一程序的臂不重跑（[`tools/hlo_audit.py`](tools/hlo_audit.py) 的程序指纹相同，上卡逐位相同）：L1 的 `rg2`、`rg4`、`usp4`（Ring、USP 的内核本来就不走 base2）用 L0 的运行，L2 的 `fsdp8`（FSDP 本来就切词嵌入表）用 L1 的。

### 命令

```bash
# 1. 输入（CPU）
python tools/make_ladder_inputs.py hf --name-map ladder/name_map_smoke16.yaml --config-from <混元 3 HF config.json> --out <HF 目录>
python tools/make_ladder_inputs.py data --parquet <fineweb-edu 第一个分片> --tokenizer <混元 3 分词器目录> --steps 3000 --ranks 8 --out <回放目录>
export HY3_YML=models/hunyuan3/alignments/04-cp-issue10/ladder/hy3-smoke16.yml \
       HY3_NAME_MAP=models/hunyuan3/alignments/04-cp-issue10/ladder/name_map_smoke16.yaml \
       HY3_FORCED=base_num_decoder_layers=16,base_emb_dim=1024,base_mlp_dim=4096,base_num_query_heads=16,base_num_kv_heads=4,base_moe_mlp_dim=256
JAX_PLATFORMS=cpu python ../../model/convert_hf_to_maxtext.py --hf <HF 目录> --out <检查点>     # 在 MaxText 镜像里
# 2. 离线编译与审计（CPU，每份约 15 s）：aot_hlo.py v5p-16 2 <目录> <级参数> <臂参数>，然后
python tools/hlo_audit.py <aot 根目录> --out results/ladder/hlo-audit.json
# 3. 上卡（每个臂一条；POD_ENV 把上面三个变量带进 Pod）
POD_ENV="HY3_YML=$HY3_YML HY3_NAME_MAP=$HY3_NAME_MAP HY3_FORCED=$HY3_FORCED JANUS_FULL_KINDS=exp_avg" \
GOLDEN=<回放目录> CKPT=<检查点>/0/items HF_MANIFEST=<HF 目录>/tensor_sha256.json \
PIP_WHEEL=gs://<POC 桶>/janus/pkg/tokamax-0.0.16-py3-none-any.whl \
bash ../../frameworks/maxtext/run.sh lbd-L2-ag2 3000 --dump-steps 1 --full --no-hooks --no-param-sha \
  learning_rate=1e-4 sa_use_base2_exp=False embedding_shard_over_data=True ici_context_parallelism=2 context_parallel_strategy=all_gather
# 4. FP64 参照（CPU，每行约 4 分钟，8 行分 8 个进程再合并）
python ../../analysis/fp64_step.py --hf <HF 目录> --golden-data <回放目录> --names-from <任一运行目录> \
  --name-map ladder/name_map_smoke16.yaml --micro 0:1 --out <fp64>/m0      # … m7，然后 --merge <fp64>/m{0..7} --out <fp64>/merged
# 5. 汇总
python tools/ladder_analyze.py --runs <运行根> --fp64 <fp64>/merged --out-dir results/ladder --series-dir results/series/ladder
```

## 阶梯实验之前的运行（[`history.md`](history.md)）

### 两类运行

| | 第 1 步对 FP64（第 2、3 节） | FineWeb 长跑（第 1、5、6、7 节） |
|---|---|---|
| 入口 | [`train_hy3.py`](../../frameworks/maxtext/train_hy3.py)，经 [`run.sh`](../../frameworks/maxtext/run.sh) 提交 | MaxText 自带的训练入口，作业由 [`tools/mkjob.py`](tools/mkjob.py) 生成 |
| 模型 | 真实权重截 4 层，`hy3-4layers.yml` 现行配置（`dense_fp32_accumulation` 与三个舍入开关打开），`mtp_num_layers=0` | 条件 A：随机初始化，`hunyuan3-smoke` 覆盖成 16 层；条件 B / E1：同左真实权重截 4 层，`hy3-4layers.yml` sha256 `305fa4b55099…` |
| 数据 | 回放 GPU golden `g1k-a` 第 1 步的全局 batch（16 条 × 4096） | FineWeb-Edu sample-10BT 第一个 parquet 分片（7.26e5 篇），HF 分词器在线切词，`packing=True`，每步 8 条 × 4096 |
| 代码 | `maxtext` hunyuan@`8ac1d175`；q/k/v 权重梯度留 FP32 的臂用 `c9f5ccdf`（`qkv_proj_wgrad_fp32=True`），最早的诊断臂用 [`tools/maxtext-qkv-wgrad-fp32-diag.patch`](tools/maxtext-qkv-wgrad-fp32-diag.patch) | 条件 A：hunyuan@`e2c18718`（同 10-29）；B / E1：hunyuan@`8ac1d175`；`E1USP4sw` 用 `c9f5ccdf` |
| 镜像 | MaxText 官方 `tpu_pre_training` 0.2.5（jax 0.11.1、libtpu 0.0.46），另装 tokamax 0.0.16（这两版 MaxText 要它；`run.sh` 的 `PIP_WHEEL`） | 同左 |
| 硬件 | v5p-8（2×2×2，2 台主机） | v5p-8；H5 续训用 v5p 2×2×4（16 片） |
| 基线并行 | FSDP 8 | A：纯 DP8；B / E1：FSDP 8（真实权重的 FP32 主权重 + Adam 约 150 GB，单片放不下完整副本） |
| CP 臂并行 | FSDP 4 × CP2、FSDP 2 × CP4、FSDP 2 × Ring 2 × Ulysses 2（`ici_fsdp_parallelism` 留 −1 自动补） | A：DP4 × CP2；B / E1：同左 |
| 精度开关 | 现行 yml；`use_tokamax_splash=True`；**`sa_use_base2_exp=False` 统一关**（「base2 开」那一臂除外）；不加 `XLA_FLAGS`（省舍入照默认开） | A / B / E1 都带 `--xla_tpu_spmd_rng_bit_generator_unsafe=false`、`--xla_allow_excess_precision=false`（同 10-29；`tools/mkjob.py` 写死）；B / E1 另：`use_tokamax_splash=True`，base2 默认开 |
| 学习率 | 恒定 1e-5（只跑 2 步） | A：峰值 3e-4、预热 300、余弦 3000 步；B：恒定 1e-5；E1：`lr_schedule_type=cosine learning_rate=1e-4 warmup_steps_fraction=0.1 learning_rate_final_fraction=0.0`，3000 步 |

### `train_hy3.py` 回放支持上下文并行（本实践加的）

- 回放原来要求「每个 batch 分片的行数 = 每 rank 的 microbatch 数」。开 CP 后 batch 维只按数据 / FSDP 轴切，这条不再成立；CP > 1 时放宽。MaxText 的梯度累积按全局 `total_weights` 归一化，行怎么分进微批不改数学。
- 负载均衡的重排在 MaxText 的数据加载器里做，回放绕过了加载器；回放里照同一个函数（`maxtext_utils.get_reorder_callable`）补上。

### 第 1 步对 FP64 的命令

```bash
export BUCKET=gs://<POC 桶> CONTEXT=<kubectl 上下文> NODEPOOL=<v5p-8 节点池> PLATFORM=v5p
export PIP_WHEEL=gs://<POC 桶>/janus/pkg/tokamax-0.0.16-py3-none-any.whl
source <(bash ../../frameworks/maxtext/stage.sh <maxtext 克隆> 8ac1d175)
O="--dump-steps 1 --full --no-hooks --no-param-sha --full-only decoder"
S="use_tokamax_splash=True sa_use_base2_exp=False context=remat query_proj=remat key_proj=remat value_proj=remat out_proj=remat \
   mlpwi_0=remat mlpwi_1=remat mlpwo=remat moe_mlpwi_0=remat moe_mlpwi_1=remat moe_mlpwo=remat moe_routing_maps=device"
bash ../../frameworks/maxtext/run.sh i10-st-b2 2 $O $S                                                  # 基线
bash ../../frameworks/maxtext/run.sh i10-st-rg2 2 $O $S ici_context_parallelism=2 context_parallel_strategy=ring
bash ../../frameworks/maxtext/run.sh i10-st-usp4 2 $O $S ici_context_parallelism=2 ici_context_usp_ulysses_parallelism=2 context_parallel_strategy=usp
LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=40960 bash ../../frameworks/maxtext/run.sh i10-st-bv40 2 $O $S   # 只改编译
XLA_DUMP=1 bash ../../frameworks/maxtext/run.sh i10-st-usp4x 2 $O $S …                                  # 要编译后的 HLO 时
python ../../analysis/cp_step1_fp64.py --fp64 <fp64_step 输出> --base <运行根>/i10-st-b2 --arm rg2=<运行根>/i10-st-rg2 … --out results/step1-cp-vs-fp64.json
python tools/qkv_split.py --fp64 <fp64_step 输出> --run base=<运行根>/i10-st-b2 --run usp4=<运行根>/i10-st-usp4 … --out results/step1-qkv-split.json
```

- 每臂完整的参数见 [`runs.yaml`](runs.yaml)。FP64 参照是 01 实践用过的同一份（`fp64_step.py`，golden 第 1 步全局 batch）。
- q/k/v 权重梯度留 FP32：`stage.sh` 打 `c9f5ccdf`（或之后的 `hunyuan`）的包，命令行加 `qkv_proj_wgrad_fp32=True`，例：

  ```bash
  source <(bash ../../frameworks/maxtext/stage.sh <maxtext 克隆> c9f5ccdf)
  bash ../../frameworks/maxtext/run.sh i10-st-usp4-sw 2 $O $S ici_context_parallelism=2 ici_context_usp_ulysses_parallelism=2 context_parallel_strategy=usp qkv_proj_wgrad_fp32=True
  ```

  最早的诊断补丁只在运行名含 `qkvf32` 时让 q/k/v 投影走 `kernel_grad_fp32`，要单独打包；开关版与它逐位相同，补丁留作历史。
- 离线编译核对开关（CPU，不占卡）：在上面的镜像里跑 [`aot_hlo.py`](../../frameworks/maxtext/aot_hlo.py)（`v5p-16`、2 步、上面的 `$S` 加 CP 参数），再用 [`hlo_screen.py compare`](../../../hunyuan4/alignments/04-tpu-self/hlo_screen.py) 比两份优化后 HLO；结果在 [`results/usp-switch-hlo.json`](results/usp-switch-hlo.json)。

### 长跑的命令

```bash
BUCKET=gs://<POC 桶> MODE=rw4 python3 tools/mkjob.py i10-e1-rg <节点池> \
  "E1RG2|lr_schedule_type=cosine learning_rate=1e-4 warmup_steps_fraction=0.1 learning_rate_final_fraction=0.0 learning_rate_schedule_steps=3000 steps=3000 ici_context_parallelism=2 context_parallel_strategy=ring|" > job.yaml
kubectl apply -f job.yaml       # 跑完从两个 Pod 的 /tmp/out 取回 <运行名>.metrics.jsonl
python ../../analysis/cp_tpu_ruler.py --ref base=results/series/E1B.json --ref vmem64=results/series/E1N1.json … --test ring=results/series/E1RG2.json --out results/e1-tpu-ruler.json
python tools/shape_stats.py --base results/series/E1B.json --arm ring=results/series/E1RG2.json … --out results/shape-E1.json
```

- `E1USP4sw`（USP + `qkv_proj_wgrad_fp32=True`）：`MT_PKG`、`MT_SHA` 换成 `stage.sh` 打的 `c9f5ccdf` 包，运行参数同 `E1USP4` 再加 `qkv_proj_wgrad_fp32=True`。
- 逐步 loss 存成 `results/series/<运行名>.json`（`first_step`、`loss`、`lr`、`raw_grad_norm`）。
- 节点池是抢占式的，被收回的运行从头重跑，不拼接。

### 显存

- 真实权重截 4 层，FSDP 8：编译期估计每设备 51.2 GB（参数 17.3 GB）；CP2 各方案 51.4–51.9 GB；单步 1.46–1.48 s。
- 随机初始化 16 层，纯 DP8：44.6 GB（参数 29.3 GB）；CP2 约 24 GB；单步 0.45–0.51 s。
- 16 片存档续跑（H5）：31.7 GB / 设备，续跑段与不中断的运行逐位相同。
- 想把权重从 CP 轴上拿掉（只按 FSDP 2 切）的 USP 臂放不下：编译要 131.2 GB 临时量。
