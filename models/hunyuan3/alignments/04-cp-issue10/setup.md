# 04 · 上下文并行的 loss 偏移 · 设置

## 两类运行

| | 第 1 步对 FP64（第 2、3 节） | FineWeb 长跑（第 1、5、6、7 节） |
|---|---|---|
| 入口 | [`train_hy3.py`](../../frameworks/maxtext/train_hy3.py)，经 [`run.sh`](../../frameworks/maxtext/run.sh) 提交 | MaxText 自带的训练入口，作业由 [`tools/mkjob.py`](tools/mkjob.py) 生成 |
| 模型 | 真实权重截 4 层，`hy3-4layers.yml` 现行配置（`dense_fp32_accumulation` 与三个舍入开关打开），`mtp_num_layers=0` | 条件 A：随机初始化，`hunyuan3-smoke` 覆盖成 16 层；条件 B / E1：同左真实权重截 4 层，`hy3-4layers.yml` sha256 `305fa4b55099…` |
| 数据 | 回放 GPU golden `g1k-a` 第 1 步的全局 batch（16 条 × 4096） | FineWeb-Edu sample-10BT 第一个 parquet 分片（7.26e5 篇），HF 分词器在线切词，`packing=True`，每步 8 条 × 4096 |
| 代码 | `maxtext` hunyuan@`8ac1d175`；诊断臂另加 [`tools/maxtext-qkv-wgrad-fp32-diag.patch`](tools/maxtext-qkv-wgrad-fp32-diag.patch) | 条件 A：hunyuan@`e2c18718`（同 10-29）；B / E1：hunyuan@`8ac1d175` |
| 镜像 | MaxText 官方 `tpu_pre_training` 0.2.5（jax 0.11.1、libtpu 0.0.46），另装 tokamax 0.0.16（这两版 MaxText 要它；`run.sh` 的 `PIP_WHEEL`） | 同左 |
| 硬件 | v5p-8（2×2×2，2 台主机） | v5p-8；H5 续训用 v5p 2×2×4（16 片） |
| 基线并行 | FSDP 8 | A：纯 DP8；B / E1：FSDP 8（真实权重的 FP32 主权重 + Adam 约 150 GB，单片放不下完整副本） |
| CP 臂并行 | FSDP 4 × CP2、FSDP 2 × CP4、FSDP 2 × Ring 2 × Ulysses 2（`ici_fsdp_parallelism` 留 −1 自动补） | A：DP4 × CP2；B / E1：同左 |
| 精度开关 | 现行 yml；`use_tokamax_splash=True`；**`sa_use_base2_exp=False` 统一关**（「base2 开」那一臂除外） | A：`--xla_tpu_spmd_rng_bit_generator_unsafe=false`、`--xla_allow_excess_precision=false`（同 10-29）；B / E1：`use_tokamax_splash=True`，base2 默认开 |
| 学习率 | 恒定 1e-5（只跑 2 步） | A：峰值 3e-4、预热 300、余弦 3000 步；B：恒定 1e-5；E1：`lr_schedule_type=cosine learning_rate=1e-4 warmup_steps_fraction=0.1 learning_rate_final_fraction=0.0`，3000 步 |

## `train_hy3.py` 回放支持上下文并行（本实践加的）

- 回放原来要求「每个 batch 分片的行数 = 每 rank 的 microbatch 数」。开 CP 后 batch 维只按数据 / FSDP 轴切，这条不再成立；CP > 1 时放宽。MaxText 的梯度累积按全局 `total_weights` 归一化，行怎么分进微批不改数学。
- 负载均衡的重排在 MaxText 的数据加载器里做，回放绕过了加载器；回放里照同一个函数（`maxtext_utils.get_reorder_callable`）补上。

## 第 1 步对 FP64 的命令

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
- 诊断补丁只在运行名含 `qkvf32` 时让 q/k/v 投影走 `kernel_grad_fp32`；打补丁的 MaxText 要单独打包，`MAXTEXT_*` 三个变量换成那个包。

## 长跑的命令

```bash
BUCKET=gs://<POC 桶> MODE=rw4 python3 tools/mkjob.py i10-e1-rg <节点池> \
  "E1RG2|lr_schedule_type=cosine learning_rate=1e-4 warmup_steps_fraction=0.1 learning_rate_final_fraction=0.0 learning_rate_schedule_steps=3000 steps=3000 ici_context_parallelism=2 context_parallel_strategy=ring|" > job.yaml
kubectl apply -f job.yaml       # 跑完从两个 Pod 的 /tmp/out 取回 <运行名>.metrics.jsonl
python ../../analysis/cp_tpu_ruler.py --ref base=results/series/E1B.json --ref vmem64=results/series/E1N1.json … --test ring=results/series/E1RG2.json --out results/e1-tpu-ruler.json
python tools/shape_stats.py --base results/series/E1B.json --arm ring=results/series/E1RG2.json … --out results/shape-E1.json
```

- 逐步 loss 存成 `results/series/<运行名>.json`（`first_step`、`loss`、`lr`、`raw_grad_norm`）。
- 节点池是抢占式的，被收回的运行从头重跑，不拼接。

## 显存

- 真实权重截 4 层，FSDP 8：编译期估计每设备 51.2 GB（参数 17.3 GB）；CP2 各方案 51.4–51.9 GB；单步 1.46–1.48 s。
- 随机初始化 16 层，纯 DP8：44.6 GB（参数 29.3 GB）；CP2 约 24 GB；单步 0.45–0.51 s。
- 16 片存档续跑（H5）：31.7 GB / 设备，续跑段与不中断的运行逐位相同。
- 想把权重从 CP 轴上拿掉（只按 FSDP 2 切）的 USP 臂放不下：编译要 131.2 GB 临时量。
