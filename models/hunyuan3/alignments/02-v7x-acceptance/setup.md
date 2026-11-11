# 02 · B200×8 → TPU v7x-4（截 4 层）· 设置

- 配置对照见 [`../../config-alignment.md`](../../config-alignment.md)。
- golden 与 01 相同：正式验收对最终 golden `rcc-a`，第 1 步比较对 `g1k-a` / `g1k-b` / `g1k-c`（第一阶段 GPU golden，不开重算，1000 步，3 份），角色见 01 [README「golden 的分工」](../01-v5p-acceptance/README.md#golden-的分工)；两边的设置写法照 [`../01-v5p-acceptance/setup.md`](../01-v5p-acceptance/setup.md)。
- 运行目录在 POC 桶里，不对外；清单见 `runs.yaml`。
- GPU golden（`g1k` / `rc` / `rcx` / `rce` / `rcec` 和最终的 `rcc`）每一份是怎么产生的、和上游默认差在哪，见 [`../../goldens.md`](../../goldens.md)。
  - `g1k`（第一阶段 GPU golden）、`rc`（GPU 生产 golden，整层重算加路由回放，即原版生产程序，5 份）；
  - `rcx`（生产 golden，输出头 dgrad 换 FP32 一次累加）；
  - `rce`（精确轨迹，BF16 权重、全程 FP32 计算）、`rcec`（真精确轨迹，精确轨迹加输出头 dgrad 分段累加）；
  - `rcc`（最终 golden，GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加，其余设置与 rc 相同，10 份）。
  - 代号的意思见[术语表](../../../../docs/glossary.md#golden-codenames)。

**本页结构**：两边 → 在哪跑 → 冒烟 → 配置换代 → 生产配置 1000 步 → L0 → L1–L4 补齐 → 已知要先定的。

## 两边

代号：`g1k-a` / `g1k-b` / `g1k-c` 是第一阶段 GPU golden 的 3 份；`t-v7x-loss30` 是 v7x 30 步冒烟；`t-loss30` 是 01 在 v5p 上的 30 步冒烟。

| | 源（golden：`g1k-a` / `g1k-b` / `g1k-c`） | 目标（被测：`t-v7x-loss30`） |
|---|---|---|
| 硬件与规模 | NVIDIA B200 × 8，单机 | TPU v7x，4 片 = 8 个 device（每片 2 个 TensorCore） |
| 拓扑 / 进程数 | 1 node，8 进程 | `2x2x1`，1 个主机进程 |
| 并行切分 | DP 8 + EP 8，TP / PP / CP 1 | FSDP 8（与 01 的 v5p 相同：v5p-8 也是 8 个 device） |
| 每步 batch | gbs 16、seq 4096 | 同（见表下） |
| 镜像 / 版本 | 同 01 | 与 01 的 `t-loss30` 同一个镜像 digest；libtpu 同一个构建（2026-08-12，`cl/963683785`），jax 0.11.1 |
| 代码 | 同 01 | 与 `t-loss30` 同一份代码包（见表下） |
| 权重 / 数据 | 同 01 | 同一份 MaxText 检查点；回放 `g1k-a` 的 `data/batches`。`data/` 下的输入指纹与 `t-loss30` 逐字节相同 |

- **每步 batch**：`per_device_batch_size` 按 device 算，8 个 device 与 v5p-8 一样，global batch 不变；`meta.config.gbs` = 16。
- **代码**：
  - `maxtext` @ `64cd9a640b8e…`、janus @ `16af8f7b116b…`；
  - 配置 sha256 `66a03bf31d35…`，命令行另加 `mtp_num_layers=0`。

## 软件版本

以 v7x 正式验收为准：TPU `t-v7x-on-1000` 对最终 golden `rcc-a`（`phase2-v7x-acceptance-rcc`）。和 01 只差硬件，软件逐项同 [01 的「软件版本」](../01-v5p-acceptance/setup.md#软件版本)：

| 侧 | 组件 | 版本 | 证据出处 |
|---|---|---|---|
| TPU | 镜像 | `tpu_pre_training`，digest `b152355a…`（同 01） | `t-v7x-on-1000` 的 `meta.json` 的 `image` |
| TPU | MaxText / janus | `hunyuan` 分支 `0622774a` / `958d173e`（同 01 的 `t-rmap-on-1000`） | `meta.json` 的 `code` |
| TPU | jax / jaxlib | 0.11.1 / 0.11.1 | `meta.json` 的 `versions` |
| TPU | libtpu | 0.0.46（TPU7x 版，构建 `cl/963683785`；按构建号认） | `meta.json` 只记构建号。同一构建号的 v7x 运行实测记的是 `libtpu==0.0.46`（device_kind TPU7x，混元 4 [`jit-miscompile-probe-v7x.json`](../../../hunyuan4/results/jit-miscompile-probe-v7x.json)） |
| TPU | tokamax | 0.0.14（镜像自带的包） | `meta.json` 不记；同 01（同一镜像 digest，01 读镜像层核过） |
| TPU | 配置 | `hy3-4layers.yml` sha256 `a238883b…`，命令行另加 `mtp_num_layers=0` 与 remat 覆盖（同 01 `t-rmap-on-1000`） | `meta.json` 的 `config`；登记见 [`runs.yaml`](runs.yaml) |
| GPU | 全部 | 同 01（golden 是同一批 `rcc-*`） | — |

结果文件 [`phase2-v7x-acceptance-rcc.results.json`](results/phase2-v7x-acceptance-rcc.results.json) 记着 `t-v7x-on-1000` 的 `meta_sha256`。**结论在这些版本上测得**；换 libtpu 或 jax 要按混元 4 [`libtpu-052.md`](../../../hunyuan4/alignments/04-tpu-self/libtpu-052.md) 的做法重测。

v5p 上实测 libtpu 0.0.46、0.0.48、0.0.51、0.0.52 有 jit 静默算错的已知问题（0.0.47、0.0.49 没测）（[环境「已定的」](../../../../docs/reference/environment.md#已定的)）。存档优化后 HLO 的扫描（混元 4 [`jit-miscompile-hlo-scan.json`](../../../hunyuan4/results/jit-miscompile-hlo-scan.json)）里混元 3 有 72 份运行，全是 v5p 上的回归与标定程序，没有一份带它的特征；正式被测 `t-rmap-on-1000`、`t-v7x-on-1000` 和 EP = 8 的运行不在扫描范围内，没有扫。

## 在哪跑

- TPU v7x 4 片切片（2x2x1，1 台主机），`run.sh` 的 `PLATFORM=v7x`。
- 提交方式见 [`../../frameworks/maxtext/README.md`](../../frameworks/maxtext/README.md)「站点配置（`JANUS_SITE_PROFILE`）」。
- 用的节点可能被平台回收（基础设施中断），回收了就重跑。
- 提交到出结果 5 分钟左右（第 1 步含编译 23 s，之后每步 0.54 s；v5p-8 是 0.91 s）。

## 冒烟：`t-v7x-loss30`（2026-11-05）

`--no-hooks --dump-steps none --no-param-sha`，30 步。逐步 lm_loss：

代号：`g1k-a` 是第一阶段 GPU golden 的第 1 份；`t-loss30` 是 v5p 30 步冒烟；`t-v7x-loss30` 是 v7x 30 步冒烟。

| 步 | GPU `g1k-a` | v5p `t-loss30` | v7x `t-v7x-loss30` | v7x − GPU | v5p − GPU | v7x − v5p |
|---|---|---|---|---|---|---|
| 1 | 11.465756 | 11.465785 | 11.465868 | +1.1e-04 | +2.9e-05 | +8.3e-05 |
| 2 | 11.235437 | 11.235693 | 11.235696 | +2.6e-04 | +2.6e-04 | +2.9e-06 |
| 5 | 9.291429 | 9.291639 | 9.291651 | +2.2e-04 | +2.1e-04 | +1.1e-05 |
| 10 | 8.620575 | 8.620508 | 8.620386 | -1.9e-04 | -6.7e-05 | -1.2e-04 |
| 20 | 7.576262 | 7.576082 | 7.576143 | -1.2e-04 | -1.8e-04 | +6.1e-05 |
| 30 | 7.270696 | 7.270776 | 7.270893 | +2.0e-04 | +8.0e-05 | +1.2e-04 |

30 步汇总（|差| 的最大值 / 平均值）：

| 对 | 最大 | 平均 |
|---|---|---|
| v7x − GPU `g1k-a` | 4.1e-4 | 1.6e-4 |
| v5p − GPU `g1k-a`（01 的 `t-loss30`） | 4.3e-4 | 1.5e-4 |
| v7x − v5p | 1.6e-4 | 7.3e-5 |
| GPU 自己跑三次（`g1k-a/b/c` 两两，参照） | 1.4e-4 ～ 2.1e-4 | 5.7e-5 ～ 6.3e-5 |

**读法**

- v7x 对 GPU 的差和 v5p 对 GPU 的差同一个量级。
- v7x 与 v5p 不逐位（30 步里没有一步相同），差在 GPU 自己的跑间噪声以内。
  - 同一份程序、同一个 libtpu 构建，在两代硬件上编出来的程序不同（切片形状、每片两个 TensorCore），不逐位是预期内的。
- 这只是冒烟，不是对齐结论（当时检查项目录还没建；之后的正式判定见 [README](README.md)「最终结论」）。

## 配置换代：打开 `dense_fp32_accumulation`（2026-11-05）

- 对齐配置 yml 从 `2a76c7590866…` 换成 `a238883b7f91…`：再打开 `dense_fp32_accumulation`。
  - 作用：注意力与稠密 MLP 投影 FP32 累加、只舍一次；
  - maxtext 要 hunyuan@0622774 或更新；
  - 原因与 v5p 上的回放验证见 `../../config-alignment.md`「收缩维被 FSDP 切开的投影的累加精度」。
- 下面这节的运行都是换代之前的配置。
- v7x 上开 / 关的步时与显存对比是 `t-v7x-st-off-100` / `t-v7x-st-on-100`（v7x 生产 rmap 配置，关 / 开这个开关各 100 步；读数见 01 README 结论）。

## 生产配置 1000 步：`t-v7x-rmap-1000-a` / `-b`（2026-11-05）

- 01 的 `t-rmap-1000`（v5p 生产配置 1000 步）的 v7x 对应：
  - 同一份代码包（maxtext `237a3270`、janus `9793b367`）；
  - yml sha256 `2a76c7590866…`（GA 2 + 三个精度开关）；
  - 命令行同样把除层输入与 `moe_routing_maps` 外的中间量全部 remat、`moe_routing_maps=device`（重算设置对齐 GPU 的 rc，即 GPU 生产 golden）。
- 每步中位数 0.96 s（v5p-8 的 t-rmap-1000 是 1.60 s）。
- `-b` 的第一次提交跑到第 70 步前后时节点被平台回收（基础设施中断），删掉 JobSet 原名重提。

### v7x 自己是否逐位（`phase2-v7x-rmap-repro`）

`-b` 对 `-a`：

- lm_loss、grad_norm、param_norm 逐步 0 格容差全过；
- `metrics.jsonl` 里 60 个字段（`step_time` 除外）1000 步全部相同；
- 输入指纹 65,000 项相同。

### 正式验收：v7x 与 v5p 对照

- **第一层**：对最终 golden 的基线 [`rcc-a`](../../../../docs/glossary.md#rcc-a)，读数见 [README](README.md) 第 3 条与 [`../../ALIGNMENT-REPORT.md`](../../ALIGNMENT-REPORT.md)。
- **第二层**：对生产 golden `rc`，见 README 第 2 条（`phase2-v7x-acceptance-rc`）。
  - 和 v5p 的 `phase2-acceptance-rc` 同号、同量级，param_norm 到第三位都一样。
  - GPU 输出层 dgrad 的长求和是 GPU 侧的事，换哪代 TPU 都会看到（01 案例 01）。
- **v7x 与 v5p 之间**：
  - 301–1000 步 loss 平均差 −1.6e-4（v7x 低），最大单步 2.5e-3；
  - param_norm 第 1000 步差 −2.4e-4。
  - 这组数没有记出处与口径（绝对差还是相对差没写），推断出自换代前配置的运行；v7x 那份逐步序列不在仓库，无法复算，不作判定依据。当前配置 v7x 对 v5p 的比较见[对齐报告](../../ALIGNMENT-REPORT.md)第 3 节（grad_norm、param_norm 都在 `rcc` 45 对的噪声内）。

## L0（`phase2-v7x-t-loss30`、`phase2-v7x-l0-state`）

- 配置逐项对照、30 步输入指纹逐位：`t-v7x-loss30`（v7x 30 步冒烟）对 `g1k-a`（第一阶段 GPU golden 的第 1 份），同 01 的 `t-loss30`（v5p 30 步冒烟）。
- 载入后的状态：`t-v7x-l0`（v7x 上 1 步，第 1 步训练之前读状态，`--no-hooks --dump-steps 1 --full`）对 `g1k-a` 第 1 步；
  生产配置 `t-v7x-rmap-1000-a` 的初始参数清单单独比（`L0.weights.init.prod`）。
- 权重转换：与 01 共用同一份 MaxText 检查点，结果是 01 的 `phase2-ckpt-convert`。
- 结果：7 项全过，计数与 v5p 的 `phase2-l0-state` 相同。
  - 初始清单 1480 项、载入后参数与主权重各 50 个张量、expert_bias 3 个、dtype 2960 项、结构 15 条。
- `t-v7x-l0` 前两次提交节点都被平台回收，第三次跑完。
  - 第 1 步 loss 11.465931（`t-v7x-loss30` 是 11.465868，配置不同：这次是生产 yml）。

## L1–L4 补齐（2026-11-05）

- 照 01 的做法与门槛（FP64 R-ratio ≤ 1.2、GPU 噪声参照、逐位）在 v7x 上重跑一遍。
- 目录都是 01 的同名目录复制后换被测（`catalogs/phase2-v7x-*.yaml`）。
- TPU 侧代码包：
  - 除观测构建一对外，都用 maxtext `237a3270` + janus `68ae6b89`（01 的 `t-st1-prod` / `t-st8-prod` / `t-nan8` / `r-ff-prod` 那一版，即下表各运行在 v5p 上的同名运行）；
  - 观测构建一对用 janus `8c3c7138`（01 的 `obs-t1` 那一版；`obs-t1` 是 v5p 观测构建 1000 步）。

| 运行 | 做什么 | 结果文件 |
|---|---|---|
| `t-v7x-st1-prod` | 生产配置 2 步，第 1 步全量状态读；派生 `t-v7x-st1-prod-gm`（动量反推梯度） | `phase2-v7x-st1-prod-state` |
| `t-v7x-st8-prod` | 9 步，第 1–8 步状态读（`--full-include` 同 01）；派生 `-gm` | `phase2-v7x-st8-opt8` |
| `r-v7x-l2` | 逐层隔离回放 `--layers 0,1,2,3 --grad` | `phase2-v7x-l2-replay-rratio`、`phase2-v7x-l1-ops-own`、`phase2-v7x-l2-replay-bitwise`（见表下） |
| `r-v7x-ff-prod` | 整网前向回放 `--full-forward` | `phase2-v7x-ff-chain`（`fwd_rratio.py --mode chain`） |
| `t-v7x-nan8` | 第 5 步梯度埋 NaN，8 步 | `phase2-v7x-t-nan8`（演练） |
| `obs-v7x-t1` / `reg-v7x-t1` | 观测构建 1000 步 / 同配置生产构建（T1 档参数） | `phase2-v7x-obs-gate`、`phase2-v7x-coverage`（用 `obs-v7x-t1` 第 1 步的清单） |
| `r-v7x-wcoff` | 同 `r-v7x-l2`（不带 `--grad`）加关窗口化的 libtpu 参数（见表下） | 诊断，不进总览（结果文件已删除，读数见 README 第 5 条） |
| `r-v7x-cont20` / `r-v7x-res20` | 连续 20 步（第 10 步存档）/ 从存档续训 11–20 步；janus `c8143c65`（01 的 `r-cont20` 那一版） | `phase2-v7x-resume`（5 项全过） |

- **`r-v7x-l2` 的结果文件**：
  - `phase2-v7x-l2-replay-rratio` 用 `fwd_rratio.py --mode isolated`；
  - `phase2-v7x-l1-ops-own` 用 `--mode ops`；
  - 反向见 `r-v7x-l2-on` 的 `phase2-v7x-l2-replay-bwd-on`。
- **`r-v7x-wcoff` 的参数**：`LIBTPU_INIT_ARGS=--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000`。

和 v5p 不同的两处（其余判定与 01 逐项相同）：

1. **前向对 FP64**：四层 R 1.57 / 1.53 / 1.56 / 1.55（v5p 1.21 / 1.22 / 1.24 / 1.23）。
   - 单算子回放里只有「RMSNorm + 投影」两项变大：
     - qkv（输入归一化 + qkv 投影）1.57（v5p 1.24）；
     - 稠密层 fc1（MLP 前归一化 + gate|up 投影）1.58（v5p 1.29）。
   - 其余都与 v5p 相当：单独的 RMSNorm 1.16、qk_norm 1.00、输出投影 1.00、路由 GEMM 0.02、门控激活 + down 1.00、注意力核 1.01、MoE 1.15。
   - 原因与 v5p 相同（收缩维被 FSDP 切开的投影编成窗口化 collective matmul，分片部分积在 BF16 里累加），v7x 上偏差更大。
   - 验证：`r-v7x-wcoff` = 同一份回放加 `LIBTPU_INIT_ARGS=--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000`（关掉窗口化）。
     - qkv 1.000、fc1 1.104，四层前向 1.059 / 1.010 / 1.012 / 1.015 都过；但同一次回放里单独的 MLP 前 RMSNorm（mlp_norm）是 1.64、超线，v5p 关窗口化同样 1.64（[README](README.md)）；
     - 单算子里 qkv、qk_norm、注意力输出投影、fc1、稠密层激活与 fc2、注意力核与 v5p 关窗口化（`phase2-l1-ops-wcoff`）到第三位相同；路由 GEMM（0.017 对 0.022）、MoE 核（0.983 对 0.924）不同，都远在 1.2 以内（README 第 5 条）。
   - 整网前向的 logits / 逐 token loss R 0.999 / 1.025（v5p 0.957 / 0.981），过。
2. **第 1 步更新后的 expert_bias**：第 1 层 65 号专家的更新方向反了（TPU +1e-3、GPU −1e-3），其余 575 个逐位；v5p 三层全部逐位。
   - 这个更新只看负载与均值的大小关系，负载贴着均值的专家一个 token 的路由不同就会翻号。
   - GPU 噪声参照是 0，所以判不过。

其他记录：

- `t-v7x-st1-prod` 的梯度（动量反推）对 GPU：第 0 层 rel_l2 2.9e-3（v5p 2.2e-3），第 3 层 0.106（v5p 0.102）。
- `r-v7x-ff-prod` 第一次提交时节点被平台回收，原名重提。

## 已知要先定的

- v7x 4 芯片是 8 个 device。FSDP 8 与 `per_device_batch_size` 沿用 v5p-8 的写法，就能让每个 device 拿到的行和 GPU rank 对应（上面的冒烟按这个跑）。
- 切分定为 FSDP 8；EP = 8 也对过一次（`t-v7x-ep8-1000`，对 `rcc-a` L4 都在噪声内，[`phase2-v7x-acceptance-rcc-ep8`](results/phase2-v7x-acceptance-rcc-ep8.report.md)）。
