[janus](../../../../README.md) › [混元 4](../../README.md) › [mini](README.md) ｜ 运行登记

# 混元 4 · T1 的运行登记（共同权重、GPU 运行、TPU 运行、噪声）

T1（hy4-mini）每一份运行是怎么来的、属于哪一组、用来干什么。

- **两个角色**：语义以 TPU 一侧 MaxText 的混元 4 实现为准（[`PROTOCOL.md`](../../PROTOCOL.md) 第 1.2 节）；判定时 GPU 是 golden / 基线、TPU 是被测，噪声尺子只用 GPU 重跑（[`PROTOCOL.md`](../../PROTOCOL.md) 第 3 节与变更记录）。
- 设置的登记在 `PROTOCOL.md` 第 1.2 节，两边每个开关见 [`config-alignment.md`](config-alignment.md)，每一处改动见 [`CHANGES-LEDGER.md`](../../CHANGES-LEDGER.md)。
- 运行目录不随仓库发布，按 janus 抓数格式（`meta.json`、`metrics.jsonl`、`data/`、`params_sha/`，第 1 步参照另有 `step_1/`）。

## 一张表看各次比较

| 比较（结果文件） | 步数 | TPU 被测 | GPU 基线 | 噪声 N | 现状 |
|---|---|---|---|---|---|
| `t1t-l4`（M28 之前） | 300 | `t1t-c` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 4、不过 1（已定责）、待定 4 |
| `t1u-l4`（M28 之前） | 1000 | `t1u-a` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 7、待定 2 |
| `t1s-l4` | 300 | `t1s-a` | `g1-a` | `g1-a … g1-h` 两两 28 对 | 过 4、不过 3、待定 2 |
| `t1-l4-nozm` | 300 | `t1n-a` | `g1-a` | 同上 | 过 2、不过 3、待定 4 |
| `t1-l4`（旧设置） | 300 | `t1-a` | `g1-a` | 同上 | 过 2、不过 5、待定 2 |
| `t1zo-l4`（**现行**，2026-11-11 起：现行 yml 的全部开关，base2 关） | 1000 | `tz-off-rep` | `g3-a` | `g3-a … g3-p` 16 份两两 120 对 | 过 7、不过 0、待定 2 |
| `t1zo300-l4`（现行，同一运行前 300 步） | 300 | `tz-off-rep` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 7、不过 0、待定 2 |
| `t1zga-l4`（上一代，t1zw-a + M50，base2 开） | 1000 | `t1zg-a` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 7、不过 0、待定 2 |
| `t1zga300-l4`（上一代，同一运行前 300 步） | 300 | `t1zg-a` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 8、不过 0、待定 1 |
| `t1zw-l4`（上一代，t1zu-a + M42b） | 1000 | `t1zw-a` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 6、不过 0、待定 3 |
| `t1zw300-l4`（上一代，同一运行前 300 步） | 300 | `t1zw-a` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 7、不过 0、待定 2 |
| `t1zgb-l4`（`t1zg-a` 的重跑，M50 修订版） | 1000 | `t1zg-b` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 7、不过 0、待定 2 |
| `t1zz-l4`（验证：t1zw-a 再关 base2；当时没采用，2026-11-11 起关 base2 成为现行，见 `t1zo-l4`） | 1000 | `t1zz-a` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 6、不过 0、待定 3 |
| `t1zz300-l4`（同一运行前 300 步） | 300 | `t1zz-a` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 6、不过 0、待定 3 |
| `t1zv-l4`（诊断代，t1zu-a + M42，已撤下） | 1000 | `t1zv-b` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 6、不过 0、待定 3 |
| `t1zv300-l4`（诊断代，同一运行前 300 步） | 300 | `t1zv-b` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 7、不过 0、待定 2 |
| `t1zu-l4`（再上一代，t1zu-b + M40 + M41） | 1000 | `t1zu-a` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 7、不过 0、待定 2 |
| `t1zu300-l4`（再上一代，同一运行前 300 步） | 300 | `t1zu-a` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 7、不过 0、待定 2 |
| `t1zr-l4`（更早一代，t1z + M36） | 1000 | `t1zr-a` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 6、不过 0、待定 3 |
| `t1zr300-l4`（更早一代，同一运行前 300 步） | 300 | `t1zr-a` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 7、不过 0、待定 2 |
| `t1zs-l4`（诊断代，t1zr + M37） | 1000 | `t1zs-a` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 3、不过 0、待定 6 |
| `t1zs300-l4`（诊断代，同一运行前 300 步） | 300 | `t1zs-a` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 4、不过 0、待定 5 |
| `t1zt-l4`（诊断代，t1zs + M34） | 1000 | `t1zt-a` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 5、不过 1（路由偏置均值）、待定 3 |
| `t1zt300-l4`（诊断代，同一运行前 300 步） | 300 | `t1zt-a` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 7、不过 0、待定 2 |
| `t1zub-l4`（诊断，t1zt + M39） | 1000 | `t1zu-b` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 4、不过 0、待定 5 |
| `t1zub300-l4`（同一运行前 300 步） | 300 | `t1zu-b` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 7、不过 0、待定 2 |
| `t1zuc-l4`（诊断，t1zu-b + M41） | 1000 | `t1zu-c` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 4、不过 0、待定 5 |
| `t1zuc300-l4`（同一运行前 300 步） | 300 | `t1zu-c` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 5、不过 1、待定 3 |
| `t1zud-l4`（诊断，t1zu-a 只关 M41） | 1000 | `t1zu-d` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 6、不过 0、待定 3 |
| `t1zud300-l4`（同一运行前 300 步） | 300 | `t1zu-d` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 7、不过 0、待定 2 |
| `t1z-l4`（上一代，M28 + M31 + M33） | 300 | `t1z-a` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 6、不过 0、待定 3 |
| `t1zk-l4`（上一代 1000 步，t1z 设置） | 1000 | `t1zk-a` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 4、不过 2（主 loss，由 M36 修掉大部分）、待定 3 |
| `t1v-l4`（再上一代，M28） | 300 | `t1v-a` | `g2-a` | `g2-a … g2-h` 两两 28 对 | 过 7、不过 1（索引器 KD，t1z 打开 M31 + M33 后消失）、待定 1 |
| `t1w-l4`（再上一代 1000 步，t1v 的设置） | 1000 | `t1w-a` | `g3-a` | `g3-a … g3-h` 两两 28 对 | 过 6、不过 1（索引器 KD，第 300 步以后 TPU 偏高；t1zr 里已过）、待定 2 |

前三代的比较目录里 `test` 是 GPU `g1-a`、`golden` 是 TPU（当时把 TPU 当基线）；2026-11-07 改为 GPU golden、TPU 被测之后，`t1t-l4` 起按新角色写。旧的三次没有按新角色重算：互换角色只把逐步相对差的符号反过来、分母从 TPU 的值换成 GPU 的值（loss 类的逐步相对差在 1e-5 量级，索引器 KD 约 1e-3），预计不改判定（推断）。各项判定见[检查清单](CHECKS.md) L4。

## 共同的初始权重

| 项 | 值 | 出处 |
|---|---|---|
| 名字 | hy4-mini 随机初始化的 HF 目录（第 0 份种子） | — |
| 怎么生成 | `model/make_tiny_hf.py --config mini --layers 7 --mtp 1 --seed 0`，权重矩阵标准差 0.006（HF `initializer_range`）；`config.json` 含发布配置里全部与维度无关的键 | 台账 J2 |
| 规模 | 427,783,489 参数（含 MTP），212 个张量 | 台账 M6 |
| 张量清单 `tensor_sha256.json` 的 sha256 | `1d869124db698cc51d3fb8fa1c2d83a00ab51066701afd81acf9a0a8af176433` | [`PROTOCOL.md`](../../PROTOCOL.md) 第 1.1 节 |
| `config.json` 的 sha256 | `24cec75c00931bf93c659de379c732df49f0bc6d6bde32487c7cb0ac45927611`（与 TPU 运行 `meta.json` 的 `hf_config_sha256` 一致） | `frameworks/maxtext/train_hy4.py` |
| 两边怎么核 | TPU 转换后反向取回 212 / 212 逐位、RoPE 去交错 32 / 32；GPU 第 1 步前逐参数等于检查点张量（载入自检） | 台账 J3、G7 |

**结构取舍**（hy4-mini 只缩宽度与个数）：

- 保留生产的每头维度和全部结构：qk_nope 192、qk_rope 64、v 256、kv_lora 512；索引器 32 头 × 128、top-k 2048；门控 MLA、可学习 sink、iHC 超连接、FP32 输出头、IndexCache、sigmoid 路由与无辅助损失偏置、共享专家、SwiGLU 截断、1 层 MTP；真实词表 120832。
- 缩小的只有宽度与个数：宽 1024、8 头、q_lora 512、稠密 FFN 3072、16 个路由专家（top-4）宽 256；16 个专家能被专家并行 4 整除。
- 7 层 = 1 稠密 + 6 MoE：IndexCache 每 4 层一组，正好一个整组（层 1 算 top-k，层 2–4 复用）加 2 层余数块（层 5 算、层 6 复用），两条代码路径都走到。
- 参数量 427,783,489（含 MTP；不含 MTP 401,841,193；嵌入 + 输出头 247,463,936），由 HF 张量形状数出。同一数法给 hy4-preview 不含 MTP 769,907,408,797，对上公开的 770B 总参数。
- 检查点读法各尺寸统一：q_pe / k_pe 交错存放、转换时去交错，latent norm eps 1e-5，索引器 k_norm eps 1e-6。

## 共同的数据

fineweb-edu `sample/10BT` 第一个分片，混元 4 分词器。GPU 写下每个 rank、每步、每个 microbatch 的 tokens / labels / loss_mask（eod 位置 loss_mask = 0），TPU 原样回放（`PROTOCOL.md` 第 1.2 节，台账 G9）。1000 步的 `g3-*` 沿同一条数据流往后取，第 1–300 步与 `g1-*`、`g2-*` 逐条相同（没有数据指纹的结果文件；`t1w-a` 前 300 步与 `t1v-a` 逐位相同是间接证据）。

## GPU 运行（判定时的 golden）

B200 × 8，Megatron-Core 0.20 + TE（NeMo 26.10 镜像所带；`meta.json` 没有记 TE 版本，未核），数据并行 8、专家并行 4（每份运行的 `meta.json` 记 `parallelism`；完整命令行没有入库），默认设置（不逐位复现）。

三代：

- `g1-*`、`g1s-*`：注意力 q·k 在 BF16 里算（janus 的 eager 实现原来的写法）。11 份的 `meta.json` 按 janus 格式改过 `steps` 一个字段（台账 G13）。
- `g2-*`、`g2s-*`：只改这一处，q·k 在 FP32 里算，其余（权重、数据、设置、代码其余部分）与 `g1-*` 相同；`meta.json` 的 `config.attn_scores_fp32` 记开关。
- `g3-*`：`g2-*` 的设置跑 1000 步。

| 代号 | 份 | 步数 | 用途 |
|---|---|---|---|
| `g1-a` | 1 | 300 | 前三代比较的 GPU 基线 |
| `g1-b … g1-h` | 7 | 300 | 同设置重跑：与 `g1-a` 一起 8 份，两两 28 对是前三代的噪声尺子 |
| `g1s-a … g1s-c` | 3 | 1 | 第 1 步全量抓数参照（不重算）；导出成 HF 布局的梯度（台账 G15）；t1s 一代第 1 步逐张量比较的 GPU 一侧 |
| `g2-a` | 1 | 300 | **300 步比较的基线**（现行 `t1zo300-l4`；上一代现行 `t1zga300-l4`、`t1mz300-l4-qkclip`；上一代及更早 `t1zw300-l4`、`t1zv300-l4`、`t1zz300-l4`、`t1zla300-l4`、`t1zud300-l4`、`t1zr300-l4`、`t1zs300-l4`、`t1zt300-l4`、`t1zub300-l4`、`t1zuc300-l4`、`t1zu300-l4`、`t1z-l4`、`t1v-l4`、`t1t-l4`） |
| `g2-b … g2-h` | 7 | 300 | 同 `g2-a` 重跑：上一行各次比较的噪声尺子 |
| `g2s-a … g2s-c` | 3 | 1 | 同 `g1s-*`，只改注意力 q·k 在 FP32 里算；t1t、t1v、t1z 与现行 t1zg 第 1 步逐张量比较的 GPU 一侧 |
| `g3-a` | 1 | 1000 | **1000 步比较的基线**（现行 `t1zo-l4`；上一代现行 `t1zga-l4`、`t1zgb-l4`、`t1mz-l4-qkclip`；上一代及更早 `t1zw-l4`、`t1zv-l4`、`t1zz-l4`、`t1zla-l4`、`t1zud-l4`、`t1zr-l4`、`t1zs-l4`、`t1zt-l4`、`t1zub-l4`、`t1zuc-l4`、`t1zu-l4`、`t1zk-l4`、`t1w-l4`、`t1u-l4`）；存下全部 1000 步的 batch 给 TPU 回放 |
| `g3-b … g3-h` | 7 | 1000 | 同 `g3-a` 重跑：上一行各次比较的噪声尺子 |
| `g2d-a` | 1 | 300 | 诊断（param_norm 定位，不进门槛）：`g2-a` 的设置，另在第 1 步与第 300 步做全量抓数（主权重前后、梯度、优化器状态；路由专家只有 rank 0 的 0–3 号）；台账 G22 |
| `g2o-a` | 1 | 8 | 诊断（优化器回放，不进门槛）：每步全量抓数，含 Muon 动量缓冲；台账 G20 |
| `g6-qk-a`、`g6-qk-b` | 2 | 见 `meta.json` | 诊断（不进门槛）：QK-Clip 逐层逐头记录，让 QK-Clip 真正触发并逐头核对；台账 G21 |
| `g7-l1-a` | 1 | 2 | 诊断（L1 逐算子 / L2 逐层，不进门槛）：`g2-a` 的设置去掉整层重算，第 1 步 rank 0 第 0 个 microbatch 逐算子抓数（`--l1-capture-step 1`）并做全量抓数；FP64 重算见 `analysis/fp64_l1_gpu.py`；台账 G23 |
| `g7-l1-c` | 1 | 1 | 诊断（L3 iHC 单层定位，不进门槛）：同 `g7-l1-a`，另存第 0 个 microbatch 的参数梯度与 iHC 混合输出的梯度（`--l1-grad-micro0`） |

## TPU 运行（被测）

v5p 2x2x2（8 个设备），MaxText 混元 4 集成分支，配置 [`frameworks/maxtext/hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml)（各代当时的版本，差别见下表「设置」一栏）。TPU 按设置分代：每代一个被测、一份同设置重跑，300 步的各代另有 4 份只改执行参数的诊断运行（t1v、t1w、t1z、t1zk、t1zr、t1zs 只有被测一份（t1w 设置另有只改执行参数的 1000 步诊断 `t1wp-1..3`））（`PROTOCOL.md` 第 1.1、3 节）：`-1` 注意力块大小 256；`-2` 专家并行 4 × FSDP 2；`-3` MoE 走不分组的稠密实现；`-4` 重算策略 `custom`、命名张量全存 HBM。

| 代号 | 设置（相对上一代） | 用途 | 结果 |
|---|---|---|---|
| `t1-a`、`t1-b`、`t1p-1 … t1p-4` | 最初的 T1 设置：路由偏置零均值更新开 | 旧设置，保留可追溯 | [`t1-l4`](results/t1-l4.report.md) |
| `t1n-a` | 路由偏置零均值关（协议变更记录「路由偏置」行，台账 G14） | t1n 一代的被测 | [`t1-l4-nozm`](results/t1-l4-nozm.report.md)、[`t1-l0-settings`](results/t1-l0-settings.report.md) |
| `t1n-b`、`t1np-1 … t1np-4` | 同 `t1n-a` | 重跑核逐位（除 `step_time` 外与 `t1n-a` 逐位相同）；诊断 | — |
| `t1s-a`、`t1s-b`、`t1sp-1 … t1sp-4` | 再打开 GPU 一侧舍入位置的四个开关与 `indexer_fp32_scores`（台账 M8–M10、M12、M23） | t1s 一代的被测、重跑（除 `step_time` 外逐位相同）、诊断 | [`t1s-l4`](results/t1s-l4.report.md)、[`t1s-l0-settings`](results/t1s-l0-settings.report.md) |
| `t1t-a`、`t1t-b`、`t1tp-1 … t1tp-4` | 再改输出头精度（`logits_matmul_precision: highest`）、重算策略（`custom`，路由索引存下）、QK-Clip 统计不含 sink、weights_proj 乘 BF16 权重（台账 M13、M25–M27） | t1t 一代；`t1t-a` 与 `t1t-b` 除 `step_time` 外逐位相同 | [`t1t-l0-settings`](results/t1t-l0-settings.report.md) |
| `t1t-c` | 同 `t1t-a`，另记 `param_norm_exact`（台账 J13） | **300 步比较的被测（M28 之前）**；除 `step_time` 与新列外与 `t1t-a` 逐位相同 | [`t1t-l4`](results/t1t-l4.report.md) |
| `t1u-a`、`t1u-b` | 与 `t1t-c` 同设置，回放 `g3-a` 的 1000 步数据 | **1000 步比较的被测（M28 之前）**（`t1u-a`）；`t1u-b` 是同设置重跑 | [`t1u-l4`](results/t1u-l4.report.md)、[`t1u-l0-settings`](results/t1u-l0-settings.report.md) |
| `t1v-a` | 再打开 `ihc_stream_sum_unrolled`（iHC 四路流逐路 FP32 乘加，台账 M28），300 步 | 再上一代 300 步比较的被测（M28 之后） | [`t1v-l4`](results/t1v-l4.report.md)、[`t1v-l0-settings`](results/t1v-l0-settings.report.md) |
| `t1z-a` | 再打开 `muon_mcore_ns_bf16`（M31）与 `transcendental_accuracy_highest`（M33），300 步 | 上一代 300 步比较的被测；L3 上一代的设置（第 1 步诊断 `t1zdiag2`） | [`t1z-l4`](results/t1z-l4.report.md)、[`t1z-l0-settings`](results/t1z-l0-settings.report.md) |
| `t1zr-a` | t1z 再开 `ihc_force_bf16_rounding`（M36），1000 步 | 更早一代比较的被测（2026-11-08 前是现行） | [`t1zr-l4`](results/t1zr-l4.report.md)、[`t1zr300-l4`](results/t1zr300-l4.report.md) |
| `t1zs-a` | t1zr 再开 M37（sigmoid / silu 的高精度写法；MaxText `7c95fabe`，含 M37 的提交 `b8d4ba8b`，M34 的代码在、配置没开），1000 步 | 诊断代的被测：带符号偏差变大、待查，不作现行比较 | [`t1zs-l4`](results/t1zs-l4.report.md)、[`t1zs300-l4`](results/t1zs300-l4.report.md) |
| `t1zt-a` | t1zs 再开 `muon_adam_accurate`（M34），即打包时 T1 配置的全部开关（MaxText `469dbb85`；与 `t1zs-a` 的 `7c95fabe` 之间只新增 hy4-mid 配置，生效配置只差 M34；没有之后加的 M38–M41），1000 步 | 诊断代的被测：1000 步路由偏置均值不过、待查，不作现行比较 | [`t1zt-l4`](results/t1zt-l4.report.md)、[`t1zt300-l4`](results/t1zt300-l4.report.md) |
| `t1zu-b` | t1zt 再加 M39（精确 sigmoid / silu 的反向改用 PyTorch 解析式，MaxText `3741e818`），M40、M41 用覆盖参数关掉；janus `2679a837`，1000 步 | 诊断：只差 M39 的对照 | [`t1zub-l4`](results/t1zub-l4.report.md)、[`t1zub300-l4`](results/t1zub300-l4.report.md) |
| `t1zu-c` | t1zu-b 再开 `mlp_fused_fc1_dgrad`（M41），M40 关；同一提交，1000 步 | 诊断：只差 M41 的对照 | [`t1zuc-l4`](results/t1zuc-l4.report.md)、[`t1zuc300-l4`](results/t1zuc300-l4.report.md) |
| `t1zu-a` | t1zu-c 再开 `moe_mcore_prob_before_fc2`（M40）；同一提交，1000 步 | 上一代比较的被测（2026-11-08 至 11-09 是现行） | [`t1zu-l4`](results/t1zu-l4.report.md)、[`t1zu300-l4`](results/t1zu300-l4.report.md) |
| `t1zu-d` | t1zu-a 只关 M41（即 t1zt + M39 + M40）；同一提交，1000 步 | 诊断：凑齐 M40 / M41 的 2×2 | [`t1zud-l4`](results/t1zud-l4.report.md)、[`t1zud300-l4`](results/t1zud300-l4.report.md) |
| `t1zv-b` | t1zu-a 再开 `splash_dsinks_fp32`（M42，sink 梯度全程 FP32），MaxText `2306c7a5`，1000 步；`t1zv-a` 未完成，同设置重跑 | 诊断代：整网离 GPU 更远，M42 从对齐配置撤下（PROTOCOL 变更记录 2026-11-09） | [`t1zv-l4`](results/t1zv-l4.report.md)、[`t1zv300-l4`](results/t1zv300-l4.report.md) |
| `t1zz-a` | t1zw-a 再关 `sa_use_base2_exp`（M43，splash 不走 base2 指数），MaxText `2c28a4f4`，1000 步（2026-11-09 yml 曾短暂写明 `false`，已改回） | 诊断代：关 base2 后整网离 GPU 更远（对 8 份 GPU 主 loss 带符号均值 r 中位数 0.51 → 2.06），对齐配置改回开，当时的现行比较仍是 `t1zw-a`（PROTOCOL 变更记录 2026-11-09） | [`t1zz-l4`](results/t1zz-l4.report.md)、[`t1zz300-l4`](results/t1zz300-l4.report.md) |
| `t1zw-a` | t1zu-a 再开 `splash_dsinks_gpu_order`（M42b，sink 梯度照 GPU 的写法），MaxText `2c28a4f4`，1000 步 | 上一代比较的被测（2026-11-09 上午是现行） | [`t1zw-l4`](results/t1zw-l4.report.md)、[`t1zw300-l4`](results/t1zw300-l4.report.md) |
| `t1zg-a` | t1zw-a 再开 `indexer_kd_generalized_kl`（M50，只改 KD 记录值），MaxText `47d31984`，即现行 yml 的全部开关，1000 步；41 个逐步量与 `t1zw-a` 逐位相同，KD、总 loss、param_norm 的记录值有末位差（参数逐位相同） | **现行比较的被测**（PROTOCOL 变更记录 2026-11-09） | [`t1zga-l4`](results/t1zga-l4.report.md)、[`t1zga300-l4`](results/t1zga300-l4.report.md)、L0 [`t1zg-l0-data`](results/t1zg-l0-data.report.md) |
| `t1zg-b` | t1zg-a 的重跑，M50 修订版（MaxText `fd30fbfb`） | 只核与 `t1zg-a` 是否一致：KD 记录值 865 步逐位相同，其余最多差 2 ulp，判定相同 | [`t1zgb-l4`](results/t1zgb-l4.report.md) |
| `t1zk-a` | t1z 的设置，1000 步 | 上一代 1000 步比较的被测 | [`t1zk-l4`](results/t1zk-l4.report.md) |
| `t1w-a` | 同 `t1v-a`，1000 步，回放 `g3-a` 的数据 | 再上一代 1000 步比较的被测（M28 之后） | [`t1w-l4`](results/t1w-l4.report.md)、[`t1w-l0-settings`](results/t1w-l0-settings.report.md) |

TPU 同设置重跑逐位：T1 之前跑两组各两份 100 步（开关打开前；Megatron-Core 对齐开关打开后），45 个逐步量逐位相同（[`hy4-mini-tpu-determinism.json`](results/hy4-mini-tpu-determinism.json)）；之后各代的重跑核对见上表。

### 诊断运行（不进 L4）

TPU 训练程序的诊断模式（`train_hy4.py --diag-step1`，台账 J5）：回放 GPU 第 1 步的 batch，用训练里同一个梯度累积函数算第 1 步梯度、不更新。

| 代号 | 设置 | 用途 | 结果 |
|---|---|---|---|
| `t1diag11` | 第 1 步诊断，t1s 一代的五个舍入开关都关，另记索引器中间量（`debug_sow_indexer`） | KD 输入的「开关关」一组（对 CPU FP64 见 `t1-l1-kd-qkclip-vs-fp64.json`） | [`t1-layer0-kd-attribution.json`](results/t1-layer0-kd-attribution.json) |
| `t1diag15` | t1s 一代的五个舍入开关 | 第 1 步逐张量梯度对 `g1s-*` | [`t1s-step1-grad-vs-fp64.json`](results/t1s-step1-grad-vs-fp64.json) |
| `t1diag16` | 同上一行，改成 dtype float32、matmul_precision highest | 同设置的 TPU FP32 版第 1 步梯度（不作判定） | [`t1-step1-grad-vs-ref.json`](results/t1-step1-grad-vs-ref.json)、[`t1-step1-indexer-grad.json`](results/t1-step1-indexer-grad.json) |
| `t1diag17` | t1t 设置（M28 之前） | 第 1 步逐张量梯度对 `g2s-*` | [`t1t-step1-grad-vs-fp64.json`](results/t1t-step1-grad-vs-fp64.json) |
| `t1diag18b` | t1t 设置改成 dtype float32、matmul_precision highest、MoE 稠密实现 | 同设置的 TPU FP32 版第 1 步梯度（不作判定） | [`t1t-step1-grad-vs-ref.json`](results/t1t-step1-grad-vs-ref.json)、[`t1t-step1-ihc-experiments.json`](results/t1t-step1-ihc-experiments.json) |
| `tw-tadiag` | t1v 设置 + `transcendental_accuracy_highest`（台账 M32，MaxText `e8dee18f`） | 第 1 步诊断：KD 第 0 行与全部参数梯度（开关打开的一组） | [`t1-step1-transcendental-accuracy.json`](results/t1-step1-transcendental-accuracy.json) |
| `tw-taref` | 同 `tw-tadiag`，改成 dtype float32、matmul_precision highest（开关也打开） | 同设置的 TPU FP32 版（不作判定） | 同上 |
| `t1diag19` | t1t + `ihc_stream_sum_unrolled` | iHC 写法的定责实验（台账 M28 的依据） | `t1t-step1-ihc-experiments.json` |
| `t1diag20` | t1t + `XLA_FLAGS=--xla_allow_excess_precision=false` | iHC 写法的定责实验（对照） | 同上 |
| `t1vdiag` | t1v 设置（t1t + `ihc_stream_sum_unrolled`） | 第 1 步逐张量梯度对 `g2s-*` | [`t1v-step1-grad-vs-fp64.json`](results/t1v-step1-grad-vs-fp64.json) |
| `t1zdiag2` | t1z 设置 | 第 1 步逐张量梯度对 `g2s-*`；L3 的上一代 | [`t1z-step1-grad-vs-fp64.json`](results/t1z-step1-grad-vs-fp64.json) |
| `tz-off-rep` | 现行配置（2026-11-11 起，base2 关；MaxText `b87a55aa`），1000 步，第 1 步全量抓数；主 loss、MTP、grad_norm 逐位等于 `t1zz-a` | 现行 L4（`t1zo-l4`、`t1zo300-l4`）；L3：第 1 步动量反推训练步梯度（`export_train_grads.py`） | [`t1zo-l4`](results/t1zo-l4.report.md)、[`t1zo-step1-train-grad-vs-fp64.json`](results/t1zo-step1-train-grad-vs-fp64.json) |
| `tz-off-qk01` | 现行配置（base2 关）τ 0.1 低阈值 QK-Clip 30 步，记逐头最大 logit（GOLDEN `g6-qk-a`） | L1 低阈值 QK-Clip 第 1 步逐头最大 logit | [`t1zo-qkclip-maxlogit`](results/t1zo-qkclip-maxlogit.results.json) |
| `mz-diag1`、`mz-opt2` | base2 开的上一代配置（MaxText `fd30fbfb`） | 上一代 L3：`mz-opt2` 的第 1 步动量反推训练步梯度，`mz-diag1` 是诊断程序梯度（旁证）；优化器一步（只比算术，与 base2 无关）仍用 `mz-opt2` | [`t1zg-step1-train-grad-vs-fp64.json`](results/t1zg-step1-train-grad-vs-fp64.json)、[`t1zg-step1-grad-vs-fp64.json`](results/t1zg-step1-grad-vs-fp64.json) |
| `mz-g1`、`mz-g1q` | 现行配置 1000 步（`mz-g1q` 多记全局最大 logit） | L4 健康信号、QK-Clip 曲线、同设置重跑（对 `t1zg-b`） | [`t1mz-health`](results/t1mz-health.results.json)、[`t1mz-l4-qkclip`](results/t1mz-l4-qkclip.report.md)、[`t1mz-determinism`](results/t1mz-determinism.report.md) |
| `mz-dump30` | 现行配置 30 步，第 1、30 步全量抓数（MaxText `fd30fbfb`） | 抓数不改数值（对 `t1zg-b`） | [`t1mz-dump-determinism`](results/t1mz-dump-determinism.report.md) |
| `mz-sp849`、`mz-sp900`；GPU `g3x-a` | `g3x-a` 是 g3 的设置重跑；TPU 从它第 849、900 步的状态各走一步 | grad_norm 尖峰段同一状态单步 | [`t1-spike-same-state.json`](results/t1-spike-same-state.json) |
| `t1zl-a` | `t1zg-a` 再开 M51（精确 log） | 验证：主 loss 带符号均值 r 1.44，M51 不进对齐配置 | [`t1zla-l4`](results/t1zla-l4.report.md) |
| `tw-qk-a` | t1v 的设置，QK-Clip 阈值 τ 改成 0.1，30 步（让 QK-Clip 真的触发） | 低阈值 QK-Clip 对 GPU `g6-qk-a/b`（不进 L4） | [`t1-qk-clip-tpu.json`](results/t1-qk-clip-tpu.json) |
| `tw-qk-ref` | 同 `tw-qk-a`，改成 dtype float32、matmul_precision highest | 同设置的 TPU FP32 版（不作判定；最大 logit 对 CPU FP64 见 `t1-l1-kd-qkclip-vs-fp64.json`） | 同上 |
| `tw-elem2`、`tw-elem3`、`tw-elem4` | 逐元素函数的 ULP（合成输入与真实输入，单独编译）；`tw-elem3` 加 HIGHEST 版与 sqrt / 除法 / 1 − β^t，`tw-elem4` 加 MaxText 开关的写法（M32 + M37、M34）；结果文件里每条检查记来源运行 | 逐元素函数检查（不进 L4） | [`tpu-elementwise-ulp.json`](results/tpu-elementwise-ulp.json) |
| `t1v-fp32` | t1v 的设置改成 dtype float32、HIGHEST、MoE 稠密实现，回放同一批数据 30 步 | TPU FP32 版的 grad_norm 轨迹（不作判定；不进 L4） | [`t1-early-grad-norm-vs-fp32.json`](results/t1-early-grad-norm-vs-fp32.json) |
| `tw-opmini-a` | hy4-mini 逐算子隔离回放（`op_replay_mini.py`）：吃 GPU `g7-l1-a` 第 1 步抓到的每个算子的输入 | L1 逐算子（不进 L4） | [`t1-l1-op-tpu.json`](results/t1-l1-op-tpu.json) |
| `mz-l2pg2` | 现行配置单层回放（`layer_replay.py --param-grads`，MaxText `fd30fbfb`）：第 2、3、4 层吃 GPU `g7-l1-c` 的同一份层输入与余切，对 CPU FP64 单层重算比参数梯度 | L3 iHC 定位（[D11](DETAILS.md#d11)） | [`t1zg-l2-param-grads.json`](results/t1zg-l2-param-grads.json) |
| `tw-opmini-c` | 同 `tw-opmini-a`，开关全开（M33 + M37） | L1 逐算子（不进 L4） | [`t1-l1-op-tpu-r2.json`](results/t1-l1-op-tpu-r2.json) |
| `tw-opmini-d` | 同 `tw-opmini-c`，再开 M40 | L1 逐算子（不进 L4） | [`t1-l1-op-tpu-r3.json`](results/t1-l1-op-tpu-r3.json) |
| `tw-optim8` | MaxText 的优化器单独跑：吃 GPU `g2o-a` 第 1–8 步的梯度与主权重 | 优化器连续 8 步回放（不进 L4） | [`t1-optim-replay-tpu.json`](results/t1-optim-replay-tpu.json) |
| `tw-optim8b`、`tw-optim8c` | 同 `tw-optim8`，再开 `muon_adam_accurate`（M34）：8b 是第一版（MaxText `64652880`），8c 是最终版（`7c95fabe`） | 同上（不进 L4） | [`t1-optim-replay-adam-acc-tpu.json`](results/t1-optim-replay-adam-acc-tpu.json)、[`t1-optim-replay-adam-acc-r2-tpu.json`](results/t1-optim-replay-adam-acc-r2-tpu.json) |
| `tw-l2-a` | L2 逐层隔离回放（`layer_replay.py`） | 逐层回放（台账 J17、J19）；在两台主机上失败，没有结果 | — |
| `tw-l2-b` | 同 `tw-l2-a`，两台主机；暴露了跨主机全局数组被闭包捕获的问题（台账 J22） | 逐层回放；同样失败（台账 J22），没有结果 | — |
| `tw-l2-f` | 同 `tw-l2-a`，修好多主机问题后，T1 当时的全部开关（MaxText `c902c147`） | 逐层回放（多主机跑通） | [`t1-l2-layer-tpu.json`](results/t1-l2-layer-tpu.json) |
| `tw-l2-g` … `tw-l2-j` | 同 `tw-l2-f`，MaxText `3741e818`；g 是 `t1zu-a` 的全部开关（M40、M41 都开），h / i / j 是 M40 / M41 的另外三种组合 | 逐层回放对 CPU FP64 单层重算（g，L2 的判定）；M40 / M41 的诊断（g–j） | [`t1-l2-layer-vs-fp64.json`](results/t1-l2-layer-vs-fp64.json)、[`t1-l2-m40-m41-tpu.json`](results/t1-l2-m40-m41-tpu.json) |
| `t1tdump3` | t1t 设置的训练运行，第 300 步做全量抓数（更新之后的主权重） | param_norm 定位：对 GPU `g2d-a` 逐张量比（不进 L4） | [`t1-param-norm-locus.json`](results/t1-param-norm-locus.json) |

### 第 1 步的 CPU FP64 参照（L3 定责用，不是任何一侧的运行）

GPU golden 同一套 Megatron-Core 代码（`frameworks/megatron/step1_ref.py`，台账 J34）在 CPU 上全程 FP64 跑第 1 步：同一份初始权重（`tensor_sha256.json` 的 sha256 `1d869124…`）、tokens.npy 第 0–15 行（与 `g2s-*` 的 `tokens_sha256_first_rows` 逐字节相同）、同一套 loss 语义。三份产物：FP64 全批（两次重跑 205 个张量逐位相同）、同一代码的 FP32 全批（自检）、第一行灌 GPU `g7-l1-a` 的 DSA / 路由选择的 FP64（灵敏度）。产物较大，不入库，比较结果见 [`t1z-step1-grad-vs-fp64.json`](results/t1z-step1-grad-vs-fp64.json)、[`t1v-step1-grad-vs-fp64.json`](results/t1v-step1-grad-vs-fp64.json)、[`t1t-step1-grad-vs-fp64.json`](results/t1t-step1-grad-vs-fp64.json)、[`t1s-step1-grad-vs-fp64.json`](results/t1s-step1-grad-vs-fp64.json)。

## 噪声：只用 GPU

- L4：同一代 GPU 8 份两两 28 对（`g1-*` 给前三代，`g2-*` 给现行 `t1zo300-l4`、上一代 `t1zga300-l4`、`t1mz300-l4-qkclip` 与 `t1t-l4`、`t1v-l4`、`t1z-l4`、`t1zr300-l4`、`t1zs300-l4`、`t1zt300-l4`、四份 `t1zu*300-l4`、`t1zv300-l4`、`t1zw300-l4`、`t1zz300-l4` 与 `t1zla300-l4`，`g3-*` 给上一代 `t1zga-l4`、`t1zgb-l4`、`t1mz-l4-qkclip` 与 `t1u-l4`、`t1w-l4`、`t1zk-l4`、`t1zr-l4`、`t1zs-l4`、`t1zt-l4`、四份 `t1zu*-l4`、`t1zv-l4`、`t1zw-l4`、`t1zz-l4` 与 `t1zla-l4`），每一对算一次三个判定量，每个判定量取 28 对里的最大值（`PROTOCOL.md` 第 3 节）。
- L4 现行 `t1zo-l4`（1000 步）用 `g3-a … g3-p` 16 份两两 120 对（2026-11-11 起）。
- L3：GPU 第 1 步参照三份两两（`g1s-*` 给 t1s 一代，`g2s-*` 给 t1t、t1v、t1z、t1zg 与现行 `tz-off-rep`）。
- TPU 同设置重跑逐位相同只作记录，TPU 只改执行参数的扰动运行只作诊断，都不进尺子。
- 各次比较的目录在 [`alignments/01-mini/catalogs/`](catalogs/)（`t1v-l4.yaml`、`t1w-l4.yaml`、`t1t-l4.yaml`、`t1u-l4.yaml`、`t1s-l4.yaml`、`t1-l4-nozm.yaml`、`t1-l4.yaml`）。现行 `t1zo-l4.yaml`（1000 步）与 `t1zo300-l4.yaml`（前 300 步，被测都是 `tz-off-rep`）；上一代现行 `t1zga-l4.yaml`、`t1zga300-l4.yaml`（被测 `t1zg-a`），重跑 `t1zgb-l4.yaml`；上一代 `t1zw-l4.yaml`、`t1zw300-l4.yaml`（被测 `t1zw-a`）；验证 `t1zz-l4.yaml`、`t1zz300-l4.yaml`（被测 `t1zz-a`）；再上一代 `t1zu-l4.yaml`、`t1zu300-l4.yaml`（被测 `t1zu-a`）；诊断代 `t1zv-l4.yaml`、`t1zv300-l4.yaml`（被测 `t1zv-b`）；更早一代 `t1zr-l4.yaml`、`t1zr300-l4.yaml`（被测 `t1zr-a`）；诊断代 `t1zs-l4.yaml`、`t1zs300-l4.yaml`、`t1zt-l4.yaml`、`t1zt300-l4.yaml`，诊断运行 `t1zub*`、`t1zuc*`、`t1zud*` 各一对（1000 步与前 300 步）；上一代 `t1z-l4.yaml`（300 步，被测 `t1z-a`）与 `t1zk-l4.yaml`（1000 步，被测 `t1zk-a`）。
