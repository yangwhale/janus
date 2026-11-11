[janus](../../README.md) › [混元 4](README.md) ｜ 知识库索引

# 混元 4 · 知识库索引

给人读的只有两份：[总览](dashboard/dashboard.md)和 [mini 对齐报告](alignments/01-mini/ALIGNMENT-REPORT.md)。下面这些文件是知识库，用来查依据、回答具体问题。知识库保留术语，看不懂的词先查本页的[术语对照表](#术语对照表)。

## 文件

| 文件 | 回答什么问题 | 什么时候查 |
|---|---|---|
| [mini 对齐细节](alignments/01-mini/DETAILS.md) | 带开关名与编号的生产建议、8 条签字条件逐条、判法变更、代号、各级现状、偏差状态索引（D 编号，机制与读数在根源一览）、开关表、测量口径、证据索引、跨尺寸看 | 报告里某个结论要追到数字与出处时 |
| [总览细节](dashboard/detail.md) | 每一格的实测值、门槛、离门槛多远；「读法」（N、r、R 等怎么算） | 总览里某一格要看推导时 |
| [mini 检查清单](alignments/01-mini/CHECKS.md) | 每一级比了哪些量、用什么门槛、逐模块的结果 | 想知道某个模块或某个量比没比、怎么比时 |
| [根源一览](ROOT-CAUSES.md) | 偏差的唯一主表。每个对不齐的来源：机制、修法、修前修后、定责依据、生产建议、代价；还没查明的与已接受的 | 问「这个差是哪来的、怎么修的」时 |
| [TPU 与 GPU 差异清单](../../docs/reference/tpu-vs-gpu-differences.md) | 两种硬件训练数值不一样的地方，逐条原理、量法、开关、生产建议 | 迁移别的模型、或问「TPU 上哪里会和 GPU 不一样」时 |
| [排查案例](cases/README.md) | 一个问题一篇：现象、机制、证据、结论 | 想看某个问题是怎么查清的时 |
| [比较协议](PROTOCOL.md) | 事先登记的比较规则、现行口径、每次改动的变更记录（含「已接受」的登记） | 问「这条规则是什么时候、为什么改的」时 |
| [改动台账](CHANGES-LEDGER.md) | 两边对原版代码的每一处改动（M 是 MaxText 一侧，G 是 GPU 一侧，J 是工具）与核验 | 查某个开关具体改了什么代码时 |
| [mini 运行登记](alignments/01-mini/goldens.md)、[`runs.yaml`](alignments/01-mini/runs.yaml) | 每一份运行是怎么来的、代号是什么 | 看到一个运行代号不认识时 |
| [mini 配置对照](alignments/01-mini/config-alignment.md)、[设置](alignments/01-mini/setup.md) | 两边每一项配置怎么对上 | 查某个设置两边是否一致时 |
| [GPU 自比](alignments/01-mini/gpu-noise/README.md) | GPU 自己重跑的波动、计划中的故障演练 | 问「判定用的波动上限从哪来」时 |
| [TPU 自比](alignments/04-tpu-self/README.md)、[实验矩阵](alignments/04-tpu-self/MATRIX.md)、[B1 梯度的跨卡归约](alignments/04-tpu-self/b1-grad-reduction.md) | 同一模型只改一样东西（并行、编译开关、代码写法、精度设置、版本、硬件），TPU 结果对 TPU golden 变不变；矩阵里是每次实验（臂）的改法、结果、机制（HLO 差在哪、第 1 步哪些梯度变）、出处与已修的 bug；B1 那页逐个查了纯数据并行与 FSDP 每次梯度跨卡求和的数据类型与舍入次数 | 问「换并行方式 / 改了代码 / 换 libtpu 后 TPU 自己变不变、算不算 bug」时 |
| [GPU 扩展](alignments/05-gpu-scale/README.md) | GPU 换机器数、换 DP / EP / PP 切法后对单机 golden 变不变；存档续跑是否逐位 | 问「多机 / 换并行后 GPU 自己变不变」「存档续跑可靠吗」时；多机怎么起见 [Megatron 一侧](frameworks/megatron/README.md#多机并行与存档) |
| [真实宽度三层](alignments/06-rw3/README.md) | hy4-preview 第 0–2 层 + MTP（3.21e10 参数），TPU 回放多机 GPU golden 的 1000 步对齐 | 问「真实宽度、带 shared 索引器层时 TPU 和 GPU 对不对得上」「这个尺寸要几片 v5p」时 |
| [历代比较](dashboard/history.md) | 各代 TPU 配置的判定 | 问「某个开关加上前后差了多少」时 |
| [支线总览](dashboard/other-sizes.md)、[真实权重截层](alignments/02-real-weight/README.md)、[中等尺寸](alignments/03-mid/README.md)、[tiny](alignments/00-tiny/README.md) | 其他尺寸的结果与报告 | 问「换尺寸、换真实权重后结论还在不在」时 |
| [MaxText 一侧](frameworks/maxtext/README.md)、[Megatron 一侧](frameworks/megatron/README.md) | 两边的配置、训练与回放、抓数格式 | 要复现或改运行时 |

## 术语对照表

左边的术语沿用混元 3、混元 4 对齐文档的叫法，各页正文和表头都这么写，右边是白话解释。只有 TPU 自比新造的几个叫法（臂、底、领头），在给人读的页里换成了实验、对照、差得最多的曲线。全仓库通用的定义与出处在[术语表](../../docs/glossary.md)。

**参照与尺子**

| 术语 | 白话 |
|---|---|
| golden | 基准运行：拿来对照的那次运行。GPU ↔ TPU 对齐里是 GPU 上同设置的标准训练（基线是其中事先指定的一份）；TPU 自比里是一次确认过、换机器重跑每一位都相同的 TPU 运行 `ts-g-a` |
| 被测 | 拿来和 golden 比的那次运行 |
| 份 / 对 | 一份是一次运行；两份配成一对比较，8 份两两配对得 28 对，16 份得 120 对 |
| N、噪声上限、尺子 | 同一个东西：GPU 自己同设置重跑能差多少（几份两两比，取最大值）。按窗口取：mini 现行比较与 TPU 自比都是前 300 步 `g2-a … g2-h` 8 份 28 对、1000 步 `g3-a … g3-p` 16 份 120 对（1000 步换 16 份：TPU 自比 2026-11-10 起，mini 现行比较 2026-11-11 起）；真实权重截层、mid 各自 8 份 28 对。2026-11-10 前 TPU 自比与 GPU 扩展的文档把它叫「刻度」；现在「刻度」只指 FP32 记录值的分辨率（如混元 3 的 param_norm 在 3042 附近一个刻度是 2.44e-4） |
| r | 差是尺子的几倍：TPU 对 GPU（或被测对 golden）的差 ÷ N；≤ 1 算对上，1–2 待定（TPU 自比叫「略超」），> 2 算不过 |
| R、R 比 | 谁离精确值远：TPU（或被测）离精确值的距离 ÷ GPU（或参照）离精确值的距离；≤ 1.2 算不比对方差 |
| 精确值、FP64 | 同一套 GPU 代码在 CPU 上用 64 位浮点重算的结果 |
| rel_L2 | 第 1 步每个梯度张量的差：‖a − b‖ ÷ ‖b‖，一个张量出一个数 |
| 窗口 | 统计哪几步。`1:301` 是第 2–300 步（第 1 步单独报） |
| 带符号均值 / MARE / 最大单步 | 一段窗口里逐步相对差的三个统计量：平均偏多少（正负不抵消的那部分）/ 平均晃多少（取绝对值再平均，即平均绝对相对误差）/ 最坏一步差多少 |
| ulp、u | 最小精度单位：一个浮点数和它下一个可表示的数之间的距离；u = 2^-8 是 16 位浮点一次舍入的相对误差 |
| 分辨率下限 | 两边误差都小于一次 16 位舍入时，比远近没有意义，直接算过 |

**判定**

| 术语 | 白话 |
|---|---|
| 逐位相同 | 每一位都相同 |
| 过 / 待定 / 不过；在噪声内 / 略超 / 超 | r ≤ 1 / r 在 1–2 之间 / r > 2。TPU 自比的四档是逐位相同、在噪声内、略超、超；第 1 步有张量丢了精度也记超。2026-11-10 前生成的结果文件里，`verdict.label` / `verdict.band` 写的是旧档名「在刻度内」「超出刻度」，对应现在的「在噪声内」「超出噪声上限」；结果文件是历史证据不改，`build_tables.py` 读的时候按新叫法显示 |
| 换了顺序（已定责） | 只是加法先后不同：被测离精确值不比参照远（R ≤ 1.2），不是 bug |
| 丢了精度（未定责） | 被测离精确值更远（R > 1.2）：某处多舍入了一次，或在低精度里做了累加 |
| 定责 | 查明一个差别来自哪边、为什么（换了顺序还是丢了精度、是哪个机制） |
| 已接受 | 查明后决定不修，写进比较协议的变更记录 |
| 留一法 | 拿 GPU 重跑自己试这把尺子：轮流一份当被测、一份当 golden，尺子用其余几份重算，看纯噪声会得出多大的 r |
| 多重比较 | 一次比较要看二十几个数（8 或 9 项检查 × 3 个统计量），任何一个过 1 就记略超；看的数越多，纯噪声碰上一个过线的机会越大 |
| 和 GPU 重跑分不开（旧写法：与重跑噪声不可区分） | 曲线最差 r 不超过 GPU 留一法的最大值，第 1 步也没有丢精度的张量；是事后解读，不改判档 |
| 事后判法 / 前瞻验证 | 看到结果之后才改的判法 / 把改好的判法事先固定，用到还没看过结果的新运行上，看它站不站得住 |
| 签字 | 宣布「TPU 和 GPU 对齐了」，要同时满足 8 条签字条件 |

**TPU 自比（04）的叫法**

| 术语 | 白话 |
|---|---|
| 臂 | 一次实验：从 golden 出发只改一样东西的那次运行。「33 个臂」就是 33 次实验 |
| 底 | 对照：有的改动要连带改一样别的东西才跑得起来，先跑一次只改连带项的运行，再拿被测和它比 |
| 领头 | r 最大的那条曲线和那个统计量，即差得最多的地方 |
| 灵敏度 | 故意埋一个已知大小的 bug，逐档加大，看多大才抓得到 |
| 正对照 / 负对照 | 故意改坏、应当被抓到的运行 / 什么都不该变的运行（如换一组机器重跑） |
| HLO、程序指纹 | HLO 是 XLA 编译出来的程序；程序指纹是去掉元数据与源码位置后对它算的哈希，指纹相同就是同一个程序，结果一定一模一样 |
| excess precision、省舍入 | 编译器把本该舍成 16 位的中间结果留在 32 位 |
| 芯片·秒 | 步时 × 芯片数，用来比多花了多少算力 |
| 先加后传 / 先传后加 | 词嵌入梯度的两种组织：纯数据并行是每张卡先把本卡 token 的梯度加进整张表、再跨卡求和；FSDP（表按隐藏维切）是先把原始 token 梯度按列换到管那一段的卡上、再一次加完（[案例 14](cases/14-ddp-fsdp-embedding.md)） |
| 只切词嵌入表 | `embedding_shard_over_data`：其余参数纯数据并行，只有词嵌入表沿数据并行轴切开，让它走先传后加（[案例 14 第二阶段](cases/14-ddp-fsdp-embedding.md#第二阶段只把词嵌入表沿数据并行轴切开)） |
| 按形状算的字节 | 从训练步 HLO 读出的集合通信字节：结果形状 × 字节宽 × 每步次数；不是线上实测，只适合互相比 |

**模型与训练**

| 术语 | 白话 |
|---|---|
| L0 / L1 / L2 / L3 / L4 | 地基（设置、数据、权重、配置）/ 单算子 / 单层 / 第一步整网 / 训练曲线 |
| T1 / T2 / T1m | mini 的对齐配置 / 真实权重截层的对齐配置 / 中等尺寸的对齐配置 |
| M / G / J / D 编号 | 改动台账里 MaxText 一侧 / GPU 一侧 / 工具的改动；对齐细节里的偏差台账行 |
| t1zg-a、g3-a、mz-*、ts-* 等 | 运行代号，查 [mini 运行登记](alignments/01-mini/goldens.md)；`ts-*` 是 TPU 自比的运行，查 [04 `runs.yaml`](alignments/04-tpu-self/runs.yaml) |
| DP / FSDP / TP / EP / PP / CP | 并行方式：数据并行（每张卡一整份模型，分数据）/ 把参数切开分到各卡的数据并行 / 张量并行（一个矩阵乘切到几张卡）/ 专家并行（MoE 的专家分到各卡）/ 流水线并行（按层分段）/ 上下文并行（一条序列切到几张卡） |
| 微批、梯度累积 | 一步的数据分几小批先后算，梯度加起来再更新；「微批 1 × 累积 2」是每批 1 条、累积 2 次 |
| iHC、hc_pre、hc_scale、hc_base、hc_fn | 超连接：多路残差流的混合；这几个是混合用的小参数（缩放、偏置、混合权重） |
| DSA、索引器、top-k | 稀疏注意力：索引器给 token 打分，只看分最高的 k 个 |
| KD | 索引器蒸馏 loss：让索引器的打分去学完整注意力的分布 |
| MTP、enorm、hnorm | 多 token 预测层；它里面的两个归一化层 |
| sink | 注意力每个头的一个可学习偏置项，不参与打分 |
| QK-Clip | 注意力打分过大时自动缩小相关权重的训练技巧 |
| base2 | 注意力的指数用 2 为底来算 |
| remat、重算 | 反向时重新算前向，省显存 |
| SparseCore | TPU 芯片上的专用小核，原本用于嵌入查表，也能接手集合通信；「all-reduce 交给 SparseCore」即让它来做跨卡求和 |
| Welch t | 比两组数的平均值有没有系统差别的统计量；\|t\| 小于约 2 时看不出有差别 |

## 文件夹地图

| 路径 | 放什么 |
|---|---|
| `ROOT-CAUSES.md`、`PROTOCOL.md`、`CHANGES-LEDGER.md` | 根源一览、比较协议（事先登记与变更记录）、改动台账 |
| `dashboard/` | `build.py` 从各尺寸的 `results/` 生成总览 `dashboard.md`、支线总览 `other-sizes.md`、历代比较 `history.md` 与各尺寸的 `CHECKS.md`；生成逻辑与混元 3 共用 `janus/report/overview.py` |
| `alignments/NN-<尺寸>/` | 一个尺寸一次对齐：报告、检查清单、配置对照、运行登记、`catalogs/`、`results/`；mini 另有 `gpu-noise/`（GPU 一侧自比） |
| [`frameworks/maxtext/`](frameworks/maxtext/README.md)、[`frameworks/megatron/`](frameworks/megatron/README.md) | TPU 侧 / GPU 侧：配置、训练与回放、提交作业、桥与核对 |
| `frameworks/hf/`、`model/`、`analysis/`、`results/` | HF 参照模型；两边共用的转换与核对；诊断脚本；跨尺寸的工具输出 |

## 代码基准

- TPU：`maxtext` 的混元 4 集成分支（每次运行的 commit 写在各尺寸的 `runs.yaml` 与报告里）。GPU ↔ TPU 的现行运行都在 jax 0.11.1、libtpu 0.0.46 上，逐项见各尺寸 `setup.md`「软件版本」（如 [mini](alignments/01-mini/setup.md#软件版本)）；换 libtpu 要按 [`libtpu-052.md`](alignments/04-tpu-self/libtpu-052.md) 重测。
- GPU：Megatron-Core 0.20 + TE（NeMo 26.10 镜像）之上的 janus 训练入口 `frameworks/megatron/train_hy4mg.py`。镜像里 CUDA 13.4、Megatron-Core 0.20.0 有运行记录；TE 2.20.2、cuBLAS 13.8 是按同一镜像推断的。

运行代号见 [mini 对齐细节「代号」](alignments/01-mini/DETAILS.md#代号)与 [`PROTOCOL.md`](PROTOCOL.md) 第 1.1 节。

## GPU 多机与并行：要点与出处

| 要点 | 出处 |
|---|---|
| 机间网络不用另装：NeMo 26.10 镜像自带 NCCL 2.31.2 和 GCP 网络插件，Pod 挂上 8 张 RDMA 网卡（DRA）、不关插件，NCCL 就用 gIB。另装 gIB 的初始化容器会把 NCCL 换成 2.30.4，torch 2.14 起不来（缺符号 `ncclGetPeerDeviceLeInfo`） | 两机冒烟日志 [`nccl-2node-smoke.txt`](alignments/05-gpu-scale/results/nccl-2node-smoke.txt) |
| 两机 16 卡 all_reduce（512 MiB，BF16）总线带宽 5.7e2–5.8e2 GB/s，all_to_all 每卡 86 GB/s | 同上 |
| PP 的 P2P 形状：Megatron-Core 按 `hidden_size` 算，混元 4 层间是 4 路流 [s, b, 4H]。Megatron-Core 自己的 mHC 开关 `enable_mhc_connections` 会在 TransformerBlock 里另做展开与收拢，且 GPTModel 不许它与 PP + MTP 同开，不能借用；改用 `variable_seq_lengths`，P2P 先交换形状 | `hy4_mcore.py` 开头；Megatron-Core 0.20 `pipeline_parallel/schedules.py` 的 `_get_pipeline_hidden_size` |
| PP 切分点要落在 full 索引器层上：shared 层复用前一个 full 层的 top-k，top-k 不跨段传。hy4-mini 只能切在第 1、5 层前；完整模型 full 层是 0、1、5、9……，可切在 1 + 4k 处 | `hy4_mcore.check_pp_split` |
| 1F1B 加整层重算时 shared 层要按 microbatch 号取 top-k：第 m 个 microbatch 的反向在第 m+1… 个的前向之后，只按层号存会取错 | 改动台账 G32 |
| 换并行方式吃的是同一批数据：样本号只由步号、microbatch 号、DP rank 决定，全局批 16 条不变 | [`data-same-batches.json`](alignments/05-gpu-scale/results/data-same-batches.json) |
| 多机重跑本身的波动比单机略大：2 台 DP 16 + EP 4 同设置 8 份两两 28 对的尺子，是单机 8 份尺子的 1.0–1.3 倍（主 loss、grad_norm、param_norm）。所以拿单机尺子量多机，8 份里有 3 份落在 1–2 倍（最大 1.76 倍，param_norm 最大单步）；反过来拿多机尺子量单机 `g3-a` 也有 2 项 1–2 倍。两组对 `g3-a` 的带符号均值没有系统偏移（Welch \|t\| ≤ 1.3） | [`yardstick-multi-vs-single.json`](alignments/05-gpu-scale/results/yardstick-multi-vs-single.json)、[`groups-single-vs-multi.json`](alignments/05-gpu-scale/results/groups-single-vs-multi.json)、[05 现状](alignments/05-gpu-scale/README.md) |
| 多机尺子怎么建：同一多机设置跑 8 份（同一个 JobSet 里串着跑，每份换端口），两两 28 对进 `multi-node-self.yaml` 那样的目录；判多机被测时用多机尺子，判「换了机器数有没有偏」时用 `gpu_scale_compare.py groups` 比两组对同一参照的带符号均值，不要拿单机尺子直接判多机 | [`catalogs/multi-node-self.yaml`](alignments/05-gpu-scale/catalogs/multi-node-self.yaml)、[`analysis/gpu_scale_compare.py`](analysis/gpu_scale_compare.py) |
| 确定性模式（`--deterministic`）几乎不慢：hy4-mini 单机第 3–20 步中位数 1.10 s / 1.09 s（`ck-a` / `ck-b`），默认设置 1.09 s（`sn-a`） | 各运行 `metrics.jsonl` 的 `step_time` |
| 存档是每个 rank 一个文件，续跑要并行设置相同、落在同一组机器上（或先把存档拷过去）；不能换并行设置续跑 | `train_hy4mg.py` 的 `save_ckpt` |
| 真实宽度第 0–2 层 + MTP：2 台 DP 16 + EP 16 每卡峰值 91.5 GiB、2.12 s / 步，对单机 DP 8 + EP 8（6 份两两 15 对的尺子，前 30 步）5 份里 3 份全在噪声内、2 份各 1 项 1.03 倍，两组没有系统偏移（Welch \|t\| ≤ 1.7）；单机 115.3 GiB、3.68 s / 步；2 台 PP 2（层 0 / 层 1–2 + MTP）+ EP 8 末段每卡 105.7 GiB、3.42 s / 步（两段不均） | `rw-a`、`rws-*`、`rw-pp2` 的 `metrics.jsonl`；[`rw-a-vs-rws.report.md`](alignments/05-gpu-scale/results/rw-a-vs-rws.report.md) |
| 真实宽度第 0–3 层 + MTP：4 台 PP 2（层 0 / 层 1–3 + MTP）+ DP 16 + EP 16 每卡峰值 93.9 GiB、2.38 s / 步；单机 DP 8 + EP 8 是 135.3 GiB、4.11 s / 步 | [`rw4-pp2-summary.json`](alignments/05-gpu-scale/results/rw4-pp2-summary.json)、`p-l4mtp` 的 `metrics.jsonl` |
| 多机 golden `rw1k-a`：真实宽度第 0–2 层 + MTP，2 台 DP 16 + EP 16，1000 步，batch 存在 `gs://<POC 桶>/janus/hunyuan4/gpu-scale/rw1k-a/data/batches/`（16 个 rank × 1 个 microbatch）。TPU 回放的读法（`frameworks/maxtext/replay.py`）核过 1000 步全部单元；回放命令 `STAGE=D3 YML=hy4-t2-d1.yml GOLDEN=…/rw1k-a bash run.sh`。D3 检查点已转换，TPU 1000 步 base2 开 / 关两臂回放已跑完（见下一节「真实宽度三层」） | [05 真实宽度截层](alignments/05-gpu-scale/README.md#真实宽度截层)、[`rw1k-a-replay-check.txt`](alignments/05-gpu-scale/results/rw1k-a-replay-check.txt) |
| 存档续跑在真实宽度上也逐位：第 0–2 层 + MTP、2 台 DP 16 + EP 16，16 个 rank 共 14512 个状态张量 sha256 全同。存档是每个 rank 一个 `torch.save` 文件，训练入口日志的 `CKPT_SAVED` 记耗时与全部 rank 的字节数 | [`ckpt-resume-rw.json`](alignments/05-gpu-scale/results/ckpt-resume-rw.json) |
| 全局批 16 条、DP 16 时每卡每步只有 1 个 microbatch，PP 没有 microbatch 填流水线：hy4-mini 4 台 PP 2 + DP 16 是 0.58 s / 步，2 台 DP 16 是 0.61–0.64 s；PP 用来放下更多层，不是提速 | 05 [`README.md`](alignments/05-gpu-scale/README.md) 的步时 |
| `mn-a`、`mn-ep16`、`mn-pp2` 的 `meta.json` 入库时把 `side` 从 `gpu-mn-*` 改成 `gpu-perturb-mn-*`（janus 要求的前缀），其余未改；之后的运行直接用新前缀 | 05 [`runs.yaml`](alignments/05-gpu-scale/runs.yaml) |

## GPU 1000 步尺子扩到 16 份：要点与出处

| 要点 | 出处 |
|---|---|
| GPU 1000 步重跑从 8 份扩到 16 份（`g3-a` … `g3-p`，同一代码包、同一参数），两两 120 对的上限是 8 份 28 对的 1.0–2.1 倍；放大最多的是带符号均值（梯度范数 1.84 倍、路由偏置均值 2.07 倍）。梯度范数的 1.84 是 2026-11-10 起的口径：逐步相对差以参照作分母（[PROTOCOL](PROTOCOL.md#7-变更记录) 变更记录 2026-11-10「曲线分母统一」），见结果文件的 `rejudge_2026_11_10` 一节；之前以两边均值作分母时是 1.70。份数少时上限会低估噪声 | [`g3x16-noise-vs-8.json`](alignments/01-mini/gpu-noise/results/g3x16-noise-vs-8.json) |
| 当时的 mini 现行比较 `t1zga-l4` 换用 16 份上限后 9 项全过（8 份时过 7、待定 2）：grad_norm 最差判定量 1.23（8 份时是带符号均值）→ 0.995（16 份时是 MARE，贴线）；2026-11-10 统一分母前是 1.25 → 0.999、路由偏置标准差 1.14 → 0.91，其余 7 项也都变小或不变。当时只作旁注；2026-11-11 起 PROTOCOL 定 1000 步一律用 16 份（前 300 步仍是 8 份 28 对），现行比较已换成 `tz-off-rep` | [`t1zga-l4-n16.report.md`](alignments/01-mini/gpu-noise/results/t1zga-l4-n16.report.md)、目录 [`t1zga-l4-n16.yaml`](alignments/01-mini/gpu-noise/catalogs/t1zga-l4-n16.yaml)；8 份版本 [`t1zga-l4.report.md`](alignments/01-mini/results/t1zga-l4.report.md) |
| TPU 自比的 1000 步臂已全部换成 16 份重判：8 份时 r 2–3 的灰区六臂落到 1.19–1.64，GPU 16 份留一法最大值 1.82，全部与重跑噪声分不开 | [04 实验矩阵「L 长跑」](alignments/04-tpu-self/MATRIX.md#l-长跑300-步在噪声内的改动1000-步会不会显形)、[`gpu-rerun-loo-r-1000.json`](alignments/04-tpu-self/results/gpu-rerun-loo-r-1000.json) |
| 「留一法最大值」有两种口径，不能互比：TPU 自比的是整次比较（300 步 8 项、1000 步 9 项检查 × 3 个量）最差那一项、golden 也轮换，16 份时 1.82；GPU 一侧的是每条检查各自的最大值、golden 固定 `g3-a`，16 份时 1.0–1.35 | [PROTOCOL](PROTOCOL.md) 变更记录 2026-11-10 |

## mini 上关 base2 为什么离 GPU 更远：要点与出处

| 要点 | 出处 |
|---|---|
| 换 GPU 16 份尺子（`g3-a` … `g3-p`，120 对）重判 1000 步：base2 开 `t1zw-a` 9 项全过（主 loss r 0.90，MARE）；关 `t1zz-a` 过 7、待定 2，主 loss r 1.55（带符号均值 7.21e-6，N 4.65e-6），8 份时是 1.91 | [`t1zwa-l4-n16`](alignments/01-mini/gpu-noise/results/t1zwa-l4-n16.report.md)、[`t1zza-l4-n16`](alignments/01-mini/gpu-noise/results/t1zza-l4-n16.report.md) |
| 两侧各补一份只改执行方式的重跑（关延迟隐藏调度 `--xla_tpu_enable_latency_hiding_scheduler=false`）：开那份 `ts-c-nolhs-1k` 主 loss r 1.24（MARE），关那份 `tz-off-nolhs` r 1.06（MARE），都是过 7、待定 2。四份主 loss 带符号均值（对 `g3-a`）：开 1.36e-6、4.53e-6，关 7.21e-6、2.96e-6，两组交叠；对 16 份各自算的中位数也交叠（开 1.62e-6、4.79e-6，关 7.48e-6、3.22e-6） | [`tsnolhs-l4-n16`](alignments/01-mini/gpu-noise/results/tsnolhs-l4-n16.report.md)、[`tzoffnolhs-l4-n16`](alignments/01-mini/gpu-noise/results/tzoffnolhs-l4-n16.report.md) |
| **结论：mini 上「关 base2 离 GPU 更远」是轨迹偶然，不是开关的效果。** 开、关各 4 份 1000 步（原一份、关延迟隐藏调度一份、vmem 上限 32 / 64 MiB 各一份，都只改执行方式），16 份尺子下主 loss r：开 0.90、1.24、1.02、1.69（中位数 1.13），关 1.55、1.06、0.93、0.95（中位数 1.01），两组完全交叠。最初的 0.36 → 1.91 恰好是开里最低的一份对关里最高的一份 | [`mini-base2-loo.json`](alignments/01-mini/gpu-noise/results/mini-base2-loo.json) |
| GPU 16 份留一法：`g3-b` … `g3-p` 轮流当被测、对 `g3-a`、噪声用其余 15 份两两 105 对，主 loss r 0.77–0.97（中位数 0.82），15 份都过。8 份 TPU 里 6 份高过全部 15 份 GPU，开、关各 3 份 | 同上 |
| 主 loss 第 2–1000 步对 GPU 逐步均值的相对差：GPU 每份对其余 15 份 −2.63e-6 到 +2.34e-6（标准差 1.33e-6）；TPU 开 +1.58e-6、+4.75e-6、+3.12e-6、+8.09e-6，关 +7.43e-6、+3.18e-6、+2.58e-6、+3.77e-6。组均值开 +4.39e-6、关 +4.24e-6，几乎相同；8 份 TPU 都偏高，是与 base2 无关的共同偏移（约 3 个 GPU 标准差），TPU 8 份之间的散布 2.3e-6 也比 GPU 大 | 同上 |
| **2026-11-11 起 mini 现行比较 = `tz-off-rep`（base2 关）**，GPU 16 份 120 对尺子：1000 步过 7、待定 2（主 loss 全程 r 1.55、后段 1.52），MTP 1.00、KD 0.77、grad_norm 0.81、param_norm 0.55、路由偏置均值 0.20、标准差 0.77；前 300 步过 7、待定 2（主 loss 1.10、grad_norm 1.23）。选 `tz-off-rep` 而不是只改执行方式的几份，是因为它是默认执行方式的那份原始运行，挑判档好的那份就是事后挑选。第 1 步训练步梯度 177 个超出 GPU 噪声、6 个比 GPU 远（5 个 iHC＋MTP enorm），其中 3 个 iHC 是关 base2 后新出现的；低阈值 QK-Clip 第 1 步逐头最大 logit 第 4 层 R 1.55（base2 开时 0.94）；L4 QK-Clip 1000 步 1.09、前 300 步 1.05；健康 6 条正常；同设置重跑 47 个逐步量逐位相同 | [`t1zo-l4`](alignments/01-mini/results/t1zo-l4.report.md)、[`t1zo300-l4`](alignments/01-mini/results/t1zo300-l4.report.md)、[`t1zo-step1-train-grad-vs-fp64.json`](alignments/01-mini/results/t1zo-step1-train-grad-vs-fp64.json)、[`t1zo-qkclip-maxlogit`](alignments/01-mini/results/t1zo-qkclip-maxlogit.results.json)、[`t1zo-l4-qkclip`](alignments/01-mini/results/t1zo-l4-qkclip.report.md) |
| mini 8 份 TPU 主 loss 全部偏高约 +4.3e-6、与 base2 无关：查下去的结果见下一节「[mini 主 loss 共同偏高](#mini-主-loss-共同偏高要点与出处)」 | [`mini-base2-loo.json`](alignments/01-mini/gpu-noise/results/mini-base2-loo.json) 的 `loss_vs_gpu_mean` |
| 只改执行方式的重复确实换了轨迹：`tz-on-vm32b` 第 1 步主 loss 与 `t1zw-a` 逐位相同，第 2 步起相差 1e-7 – 6e-6。另一份 `tz-on-vm32` 跑到一半节点被收回（spot），改名重跑 | `tz-on-vm32b`、`t1zw-a` 的 metrics.jsonl |
| 只改执行方式就能让主 loss 带符号均值挪 3.2e-6（开）、4.3e-6（关），和 base2 开关造成的 5.9e-6 同一量级：mini 上「关 base2 离 GPU 更远」分不开是开关的效果还是轨迹偶然 | 同上，两两直接相减（`janus.checks.series.window_stats`，窗口第 2–1000 步） |
| 原样重跑关 base2 的那一份（`tz-off-rep`，TPU 自比的代码包，MaxText `b87a55aa`）：主 loss、MTP、grad_norm 1000 步逐位与 `t1zz-a`（MaxText `2c28a4f4`）相同；KD 记录值按 M50 改记，不同；`param_norm_exact` 等 3 个单独编译的诊断列个别步末位不同。TPU 原样重跑测不出轨迹偶然，所以「重复」用的是只改执行方式的那一份 | `tz-off-rep` 的 metrics.jsonl |
| mini 第 1 步关 / 开 base2 对 CPU FP64：主 loss 开 −2.14e-6、关 −3.68e-6，GPU 8 份 −2.46e-6 到 +4.7e-7，都在一个量级；逐张量梯度（205 个）对 FP64 的 rel_L2 中位数开 6.32e-3、关 6.40e-3，93 个关后更近，各模块的比值都在 0.99–1.02；开、关两份之间 3.6e-3，和 GPU 三份之间（2.7e-3）同量级。mini 的 logits 小（第 1 步最大 0.129），base2 的放大不起作用，关掉只相当于换一处舍入 | [`t1-step1-base2-onoff-vs-fp64.json`](alignments/01-mini/results/t1-step1-base2-onoff-vs-fp64.json) |
| 和真实宽度上索引器 KD「关后离 GPU 更远、离 FP64 更近」不是同一类：那里第 1 步 GPU 的 KD 对 FP64 就偏高 8.3e-4（TPU 开 2.9e-4、关 1.7e-4），是 GPU 一侧有偏差；mini 第 1 步 GPU 的主 loss 对 FP64 没有偏差（在 ±2.5e-6 内），两种 TPU 也没有 | [`step1-base2-off.json`](alignments/06-rw3/results/step1-base2-off.json)、上一行 |

## mini 主 loss 共同偏高：要点与出处

问题（[D30](alignments/01-mini/DETAILS.md#d30)、[根源一览「还没查明」](ROOT-CAUSES.md#u-mini-tpu-high)）：现行比较主 loss 带符号均值 r 1.55，开、关 base2 各 4 份 TPU 主 loss 都比 GPU 16 份均值高，平均 +4.3e-6。下面每条都是实测；推出来的写「推断」。

| 要点 | 出处 |
|---|---|
| **偏差随步数怎么变**（TPU 8 份对 GPU 16 份逐步均值，25 步分块；GPU 留一法的标准差作尺子）：第 1 步 −2.0e-6（8 份都偏低）；第 51–90 步一个小鼓包，8 份都偏高，最高的第 71–80 步 +7.2e-6；第 100–550 步在 0 附近；**第 550 步起一路变大**：第 601–625 步 +4.1e-6、第 751–775 步 +1.1e-5、第 976–1000 步 +1.8e-5（每块 8 份里 7–8 份偏高，GPU 尺子 2–7e-6）。第 2–500 步正负相抵，2–1000 步的 +4.3e-6 几乎全部来自第 501–1000 步。学习率是常数，跟它无关；分块偏差与步数、GPU param_norm 同步（第 501 步起相关 0.96），与梯度范数无关（−0.11） | [`mini-main-loss-drift.json`](alignments/01-mini/results/mini-main-loss-drift.json)、`analysis/mini_main_loss_drift.py` |
| **MTP 反着走**：同一批运行的 MTP loss 第 101 步起偏低，第 751–1000 步 −3.2e-5（8 份里 8 份偏低；GPU 尺子 1.3e-5）。MTP 判档能过，是因为 MTP 的尺子比主 loss 宽 4 倍，不是因为没偏 | 同上 |
| **不是前向的差**：同一份权重上 TPU、CPU FP64（GPU 同一套代码）、CPU BF16（GPU 写法）各算同几批，6 个状态（TPU 第 301、501、751、1000 步，GPU 第 849、900 步）共 19 批：主 loss 对 FP64，TPU +1.36e-6 ± 0.68e-6，GPU 写法 +0.53e-6 ± 1.15e-6，TPU − GPU 写法 +0.83e-6 ± 1.15e-6（均值 ± 标准误）。对第 751–1000 步轨迹上的 +1.3e-5，前向的差在一成左右或更少。MTP 前向 TPU 比 GPU 写法低 3.1e-6 ± 1.1e-6，是后段 MTP 偏低的约一成 | [`mini-main-loss-same-state.json`](alignments/01-mini/results/mini-main-loss-same-state.json)、`analysis/mini_main_loss_same_state.py`、`analysis/cpu_bf16_gpu_forward.py` |
| **真 GPU 的同状态单批读数不能单独用**：GPU 第 849、900 步状态上真 GPU 重跑主 loss 对 FP64 −3.2e-6、−7.2e-6（各 2 份平均），看着像「GPU 前向偏低」；同一批上 CPU BF16 写法也偏低（−8.4e-6、−3.4e-6），但多算 3 批后平均回到 0（−0.03e-6、+2.4e-6）。原因是离散选择：把 GPU 那一次的路由与 DSA top-k 灌进 FP64 前向，第 900 步主 loss 就移 −9.9e-6（GPU 实测 −8.8e-6），第 849 步移 −1.1e-6（GPU −4.4e-6）。同状态前向对照至少要几批平均 | [`mini-same-state-gpu-choices-inject.json`](alignments/01-mini/results/mini-same-state-gpu-choices-inject.json)（`step1_ref.py --inject-dir`） |
| **不随 TPU 的执行方式、版本、硬件变**：TPU 自比的 35 个 1000 步臂（libtpu 0.0.45–0.0.48、v7x、16 设备、张量并行、数据并行、重算三种、普通注意力、稠密 MoE、Muon 正交化 FP32、iHC 不强制舍入、SwiGLU 少舍一次、近似超越函数、BF16 累加等）主 loss 第 2–1000 步全部为正（+2.4e-6 到 +7.9e-6），第 751–1000 步全部为正（+5.2e-6 到 +2.2e-5），MTP 第 751–1000 步 35 个臂全部为负；连同历代 T1 配置的 20 份与本次 15 份诊断运行，70 份 TPU 运行第 751–1000 步主 loss 都为正（+3.0e-6 到 +4.1e-5，最大的是关省舍入那份）。第 1 步梯度相对差到 6e-3 的改动都去不掉它，推断不是 TPU 某一处舍入 | `mini-main-loss-drift.json` 的 `variants` |
| **主 loss 与 MTP 之间极敏感**：只改 MTP loss 系数（0.1 → 0.099 / 0.101，各 1 份，MTP 记录值按系数换回），主 loss 第 2–1000 步 −3.6e-5 / +4.6e-5，MTP −7.4e-4 / +8.0e-4（按 0.1 的口径换算后对 GPU 均值 +7.4e-4 / −8.0e-4）。TPU 的偏差若全算成 MTP 权重的差，主 loss 要 +0.1%、MTP 只要 +0.02%，两者对不上，所以不是单纯的 MTP 权重差（推断） | `mlh-mtp099`、`mlh-mtp101` 的 metrics.jsonl |
| **精度本身对轨迹影响很大**：TPU 整网 FP32 计算（`dtype=float32`、`matmul_precision=highest`，关掉照 GPU 舍 BF16 的几个开关）1000 步主 loss 比 GPU 低 7.1e-4；关掉编译器省舍入（`--xla_allow_excess_precision=false`）高 4.1e-5。+4.3e-6 是这把尺子上的百分之一量级 | `mlh-hp32b`、`mlh-noexc` |
| **前段鼓包的根因：hc_head 收拢之后那次 BF16 舍入 TPU 没做**。GPU 在 hc_head 把四路流加权收成一路后 `.to(bfloat16)` 再进末层 RMSNorm；TPU 写的是 `astype(bf16)`，编译器在省舍入下把这一来回省掉（同第 16 行 iHC 的情形，推断：没看 LLO）。新开关 `hc_head_force_bf16_rounding`（M52，用 reduce_precision 强制舍入，转置同时舍余切）：4 份运行（两份单开、两份与 MTP 残差舍入同开）第 75 步 param_norm 对 GPU 均值 −1.9e-6 到 +7.1e-6，其余 66 份 TPU 运行全部在 −1.1e-5 到 −4.8e-5（第 75 步 GPU 16 份标准差 2.5e-6）；第 51–100 步主 loss −2.4e-6、+0.2e-6、+1.8e-6、+0.3e-6，其余 66 份平均 +4.4e-6、最低 +1.2e-6。第 751–1000 步不变（+1.0e-5 到 +2.2e-5） | `mlh-hc16-a/b`、`mlh-mreshc-a/b`；`mini-main-loss-drift.json` 的 `variants` |
| **后段排除过的候选**（每个 2 份 1000 步，第 751–1000 步主 loss 都落在上面 70 份 TPU 运行的范围里）：MTP loss 不给 hc_head 参数梯度（`mtp_hc_head_param_stop_gradient`）+1.9e-5 / +1.4e-5；MTP 与输出头共用同一次末层 norm（`mtp_input_shared_final_norm`，GPU 的接法）+1.1e-5 / +1.5e-5；RoPE 改成 Megatron-Core 不融合的 BF16 写法（`rope_bf16_gpu_form`，含索引器）+1.9e-5 / +7.4e-6；MTP 层的普通残差强制舍 BF16（`plain_residual_force_bf16_rounding`）+1.6e-5 / +1.4e-5；RMSNorm 输出的余切强制舍 BF16（`rmsnorm_cotangent_bf16_rounding`）+4.9e-6 / +1.4e-5；hc_head 舍入（上一条）+1.0e-5 / +2.2e-5 | 同上 |
| **第 1 步梯度按主 loss / MTP 两部分拆开**（FP64 全量减 `--mtp off`）：hc_head 三个参数的梯度里，GPU 三份把 MTP 那一部分丢掉一大半（组中位数 β −0.26 到 −0.61；`hc_head_fn` −0.51 到 −1.0），TPU 留着（β +0.31，`hc_head_fn` +0.04）；其余各组两部分的放大量两边接近，GPU 三份之间的散布和两边的差同量级。TPU 上把 MTP 对 hc_head 参数的梯度整个拿掉，后段不变（上一条第一项），所以这一处不是后段的原因 | [`mini-step1-grad-main-mtp-split.json`](alignments/01-mini/results/mini-step1-grad-main-mtp-split.json)、`analysis/step1_grad_main_mtp_split.py` |
| **三类小参数的梯度缺口要用几份 GPU 写法样本判**：GPU 第 849 步状态上只有一份真 GPU 梯度，拿它一份当 R 的分母会把它的运气算进来。另用 CPU BF16 的 GPU 写法（GPU 同一套代码，每行落地一次再 FP32 累加）多出 2 份样本。按类别取 R 中位数（分母是真 GPU / CPU 样本）：sink TPU 1.33 / 1.04，CPU 样本对真 GPU 也有 1.15，8 个里 4 个 > 1.2，所以 sink 没有缺口，之前的 R 1.2–2.1 是那份真 GPU 恰好准；norm 1.11 / 1.08，几份 GPU 写法样本之间都是 1.00，是真缺口；iHC 1.16 / 1.32，但真 GPU 对 CPU 样本也有 1.28，分不清；其余参数 1.00 / 1.02 | [`mini-s849-grad-classes.json`](alignments/01-mini/results/mini-s849-grad-classes.json)、`analysis/same_state_grad_classes.py`、`cpu_bf16_gpu_forward.py --all-grads` |
| **sink 的梯度已经是 GPU 写法，没有可拉回的地方**：sink 梯度照 Megatron-Core 的写法算（M47，单层探针对 GPU 写法 R 1.000）；会改 sink 输入的 q 预缩放，在 mini 上是精确的（softmax_scale 是 1/16，BF16 里乘 2 的幂不舍入）。所以这一类没有开关可做，也没上卡 | 台账 M47；本节上一条 |
| **norm 一类：开关能把梯度拉回 GPU 水平，后段不变**：现成开关 `rmsnorm_scale_grad_gpu_order`（M48，dγ 每条序列舍一次 BF16 再 FP32 相加，同 TE + Megatron 的 main_grad）在第 849 步状态上让 MTP 四个 norm 的 R 从 1.34–1.64 降到 1.02–1.04，全类中位数 1.11 → 1.03（40 个里 1 个 > 1.2），前向逐位不变。1000 步两份，第 551–1000 步对 GPU 16 份均值 +8.5e-6、+9.8e-6，后段斜率每 100 步 +3.0e-6、+4.7e-6，GPU 留一法斜率 −3.9e-6 到 +1.9e-6：后段照飘 | [`mini-s849-grad-classes.json`](alignments/01-mini/results/mini-s849-grad-classes.json)、[`mini-loss-isolation.json`](alignments/01-mini/results/mini-loss-isolation.json) 的 `nrm`；`lgd-nrm-a/b` |
| **iHC 一类：流梯度照 GPU 的舍入顺序（M58），结果还没定**：GPU 上一个 BF16 的流在一层里有三处用到，autograd 把三份 BF16 梯度按「加权和 + 门控」再加「残差」的分组加起来，每加一次舍一次（CPU 上用 GPU 同一段代码逐位核过）；分支输出的梯度也舍 BF16。TPU 在 excess precision 下这几处可能不舍，开关 `ihc_stream_grad_gpu_order` 把它们补上；顺带发现 XLA 原来把分支输出留在 FP32，开关补上这次舍入后第 849 步前向 loss 相对变了 +4.3e-6。第 849 步 iHC 类 R 只从 1.16 降到 1.15（对 CPU 样本 1.32 → 1.24）。1000 步两份，第 751–1000 步 +7.9e-6、+4.3e-7，后一份低于以往 70 份 TPU 运行的最小值 +3.0e-6；第 551–1000 步合起来 +3.3e-6（GPU 尺子 2.1e-6），斜率 +2.4e-6、−0.2e-6，都在 GPU 留一法范围里。两份之间差得大，还在补样本 | `analysis/ihc_stream_grad_order_probe.py`、[`mini-s849-grad-classes.json`](alignments/01-mini/results/mini-s849-grad-classes.json)、[`mini-loss-isolation.json`](alignments/01-mini/results/mini-loss-isolation.json) 的 `ihc`；`lgd-ihc-a/b` |
| **隔离实验：关 MTP、关 KD、两个都关，后段都还飘，所以不是这两项 loss 造成的**。两侧同一语义：系数设 0（GPU `--mtp-factor=0` / `--indexer-loss=0`，TPU `mtp_loss_scaling_factor` / `indexer_loss_scaling_factor` 设 0）。关 MTP 时 MTP 层只做前向、梯度为零；关 KD 时两侧索引器都冻结（Megatron `requires_grad=False`，MaxText 零梯度）。每种配置 GPU 4 份、TPU 2 份，TPU 对同配置 GPU 均值、第 551–1000 步（GPU 只有 4 份、6 对，尺子粗）：对照组（现行配置，同一条新的提交路径）TPU +1.48e-5，GPU 对 g3 16 份均值 −2.3e-6（尺子 2.1e-6），复现成立；A 关 MTP +1.16e-5（GPU 留一法 sd 1.3e-6；两份斜率 +4.3e-6、+2.3e-6，GPU 范围 −0.7e-6 到 +0.4e-6）；B 关 KD +9.0e-6（sd 3.4e-6；斜率 +0.5e-6、+4.0e-6）；C 都关 +8.7e-6（sd 2.0e-6；斜率 +1.2e-6、+2.7e-6，GPU 范围 −0.7e-6 到 +1.0e-6）。摘掉两项后幅度约是对照的六成，可能是这两项放大了它，也可能只是样本少（推断） | [`mini-loss-isolation.json`](alignments/01-mini/results/mini-loss-isolation.json)；GPU `gi-*`、TPU `ti-*`（`runs.yaml`） |
| **MTP 先动、主 loss 后动是相关不是因果**：现行 8 份 TPU 里，MTP 从约第 100 步起 8 份里 7 份偏低，主 loss 第 551 步以前正负参半，第 601–650 步两者一起跳（主 loss +4.1e-6，MTP −3.8e-5）。但 MTP 整个关掉以后主 loss 尾部照样在（上一条）。改 MTP 系数（0.1 → 0.099 / 0.101）让主 loss 挪 −3.6e-5 / +4.6e-5，只说明主 loss 对 MTP 权重敏感，TPU 的尾部不是「MTP 权重的等效差」 | [`mini-loss-isolation.json`](alignments/01-mini/results/mini-loss-isolation.json)；本节「主 loss 与 MTP 之间极敏感」一条 |
| **索引器 KD 也是 TPU 偏低**：现行 8 份 TPU 的 KD 从约第 200 步起比 GPU 低（8 份里 7 份），和 MTP 一样是「辅助 loss 低、主 loss 高」；KD 的梯度不进主干 | [`mini-loss-isolation.json`](alignments/01-mini/results/mini-loss-isolation.json) 的 `cur8` |
| **GPU 一侧索引器 RoPE 改 FP32 不改变尾部**：真实宽度上 GPU 的 KD 偏高约六成来自索引器 RoPE 在 BF16 里算（06-rw3）。mini 上 GPU 照同样写法改成 FP32（环境变量 `HY4_INDEXER_ROPE_FP32=1`，台账 G34）跑 1 份：它自己对 g3 16 份均值 +2.4e-6（在噪声里），TPU 对照对它第 551–1000 步仍是 +1.24e-5 | [`mini-loss-isolation.json`](alignments/01-mini/results/mini-loss-isolation.json) 的 `irf`、`irfgpu`；`gi-r-1` |
| **libtpu 0.0.52 一样飘**：TPU 自比的 0.0.52 golden `ts-g52-a`（与重跑 `ts-g52-c` 逐位相同，只算 1 份；base2 开的那一代配置）对 GPU 16 份均值，第 551–1000 步 +7.7e-6、斜率 +2.9e-6；同配置 0.0.46 的 4 份 +9.8e-6、斜率 −0.7e-6 到 +6.5e-6 | [`mini-loss-isolation.json`](alignments/01-mini/results/mini-loss-isolation.json) 的 `lt52on`、`lt46on` |
| **和 GPU 第 1 步主 loss 的偏差不是同一件事**：真实宽度三层 GPU 第 1 步主 loss 对 FP64 +5.3e-5，是第 1 步就有的前向偏差（索引器 RoPE 用 BF16 只解释 KD，不进主 loss；主 loss 这份还没全部定位）；mini 第 1 步 GPU 对 FP64 在 ±2.5e-6 内，TPU 的偏高是训练里攒出来的、同状态前向没有。两者的共同点只有「都和 BF16 舍入位置有关」 | 06 [README](alignments/06-rw3/README.md)；本节第 1、3 条 |

## 真实宽度三层：要点与出处

| 要点 | 出处 |
|---|---|
| D3（3.21e10 参数）在 v5p 16 片上跑得起来，训练步程序每设备预留 72.8 GB 临时量；按 T2 实测线性外推出的「8 片放得下」不成立（多出的两层 MoE 让临时量远大于 T2 的 12.9 GB）。编译期分析对 T2 B 给 73.1 GB、实测峰值 26.5 GB，两种估法都不可靠，片数要实跑看 | [06 设置「显存：要几片」](alignments/06-rw3/setup.md#显存要几片) |
| D3 检查点在 CPU 上转：读 111 个 HF 张量、搬 132 个叶子 1279 s，存 Orbax 443 s，产物 56 GB；反向取回逐位核过 | [`hy4-d3-l0-ckpt-convert`](alignments/06-rw3/results/hy4-d3-l0-ckpt-convert.report.md) |
| 第 1 步主 loss 对 CPU FP64：GPU +5.76e-5、TPU −9.87e-5（R 1.71）；GPU 同设置 4 份都在 +5.1e-5 到 +5.8e-5。真实宽度截层 B 也是 GPU 偏高、TPU 偏低（[02 D20](alignments/02-real-weight/ALIGNMENT-REPORT.md#d20)） | [`step1-loss-vs-fp64.json`](alignments/06-rw3/results/step1-loss-vs-fp64.json) |
| D3 第 1 步 FP64 只前向（`step1_ref.py --forward-only --expert-store ram`）：每行 72–100 s（40 核），峰值 RSS 257.7 GiB；16 行都算完后写输出时超出 300 GiB 的上限被杀，全局 loss 从每行打印的 MICRO 记录合成。被杀的原因：只前向时也把专家梯度从累加器读回，没有梯度就建 216 GiB 的 FP64 零张量；峰值 257.7 GiB 则是专家先载成 FP64 再转 BF16。两处都已改（台账 J47），现在 D3 只前向峰值 109 GiB、每行约 67 s，原样参照第 0 行与旧记录逐位相同 | [`d3-step1-fp64-forward.micro.jsonl`](alignments/06-rw3/results/d3-step1-fp64-forward.micro.jsonl)、[`analysis/step1_loss_vs_fp64.py`](analysis/step1_loss_vs_fp64.py) |
| spot v5p 会被整片收走：2026-11-09 夜里同一区域全部 v5p spot 节点两次在几分钟内一起被抢（间隔约 1 小时），跑到一半的 1000 步运行连同 Pod 里的运行目录一起丢了。TPU 长跑要分段存档（整个训练状态存到 GCS，被抢从最新存档接着跑），并边跑边把 `metrics.jsonl` 拷出来 | 06 [`README.md`](alignments/06-rw3/README.md)、[`runs.yaml`](alignments/06-rw3/runs.yaml) |
| TPU 长跑的分段存档：`train_hy4.py --save-every N --save-dir gs://… --resume …/step_N` 把整个训练状态（参数、优化器状态、路由偏置）用 Orbax 存到 GCS；Orbax 不肯存大小为 0 的数组（被掩掉的优化器状态），要跳过。D3 一份 2.37e11 字节、16 片存 57 s。存档不改数值：存过档的 `trw-e` 与没存档的 `trw-a` 在第 147 步逐位相同。从存档续跑在 TPU 上逐位相同：第 100 步存档续到第 250 步，150 步与不中断的运行逐位相同；1000 步运行被收两次后续跑，各段重叠的 81 步逐位相同 | 06 [`README.md`](alignments/06-rw3/README.md#从存档续跑为什么装不下已修) |
| MTP loss 的单批对照分不出 2e-5 量级的差：真实宽度 D3 上 GPU 同一份权重、同一批重跑，MTP loss 之间差 5.5e-5（均方根），主 loss 只差 3.7e-6；TPU 换一个单独编译的前向，MTP 也动 2.5e-5。同状态对照 MTP 要冻住权重（学习率、路由偏置更新率都设 0）连算几十批：第 900 步 32 批的结果是 TPU 低 1.02e-5 ± 7.5e-6，不显著，MTP 的后段偏低主要是轨迹差 | 06 [`rw3-mtp-same-state.json`](alignments/06-rw3/results/rw3-mtp-same-state.json) |
| 读档时别留旧状态的引用：`state.to_pure_dict()` 的叶子就是状态数组本身，不是拷贝。拿它取形状、读档后不释放，旧状态就整份留在设备上（D3 每设备多 1.64e10 字节，训练步的临时区装不下）。只留 `ShapeDtypeStruct`、先 `delete()` 旧缓冲再读，读档前后的活数组字节数应相同（`meta.json` 的 `memory_probe`） | 06 [`resume-check.json`](alignments/06-rw3/results/resume-check.json) |
| 第 1 步主 loss TPU 偏低定责到 splash 的 base2 模式：内核 `q *= LOG2E` 里常数随 BF16 的 q 舍成 1.4453125（大 1.81e-3），exp2 换回自然底与 sink 按精确值，等于注意力 logits（不含 sink）整体乘 1.0018。CPU FP64 前向只加这一处：D3 主 loss 移动 −9.69e-5（TPU 实测 −9.87e-5，扣掉后 −1.8e-6，R 1.85 → 0.03）；B 移动 −8.41e-5，TPU 只关 base2 的已有对照移动 −8.79e-5（−8.06e-5 → +7.3e-6）。完整模拟（再加 q 的逐元素舍入）减「只把 q 舍 BF16」的对照：D3 −9.88e-5（TPU 扣掉后 +1.1e-7）、B −8.28e-5，逐元素那一半只有 ±2e-6。逐层只加 κ：第 0 / 1 / 2 层 −3.32e-5 / −5.77e-5 / −8.4e-6，MTP 层只动 MTP loss（−4.47e-5） | 06 [README「第 1 步主 loss 为什么 TPU 偏低」](alignments/06-rw3/README.md#第-1-步主-loss-为什么-tpu-偏低)、[`step1-base2-emulation.json`](alignments/06-rw3/results/step1-base2-emulation.json)、[`analysis/step1_base2_emulation.py`](analysis/step1_base2_emulation.py) |
| base2 内核的语义在 CPU 上核过：Pallas interpret 跑 tokamax 0.0.17 与 MaxText 树里的副本，base2 的 logsumexp 对「q × 1.4453125 舍 BF16 再 ÷ log2 e」的 FP64 参照最大差 8.1e-7；对只乘 κ 的参照，带符号均值差 −1.8e-5（剩下的是 q 那一次逐元素舍入）。TPU 训练镜像（digest `b152355a…`）里装的是 tokamax 0.0.14（2026-11-11 读镜像层里的 dist-info 核过），MaxText `b87a55aa` 的 splash 从装好的包导入；0.0.14 与 0.0.17 的 base2 那段是否相同没核 | [`splash-base2-probe.json`](alignments/06-rw3/results/splash-base2-probe.json)、[`frameworks/maxtext/splash_base2_probe.py`](frameworks/maxtext/splash_base2_probe.py) |
| base2 放大只在真实权重上显形：效果和 logits 大小成正比，mini 第 1 步全局最大 logit 0.129，真实权重截层 B 逐头最大 14.5、中位数约 7 | [`t1-qkclip-same-state.json`](alignments/01-mini/results/t1-qkclip-same-state.json)；B 的 `tb6diag` 第 1 步记录 |
| 截层 B 的 TPU 第 1 步只有 `diag_step1.json` 的总 loss，主 loss 要减 KD 项：KD 项 = Σ 各层 indexer_loss_raw ÷ 累积份数 × 每个 microbatch 的 token 位置数（`targets.size`，B 是 32768）÷ 全局有效 token 数（`indexer_kd_mcore_weight` 开时）；D3 的 `trw-e` 用同一公式，总 loss 对上到 4e-7 | [`step1-base2-observed.json`](alignments/06-rw3/results/step1-base2-observed.json) |
| D3 第 1 步 TPU 只关 `sa_use_base2_exp`（`b2s1-bt`，其余同 `trw-e`、同一代码包）：主 loss 对 FP64 −9.87e-5 → −1.24e-6，R 1.85 → 0.02；移动 +9.75e-5，是 CPU 模拟预测的 1.006 倍（只加放大）、0.99 倍（完整模拟）。MTP loss −9.88e-5 → −6.92e-5，没按预测（+1.31e-4）移动，落在 GPU 4 份的范围里 | [`step1-base2-off.json`](alignments/06-rw3/results/step1-base2-off.json)、06 [README](alignments/06-rw3/README.md#tpu-实测d3-只关-base2) |
| D3 第 1 步逐张量梯度（路由专家以外 102 个张量）对 CPU FP64：base2 开 → 关，中位数 2.03e-3 → 1.72e-3，86 个更近；注意力 2.44e-3 → 1.78e-3、norm 2.80e-3 → 2.18e-3、索引器 2.43e-3 → 2.41e-3（不动）；关后变远的 11 个里最大的是 MTP 层 sink 4.81e-3 → 9.37e-3。对 GPU 的 R 见下文 `r3s1-*` 那两条 | [`step1-base2-off.json`](alignments/06-rw3/results/step1-base2-off.json) |
| D3 base2 开 / 关 250 步回放（`b2r-on` / `b2r-off`，`rw1k-a` 作基线、GPU 8 份两两 28 对作尺子）：开过 2、待定 2、不过 5，关过 4、待定 2、不过 3。主 loss 全程 r 11.2 → 7.32、后段 2.82 → 2.72；MTP 3.14 → 0.99、后段 2.75 → 1.74；param_norm 2.45 → 0.67；索引器 KD 1.62 → 2.33；路由偏置均值 1.05 → 1.55；grad_norm、偏置标准差两臂都过 | [`rw3-base2-replay.json`](alignments/06-rw3/results/rw3-base2-replay.json)、06 [README](alignments/06-rw3/README.md#base2-开--关的-250-步回放) |
| D3 base2 开 / 关 1000 步回放（`rfx-on` / `rfx-off`，同一代码包，`rw1k-a` 作基线、GPU 8 份两两 28 对作尺子）：开过 2、待定 4、不过 3（MTP 两个窗口、索引器 KD），关过 3、待定 6、不过 0。主 loss 全程 r 1.87 → 1.20、后段 1.24 → 1.02，两臂都偏低，第 251–500 步回到噪声、之后又是 1–1.4 个 N；MTP 全程 3.17 → 1.85；索引器 KD 2.06（偏高）→ 1.49（偏低）。`rfx-off` 除两项切分参数外与现行对齐 yml 逐键相同 | [`rw3-1k-replay.json`](alignments/06-rw3/results/rw3-1k-replay.json)、06 [README](alignments/06-rw3/README.md#1000-步回放base2-开--关) |
| 两臂直接相减（开 − 关，分母关那一臂，再除以同窗口的 GPU 噪声上限）：主 loss 1–164 步 −3.45e-5（7.1 个 N）、1–250 步 −2.08e-5（3.9 个 N）、101–250 步 −9.1e-7（0.1 个 N）；MTP 1–250 步 −5.70e-5（2.2 个 N）；索引器 KD +3.80e-4（2.5 个 N）；param_norm −2.69e-9（1.9 个 N）；grad_norm 0.75 个 N。base2 对主 loss 的影响集中在前约 100 步 | [`rw3-base2-replay.json`](alignments/06-rw3/results/rw3-base2-replay.json) |
| 主 loss 后段（101–250 步）对 GPU 两臂都偏低 −2.5e-5（2.7 个 N），与 base2 无关；第 1 步上，关掉 base2 的 TPU 对 GPU 的 −5.9e-5 与 GPU 自己对 FP64 的 +5.3e-5 相当。索引器 KD 第 1 步对 FP64：GPU +8.3e-4（4 份 +8.1e-4 – +8.4e-4）、TPU 开 +2.9e-4、关 +1.7e-4 | [`rw3-base2-replay.json`](alignments/06-rw3/results/rw3-base2-replay.json)；FP64 参照 `d3-fp64-grad` 的 `indexer_loss` |
| 同一配置在同一种切片上重跑逐位相同：`b2r-on` 的前 164 步与 `trw-e` 逐位相同（全部记录列，不含步时）；`b2r-off` 第 1、2 步与 `b2s1-bt` 逐位相同 | `b2r-on`、`b2r-off` 的 metrics.jsonl |
| 改正在跑的 bash 脚本会让它在收尾时报语法错：bash 按偏移逐段读脚本，`tpu1.sh` 被改后 `b2r-off` 收尾报 `unexpected EOF`，没删 JobSet（运行目录已取回）。要改就另存新文件再换名 | `b2r-off` 的提交日志 |
| `--diag-step1` 在 D3 / v5p 16 片上装不下：梯度程序要预留 74.22G，只有 72.93G（训练状态里的优化器状态占着）。`--diag-free-opt` 先释放优化器状态的设备数组（全局 1.34e11 字节，每设备 8.4 GB），只把模型变量传进梯度函数；CPU 冒烟（阶段 D）开与不开梯度文件 sha256 相同 | `b2s1-bd` 的 Pod 日志；`train_hy4.py --diag-free-opt` |
| `gradient_accumulation_steps=1` 时 `--diag-step1` 的梯度比训练步小 N 倍（N = 全局有效 token 数，D3 第 1 步 65453）：MaxText 的梯度累积函数又除了一次。乘回 N 后 grad_norm 与训练步 raw_grad_norm 差 1.3e-5–1.7e-5，诊断的总 loss 与训练步差 2.1e-6：诊断与训练是两份编译，不逐位相同。本节 D3 第 1 步的读数里，主 loss 用训练步记录（`trw-e`、`b2s1-bt`），逐张量梯度与 R 用诊断程序的梯度（`b2s1-ad2`、`b2s1-bd2`） | [`step1-base2-off.json`](alignments/06-rw3/results/step1-base2-off.json) 的 `tpu_scale`；`b2s1-bt` / `trw-e` 第 1 步 |
| D3 CPU FP64 整网反向：`step1_ref.py --expert-store ram --keep-experts 0 --expert-grads-kept-only --recompute`，路由专家只算第 0 号的权重梯度（全部专家的 FP64 累加器要 216 GiB），峰值 RSS 220.8 GiB；每行前向 88–116 s、反向 187–206 s（40 核），16 行 80 分钟；逐行前向 loss 与只前向的参照相同（差 ≤ 1.5e-11 绝对） | `d3-fp64-grad`（[runs.yaml](alignments/06-rw3/runs.yaml)） |
| `step1_ref.py --emulate-base2 full`（或 `q16`）和 `--audit` 一起用会静默退化成 `kappa`：`_unforced` 只退出最上面一层 dispatch mode，带审计时退出的是审计，ForceDtype 还在，q 的 `.to(bfloat16)` 被改回 FP64。D3 的 `d3-fp64-full-grad` 主 loss 与只前向的 κ 模拟逐位相同（12.013328636119082）。脚本现在拒绝这种组合 | `d3-fp64-full-grad`（[runs.yaml](alignments/06-rw3/runs.yaml)）；`step1_ref.py` |
| 真实宽度三层 GPU 第 1 步逐张量抓数（单机 DP 8 + EP 8 同设置 3 份 `r3s1-*`，`--dump-steps 1 --full --full-kinds grad+param --no-dump-acts --recompute`，每份 168–171 s，每卡峰值 104.4 GiB）：主 loss 对 FP64 +5.57e-5 – +5.65e-5、KD +8.15e-4 – +8.35e-4，与 2 台 `rw1k-*` 同号同量级 | [`step1-tpu-gpu-vs-fp64.json`](alignments/06-rw3/results/step1-tpu-gpu-vs-fp64.json)、`r3s1-*` 的 metrics.jsonl |
| D3 第 1 步 R（TPU 对 FP64 ÷ GPU 3 份中位数对 FP64，102 个非路由专家张量）：中位数 base2 开 1.06、关 0.92；R > 1.2 开 38、关 25；norm 一组关后仍 1.45（20 个里 12 个 > 1.2，最大第 2 层 `kv_a_layernorm` 2.52）；索引器 0.64 | [`step1-tpu-gpu-vs-fp64.json`](alignments/06-rw3/results/step1-tpu-gpu-vs-fp64.json) |
| MTP 层 sink 梯度关 base2 后 4.81e-3 → 9.37e-3 不是真变坏：梯度范数 2.4e-4（第 2 层 sink 9.8e-3），GPU 三份对 FP64 是 5.33e-3 – 7.26e-3、两两最大 8.27e-3；TPU 关那份离最近一份 GPU 7.67e-3，0.93 个 N | [`step1-tpu-gpu-vs-fp64.json`](alignments/06-rw3/results/step1-tpu-gpu-vs-fp64.json) |
| 后段主 loss 偏低的时间走势（TPU 对 GPU 8 份逐步均值，窗口平均）：base2 关第 1–10 步 −5.91e-5、11–50 −5.00e-5、51–100 −7.21e-5、101–150 −4.55e-5、151–200 −3.04e-5、201–250 −5.97e-6（GPU 8 份窗口均值的散布 1.3e-6 – 5.7e-6）；base2 开从 −1.50e-4 降到 −1.5e-6。param_norm、路由偏置、索引器 KD 都没有更早动 | [`rw3-base2-replay.json`](alignments/06-rw3/results/rw3-base2-replay.json)、06 [README](alignments/06-rw3/README.md#后段主-loss-偏低不是前向的差是轨迹的差) |
| 同一份权重（TPU base2 关第 150 步之后）上第 151 步前向：CPU FP64 6.594300；TPU 训练步 +3.68e-5（单独编译的前向 +4.30e-5）；GPU 2 台两份 +2.11e-5、+2.22e-5、单机一份 6.594448（+2.25e-5）。同权重 TPU 比 GPU 高 1.5e-5。GPU 8 份第 151 步自己状态的均值 6.594587（单步散布 1.9e-5），按 GPU 前向 TPU 这份权重低 2.2e-5 | [`rw3-step151-same-state.json`](alignments/06-rw3/results/rw3-step151-same-state.json) |
| 训练后的同状态对照要用 GPU 训练时真用的权重：GPU 的输出头与索引器 k_norm 留 FP32（`mark_keep_in_fp32`），HF 却存 BF16。快照把这两处舍成 BF16 时，FP64 主 loss 升 1.775e-4、TPU 舍同样两处升 1.780e-4（只有输出头起作用；k_norm 只动 KD +5.45e-3）；其余各组舍 BF16 对主 loss 是 0 或 −1.4e-6（TPU 计算时本来就先舍 BF16）。base2 开那一臂同样：输出头 +1.76e-4 | [`rw3-step151-same-state.json`](alignments/06-rw3/results/rw3-step151-same-state.json)、[`b2e-off151`](alignments/06-rw3/results/b2e-off151-eval-variants-step151.json)、[`b2e-on151`](alignments/06-rw3/results/b2e-on151-eval-variants-step151.json) |
| TPU 从 HF 快照起步算任一步：`train_hy4.py --hf-snapshot-at N`（第 1 步 111 个张量与 HF 原件逐位相同）导出、`--param-override <多个文件>` + `--bias-override hf:<文件>` 换回、`--diag-step1 --first-step N` 算梯度；`--eval-variants-at N` 对同一份权重做「部分参数舍 BF16」的前向。GPU 一侧从 HF 目录载入，tokens 用第 (N−1)·16 行起的切片（GPU 第 1 步 = 第 N 步的批） | `train_hy4.py`、[runs.yaml](alignments/06-rw3/runs.yaml) |
| GPU 索引器 RoPE 改到 FP32 里算（诊断，`irf-a`、`irf-b` 各 250 步）：索引器 KD 带符号均值 −4.18e-4、−4.70e-4（后段），落到 TPU 一侧；主 loss r 0.88、0.86，后段对 GPU 8 份均值 +1.8e-6、−1.2e-6。KD 偏高在 GPU 一侧且来自这一处，但不是主 loss 轨迹差的原因 | [`irf-a`](alignments/06-rw3/results/rw3-l4-b2r250-irf-a.report.md)、[`irf-b`](alignments/06-rw3/results/rw3-l4-b2r250-irf-b.report.md) |
| 同一份权重上的梯度范数：TPU 8.851，GPU 2 台 8.666、8.659、单机 8.649；GPU 8 份在自己状态上 8.93–9.00。差别主要在 iHC 一组（GPU 平方和 60.5，按 TPU 总范数反推约 63.8）；第 1 步 iHC 平方和两边只差 0.27% | 同状态 GPU 的 metrics.jsonl；`b2s1-ad2` / `r3s1-*` |
| 同一份权重（第 150 步之后）上第 151 步的逐张量梯度：TPU 对 GPU 超出 GPU 噪声的倍数中位数，iHC 7.5（21 个里 19 个 > 2），注意力 2.0、norm 2.1、MLP 1.8；对 CPU FP64 的 R 中位数全部 1.03、iHC 0.90。iHC 梯度平方和 FP64 62.06、TPU 63.80、GPU 60.27 / 60.40；梯度范数 FP64 8.751、TPU 8.850、GPU 8.649 / 8.666 | [`rw3-step151-same-state-grads.json`](alignments/06-rw3/results/rw3-step151-same-state-grads.json) |
| 关 TPU 编译器省舍入（`XLA_FLAGS=--xla_allow_excess_precision=false`）后同一份权重的 iHC 平方和 59.93、梯度范数 8.630，落到 GPU 旁边；但其余各组离 FP64 更远（R 中位数 1.03 → 1.30，R > 1.2 的 27 → 64 个）。iHC 上 TPU 与 GPU 的差来自 GPU（eager）一定舍成 BF16、TPU 编译器省掉的那几次舍入 | [`rw3-step151-same-state-grads-noexc.json`](alignments/06-rw3/results/rw3-step151-same-state-grads-noexc.json) |
| base2 关再加 `XLA_FLAGS=--xla_allow_excess_precision=false` 跑 250 步（`b2r-noexc`）：主 loss 第 1–250 步 r 7.32 → 29.6、带符号均值 −3.90e-5 → −1.58e-4，后段 r 2.72 → 13.2；MTP r 0.99 → 5.79；索引器 KD 从 −3.53e-4 翻到 +4.63e-4。同权重上 iHC 梯度靠向 GPU，但轨迹离 GPU 更远：iHC 上省掉的舍入不是后段偏低的原因 | [`rw3-l4-b2r250-noexc.report.md`](alignments/06-rw3/results/rw3-l4-b2r250-noexc.report.md)、[`rw3-b2r-noexc-vs-off.json`](alignments/06-rw3/results/rw3-b2r-noexc-vs-off.json) |
| D3 第 1 步 GPU 主 loss 对 FP64 +5.3e-5（CPU 上按 GPU 写法的 BF16 前向复现 +5.21e-5）没有单一来源。FP64 里只加一类 BF16 舍入：MLA RoPE（q_pe/k_pe）+8.74e-6（17%），线性层+norm+注意力核输出 +1.26e-5（24%，只舍线性层 −7.65e-6），层间残差流 +5.02e-6（10%），输出头输入 +1.87e-6，k_pe 存储 +1.45e-6，注意力概率 +1.6e-7。合计约 55%，其余推断在逐元素运算 | [`step1-main-loss-bf16-sites.json`](alignments/06-rw3/results/step1-main-loss-bf16-sites.json) |
| D3 第 151 步的 FP64 整网反向（v2 快照，路由专家只算第 0 号）：16 行，每行前向 106–142 s、反向 210–435 s（与别的 FP64 作业争 CPU 时更慢），峰值 RSS 约 220 GiB；主 loss 与只前向的 v2 参照相同（6.59429960743114） | `s151-fp64-grad`（[runs.yaml](alignments/06-rw3/runs.yaml)） |
| TPU 从 HF 快照起步的 `--param-override` 在 Pod 里先把每个 gs:// 文件下载到本地盘：D3 快照 67 GB，Pod 盘 95 GB，梯度文件再写 12.4 GB 就满（作业在写运行目录前退出，日志也取不回）。现在读完一个文件就删本地副本 | `b2d-off151` |
| D3 `--diag-step1` 取回梯度慢：梯度算完到写完 `diag_grads.safetensors` 约 21 分钟，进程 0 的主机内存到 258 GiB（主机 441 GiB）；路由专家每层一个融合张量（FP32 约 39 GB）也要取回算平方和。只存路由专家以外的张量（`--full-only` 列出模块名）文件 12.4 GB | `b2s1-bd2`、`b2s1-ad2` |
| 真实宽度三层前 164 步：TPU 主 loss、MTP、索引器 KD、param_norm 都同号偏低，主 loss 带符号均值是同窗口 GPU 尺子的 17.8 倍；与第 1 步对 FP64 的方向一致（TPU 偏低、GPU 偏高） | [`rw3-l4-first164`](alignments/06-rw3/results/rw3-l4-first164.report.md) |
