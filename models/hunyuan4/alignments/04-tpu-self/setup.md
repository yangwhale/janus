[janus](../../../../README.md) › [混元 4](../../README.md) › [TPU 自比](README.md) ｜ 设置

# 混元 4 · 04 · TPU 自比 · 设置

## golden

| 项 | 值 |
|---|---|
| 模型 | hy4-mini（7 层 + 1 层 MTP，427,783,489 参数，随机初始化；同 [mini](../01-mini/setup.md)） |
| 硬件 | v5p 8 片（2x2x2，2 台主机，8 个 JAX 设备） |
| 并行 | FSDP 8；每设备微批 1 × 梯度累积 2，全局批 16 × 4096 |
| 程序 | MaxText，`maxtext` 混元 4 集成分支 `b87a55aa` |
| 配置 | [`frameworks/maxtext/hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml) 在 golden 跑的时候的版本（sha256 见 golden 的 `meta.json`），命令行不加改开关的覆盖参数。那时 `sa_use_base2_exp: true` |
| base2 | **自比的 golden 仍是开 base2 的一代（`ts-g-a`），对齐配置 2026-11-11 起已改成关，两者不冲突**：自比比的是同一个起点上只改一样东西前后的差，不依赖 base2 取什么值；golden 和各臂都是开 base2，比较内部一致。关 base2 的同一起点已有几份（`tz-off-nolhs`、`tz-off-vm32`、`tz-off-vm64`），要用关 base2 作底比时另写目录（PROTOCOL 变更记录 2026-11-11） |
| 初始权重、数据 | 同 mini：同一份 HF 目录转换的检查点；原样回放 GPU `g3-a` 写下的 1000 步 batch 与 loss_mask |
| 镜像 | MaxText `tpu_pre_training`（digest 见 `frameworks/maxtext/run.sh`） |
| 学习率 | 恒定 1e-5，所以 1000 步运行的前 300 步与 300 步运行的学习率相同 |

命令（[`frameworks/maxtext/README.md`](../../frameworks/maxtext/README.md) 的写法）：

```bash
XLA_DUMP=1 YML=hy4-mini-t1.yml STAGE=mini GOLDEN=<GPU g3-a 运行目录> bash run.sh <运行名> 1000 --dump-steps 1 --full
```

- `--dump-steps 1 --full`：第 1 步全量抓数（更新前参数、两个动量、更新后主权重、路由偏置）。第 1 步训练梯度从动量反推。
- `XLA_DUMP=1`：训练步的优化后 HLO 拷进运行目录 `hlo/`，`meta.json` 的 `hlo.fingerprint` 记去掉元数据后的指纹。

## 变种

每个变种从 golden 的命令出发，只改一样东西（一个覆盖参数、一个环境变量、一个 MaxText 提交、换切片、换版本或换硬件），跑 300 步，第 1 步全量抓数、存训练步 HLO；代表臂另跑 1000 步。

| 改什么 | 怎么传 |
|---|---|
| MaxText 配置 | 命令行末尾 `key=value` |
| XLA / libtpu 开关 | `XLA_FLAGS`、`LIBTPU_INIT_ARGS` |
| libtpu / jax 版本 | `PIP_EXTRA="libtpu==<版本>"`（Pod 里先 `pip install --no-deps`，运行记录 `meta.versions.libtpu_pip` 记下版本号） |
| 代码写法 | MaxText 上从同一提交拉一个只改一处的提交，用它打包 |
| 硬件 | `PLATFORM=v5p-16`（2x2x4）、`v7x`（2x2x1，8 个 TensorCore）、`v7x-16`（2x2x2） |

## 比较

`analysis/tpu_self_compare.py` 对 golden 比，判三档：

| 档 | 条件 |
|---|---|
| 逐位相同 | 被测跑过的每一步，两边都记的逐步量（步时除外）逐位相同；第 1 步全量抓数每个张量 sha256 相同 |
| 在噪声内 | 不逐位；前 300 步每条曲线的三个判定量 ≤ GPU 同设置 8 份两两 28 对的上限；第 1 步训练梯度超出 GPU 尺子的张量都能对 CPU FP64 定责（见下） |
| 超出噪声上限 | 其余；曲线最差 r ≤ 2 且第 1 步没有定不了责的张量记「略超」，否则记「超」 |

- 第 1 步训练梯度从全量抓数的动量反推（Adam m / (1 − β1)、Muon buf / (1 − β)，再乘回裁剪系数）。尺子 = 该张量上 GPU 三份第 1 步两两的最大相对差，不低于分辨率下限 2^-20（反推带进的 FP32 舍入）。
- 定责：被测离 FP64 的距离 ÷ max(golden 离 FP64, GPU 三份离 FP64 的中位数) ≤ 1.2 记「换了顺序」；超过的记「丢了精度」。iHC 小参数单列（01-mini D11 已接受），不挡。
- 解读列「和 GPU 重跑比」（不改判档）：曲线最差 r ≤ GPU 留一法最大值（300 步 8 份 2.47、1000 步 16 份 1.82）且第 1 步没有定不了责的张量 → 和 GPU 重跑分不开；贴着最大值（5% 以内）记临界（PROTOCOL 变更记录 2026-11-09 登记，前瞻验证中）。
- 另报：HLO 指纹（文件 sha256、去掉元数据的指纹、再去掉源码位置表的程序指纹）、各类指令个数之差；第 1 步梯度按层汇总；步时、显存峰值。
- 曲线目录：`catalogs/vs-golden.yaml`（300 步）、`vs-golden-1000.yaml`（1000 步，尺子 GPU g3）、`vs-<底>.yaml`（对某个只改了连带项的底比）。

```bash
python models/hunyuan4/analysis/tpu_self_compare.py --root <运行目录根> --golden ts-g-a --test <被测> \
  --catalog models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml \
  --gpu-step1 g2s-a=<GPU 第 1 步梯度> g2s-b=<…> g2s-c=<…> --fp64 <CPU FP64 第 1 步参照目录> \
  --out models/hunyuan4/alignments/04-tpu-self/results/<被测>-vs-golden
python models/hunyuan4/alignments/04-tpu-self/build_tables.py   # 重画 README 的分堆表与 MATRIX.md 的结果表、脾气表
```

## 判档怎么读

每次比较分成四档。

| 档 | 意思 | 该怎么看 |
|---|---|---|
| 逐位相同 | 每一步记下的数、第 1 步的全部状态，每一位都相同 | 改动没碰数值 |
| 在噪声内 | 不逐位，但差别不超过 GPU 自己重跑的波动（r ≤ 1）；第 1 步变了的梯度离精确值不比原来远 | 只是换了加法顺序，不是 bug |
| 略超 | 曲线最差 r 在 1–2 之间 | 多是梯度范数或路由偏置标准差某一项略过 1。golden 拿去和 GPU 比，本身就是这两项略超（r 1.23、1.14），所以这一档一般也是换了顺序 |
| 超 | r > 2，或第 1 步有梯度离精确值远了 1.2 倍以上 | 改到了精度，按 bug 查 |

第 1 步的判法：对每个变了的梯度张量，看被测离精确值有多远，和 golden、GPU 里离得较远的那个比。不比它远，记「换了顺序」：只是加法先后不同，不是 bug。远出 1.2 倍以上，记「丢了精度」：某处多舍入了一次，或者在低精度里做了累加。查清差别从哪来、属于哪一种，叫**定责**。精确值是 CPU 上用 64 位浮点算的第 1 步梯度。

**GPU 自己重跑也常常「略超」。** 拿 GPU 同设置的几份互相比，用同一套判法试这把尺子：轮流挑一份当被测、一份当 golden，尺子用其余几份重算，这种做法叫**留一法**。300 步 8 份共 56 种组合，最差 r 超过 1 的占 5.9e-1，超过 2 的占 1.1e-1，最大 2.47（[`gpu-rerun-loo-r-300.json`](results/gpu-rerun-loo-r-300.json)）；1000 步 16 份共 240 种组合，超过 1 的占 2.2e-1，没有超过 2 的，最大 1.82（[`gpu-rerun-loo-r-1000.json`](results/gpu-rerun-loo-r-1000.json)）。

原因是一次要看的数太多：300 步看 8 项（1000 步 9 项），每项 3 个量，二十几个数里任何一个过 1 就记略超，纯噪声也很容易碰上一个，这叫**多重比较**。所以表里另有一列「和 GPU 重跑比」：曲线最差 r 不超过 GPU 留一法的最大值（300 步 2.47、1000 步 1.82），第 1 步也没有离精确值更远的梯度，就写「和 GPU 重跑分不开」。这一列是事后解读，不改判档。

曲线指前 300 步的主 loss、MTP loss、索引器 loss、梯度范数、参数范数、路由偏置。每条曲线看三个判定量：**带符号均值**（平均偏多少，正负不抵消的那部分）、**MARE**（平均绝对相对差，即平均晃多少）、**最大单步**（最坏一步差多少）。

### golden 的读数

| 项 | 值 |
|---|---|
| 配置 | mini 现行对齐配置（[`hy4-mini-t1.yml`](../../frameworks/maxtext/hy4-mini-t1.yml)），v5p 8 个设备，FSDP 8，每设备微批 1 × 累积 2 |
| 复现 | 换一组机器重跑，逐步量、第 1 步全部状态、最终参数、编译产物都逐位相同；只跑 300 步也和前 300 步逐位相同 |
| 对 GPU | 1000 步 9 项过 7、略超 2（尺子 GPU 8 份），与 mini 现行比较相同；主 loss 的 MARE：TPU 8.05e-6，GPU 波动上限 8.24e-6，r 0.98。尺子换成 16 份后，mini 现行比较 9 项全过，见[知识库](../../knowledge.md#gpu-1000-步尺子扩到-16-份要点与出处) |
| 步时 | 0.494 s |
