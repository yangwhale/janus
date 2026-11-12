[janus](../../README.md) › [文档](../README.md) › 参考 · 抓数 ｜ 上手步骤见[第 4 章 逐级对齐](../guide/04-level-by-level.md)

# 抓数：抓什么、怎么抓、怎么证明没抓坏

## 一句话

两边在训练里把中间量按同一种格式写出来，janus 只读这种格式。抓数有两条铁律：

- **存原始位模式**；
- **开不开钩子结果逐位相同**。

格式的字段定义（目录结构、`meta.json`、`metrics.jsonl`、张量索引、规范名）以 [`reference/dump-format.md`](dump-format.md) 为准，这里讲做法和道理。

## 抓什么

| 内容 | 频率 | 用在哪 |
|---|---|---|
| 逐步标量（见注 1） | 每步 | L4 |
| 每个 rank、每个 microbatch 的输入指纹（见注 2） | 每步 | L0 数据 |
| 每个 rank 持有的每个参数与浮点 buffer 的 sha256（全量） | 训练前、训练后 | L0 权重、DP 副本一致性 |
| 实例化出来的模块树与参数清单 | 训练前 | L0 结构 |
| 张量（见注 3） | 指定的步（默认第一步） | L2、L3 |
| 全量时另抓（见注 4） | 指定的步 | L1 优化器重放、L2、L3 |

注 1（逐步标量）：
- 各项 loss、学习率、实际生效的权重衰减；
- 梯度范数、参数范数、梯度零元素数、是否跳过更新；
- 每层路由偏置均值 / 标准差、专家负载。

注 2（输入指纹）：tokens / labels / loss_mask / position_ids 的 sha256 与非 padding 数。

注 3（张量）：
- 逐层各子模块的前向输出；
- 路由（logits / 概率 / 选中的专家 / 专家计数）；
- 逐 token loss、输入 token。

注 4（全量时另抓）：
- 子模块输出的梯度；
- 逐参数梯度（进优化器前、已归约）；
- 更新前后的 FP32 主权重、Adam 两个动量；
- 更新前后的 buffer。

## 为什么张量只在第一步逐层抓

逐层比较要求两边权重相同，只有第一步满足。

- 训练一步后两边权重就各自走了；
- 第 1000 步两次 GPU 运行的激活 rel_L2 已到 0.15，见 `levels.md`。

后面的步只抓标量。

要把第一步的误差隔离到单层，用**逐层注入**：把参照运行上一层的输出喂给被测这一层（GPU 侧已实现，见下）。

## 三条规矩，以及为什么

1. **存原始 dtype 的位模式**。BF16 就存 BF16，不先升 FP32。
   - 否则 ULP 数不准（BF16 按 FP32 数会放大 65536 倍），逐位比较也失去意义。
2. **开钩子与不开钩子逐位相同**。抓数改了数值，抓到的就不是要比的那个东西。每种配置都要先过这一关（`determinism` 检查）。
3. **抓数时独占节点**。别的作业抢资源可能改变执行（kernel 选择、通信时序）。

另外：

- 浮点标量写成能往返还原的十进制，不截断；日志里只打几位小数的值不能当数据源。
- 产物写完即只读，重跑用新的运行名。

## GPU 侧（Megatron）：回调怎么挂

实现：`janus/dump/megatron.py` 的 `JanusDump`，以 Megatron-Bridge 训练回调的形式挂上，**不改 Megatron 源码**。

### 挂法与各钩子

```python
from janus.dump.megatron import JanusDump, wrap_forward_step
cb = JanusDump(out_dir, meta, dump_steps={1}, full=True)
pretrain(cfg, wrap_forward_step(forward_step, cb), callbacks=[cb])
```

- 前向输出：在各子模块上挂 forward hook，框架模块名按 `janus/dump/names.py` 换成规范名（如 `layers.2.attn.proj.out`）。
- 反向：全量抓数时，在前向输出上注册梯度 hook，得到 `<激活名>.grad`。
- 参数梯度、主权重、动量：包一层优化器的 `step`，在更新前后各读一次。
- 路由：包住路由的门控计算，抓 logits、选中的专家与计数。
- 输入指纹：包住数据迭代器，每个 microbatch 记 sha256。
- 激活重算开着时，同一步第二次前向另存为 `<名>.recompute`，用来比「重算的前向」和「首次前向」。
  - 路由另存 `router.logits.recompute`、`router.topk_idx.recompute`、`router.probs.recompute`；
  - 开路由回放时用来核对重算用的专家与首次前向逐位相同。
- 逐层注入（`inject_from`）：在抓数那一步，第 i 层的输入换成参照运行同一步的 `layers.{i-1}.out`，第 0 层换成 `embed.out`。

混元 3 的训练入口已经接好：

```bash
models/hunyuan3/frameworks/megatron/run.sh <运行名> <步数> --janus-run <目录> --side golden-<n> --dump-steps 1 [--full] [--deterministic]
```

参数说明见 `models/hunyuan3/frameworks/megatron/README.md`。

### GPU 侧的限制

都在第一阶段碰到过：

- 抓张量只抓 rank 0 的第一个 microbatch。
- 全量抓数要关分布式优化器（每个 rank 上要有完整的主权重和已归约梯度）。
  - 对照运行的优化器配置要一致，否则比出来的是配置差。
- 专家参数默认只有 rank 0 持有的那部分。
  - 全量抓数默认每层只留 rank 0 本地专家的前两个的权重（其余太大）；
  - 参数 sha256 清单仍是全量；
  - `gather_experts=True`（默认关）时，rank 0 所在专家并行组里其他 rank 的本地专家在全量抓数步点对点收到 rank 0，按全局专家号命名（`...linear_fc1.weight<ep_rank·本地专家数 + j>`，rank 0 上本地号即全局号），各种类（梯度、主权重前后、BF16 参数、优化器状态）都收；
  - `full_kinds`（默认全写）限定全量抓数写哪几种，`forward_acts=False` 不抓前向激活：只要参数 / 状态时用来控制体积。
- 专家并行分发后的缓冲区（`mlp.experts.out`）行数取决于别的 rank，跨运行不能逐位比。
  - 比 token 空间的 `mlp.out`。
- 开张量并行 / 上下文并行时，rank 0 上的激活是切开的。
  - 不逐元素比，只比整网标量（`phase1-tp-cp`）。
- 参数梯度用 Megatron 原名（`grad.decoder.layers.<i>....`）。
  - 两边的名字与布局映射在 `models/hunyuan3/model/name_map.yaml`；
  - TPU 侧抓数按它把状态写成同名、同布局的张量，所以跨侧直接按名字比。

### GPU 侧已验证不改数值

- 开抓数回调的确定性运行与不开的逐步逐位相同（`models/hunyuan3/frameworks/megatron/README.md`「已验证」）；
- 轻量抓数的运行对全量抓数的运行（两边优化器配置一致），30 步逐位相同（`phase1-hooks`）。
  - 轻量抓数：第 1 步只抓前向张量，不挂梯度钩子、不包优化器；
  - 逐位相同的量：数据指纹与 loss、梯度范数、参数范数、零元素数、学习率、路由偏置。

## TPU 侧（MaxText）：原则

三条原则（实现见下一节）：

1. **生产图里只抓本来就落到 HBM 的值**。
   - 层与层之间的激活、参数、梯度、优化器状态本来就要写回 HBM，把它们作为额外输出取出来，原则上不需要改动计算。
   - 这和 GPU 侧不同：GPU 的 hook 只是在已经算好的张量上读一下。
   - TPU 是整图编译，往图里加输出仍可能改变编译器的融合与调度决策，从而改变舍入位置。所以有第 2 条。
2. **每种配置先过「开不开钩子逐位相同」**。
   - 换模型尺寸、并行方式、编译开关，都要重新证明一次，不能沿用别的配置的结论。
3. **融合内部的值不在生产图里抓**。
   - 只在片上内存（VMEM）里存在、从不写回 HBM 的中间值（例如一个融合 kernel 内部的部分和），硬抓会逼编译器把它写出去，图就变了。
   - 这类值改用两种办法：
     - 读编译后的 HLO，看舍入点和融合边界；
     - 或者单独编一个**诊断版**（拆开融合）来抓，诊断版的结论只用于定位，不当生产图的结论。
   - 报告里要标明哪些量是从诊断版或 HLO 得出的。

## TPU 侧（MaxText）：怎么抓

- 实现：`janus/dump/maxtext.py`。
- 混元 3 的训练入口：`models/hunyuan3/frameworks/maxtext/train_hy3.py`（用法与已核对的项见 `models/hunyuan3/frameworks/maxtext/README.md`）。
- **不改 MaxText 源码**：训练循环自己写，调 MaxText 的 `setup_train_loop` 与 `train_step`。

### 怎么取值


- **状态类，零扰动**：
  - 参数（FP32 主权重）、Adam 两个动量、路由偏置都在训练状态里，训练步与训练步之间从设备读出；
  - 逐步标量取 MaxText 训练步本来就输出的 loss 与范数；
  - 学习率、权重衰减在主机上按配置算。
- **图内的值，作为额外输出**：只在抓数步换一个加钩子的训练步。
  - 激活：按模块实例路径打标记，给几个模块类的 `__call__` 套壳，用 nnx 的 `sow` 存成 Intermediate，随 MaxText 原有的 Intermediate 通道带出；
  - 参数梯度：拦 `value_and_grad` 的输出，取 MaxText 自己算出的未裁剪梯度。
  - 额外输出放进训练步返回的 metrics，训练状态的结构不变。
  - 只留设备 0 的那几行，在图里切片。

### 每个抓数步自检

- **每个抓数步自检**（原则 2）：
  - 往下训练一律用不加钩子的程序的输出；
  - 加钩子的程序在同一输入状态的副本上跑、只取额外输出；
  - 两者输出的训练状态与 loss 逐叶子逐位比，结果写进 `meta.json` 的 `hooks.selfcheck`；
  - 不逐位时额外输出标为诊断值。
- 抓数档位分几档（`TIERS`），`--probe-tiers` 一次测完哪一档在当前配置上不改数值。
- v5p 上「全量」档实测**不**逐位，CPU 上逐位——原则 2 说的「每种配置先证明」在这里是实打实的。
  - 根因已查到：两份程序优化前的 HLO 只差多出的输出，是 XLA 为了物化这些输出改了 fusion 划分，FP32 规约的分组随之改变。
  - 出处：`models/hunyuan3/frameworks/maxtext/README.md`「上 TPU 才能核对的九项：结果与现状」第 2 条。
  - 所以在 TPU 上，「加输出」本身就可能改数值。
  - 要的是生产图里的值时，改用把被测部分单独编译的回放（逐层隔离回放）。
- 另外整次运行再和一个不加钩子的运行逐步比（`L0.hooks.determinism`）。

### 不用钩子的正式证据、名字与诊断版

- **不用钩子也能拿到的正式证据**，三条路：
  - 梯度：从训练步之间读出的 Adam 一阶动量反推（g = (m_k − β1·m_{k−1}) / (1 − β1) / 裁剪系数，零扰动）；
  - 前向：用把被测部分单独编译的回放（逐层、单算子、整网只前向），回放程序本身就是测量对象；
  - 「观测构建」：每一步都跑同一个加钩子的程序，先证明它和生产构建的差在 TPU 执行噪声以内（标定三关），再把它的中间值当正式证据。
  - 三条路的用法见 `models/hunyuan3/frameworks/maxtext/README.md`「在 TPU 上拿第 1 步的正式证据」。
- **名字与布局**：
  - 参数类张量换成 Megatron 名字与布局（qkv 按 KV 组交错、gate|up 拼接、专家按 EP8 的持有关系换本地号）；
  - 参数清单也按 GPU 的 rank 分；
  - 激活写规范名、转成 GPU 的 [S, B, …] 布局。
- **诊断版**（原则 3）：`models/hunyuan3/frameworks/maxtext/probe_forward.py` 单独编一个只有前向的程序。
  - 用来核对名字与布局，结论不当生产图的结论。

### TPU 侧的限制

- 激活的梯度（`<名>.grad`）不抓：余切量不落在任何模块输出上，要往图里插探针输入才能拿到，会改图。
- 路由的 top-k 在稀疏路径的 `shard_map` 里抓不出来。
  - `topk_idx` / `probs` / `counts` 由图内抓到的同一份 FP32 分数在主机上重算（派生量，名字映射里注明）。
- 混元 3：训练步里的跨平台逐层注入、跨平台路由回放（把 GPU 选中的专家喂给 TPU）还没有。
  - 原因：扫描层里拿不到层号；混元 3 的层还没接 MaxText 的 `forced_routed_experts`。
  - 混元 4 第 1 步已有「灌 GPU 的选择」，只作诊断、不进判定（[差异清单 G12](tpu-vs-gpu-differences.md#g12-第-1-步正式判定用-tpu-自选的路由灌-gpu-的选择只作诊断)）。
  - 它和 GPU 生产 golden [`rc`](../glossary.md#rc) / TPU `rmap` 里的「路由回放」不是一回事。
  - 后者是同一次运行里重算沿用自己首次前向选中的专家，两侧都已实现（[`levels.md`](levels.md#l2--单层逐层逐子模块)「L2 · 单层：逐层、逐子模块」）。
- 加钩子的训练步只支持不做梯度累积。

### 混元 4 怎么抓

训练入口 `models/hunyuan4/frameworks/maxtext/train_hy4.py`（参数见文件头与 `--help`）。和混元 3 的不同在于不加钩子，图内的值不进训练步：

- **取什么**：状态类全量抓数（`--full`）：更新前后的 FP32 主权重、Adam 两个动量、路由偏置，写 GPU golden 的名字与 HF 布局；逐步另记 loss、范数、分组参数平方和、输入指纹。
- **何时取**：`--dump-steps` / `--full-steps` 指定的步，在训练步与训练步之间从训练状态读出，训练步本身不变。
- **第 1 步梯度**：正式 L3 用训练步实际用的梯度，由第 1 步全量抓数里的 Adam 一阶动量反推（相对误差约 1e-7）。文件完整性核导出文件的 sha256 与反推梯度的全局范数（C 口径之前的配置 `tz-off-rep` 上做过：[`t1zo-step1-train-grad-file-check.json`](../../models/hunyuan4/alignments/01-mini/results/t1zo-step1-train-grad-file-check.json)；混元 4 [`PROTOCOL.md`](../../models/hunyuan4/PROTOCOL.md#0-现行口径按变更记录汇总) 第 0 节；现行 C 口径的第 1 步全量抓数 `s1f-c` 上也做了：728 个张量 sha256 无不符，反推范数相对差 4.3e-7，[`t1zc-step1-train-grad-file-check.json`](../../models/hunyuan4/alignments/01-mini/results/t1zc-step1-train-grad-file-check.json)）。
  - `--diag-step1` 单独编一个只算梯度、不更新的程序（与训练同一个梯度累积函数），写 HF 名字与布局的 FP32 梯度后退出。它和训练步分开编译，同一状态、同一批数据的梯度不逐位相同（205 个张量相对差中位数 2.9e-4），只作旁证（PROTOCOL 变更记录 2026-11-09）。
- **激活**：梯度累积 2 时训练步里不抓（同混元 3）；接线核对用 `probe_forward.py` 的诊断前向。
- **自检**：抓与不抓整次运行逐步比。C 口径之前的配置上：第 1 步全量抓数的 `tz-off-rep` 对不抓的 `tz-off-qkc` 1000 步 47 个逐步量逐位相同（[`t1zo-determinism`](../../models/hunyuan4/alignments/01-mini/results/t1zo-determinism.report.md)）。换配置要重做这一条：C 口径上没重做。`s1f-c` 是单独跑的 1 步抓数，它与不抓的 `tc-a` 只核了第 1 步 `raw_grad_norm` 与裁剪后范数两个量，逐位相同（[mini `goldens.md`](../../models/hunyuan4/alignments/01-mini/goldens.md) `s1f-c` 一行；`tc-a` 一侧读的是运行目录里的 `metrics.jsonl`，不在仓库里）；loss 等其余逐步量，以及抓数改没改程序，都没核。

## 抓完以后：两侧共同的检查

- 每个张量的 sha256 必须能复算；缺文件、校验和不符、同一步里张量重名，读入就报 error。
- 运行目录写完最后一个文件后创建 `DONE`；没有 `DONE` 的目录视为没写完，janus 拒绝读。
- `coverage` 检查确认「该比的都比到了」：期望清单里缺一个，就报 error，不报一致。
