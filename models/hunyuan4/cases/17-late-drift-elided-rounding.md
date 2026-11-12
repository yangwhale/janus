[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 17

# 案例 17：后段为什么慢慢偏开：iHC 流梯度、被省掉的 BF16 舍入，以及一次扫全图找同类

混元 4 mini 训练 1000 步，TPU 的主 loss 在第 100–550 步和 GPU 分不开（更早还有一个小鼓包，另有原因），第 550 步以后一路比 GPU 高，到第 976–1000 步高 1.8e-5。前向没差，MTP、KD、libtpu 版本都不是原因。去掉 iHC，偏差就消失。再往下拆，主因是 iHC 流梯度的一处舍入写法：GPU 每加一次舍一次 BF16，TPU 在 FP32 里加完才舍。这个结论两头都验证过。
补上这处舍入以后还有残余，于是换了个办法：不再一处一处猜，而是**先编译、再扫描**。把整个训练步离线编译出来，在编译产物里精准匹配「代码里声明了要舍 BF16、编译器却在融合里把它省掉」这一种模式，一次找出全图所有同类位置，问题的范围就清楚了。

> **状态：主因已定位（实测，两头验证）；残余在查。** 对齐配置从 2026-11-11 起是 C 口径（三个「照 GPU 一定舍 BF16」的开关全开，见下「先补背景」），主 loss 判档刚好过线，但后段的残余仍看得出来。全局清单给出了残余的候选范围（三类位置）。GPU 上对应的两个诊断开关（ATTNIN、WGRAD）在跑，没有结果之前不下结论。

> **推荐先读[第九幕「全局清单」](#第九幕全局清单先编译再扫描)。** 这一步的方法不依赖混元 4，任何「TPU 和 GPU 舍入位置对不齐」的问题都能直接拿去用；工具是 [`analysis/elided_rounding_scan.py`](../analysis/elided_rounding_scan.py)。

## 先补背景：`astype` 会被省掉，`reduce_precision` 不会

混元 4 的计算习惯是「FP32 算、BF16 存」：一段运算在 FP32 里做完，结果舍成 BF16 再交给下一步。GPU（PyTorch）上写 `.to(bfloat16)`，这次舍入一定发生。MaxText（JAX）里对应的写法是 `x.astype(bf16)`，编译后变成一对 `convert(f32→bf16)` → `convert(bf16→f32)`（下一步又要 FP32 时）。

TPU 的编译器默认开着 excess precision：允许把中间结果留在 FP32、少舍几次。这对来回 convert 落在同一颗融合 kernel 里时，编译器直接拿 FP32 的值往下算，这次舍入就没了。`jax.lax.reduce_precision(x, 8, 7)` 是另一种写法：值还是 FP32，但尾数被截到 BF16 的 7 位。编译器把它当成一次真的运算，**不会省**。所以要「照 GPU 一定舍」，就把 `astype` 换成 `reduce_precision`。

本案例反复出现三个这样的开关，都在 `maxtext` 的混元 4 分支上，都只为照 GPU 的舍入位置（编号是[变更台账](../CHANGES-LEDGER.md)里的号）：

| 开关 | 改了哪里、改成什么 | 后文简称 |
|---|---|---|
| `ihc_force_bf16_rounding`（M36） | iHC（超连接）每层两处：送进分支的四路加权和（分支输入），以及残差更新后的流。原来是 `astype(bf16)`，换成 `reduce_precision` | iHC 强制舍 |
| `hc_head_force_bf16_rounding`（M52） | hc_head 把四路流收成一路以后、进末层 RMSNorm 之前，强制舍 BF16（`reduce_precision`）。它的转置把反向传回来的梯度（余切）也舍一次。GPU 这里是 `.to(bfloat16)` | hc_head 强制舍 |
| `ihc_stream_grad_gpu_order`（M58） | 两部分。① 流梯度：用一个 `custom_vjp` 函数 `stream_fanout` 把流复制成 3 份（加权和、门控、残差三处各用一份；hc_head 里是 2 份）。反向时每份的梯度先舍 BF16，再按 GPU autograd 的顺序逐个相加，每加一次舍一次，全用 `reduce_precision`。② 分支输出：注意力、MLP 的输出转 FP32 后用 `reduce_precision` 舍到 BF16 网格，它的梯度同样被舍。M59 `ihc_stream_grad_gpu_order_parts` 可以只开 ① 或只开 ② | 流梯度照 GPU 舍 |

**C 口径**是 2026-11-11 起的对齐配置，就是这三处全开：

| 配置 | iHC 强制舍（M36） | hc_head 强制舍（M52） | 流梯度照 GPU 舍（M58） |
|---|---|---|---|
| 对照（C 口径之前的对齐配置） | 开 | 关 | 关 |
| 只开 M58 | 开 | 关 | 开 |
| 只开 M52 | 开 | 开 | 关 |
| **C 口径** | 开 | 开 | 开 |

下文「只开 M58」「只开 M52」都是在 M36 已开的基础上。

## 先看结论：三层

**第一层：主因是 iHC 流梯度的舍入写法（实测，两头验证）。**

iHC（超连接）把每层的隐藏状态扩成四路「流」。GPU 上一路 BF16 的流在一层里有三处用到，反向时 autograd 把三份 BF16 梯度按「加权和 + 门控」再加「残差」的分组加起来，每加一次舍一次 BF16（CPU 上用 GPU 同一段代码逐位核过）。TPU 的代码里同样写了 BF16，但 excess precision（编译器允许把中间结果留在 FP32、少舍几次的开关，默认开）让编译器在融合里把这几次舍入省掉，三份梯度实际在 FP32 里加完才舍。

两头各改一处：

| 改哪边 | 改成什么 | 第 551–1000 步主 loss 对 GPU 16 份均值 | 份数 | 检验 |
|---|---|---|---|---|
| TPU（不开 M52） | 照 GPU 补上舍入（`ihc_stream_grad_gpu_order`，M58） | +9.4e-6 → +1.9e-6 | 对照 4 份、改后 5 份 | 置换检验 p 0.016 |
| GPU | 照 TPU 不舍（`HY4_TPU_LIKE_STREAM`） | GPU 自己对 GPU 留一法 +3.9e-6，4 份都正 | 4 份 | p 0.0025 |

TPU 照 GPU 写法就平，GPU 照 TPU 写法就飘。两边都去掉 iHC 后，两边都不飘。

**第二层：TPU 更准，但对不齐。**

按单次运算看，三份梯度在 FP32 里加完只舍一次，比每加一次舍一次少舍两次，离精确值更近。这是按舍入次数推出来的；整条训练轨迹上谁离精确值更近，没有测。TPU 没有算错，只是和 GPU 的舍入位置不同。对齐要的是「同样的舍入位置」，所以 TPU 这边要补上这些舍入。生产训练不需要补（[根源一览第 30 行](../ROOT-CAUSES.md#r30)：不开）。

**第三层：残余，以及全局清单给出的范围。**

C 口径下补全了 iHC、hc_head 两处，后段仍剩约 +7e-6：6 份第 551–1000 步 +6.9e-6，6 份里 5 份尾部在，和不修的对照分不开（p 0.32）。残余的机制还没查清，下面这几条都是查到的事实：

- M52（hc_head 输出照 GPU 舍）一开，M58 的作用就测不出来了。
- 拆到最后，两者的相冲点在 hc_head 那一次余切舍入上。
- GPU 上拿掉这次舍入，前段出现鼓包、后段反而变低。「只开 M58 时后段变平」里有一部分是两处误差互相抵消，不全是修好了。

于是不再逐处猜，改成把整步程序扫一遍：凡是「代码写了 f32→bf16、同一颗融合里又转回 f32」的位置都列出来。扣掉已经用 reduce_precision 补上的 iHC、hc_head，HLO 里还有 272 处往返。按类别归下来，值得查的是三类：

1. 分支输入的梯度合流，典型是注意力输入：q、kv 下投影和输出门三份梯度在 FP32 里相加；
2. 各线性层每个微批的 dW 不舍 BF16（GPU golden 关了 `gradient_accumulation_fusion`，每个微批的 dW 先出 BF16）；
3. 前向的几处：输出门、fc1、路由专家的 act×up。

其中整颗 kernel 只含这种往返、不含任何真舍入的有 41 颗，LLO 里打包指令全部是 0 条，确认舍入确实被省掉了。残余是不是来自这三类，要等 GPU 上的诊断开关跑完；在那之前，它们只是候选。

## 怎么查到的

下面按时间顺序讲。所有数字出自 [mini 对齐细节 D30](../alignments/01-mini/DETAILS.md#d30)、[知识库「mini 主 loss 共同偏高」](../knowledge.md#mini-主-loss-共同偏高要点与出处)与那里列的结果文件；PR 是 。

尺子统一是：TPU 对 GPU 16 份的逐步均值，取第 551–1000 步（「后段」）的均值；GPU 自己的波动用 16 份轮流当被测的留一法量。两组运行之间的差用置换检验（每格 4–6 份，换执行方式的重复：默认、scoped VMEM 64 / 32 MiB、关延迟隐藏调度等）。

## 第一幕：现象，以及跑到 2000 步

C 口径之前的现行比较，主 loss 带符号均值 r 1.55（r = 差 ÷ GPU 重跑的噪声上限）。开、关 base2 各 4 份 TPU，主 loss 都比 GPU 16 份均值高，平均 +4.3e-6。

按 25 步分块看偏差怎么随步数变（TPU 8 份对 GPU 16 份，[`mini-main-loss-drift.json`](../alignments/01-mini/results/mini-main-loss-drift.json)）：

| 步 | 偏差 | 读法 |
|---|---|---|
| 第 1 步 | −2.0e-6 | 8 份都偏低 |
| 第 51–90 步 | 最高 +7.2e-6（第 71–80 步） | 一个小鼓包，8 份都偏高 |
| 第 100–550 步 | 0 附近 | |
| 第 601–625 步 | +4.1e-6 | 第 550 步起一路变大 |
| 第 751–775 步 | +1.1e-5 | |
| 第 976–1000 步 | +1.8e-5 | 每块 8 份里 7–8 份偏高；GPU 尺子 2–7e-6 |

2–1000 步的 +4.3e-6 几乎全部来自第 501–1000 步。学习率是常数，和它无关。分块偏差与 GPU 的 param_norm 同步（第 501 步起相关 0.96），与梯度范数无关（−0.11）。同一批运行的 MTP loss 反方向走，第 751–1000 步 −3.2e-5。

先确认两件事。

- **不是前向的差。** 同一份权重上 TPU、CPU FP64、CPU BF16（GPU 写法）各算同几批：6 个状态 19 批，TPU 比 GPU 写法高 +0.83e-6 ± 1.15e-6，不到后段差距的一成（[`mini-main-loss-same-state.json`](../alignments/01-mini/results/mini-main-loss-same-state.json)）。所以偏差是训练里攒出来的，要查更新路径，不是查前向。
- **不会越飘越大。** 延长到 2000 步，TPU 1 份对 GPU 2 份：第 1001–1250 步 +2.9e-5 到顶，之后三段 +5.4e-6、+7.4e-6、+4.1e-6，落回 GPU 两份的范围（[`mini-ihc-ablation.json`](../alignments/01-mini/results/mini-ihc-ablation.json) 的 `x2000`；2 份只有 1 对，尺子很粗）。

前段那个小鼓包另有原因：hc_head 把四路流收成一路后那次 BF16 舍入，TPU 没做。补上它的开关是 `hc_head_force_bf16_rounding`（M52，[根源一览第 28 行](../ROOT-CAUSES.md#r28)）。这个开关后面还会出场。

## 第二幕：隔离实验，排除 MTP、KD、GPU 索引器 RoPE、libtpu

后段先从「是不是某项 loss 带出来的」查起（[`mini-loss-isolation.json`](../alignments/01-mini/results/mini-loss-isolation.json)）。两侧同一语义把系数设 0，每种配置 GPU 4 份、TPU 2 份，TPU 对同配置 GPU 均值：

| 配置 | 后段 | 两份的后段斜率 | GPU 留一法斜率范围 |
|---|---|---|---|
| 对照（同一条提交路径） | +1.48e-5 | — | — |
| A 关 MTP | +1.16e-5 | +4.3e-6、+2.3e-6 | −0.7e-6 到 +0.4e-6 |
| B 关 KD | +9.0e-6 | +0.5e-6、+4.0e-6 | — |
| C 都关 | +8.7e-6 | +1.2e-6、+2.7e-6 | −0.7e-6 到 +1.0e-6 |

三种都还飘，所以不是这两项 loss 造成的。另外两个候选也排除了：

- **GPU 索引器 RoPE。** 真实宽度上 GPU 的 KD 偏高约六成来自索引器 RoPE 在 BF16 里算。mini 上 GPU 照同样写法改成 FP32（`HY4_INDEXER_ROPE_FP32`，台账 G34）跑 1 份，TPU 对它后段仍是 +1.24e-5。
- **libtpu 版本。** 0.0.52 的 golden 对 GPU 16 份，后段 +7.7e-6，同配置 0.0.46 的 4 份 +9.8e-6，一样飘。

在这之前（#385）还排除了五个候选（每个 2 份，后段都落在 70 份 TPU 运行的范围里）：MTP 不给 hc_head 参数梯度、MTP 与输出头共用末层 norm、RoPE 照 GPU 的 BF16 写法、MTP 残差强制舍 BF16、RMSNorm 输出的余切强制舍 BF16。TPU 自比的 35 个 1000 步运行换了 libtpu、v7x、并行方式、重算、注意力实现等，第 751–1000 步主 loss 全部为正；第 1 步梯度相对差到 6e-3 的改动都去不掉它。

## 第三幕：结构消融，去掉 iHC 尾部就消失

剩下的嫌疑在模型结构里。最直接的办法是两边一起把 iHC 拿掉，看还飘不飘：

- GPU 用 Megatron-Core 的 TransformerLayer 加 bias-dropout-add（与 MTP 层同一种写法），MaxText 用 `mhc_expansion_rate=1`。
- 随机初始化用 `make_tiny_hf.py --no-ihc`，其余 167 个张量与 `hy4-mini-seed0` 逐位相同；CPU 上同一份权重、同一批数据，两边第 1 步 loss 相对差 −1.9e-6。

1000 步 GPU 3 份、TPU 2 份：TPU 后段 +1.9e-6、+0.7e-6，斜率 +0.5e-6、+0.15e-6，都在 GPU 范围（−1.0e-6 到 +1.2e-6）里（[`mini-ihc-ablation.json`](../alignments/01-mini/results/mini-ihc-ablation.json) 的 `noihc`；GPU 3 份只有 3 对，尺子粗）。

范围从整个模型缩到 iHC 一带。

## 第四幕：开关拆半

iHC 里 TPU 和 GPU 不同的地方，读代码能列出来：GPU 上一路 BF16 的流在一层里有三处用到，三份梯度各舍 BF16、在 BF16 里相加，每加一次舍一次；分支输出（注意力、MLP 的输出）也是 BF16。TPU 的代码写的是 `astype(bf16)`，编译器在 excess precision 下可以把这几处的往返省掉。

开关 `ihc_stream_grad_gpu_order`（M58）用 reduce_precision 把这几处舍入补上。编译器不会省 reduce_precision，这是它和 `astype` 的区别。它又能拆成两半（M59）：

| 配置（C 口径之前，只开 M36 的基础上） | 后段 | 份数 |
|---|---|---|
| 对照 | +9.4e-6，4 份尾部都在 | 4 |
| M58 全套 | +1.9e-6，2 份尾部在 | 5 |
| 只补流梯度 | +0.6e-6，斜率平 | 1 |
| 只补分支输出 | +3.0e-6，斜率平 | 1 |

M58 全套对对照差 −7.5e-6，置换检验 p 0.016。libtpu 0.0.52 上开 M58，后段从 +1.19e-5 降到 +5.2e-6。两半各 1 份，只说明两半都在起作用，分不出谁大。

## 第五幕：两头验证

单看 TPU 一侧，「补上舍入后变平」也可能是巧合。所以反过来在 GPU 上做：GPU 一侧加三个诊断开关（台账 G37），各只把一处改成 TPU 的样子，对 GPU 16 份留一法：

| GPU 开关 | 改成 TPU 的什么 | 后段 | 份数 | 检验 |
|---|---|---|---|---|
| `HY4_TPU_LIKE_STREAM` | 流转 FP32 给三处用，梯度在 FP32 里加完只舍一次 | +3.9e-6，4 份都正（+3.2e-6 到 +4.8e-6），斜率 +2.2e-6 | 4 | p 0.0025 |
| `HY4_TPU_LIKE_HEAD` | hc_head 收拢输出不舍、FP32 进末层 norm | +0.2e-6；但第 51–100 步 +3.7e-6 | 4 | 后段 p 0.89；前段 p 0.001 |
| 两处都改 | | +3.5e-6 | 4 | p 0.015 |

GPU 照 TPU 写法流梯度，GPU 自己就出现尾部。GPU 留一法 16 份里后段 ≥ +3.0e-6 的只有 3 份，只改流梯度那 4 份全在其中。只改 hc_head 则正好复现 TPU 的前段鼓包。

至此两头都验证了：

- TPU 照 GPU 补舍入，尾部下去；
- GPU 照 TPU 不舍，尾部出现；
- 两边都去掉 iHC，都不飘。

主因定为 iHC 流梯度的舍入写法（[根源一览第 30 行](../ROOT-CAUSES.md#r30)）。

## 第六幕：C 口径没修好，M52 和 M58 相冲

对齐配置随后改成 C 口径（项目负责人定）：M36（iHC 分支输入与流照 GPU 舍）、M52（hc_head 输出照 GPU 舍）、M58（iHC 流梯度照 GPU 舍）一起开，三处都是 GPU 的写法。照理前段、后段都该好。实测前段好了，后段没好（撤回了 #411 的「已定位」，#430、#438、#448 补齐份数和拆格）：

| 配置 | 后段 | 尾部在的份数 | 对对照（只开 M36） |
|---|---|---|---|
| 只开 M58 | +1.9e-6 | 5 份里 2 份 | p 0.016 |
| 只开 M52 | +12.5e-6 | 4 份都在 | 前段鼓包消失（p 0.029） |
| C 口径（M36 + M52 + M58） | +6.9e-6 | 6 份里 5 份 | p 0.32，分不开 |

C 口径比只开 M58 大 +5.0e-6（p 0.024）。M52 开着时，再补 M58 全套或任一半都测不出作用（[`mini-factorial-c-parts-vs-m52.json`](../alignments/01-mini/results/mini-factorial-c-parts-vs-m52.json)）：

| 在 M36 + M52 之上，各 4 份 | 后段 | 对只开 M52 |
|---|---|---|
| M58 全套 | +6.7e-6 | −5.9e-6，p 0.14 |
| 只补流梯度一半 | +7.9e-6 | −4.6e-6，p 0.14 |
| 只补分支输出一半 | +10.8e-6 | −1.7e-6，p 0.54 |

再把 M52 拆开（各 4 份，台账 M62，[`mini-factorial-c-ab-vs-m52.json`](../alignments/01-mini/results/mini-factorial-c-ab-vs-m52.json)）：

- **(a) M52 只在前向舍，余切原样传回**：后段 +3.7e-6，对只开 M52 p 0.029，和只开 M58 分不开；但前段鼓包回来了。
- **(b) M58 不覆盖 hc_head**（配完整 M52）：后段 +4.3e-6，p 0.057；前段鼓包没有。

推断：M52 消前段鼓包，靠的是它在 hc_head 输出处舍余切；同一次舍入又让 M58 在后段的作用不见了。相冲点在 hc_head 输出处的余切舍入。

注意：这个相冲在 GPU 上看不到。GPU 上两处一起改（上一幕的「两处都改」）和只改流梯度差不多。

## 第七幕：CPU 逐位对照，和 v5p 上编出来的 LLO

相冲点落在 hc_head 加末层 norm 这一小段。先在 CPU 上把这一段三方各算一遍反向，逐位比（[`hc-head-order-probe.json`](../alignments/01-mini/results/hc-head-order-probe.json)，[知识库「CPU 上逐位对照」](../knowledge.md#hc_head-加末层-norm-的反向cpu-上逐位对照)）。三方是：GPU 照 `Hy4FinalNorm` 逐行写、MaxText 真模块关省舍入、MaxText 真模块开省舍入。

- 语义上唯一实质不同的一步：MaxText 给 MTP 另做一次末层 norm，两路各做一次 norm 反向再相加；GPU 是先把两路梯度加起来，只做一次 norm 反向。
- C 口径再开 `mtp_input_shared_final_norm`（M53）后，和 GPU 只差 FP32 累加顺序：流梯度 dx 2.1e-5，逐位相同 99.993%。
- 上卡 4 份：C + 共用 norm 后段 +5.0e-6，对 C 口径 p 0.40，后段没变。

语义对齐到只差累加顺序，后段照旧。剩下的只能是 TPU 后端实际省掉了哪些舍入，这要看编译产物。

于是把这一段写成最小复现，CPU 上为 v5p 离线编译，读优化后 HLO 和 LLO（[`hc-head-llo-probe.json`](../alignments/01-mini/results/hc-head-llo-probe.json)，[知识库「同一段在 v5p 上编出来」](../knowledge.md#同一段在-v5p-上编出来哪几处舍入被省掉)）。读法沿用 iHC 那次（[TPU 自比 · 影响逐件 05](../alignments/04-tpu-self/impacts/05-ihc-bf16-rounding.md)）：

> 开着 excess precision 时，**融合内部的来回 convert 不产生舍入指令**，跨融合边界的值按声明的 BF16 写回内存。HLO 里 `convert(f32→bf16)` 紧跟 `convert(bf16→f32)` 还在，LLO 里那颗 kernel 一条 BF16 打包指令（`vpack.*.bf16`）都没有。加上 reduce_precision 后，同一颗 kernel 里出现打包指令（iHC 的均方 kernel 多 256 条，hc_head 这一颗多 688 条）。

读出来的结果：

- C 口径和只开 M58 在这一段只差一处：**进 hc_head 收拢的梯度舍不舍 BF16**。C 舍（和 GPU 一样），只开 M58 留 FP32。
- 两路 norm dx 各自那次转 BF16，四种配置都被省掉了。这和 GPU 不同，但各配置一样，解释不了它们之间的差。

这一段 C 更像 GPU，后段却是不舍的那两种（只开 M58、(a) 格）更低。按「哪一处没照 GPU 舍」，在这一段找不到相冲的原因。

## 第八幕：GPU HEADCOT，发现是抵消

C 和只开 M58 在 TPU 上只差这一次舍入。那就在 GPU 上把这一次舍入拿掉，看 GPU 会怎样。GPU 诊断开关 `HY4_TPU_LIKE_HEADCOT=1`：末层 norm 的反向 dx 留 FP32 直接给收拢输出，其余不变。4 份，对 GPU 16 份留一法（[`mini-gpu-headcot.json`](../alignments/01-mini/results/mini-gpu-headcot.json)）：

| | 第 51–100 步（前段） | 第 551–1000 步（后段） |
|---|---|---|
| GPU 拿掉这次舍入 | +4.9e-6（p 0.0002） | −2.9e-6，4 份都负（p 0.018） |
| TPU 只开 M58（这次不舍） | +6.1e-6 | +1.7e-6 |
| TPU C 口径（这次舍） | −0.5e-6 | +6.7e-6 |

方向一样：这次舍入拿掉，前段出鼓包，后段被往下压。推断：

1. M52 消前段鼓包，靠的就是这次余切舍入。
2. 只开 M58 时后段变平，有一部分是这次舍入没照 GPU 做带来的下压，**是两处误差抵消，不全是修好了**。
3. C 口径在这一段和 GPU 一致，它后段剩下的约 +7e-6 来自这一段以外。

问题于是变了：前面一直在 iHC 到 hc_head 一段里逐处找「哪里没照 GPU 舍」，这一段已经找干净了。下一处在哪，凭读代码一处一处猜，猜不完。

## 第九幕：全局清单：先编译、再扫描

> **这一步是整个排查里最有价值的方法。** 前八幕每次只查一处：读代码、写开关、上卡每格跑 4 份 1000 步。这一步换成在整个训练步的编译产物里，按一个精确的模式一次扫出所有同类位置。不占卡：只要 HLO 时编译约 2 分钟，带 LLO 确认约 15 分钟；扫描本身只要 HLO 时约 1 秒，读 LLO 时约 11 秒。

### 为什么能这么做

第七幕里有一条规律，是在 iHC、hc_head 两个最小复现上实测出来的：

- 代码里写的 `x.astype(bf16)` 再用 FP32 去算，编译到优化后 HLO 里是一对 `convert(f32→bf16)` → `convert(bf16→f32)`。**这对往返就是代码在说「这里该舍一次 BF16」。**
- 开着 excess precision 时，这对往返如果落在同一颗融合里，后端不产生舍入指令，直接拿 FP32 的值用。
- 只有跨融合边界写回内存、或喂给 BF16 矩阵乘的值，才真的舍了。

HLO 里 convert 还在，所以能从 HLO 认出「声明了」；LLO 里没有打包指令，所以能确认「被省掉了」。两层合起来，就是一个可以机械匹配的模式。

这条规律反过来就是 C 口径里那些开关的作用：reduce_precision 编译器省不掉。M36、M52、M58 就是把这种往返换成了 reduce_precision。

### 怎么做

**1. 离线编译整步，不占卡。** 用 [`frameworks/maxtext/aot_hlo.py`](../frameworks/maxtext/aot_hlo.py) 在 CPU 上为 v5p-16 编 C 口径的训练步，和上卡编的是同一个程序（这个工具的来历见[案例 14「怎么从编译产物看通信」](14-ddp-fsdp-embedding.md#怎么从编译产物看通信)）。要 LLO 时，再让 libtpu 把每颗 kernel 的 `final_bundles` 存下来。

| | 只要 HLO | HLO + LLO |
|---|---|---|
| 编译耗时 | 约 110 s | 830–1010 s |
| 进程峰值内存 | 约 15 GiB | 41–43 GiB |
| 产物 | 优化后 HLO 一份（约 35 MB） | 另有 LLO 目录约 720 MB，1666 颗 kernel（相同的 kernel 只编一次） |

**2. 在 HLO 里匹配。** [`analysis/elided_rounding_scan.py`](../analysis/elided_rounding_scan.py) 对每个计算（融合的子计算）逐条看：

| 规则 | 判成 |
|---|---|
| `y = bf16 convert(x)`，x 是 f32，且 y 在**同一个计算里**被 `f32 convert(y)` 用到（中间可以隔着 bitcast、reshape、copy、transpose、broadcast、slice 这类只搬数据的 BF16 运算） | **省掉**（elided）：声明了舍入，同一颗融合里又转回 FP32 |
| 转回 f32 之后紧跟一个舍到 BF16 网格的 `reduce-precision(exponent_bits=8, mantissa_bits=7)` | **已补**（covered）：有人用 reduce_precision 补上了，不算省掉 |
| y 作为融合的输出写出、喂给矩阵乘或卷积、或给其他 BF16 运算 | **舍了**（kept） |
| x 本身已在 BF16 网格上（x 是 reduce-precision 的结果，或刚从 BF16 转上来） | 跳过：来回是精确的，舍不舍都一样 |

每处再从 metadata 归类：

- 源码位置：取 HLO 栈帧表里最内层的 MaxText 文件和行号；
- 区域：op_name 里 `while/body` 的层数，2 层是层 scan 里，1 层是 scan 外（MTP 块、hc_head、嵌入、loss），0 层是优化器；
- 前向还是反向：op_name 里有没有 `transpose(`；
- 位置类：按源码行和形状（有没有序列维 4096 或 16384）归成「权重梯度」「分支输入梯度合流」「前向线性层输出」等类。

一处真实的命中长这样（反向里重算的输出门，删去布局和 metadata）：

```text
%convolution.2585.clone.3 = f32[4096,8,256] convolution(%fusion.5209.clone.3, %fusion.5539.clone.3), ...
%convert_element_type.27396.clone.3 = bf16[4096,8,256] convert(%convolution.2585.clone.3)        ← 声明：舍 BF16
%convert_element_type.25993.clone.3 = f32[4096,8,256] convert(%convert_element_type.27396.clone.3)  ← 同一颗融合里转回
%abs.1481.clone.3 = f32[4096,8,256] abs(%convert_element_type.25993.clone.3)                     ← 拿去做 sigmoid
```

**3. 用 LLO 确认。** 挑出「整颗 kernel 只含『省掉』、没有『舍了』、没有 reduce_precision」的 kernel，数它 LLO 里 `vpack.*.bf16` 的条数。按上面的规律，应该一条都没有。

负对照：同一配置加 `--xla_allow_excess_precision=false` 再编一份。这时编译器不许省舍入，同一批 kernel 应该全部有打包指令。

### 得到了什么

C 口径训练步（`hy4-mini-t1.yml`，v5p-16，libtpu 0.0.46，jax 0.11.1），按类的数字来自 MaxText `98a80d92` 的那次编译。在 MaxText `8d3d7561` 的 C 口径编译上复算，按类数字逐个相同。结果文件是 [`mini-c-elided-rounding-scan.json`](../alignments/01-mini/results/mini-c-elided-rounding-scan.json)：

- **HLO：往返 272 处**（单位是「往返的使用」，一个 convert 在同一计算里每被转回一次算一处，不是 kernel 数）；
- 已补的 55 处（covered：转回 FP32 后紧跟 reduce_precision）不在其中，其中 30 处在 `mhc.py`（iHC 与 hc_head），其余在 norm 和线性层；
- 输入已在 BF16 网格上、来回精确的 1475 处已跳过；
- iHC、hc_head 在 272 处里一处都没有：那几处的往返都已被 M36、M52、M58 的 reduce_precision 补上或换掉。

| 位置类 | 往返（处） | 层 scan 里 / scan 外 / XLA 自插 | GPU 那边 | 方向 | 优先级 |
|---|---|---|---|---|---|
| ① 分支输入的梯度合流（注意力输入处 q、kv 下投影和输出门三份梯度；另有 MoE 路由输入 7 处、XLA 自插 17 处） | 70 | 30 / 23 / 17 | 三份 BF16 梯度在 BF16 里相加 | 反向 | **高** |
| ② 各线性层每个微批的 dW | 75 | 45 / 28 / 2 | 关了 `gradient_accumulation_fusion`：每个微批的 dW 先出 BF16，再加进 FP32 的 main_grad | 反向 | **高** |
| ③a 前向线性层输出（fc1 等） | 51 | 33 / 18 / — | 线性层输出 BF16 | 前向 | 中 |
| ③b 注意力输出门 | 34 | 25 / 9 / — | 同上 | 前向 | 中 |
| ③c 路由专家 act×up | 13（另有反向 7） | 8 / 5 / — | 同上 | 前向 | 中 |
| norm 输入梯度、norm 输出 | 7、6 | scan 外 | | | 低 |
| 索引器掩码 | 9 | scan 外 | | | 无害：值是 0 和 −∞ 这类在 BF16 上精确的数 |

优先级按三条排：GPU 那边会舍；TPU 这边省掉了；在反向的梯度路径上。前两类三条都满足。iHC 流梯度这个主因本来就属于「梯度合流」这一种（三份梯度合流），①是同一种模式在分支输入处又出现；②在每层每个线性层上都有。①里注意力、MLP 各占多少，扫描器只靠源码行和形状分不开（多数命中都在 `linears.py` 的同一行），GPU 上的 ATTNIN 开关只覆盖注意力输入这一处。

**LLO 确认**（MaxText `8d3d7561` 的 C 口径编译，带 LLO）：整颗 kernel 只含往返的有 41 颗，41 颗的打包指令全部是 0 条。

- 这 41 颗里，34 颗是②每微批 dW，4 颗是索引器掩码，3 颗是 norm。
- 负对照：同配置关 excess precision 再编，同样挑法挑出 35 颗，没有一颗是 0 条。
- ①和③的往返都落在同时含有真舍入的 kernel 里。那颗 kernel 的打包指令条数没法单独归给这几处往返，所以它们的证据是 HLO 规则加上 iHC、hc_head 两个最小复现确立的规律，没有逐颗 LLO 核对。

### 这个方法的边界

- **只找得到一类：代码声明了、编译器省掉了。** 代码里根本没写舍入（例如一开始就用 FP32 算、GPU 却舍了 BF16）的地方，HLO 里没有 convert，扫不出来，要靠读代码对照 GPU。累加顺序不同、近似超越函数这类差，也不在这个模式里。
- **GPU 自己会不会在这些位置舍，要对照 GPU 代码。** 清单只说 TPU 这边省掉了，不说 GPU 那边舍了。例如②成立，是因为 GPU golden 关了 `gradient_accumulation_fusion`（`frameworks/megatron/train_hy4mg.py`）；如果开着，GPU 的 dW 直接在 FP32 里累加，这一类就不算差异。
- **找到的是候选，因果要上卡验证。** 272 处里哪些真的影响轨迹，要在 GPU 上照 TPU 的样子改（HEADCOT、STREAM 那种诊断开关）或在 TPU 上补 reduce_precision，各跑几份 1000 步看。本案例写到这里时，GPU 上 ATTNIN（①）、WGRAD（②）两个开关的实验还在跑。
- **分类规则是按混元 4 mini 定的经验规则。** 位置类靠源码行号和形状（序列 4096、路由 token 16384）判，换模型、换 MaxText 提交要重核。栈帧表是截短的，只看得到最内层几帧；层 scan 外的未进 scan 的层和 MTP 块分不开。op_name 里带 `transpose(` 的也包括反向里重算的前向（`checkpoint/rematted_computation`），前向、反向要结合这一点读。
- **LLO 只能确认「整颗只含往返」的 kernel。** 混合 kernel 的打包指令来自别的真舍入，条数不能拆开归属。
- **结论绑定编译器版本。** 融合怎么切随 libtpu 版本、芯片代际变（案例 15 就是融合变了导致舍入次数变了），换版本要重扫。

## 在自己的模型上怎么用

前提：MaxText 训练镜像（jax、libtpu 和上卡一致；镜像 digest 见 `frameworks/maxtext/run.sh`），CPU 机器，内存 ≥ 48 GiB（要 LLO 时）。

**1. 离线编译，存 HLO 和 LLO。**

```bash
OUT=/tmp/elided/c
JAX_PLATFORMS=cpu ALLOW_MULTIPLE_LIBTPU_LOAD=1 TPU_SKIP_MDS_QUERY=1 \
HY4_YML=models/hunyuan4/frameworks/maxtext/hy4-mini-t1.yml \
XLA_FLAGS="--xla_dump_to=$OUT/xla --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*" \
LIBTPU_INIT_ARGS="--xla_jf_dump_to=$OUT/jf --xla_jf_dump_llo_text=true" \
  python models/hunyuan4/frameworks/maxtext/aot_hlo.py --stage mini --topology v5p-16 --steps 1000 --out $OUT
```

只要清单、不要 LLO 确认时，去掉 `LIBTPU_INIT_ARGS` 那一行，编译快 8 倍左右。已经在用 TPU 自比[跑前离线筛](../alignments/04-tpu-self/setup.md#跑前离线筛)的，把 `--xla_jf_dump_to=/out/jf --xla_jf_dump_llo_text=true` 写进臂的 `libtpu_args`，`hlo_screen.py run` 的缓存目录里就会有 `train_step.after_optimizations.txt` 和 `jf/`，直接拿来扫。

**2. 扫描。**

```bash
python models/hunyuan4/analysis/elided_rounding_scan.py \
  --hlo $OUT/xla/module_*.jit_train_step.*.after_optimizations.txt \
  --jf $OUT/jf --out $OUT/elided.json --positions
```

`--jf` 可以不给（只做 HLO 匹配）；`--out` 不给就只打印。

屏幕上先打印 LLO 核对（只含往返的 kernel 数、其中 0 条打包的数），再打印按模块的计数和逐个源码位置。JSON 里：

| 字段 | 内容 |
|---|---|
| `llo_check` | 只含往返的 kernel 数、其中打包指令 0 条的数、非 0 的例子 |
| `elided_uses_by_class` | 按位置类、按区域（层 scan 里 / scan 外 / XLA 自插）的往返处数 |
| `elided_uses_by_module` | 按区域 / 模块的往返处数 |
| `sites` | 每个源码位置一行：省掉、舍了、已补各几处，前向、反向各几处，形状，所在的 kernel 和每颗的打包指令条数 |
| `skipped_exact_converts` | 输入已在 BF16 网格上、来回精确而跳过的处数 |
| `positions`（加 `--positions` 时） | 逐处清单：每一处往返（省掉或已补）所在的计算、kernel、区域 / 模块、源码位置、前向还是反向、省掉 / 已补 / 舍了各几路、位置类、那颗 kernel 的打包指令条数、是不是「整颗只含往返」 |

**3. 负对照（建议做一次）。** 同一条命令在 `XLA_FLAGS` 里加 `--xla_allow_excess_precision=false` 再编、再扫，`llo_check` 里 0 条打包的 kernel 应该变成 0 颗。不是 0 就说明规律在你的版本上不成立，清单不能直接用。

**4. 读清单。** 逐类问三个问题：GPU 那边在这里舍不舍（读 GPU 代码、看训练参数）；在不在反向梯度路径上；有没有现成的 reduce_precision 开关能补。三条都满足的，排进上卡验证。验证照第五幕做两头：TPU 补上舍入看偏差下不下去，GPU 照 TPU 不舍看偏差出不出来。

换模型时，`classify_position` 里的源码行号和形状要按自己的模型改；`elided_uses_by_module` 与 `sites` 不依赖这些规则，可以先看。

## 实测与推断

| 结论 | 性质 |
|---|---|
| 后段偏差从第 550 步起变大，2000 步时回落 | 实测 |
| 不是前向差；不是 MTP、KD、GPU 索引器 RoPE、libtpu 版本 | 实测 |
| 两边去掉 iHC 后不飘 | 实测（GPU 3 份、TPU 2 份，尺子粗） |
| 主因是 iHC 流梯度的舍入写法 | 实测，两头验证（TPU p 0.016，GPU p 0.0025） |
| TPU 的写法单次运算离精确值更近 | 推断（按舍入次数）；轨迹上谁更近没测 |
| M52 与 M58 相冲在 hc_head 输出处的余切舍入 | 拆格是实测；「相冲点在这里」是推断 |
| 只开 M58 的后段变平有一部分是抵消 | 推断（GPU HEADCOT 方向一致，实测） |
| 融合内部的来回 convert 不产生舍入指令 | 实测（iHC、hc_head 两个最小复现；整步 41 颗只含往返的 kernel 全 0 条，负对照 35 颗全非 0） |
| 全图 272 处往返、三类位置 | 实测（HLO 匹配） |
| 残余来自这三类 | **在查**（GPU ATTNIN、WGRAD 在跑） |

## 这次排查用到的 janus 能力

| 能力 | 在本案例中的作用 |
|---|---|
| 逐步均值对 GPU 16 份、25 步分块、GPU 留一法尺子（[`mini_main_loss_drift.py`](../analysis/mini_main_loss_drift.py)、[`mini_loss_isolation.py`](../analysis/mini_loss_isolation.py)） | 看出偏差集中在后段、斜率超出 GPU 范围 |
| 同状态前向（[`mini_main_loss_same_state.py`](../analysis/mini_main_loss_same_state.py)、[`cpu_bf16_gpu_forward.py`](../analysis/cpu_bf16_gpu_forward.py)） | 排除前向，把问题定为训练里攒出来的 |
| 单因素交叉与置换检验（[`mini_factorial_cells.py`](../analysis/mini_factorial_cells.py)，`--control gpu-loo`） | 每格 4–6 份，给出 p 值，区分真作用和轨迹偶然 |
| GPU 一侧的诊断开关（台账 G37：STREAM、HEAD、BRANCH、HEADCOT） | 两头验证；发现抵消 |
| CPU 逐位对照（[`hc_head_order_probe.py`](../analysis/hc_head_order_probe.py)） | 把 hc_head 一段的语义差缩到只差累加顺序 |
| 最小复现的 HLO / LLO 计数（[`ihc_llo_probe.py`](../analysis/ihc_llo_probe.py)、[`hc_head_llo_probe.py`](../analysis/hc_head_llo_probe.py)） | 确立「融合内往返不舍」的规律 |
| 整步离线编译（[`aot_hlo.py`](../frameworks/maxtext/aot_hlo.py)、[`hlo_screen.py`](../alignments/04-tpu-self/hlo_screen.py)） | 不占卡拿到上卡同一个程序 |
| **全图扫描（[`elided_rounding_scan.py`](../analysis/elided_rounding_scan.py)）** | **一次列出全部被省掉的舍入，给出残余的候选范围** |
