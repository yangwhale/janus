[janus](../../../../README.md) › [混元 4](../../README.md) › [TPU 自比](README.md) › [实验矩阵](MATRIX.md) ｜ B1 梯度的跨卡归约

# 混元 4 · 04 · TPU 自比 · B1 纯数据并行：梯度怎么跨卡求和，为什么只有词嵌入梯度丢精度

纯数据并行是每张卡放一整份模型、各算一部分数据，最后把梯度跨卡加起来；FSDP 是把模型切开分到各张卡上。纯数据并行 8 卡（`ts-dp8`）第 1 步只有词嵌入梯度离精确值远（精确值指 CPU 上用 FP64 算的梯度）：离 FP64 1.51e-2，R 3.62。R 的分母按定责规则取 max(基准运行 golden `ts-g-a`（FSDP 8 卡）4.00e-3, GPU 三份中位数 4.17e-3)，这里是 GPU；只对 golden 比是 3.77 倍。其余 204 个张量和 golden 一样近。

这一页把两边编译出来的训练步程序（XLA 优化后的 HLO）里每一次梯度跨卡求和逐个查清：用什么数据类型加、在不在梯度累积循环里、对应哪个参数、跨卡前后各舍入几次，再说明词嵌入为什么单独丢精度。

**一句话答案**：大部分权重在两边舍入的次数一样，只是加法顺序不同；词嵌入的毛病主要出在每张卡自己那次 BF16 累加，不在跨卡求和。

第一至六节只读编译产物与第 1 步抓数，没有新跑；第七节是新做的单算子实验（v5p 上真跑，另在 CPU 上为 TPU 交叉编译看各 pass 与后端代码）。用到的运行：`ts-dp8`、`ts-g-a`、`ts-dp8-ga1`（B1d）、`ts-f16-dp16`（B2e）、`ts-f16`（B2）、`ts-c-noexcess`（C 关 excess precision）、`ts-dp8-embf32`（B1b，修复前）、`ts-fx-dp8-embf32`（B1f，修复后）。HLO 在各自运行目录的 `hlo/`；`ts-dp8` 另有优化前 HLO。

每条结论标明来源：

| 标记 | 意思 |
|---|---|
| 【HLO】 | 优化后 HLO 原文里直接看得到 |
| 【数据】 | 用第 1 步抓数（从动量反推的梯度）对 CPU FP64 参照算出来 |
| 【推断】 | 由前两者推出，HLO 里看不到 |
| 【实测】 | 第七节的单算子实验直接量到 |

文中的通信和编译术语：

| 词 | 白话 |
|---|---|
| all-reduce | 把各张卡上同一份数据加起来，每张卡都拿到总和 |
| reduce-scatter | 加起来以后，每张卡只留自己那一段 |
| all-gather | 每张卡把自己那一段发给大家，拼成整份 |
| all-to-all、collective-permute | 卡之间互相交换数据块 / 点对点传一块数据，只搬运、不求和 |
| 窗口化 reduce-scatter | 编译器把 reduce-scatter 拆进矩阵乘的循环：部分和在卡之间一跳一跳地传，每到一张卡就加上本卡算出的那一份 |
| scatter-add | 按 token 号把每个 token 的梯度加到词嵌入表的对应行 |
| 归约函数 | 求和时两两相加用的那个加法，它的类型决定在什么精度里加 |
| excess precision（省舍入） | 编译器允许把本该舍成 BF16 的中间结果留在 FP32，少舍几次 |
| MXU | TPU 的矩阵乘单元 |
| 尺子 | 尺子：GPU 第 1 步重跑三份之间本来就有的差 |
| 定责为换了顺序 | 离精确值不比 golden 远，只是加法先后不同 |

## 结论

1. **纯数据并行下，大部分矩阵乘权重的跨卡求和是 FP32，求完和才舍成 BF16；真正以 BF16 跨卡求和的是路由专家、索引器 `wk`/`wq_b` 和词嵌入。** 32 个 all-reduce 里，归约函数是 BF16 加法的有 10 个（路由专家 6 个、索引器 2 个、词嵌入 2 个）。注意力 6 个投影和共享专家 3 个矩阵（第 1–6 层）的 all-reduce 输入 FP32、归约函数 FP32、输出 BF16。其余都是 FP32。
2. **这几类权重在 FSDP 下的数据类型和舍入次数与纯数据并行一样。** 路由专家在 FSDP 下也是 8 路 BF16 all-reduce；索引器 `wk`/`wq_b` 是 BF16 部分和的窗口化 reduce-scatter；FP32→BF16 那一类在 FSDP 下是 FP32 部分和的窗口化 reduce-scatter，求完和舍一次 BF16。所以这 204 个张量只是换了求和顺序：矩阵乘、路由专家、索引器、RMSNorm 这几类第 1 步离 FP64 的比值（纯数据并行 ÷ golden）在 0.87–1.07；204 个张量里除 6 个 iHC 小参数（数值上病态，本来就定不了责，01-mini D11）外，全部在噪声内或定责为换了顺序。
3. **词嵌入丢精度的主因不在 8 路 BF16 all-reduce，而在每张卡本地那次 BF16 scatter-add 的累加。** 两边 HLO 里 scatter 都声明为 BF16、归约函数都是 BF16 加；差别是 FSDP 先按 token 号排序、清零在融合内部，纯数据并行不排序、往外面传进来的零缓冲上加。多出来的误差集中在 39 个高频 token 的行，随每张卡一次 scatter 的 token 数涨（4096 个 1.57e-2，8192 个 3.07e-2），与 all-reduce 是 8 路还是 16 路无关；而 FSDP 关掉 excess precision 后，scatter 的 HLO 一字不差，高频行却远 35 倍。
   【实测】（第七节）scatter 标了 `indices_are_sorted=true` 时，TPU 后端换一种写法：每行的连续更新在片上 FP32 累加器里加完、写出时舍一次 BF16，前提是 excess precision 开着；没标时每个更新都对 BF16 输出读、加、舍、写一次，开不开 excess precision 都一样。排序是编译器 pass `tpu_scatter_gather_optimizer` 插的，条件是这一次 scatter 的下标个数超过表行数的 1/8（词表 120832 行即超过 15104 个），和切不切分无关：FSDP 每张卡的 scatter 收的是 8 张卡全部 32768 个 token，过线；纯数据并行每张卡 4096 个（不累积时 8192 个），不过线。
4. **修复（`embedding_grad_fp32`）后**，词嵌入两个 `bf16[120832,1024]` all-reduce 变成循环里一个 `f32[120832,1024]` all-reduce（主输入与 MTP 输入合进一次 FP32 scatter），每步执行 2 次。修复前那份把这个 FP32 all-reduce 挪到了循环外，循环体里用 `select(convert(partition-id), 0, 累加器)` 把非 0 号分区的累加器每次清零。

## 一、纯数据并行的 32 个 all-reduce

【HLO】梯度累积循环是 `%while.2471`（条件 `i < 2`，`i` 从 0 起），每步执行 2 次；循环外的 all-reduce 每步执行 1 次。所有 all-reduce 都在 8 张卡上（`replica_groups` 8 路）。

【HLO + 推断】「对应梯度」由形状、`metadata` 里 `op_name` 与源码调用栈（栈帧表）回溯得到：调用栈区分第 0 层（稠密层）、第 1–4 层（扫描的 MoE 块）、第 5–6 层（余下两层）、MTP 与输出头；第 1–4 层、5–6 层内部按反向的出现顺序（后面的层先出现）认层号，有索引器参数的那一层就是第 1、5 层，两者对得上。

| HLO 名 | 位置 · 每步次数 | 元素类型（入 → 出） | 归约函数 | 对应梯度 |
|---|---|---|---|---|
| `all-reduce.1160` | 循环内 · 2 | f32[] | f32 加 | 非梯度：MTP 的标量 |
| `all-reduce.1159` | 循环内 · 2 | s32[] | s32 加 | 非梯度：loss 的 token 计数 |
| `all-reduce.1178` | 循环内 · 2 | s32[16] × 4 | s32 加 | 非梯度：专家计数 |
| `all-reduce.1175` | 循环内 · 2 | f32[1024,120832] | f32 加 | 输出头 `logits_dense`（主与 MTP 两路已在本卡相加） |
| `all-reduce.1165` | 循环内 · 2 | bf16 × 4 | bf16 加 | 第 1、5 层索引器 `wk` `bf16[1024,128]`、`wq_b` `bf16[512,32,128]` |
| `all-reduce.1177` | 循环内 · 2 | bf16[120832,1024] | bf16 加 | 词嵌入（MTP 输入那一路） |
| `all-reduce.1162` | 循环内 · 2 | bf16 × 3 | bf16 加 | 第 6 层路由专家 `wi_0`、`wi_1` `bf16[16,1024,256]`，`wo` `bf16[16,256,1024]` |
| `psum.458` | 循环内 · 2 | f32[8] | f32 加 | 第 6 层 `sinks` |
| `all-reduce.1161` | 循环内 · 2 | 24 个：f32→bf16 9 个，f32 15 个 | f32 加 | 第 6 层：9 个矩阵乘（`wq_a`、`wq_b`、`wkv_a`、`wkv_b`、`out`、`gate_proj`、共享专家 `wi_0`/`wi_1`/`wo`）f32→bf16；路由门 `gate`、iHC 10 个、两个 RMSNorm scale、`q_norm`、`kv_norm` 为 f32 |
| `all-reduce.1164` | 循环内 · 2 | bf16 × 3 | bf16 加 | 第 5 层路由专家 |
| `psum.459` | 循环内 · 2 | f32[8] | f32 加 | 第 5 层 `sinks` |
| `all-reduce.1163` | 循环内 · 2 | 27 个：f32→bf16 9 个，f32 18 个 | f32 加 | 第 5 层：同 `all-reduce.1161`，另加索引器 `k_norm` scale/bias、`weights_proj`（f32） |
| `all-reduce.1167` `.1169` `.1171` `.1173` | 循环内 · 2 | bf16 × 3 | bf16 加 | 第 4、3、2、1 层路由专家 |
| `psum.460`–`psum.463` | 循环内 · 2 | f32[8] | f32 加 | 第 4、3、2、1 层 `sinks` |
| `all-reduce.1166` `.1168` `.1170` | 循环内 · 2 | 24 个：f32→bf16 9 个，f32 15 个 | f32 加 | 第 4、3、2 层，组成同 `all-reduce.1161` |
| `all-reduce.1172` | 循环内 · 2 | 27 个：f32→bf16 9 个，f32 18 个 | f32 加 | 第 1 层，组成同 `all-reduce.1163` |
| `all-reduce.1174` | 循环内 · 2 | 24 个 f32 | f32 加 | 第 0 层与 MTP 的 RMSNorm 类 scale（第 0 层 4 个、MTP 6 个），输出头那段算出的 3 个 `f32[1024]` scale（逐个对应未定）；另 11 个标量是索引器 KD loss 等记录量，非梯度 |
| `all-reduce.1176` | 循环内 · 2 | bf16[120832,1024] | bf16 加 | 词嵌入（主输入那一路） |
| `all-reduce.1136` | 循环外 · 1 | f32 × 2 | f32 加 | 第 0 层、MTP 的 `sinks` |
| `all-reduce.1137` | 循环外 · 1 | f32 × 9 | f32 最大值 | 非梯度：QK-Clip 的最大 logit |
| `all-reduce.1133` | 循环外 · 1 | s32[2,16] × 3 | s32 加 | 非梯度：路由计数 |
| `all-reduce.1138` | 循环外 · 1 | f32 累加器 → bf16 | bf16 加 | 第 0 层索引器 `wk`、`wq_b` |
| `all-reduce.1135` | 循环外 · 1 | f32 × 7 | f32 加 | MTP：路由专家 `wo`、索引器 `wq_b`、注意力 `out`/`wkv_a`/`wkv_b`/`wq_a`/`wq_b` |
| `all-reduce.1134` | 循环外 · 1 | f32 × 39 | f32 加 | 第 0 层：iHC 12 个、稠密 MLP `wi_0`/`wi_1`/`wo`、注意力 6 个投影、索引器 `k_norm` ×2、`weights_proj`；`hc_head` 3 个；MTP：`projection`、路由门、路由专家 `wi_0`/`wi_1`、共享专家 3 个、`gate_proj`、索引器 `k_norm` ×2、`weights_proj`、`wk` |

几点【HLO】细节：

- 「f32→bf16」是同一条 all-reduce 指令的输入为 f32、输出为 bf16（如 `all-reduce.1161` 的第 1 个元素，输入 `fusion.15776 = f32[1024,256]`，输出 `bf16[1024,256]`），归约函数 `add.107.clone` 是 f32 加。优化前 HLO 里这段是 `dot`（输出 f32）→ `convert` 成 bf16 → `convert` 回 f32；切分时 all-reduce 插在 `dot` 之后。【推断】编译器把后面那次 `convert` 并进了 all-reduce：先在 FP32 里跨卡求和，再舍一次 BF16。
- 第 0 层和 MTP 的梯度 all-reduce 被挪到了循环外（`all-reduce.1134`/`.1135`/`.1136`/`.1138`）：循环体里只在本卡累加，例如 MTP 的 `wkv_b`：`convolution`（f32）→ `convert` bf16 → `convert` f32 → 加到累加器（第 0 个微批用 `select(i == 0, 0, 累加器)` 清零）。也就是**本卡结果先舍一次 BF16，再做跨卡 FP32 求和**，舍入落在跨卡之前。
- `all-reduce.1138` 的输入是循环出来的 f32 累加器，经一条改类型的 `copy`（f32 → bf16）后做 BF16 all-reduce，出来再 `convert` 回 f32。
- RMSNorm 类 scale 的 all-reduce 是 f32，出来紧接一条 `reduce-precision`（舍成 BF16 的精度）再加到累加器，即先 FP32 跨卡求和、再舍一次。这次舍入来自 `rmsnorm_scale_grad_fp32=false`。
- 本次运行 `logits_dense_wgrad_fp32`、`rmsnorm_scale_grad_fp32`、`embedding_grad_fp32` 都是 false（运行 `meta.json`）。输出头梯度是 FP32，是因为 `logits_dot_in_fp32=true`、`logits_matmul_precision=highest`（`all-reduce.1175` 的来源是 `operand_precision={highest,highest}` 的 f32 卷积）。
- 索引器 `wk`/`wq_b` 是 BF16：MaxText 里这两个 `DenseGeneral` 没有接 `dense_fp32_accumulation`，反向卷积直接输出 bf16。路由专家是 BF16：梯度来自 gmm 内核（输出 bf16），在 `shard_map` 里显式 `psum`。

## 二、FSDP 一侧

【HLO】golden 有 38 个 all-reduce、0 个 reduce-scatter、622 个 all-gather、900 对 collective-permute。权重梯度的跨卡求和分两种：

| 机制 | HLO | 数据类型 | 位置 · 每步次数 | 对应梯度 |
|---|---|---|---|---|
| 窗口化 reduce-scatter（循环体里双向 collective-permute 传部分和、本卡矩阵乘结果逐跳加上去） | 94 个 `windowed_dot_general_body_rs.*` 循环 | 86 个 f32 部分和，8 个 bf16 部分和 | 梯度累积循环内 · 2 | f32：各层注意力 6 个投影、MoE 层路由门与共享专家 3 个、第 0 层稠密 MLP 3 个、索引器 `weights_proj`、MTP `projection`、输出头（主与 MTP 各一）；bf16：第 0、1、5 层与 MTP 的索引器 `wk`、`wq_b` |
| all-reduce，BF16 | `all-reduce.1990` `.1992` `.1994` `.1996` `.1998` `.2000` | bf16，归约函数 bf16 加，8 路 | 循环内 · 2 | 第 6–1 层路由专家（`.2000` 另含 MTP 路由专家） |
| all-reduce，FP32 | `all-reduce.1991` `.1993` `.1995` `.1997` `.1999` `.2001`、`.2002`、`psum.467`–`.472` | f32 | 循环内 · 2 | 第 6–1 层 iHC、RMSNorm 类 scale、索引器 `k_norm`；第 0 层、MTP、输出头前的 norm scale；各层 `sinks` |
| all-reduce，FP32 | `all-reduce.1969`、`.1970` | f32 | 循环外 · 1 | 第 0 层 iHC、索引器 `k_norm`、`hc_head`、第 0 层与 MTP 的 `sinks`；`.1969` 另含 115 个梯度裁剪用的范数部分和 |
| 其余 all-reduce | `all-reduce.1957`–`.1968`、`.1971`、`.1956`、`.1988`、`.1989`、`.2003` | — | — | 非梯度：Muon 正交化的矩阵乘部分和与范数、最大 logit、计数、loss |

- 窗口化 reduce-scatter 的 f32 部分和求完后，例如 `wkv_a`：两个方向的部分和相加（f32）→ `convert` bf16 → `convert` f32 → 加到累加器，舍一次 BF16。路由门、`weights_proj`、输出头这几个 f32 循环出来直接加到累加器，不舍。
- 词嵌入在 FSDP 下没有跨卡求和：token 号 `all-gather` 到每张卡、本卡 128 列上 8 张卡全部 token 的梯度经 `all-to-all` 到齐（bf16，原样搬运），然后在本卡做一次 scatter（见第四节）。

## 三、按类别对照：数据类型、通信、舍入次数

每个微批、每个元素。「舍」指舍成 BF16。

| 类别 | 纯数据并行 | FSDP | 第 1 步离 FP64 的比值（纯数据并行 ÷ golden，该类张量的范围） |
|---|---|---|---|
| 第 1–6 层注意力 6 个投影、共享专家 3 个 | MXU 累加本卡 4096 个 token（f32 输出）→ all-reduce：FP32 跨卡求和后舍 1 次 | MXU 累加（f32 输出）→ f32 部分和环上相加 → 舍 1 次 | 0.88–1.07 |
| 第 0 层、MTP 的同类矩阵乘 | 本卡结果先舍 1 次 → 两个微批 f32 累加 → 循环外 FP32 all-reduce | 同上一行 | 0.91–0.99（第 0 层）、0.94–1.02（MTP） |
| 第 1–6 层路由专家 | gmm 内核输出 bf16（舍 1 次）→ 8 路 BF16 all-reduce → f32 | 相同 | 0.95–1.03 |
| 索引器 `wk`、`wq_b` | 卷积输出 bf16（舍 1 次）→ BF16 all-reduce | bf16 部分和环上逐跳相加 | 0.91–1.07 |
| RMSNorm 类 scale | FP32 求和 → FP32 all-reduce → 舍 1 次 | 相同 | 0.87–1.05 |
| 路由门、`weights_proj`、输出头、iHC、索引器 `k_norm`、`sinks` | 全程 FP32 | 全程 FP32 | 0.63–1.25（iHC 的 `hc_scale`/`hc_base` 除外）；在噪声内或定责为换了顺序，6 个 iHC 小参数不定责（D11） |
| 词嵌入 | 本卡 BF16 scatter（不排序）→ 8 路 BF16 all-reduce → f32 | 排序后 BF16 scatter（8 张卡全部 token、本卡 128 列），无跨卡求和 | 3.77 |

- 【HLO】矩阵乘类权重梯度的卷积输入是 bf16；接了 `dense_fp32_accumulation` 的输出 f32，索引器的输出 bf16。【推断】MXU 按 FP32 累加乘积，f32 输出不舍，bf16 输出只在写出时舍一次。
- 【HLO】第 1–6 层矩阵乘两边都是「FP32 跨卡求和、求完舍一次」；路由专家两边是同一种 BF16 all-reduce；索引器两边都以 BF16 跨卡。舍入次数相同，只差求和顺序。
- 第 0 层与 MTP 在纯数据并行下舍入落在跨卡之前（8 份本卡结果各舍一次），和 FSDP 不同；【数据】离 FP64 的比值 0.91–0.99、0.94–1.02，没有变远。
- 所以「纯数据并行下很多权重也是 BF16 跨卡求和」是对的，但它们在 FSDP 下同样是 BF16，不构成差别。

## 四、词嵌入为什么单独丢精度

### HLO 里两边的 scatter

【HLO】纯数据并行，每个微批两次（主输入、MTP 输入），本卡 4096 个 token：

```
%scatter_fusion.1 = bf16[120832,1024] fusion(%broadcast.31183.clone.1, %bitcast.26351, %fusion.16171), kind=kCustom
    // 内层 %scatter.1095：updates bf16[4096,1024]，to_apply=%region_960.1060.clone（bf16 add），没有 indices_are_sorted
%all-reduce.1177 = bf16[120832,1024] all-reduce(%scatter_fusion.1), to_apply=%region_960.1060.clone.1.clone   // bf16 add
%select_add_fusion.230 = f32[120832,1024] fusion(%wide.wide.param.2#254, %all-reduce.1176, %all-reduce.1177, %compare.2310)
    // 两路各 convert 成 f32 相加，再加 select(i == 0, 0, 累加器)
```

【HLO】FSDP，每个微批两次，本卡 128 列、8 张卡全部 32768 个 token：

```
%all-gather.2539 = s32[8,4096,1] all-gather(...)                       // 8 张卡的 token 号
%sort.1000 = (s32[32768], s32[32768]) sort(%bitcast.33260, %iota.3364)  // op_name 是 …/scatter-add：编译器为 scatter 插的排序
%all-to-all.23 = bf16[1,4096,8,128] all-to-all(...)                     // 本卡 128 列上全部 token 的梯度
%scatter_custom_fusion.446 = bf16[120832,128] fusion(%sort.1000#0, %constant.7799.clone.100..sunk, %bitcast.33181, %broadcast_clamp_fusion.21), kind=kCustom
    // 内层先 broadcast 出 bf16 的 0，再 %scatter.1098：indices_are_sorted=true，to_apply 是 bf16 add
```

- 两边 scatter 的输出、归约函数都是 BF16；把结果加进 FP32 累加器的是另一个融合。**两边 HLO 里都看不到 excess precision 让 scatter 留在 FP32**（类型上没有任何 f32）。
- 看得到的差别：FSDP 先排序、标 `indices_are_sorted=true`、清零在融合内部；纯数据并行不排序、零缓冲是融合外的 `broadcast` 传进来的。
- 关掉 excess precision 的 FSDP（`ts-c-noexcess`）里，这两个 scatter 融合和归约函数与 golden 逐字相同。

### 数据：误差集中在高频 token，随本卡 token 数涨

【数据】词嵌入梯度按行分组，每组算 ‖被测 − FP64‖ ÷ ‖FP64‖。行的分组用这一步 16 条序列的 token（主输入与 MTP 输入合计出现次数；「只在 1 张卡上」按纯数据并行 8 的分配：第 d 张卡第 m 个微批是第 d + 8m 条序列）。数据在 [`results/emb-grad-row-groups.json`](results/emb-grad-row-groups.json)，脚本 [`analysis/emb_grad_row_groups.py`](../../analysis/emb_grad_row_groups.py)。

| 行 | 行数 | 占梯度范数平方 | golden（FSDP 8） | 纯数据并行 8，累积 2（本卡 4096 token/次） | 纯数据并行 16，累积 1（4096） | 纯数据并行 8，累积 1（8192） | 修法①（全程 FP32） | 修法②（本卡 FP32、跨卡 BF16） | FSDP 16 | FSDP 8 关 excess precision |
|---|---|---|---|---|---|---|---|---|---|---|
| 每个微批只在 1 张卡上出现 | 7638 | 1.3e-2 | 9.31e-3 | 9.41e-3 | 9.25e-3 | 9.16e-3 | 9.31e-3 | 9.38e-3 | 9.50e-3 | 1.02e-2 |
| ≥ 2 张卡、出现 ≤ 300 次 | 3234 | 8.9e-2 | 7.77e-3 | 8.31e-3 | 8.33e-3 | 8.26e-3 | 7.28e-3 | 7.66e-3 | 7.90e-3 | 9.58e-3 |
| 出现 ≥ 301 次（308–5343 次） | 39 | 0.90 | 3.26e-3 | 1.57e-2 | 1.59e-2 | 3.07e-2 | 3.08e-3 | 3.60e-3 | 3.68e-3 | 1.15e-1 |
| 全部 | 10911 | 1 | 4.00e-3 | 1.51e-2 | 1.53e-2 | 2.92e-2 | 3.79e-3 | 4.24e-3 | 4.34e-3 | 1.09e-1 |

- 只在 1 张卡上出现的行，BF16 all-reduce 加的是 0，是精确的；这些行纯数据并行和 FSDP 一样（9.41e-3 对 9.31e-3）。
- 多出来的误差几乎全在 39 个高频 token 的行上（它们占梯度范数平方的 0.90）：一个这样的 token 在一张卡、一个微批里出现中位数 74 次，最多 458 次（每个 token 取出现最多的那张卡、那个微批；主输入与 MTP 输入两次 scatter 合计，单算主输入是 37 次、229 次）。
- 本卡一次 scatter 的 token 数从 4096 翻到 8192，高频行误差从 1.57e-2 翻到 3.07e-2；all-reduce 从 8 路换成 16 路（每卡仍 4096），误差不变（1.59e-2）。误差跟着本卡累加的次数走，不跟跨卡求和的路数走。
- 出现次数 ≤ 300、在多张卡上的行，纯数据并行比 golden 远 1.07 倍（8.31e-3 对 7.77e-3）。这是本卡不排序的逐次舍入和 8 路 BF16 all-reduce 合在一起的总差，不是 all-reduce 单独的贡献。拆开要看两种修法：修法② 本卡 FP32、跨卡仍 BF16，这组是 7.66e-3（golden 的 0.98 倍）；修法① 全程 FP32，7.28e-3。所以 BF16 跨卡求和在这组约 1.05 倍（7.66e-3 对 7.28e-3），本卡逐次舍入约 1.09 倍（8.31e-3 对 7.66e-3），两项都小；高频行里 BF16 跨卡求和约 1.17 倍（3.60e-3 对 3.08e-3），本卡逐次舍入 4.4 倍（1.57e-2 对 3.60e-3）。
- FSDP 的 scatter 一次要累加 8 张卡的全部 token（FSDP 16 是 16 张卡的），每行的重复次数是纯数据并行的 8–16 倍，高频行却和 FP32 一样近；同一个 scatter 关掉 excess precision 后远 35 倍。

### 机制（第七节实测）

- 纯数据并行那份 scatter 没标排序，后端对每个更新都把输出里那一行 BF16 读出来、加上、舍成 BF16 写回，同一个 token 的几十到几百次更新各舍一次，高频行的误差因此随本卡 token 数增长。8 路 BF16 all-reduce 是次要的。
- FSDP 那份由编译器先排序、标 `indices_are_sorted=true`，后端用另一种写法：同一行的一段连续更新在片上的 FP32 累加器里加完，写出时舍一次；excess precision 关掉后，同样的程序在每次加法后多一次舍成 BF16（排序后 32768 个更新顺序累加，所以更糟）。这一层在 HLO 里看不到，第七节在后端代码（LLO）里看到了。
- 原先的推断「排好序的 scatter 在 excess precision 下留在 FP32 里累加」成立，要补两处：起作用的是 `indices_are_sorted` 这个标记选中的写法，光把数据排好序不标没有用；累加器在片上 VMEM，不一定在寄存器里。「零缓冲从外面传进来」不是原因，是不排序那种写法要先清零再就地加的结果。
- 所以之前「纯数据并行把词嵌入梯度改成 BF16 跨卡求和，所以丢精度」的说法不准：跨卡求和确实是 BF16，但主要的损失在跨卡之前。修法 `embedding_grad_fp32` 把 scatter 和 all-reduce 都换成 FP32，两处一起解决。

## 五、`embedding_grad_fp32`：修复前后

【HLO】修复后（`ts-fx-dp8-embf32`）：主输入与 MTP 输入合成一次 FP32 scatter（updates `f32[2,4096,1024]`），all-reduce 留在循环里，每步 2 次：

```
%scatter_fusion = f32[120832,1024] fusion(%broadcast.31307, %bitcast.26351, %convert.6800), kind=kCustom   // %broadcast.31307 是全 0
%all-reduce.1244 = f32[120832,1024] all-reduce(%scatter_fusion), to_apply=%region_961.1061.clone.1.clone  // f32 add
%select_add_fusion.499 = f32[120832,1024] fusion(%all-reduce.1244, %wide.wide.param.2#254, %compare.2412)
    // all-reduce 结果 + select(i == 0, 0, 累加器)；%compare.2412 = compare(i, 0), direction=EQ
```

【HLO】修复前（`ts-dp8-embf32`）：all-reduce 挪到循环外，每步 1 次；循环体里每次迭代都对非 0 号分区把累加器清零，再往累加器里 scatter 本微批：

```
// 循环体 %wide.wide.region_0.1074_spmd.clone.clone.sunk 内
%partition-id.247 = u32[] partition-id()
%convert.6873 = pred[] convert(%partition-id.247)
%broadcast_select_fusion.436 = f32[120832,1024] fusion(%wide.wide.param.2#254, %convert.6873), kind=kLoop, calls=%fused_computation.887.clone.1
    //   %select.4531 = f32[120832,1024] select(%broadcast.30811, %broadcast.30810, %param_0.67484)
    //   即 select(partition-id != 0, 0, 累加器)
%scatter_fusion = f32[120832,1024] fusion(%broadcast_select_fusion.436, %bitcast.26476, %convert.6875), kind=kCustom
    // 循环出口 #254 就是它
// 循环外
%all-reduce.521 = f32[120832,1024] all-reduce(%while.2471#254), to_apply=%region_961.1061.clone.1.clone
```

- 结果：0 号分区保留两个微批，其余 7 个分区只剩最后一个微批，all-reduce 后整体少了一块（第 1 步词嵌入梯度范数 0.51 倍）。
- 同一个程序里第 0 层、MTP 的梯度 all-reduce 也被挪到了循环外，但它们的累加器清零是 `select(i == 0, 0, 累加器)`，没有 `partition-id`，结果正确。【推断】出错的是这份梯度特有的写法：scatter 直接加进循环携带的累加器。

## 六、顺带更正：HLO 指令计数

`analysis/tpu_self_compare.py` 数指令时，元组形状里只要有 `/*index=5*/` 这样的注释（6 个元素及以上的元组）就漏数，合并后的大 all-reduce 和多输出融合都没数进去。正则表达式已修（单测 `tests/models/hunyuan4/test_tpu_self_compare_hlo_ops.py`）。

已提交的 `results/*.json` 里 `hlo.op_counts_diff` 是旧算法数的，照规矩不改；文档里引用过的几处，用修好的算法对各运行的 HLO 重数如下：

| 出处 | 原来写的 | 重数 |
|---|---|---|
| B1 `ts-dp8` 对 golden | all-reduce 16 → 22 | 38 → 32 |
| C6 `ts-c-scar` 对 golden | all-reduce 16 → 23 | 38 → 55 |
| D `einsum` → 广播乘 | 融合少 19 个 | 少 18 个（convert 少 50 个不变） |
| H v7x 对 v5p | 融合多 2931 个 | 多 2941 个（reduce-scatter 21 个不变） |

## 七、单算子实测：scatter 什么时候被排序、排序后为什么不丢精度

把词嵌入反向那一个 BF16 scatter-add 单拿出来：表 `bf16[120832,1024]`，下标是 golden 第 1 步全局批的真实 token（取前 N 个），更新量是 N(0, 1e-3²) 的随机 BF16。对 numpy FP64 算 rel_L2，并和两种 CPU 仿真逐行逐位比：

- **E1**：每行按下标出现的先后在 FP32 里累加，最后舍一次 BF16；
- **E2**：同一顺序，每加一次舍一次 BF16。

脚本 [`analysis/emb_scatter_sort_probe.py`](../../analysis/emb_scatter_sort_probe.py)。v5p 上真跑的结果在 [`results/emb-scatter-sort-probe-tpu.json`](results/emb-scatter-sort-probe-tpu.json)（运行 `embsort`，镜像与 golden 相同，libtpu 同一构建）；CPU 上为 TPU 交叉编译（只编不跑）看各 pass 和后端代码的结果在 [`results/emb-scatter-sort-aot.json`](results/emb-scatter-sort-aot.json)。

### 四格（N = 4096，纯数据并行每张卡一次 scatter 的量）

| | excess precision 开 | excess precision 关 |
|---|---|---|
| 不排序（训练里的写法） | 6.75e-3，100% 的行逐位等于 E2 | 6.75e-3，等于 E2 |
| 下标排好序、标 `indices_are_sorted=True` | **1.48e-3，100% 等于 E1** | 6.75e-3，等于 E2 |

对照：FP32 scatter 2.6e-9；FP64 是参照本身。出现 ≥ 100 次的行（3 行）：不排序 1.46e-2，排序 + 开 1.70e-3。

- 只有「排序 + excess precision 开」一格按 FP32 累加，和预期一致。其余三格逐位就是每加一次舍一次。
- **起作用的是标记，不是数据的顺序。** 下标排好序但不标 `indices_are_sorted`，结果仍逐位等于 E2（`presorted_nf`）。零缓冲从 jit 外面传进来（`unsorted_buf`，模拟纯数据并行训练步里融合外的零缓冲），结果和不传一样。

### 随一次 scatter 的下标个数 N 怎么变

rel_L2（全部被更新的行），每格都逐位等于括号里的仿真：

| N | 被更新的行 / 其中出现 ≥ 100 次的 | 不排序，excess 开 | 排序 + 标记，excess 开 | excess 关（排不排都一样） | FP32 |
|---|---|---|---|---|---|
| 2048 | 945 / 0 | 4.59e-3（E2） | 1.37e-3（E1） | 4.59e-3（E2） | 1.6e-9 |
| 4096 | 1426 / 3 | 6.75e-3（E2） | 1.48e-3（E1） | 6.75e-3（E2） | 2.6e-9 |
| 8192 | 2483 / 8 | 9.14e-3（E2） | 1.51e-3（E1） | 9.14e-3（E2） | 4.3e-9 |
| 16384 | 4195 / 14 | **1.55e-3（E1，编译器插了排序）** | 1.55e-3（E1） | 1.29e-2（E2） | 7.4e-9 |
| 32768 | 6956 / 28 | **1.58e-3（E1，编译器插了排序）** | 1.58e-3（E1） | 1.76e-2（E2） | 1.1e-8 |

- 不排序时误差随 N 涨，和训练里纯数据并行「本卡 token 数翻倍、误差跟着涨」同向（第四节）。按 FP32 累加的那一列几乎不随 N 变。
- 16384 起，没标排序的 scatter 也变成 E1：优化后 HLO 里多了 `sort`，scatter 被标上 `indices_are_sorted=true`，和 FSDP 的训练步一样。
- 随机更新量下的绝对数和训练里的不能直接对：训练里同一个 token 的梯度彼此相关，和长得更快（推断）。这里只用来定「按哪种精度累加」。

### 8 个设备上照两种切法各跑一份（N 每张卡 4096，全局 32768）

| | excess 开 | excess 关 | 优化后 HLO |
|---|---|---|---|
| FSDP：表按列切到 8 张卡，每张卡拿全部 32768 个下标 | 1.58e-3 | 1.76e-2 | 每张卡的程序里有 `sort`，scatter 标 sorted |
| 纯数据并行：每张卡本地 4096 个 scatter 整表，再 BF16 跨卡求和 | 7.17e-3 | 7.17e-3 | 没有 `sort` |
| 纯数据并行，先在 JAX 里排序、标 `indices_are_sorted=True` | 2.86e-3 | 7.17e-3 | 有 `sort`，scatter 标 sorted |

纯数据并行加了排序后还比 FSDP 远一些（2.86e-3 对 1.58e-3），差在每张卡的部分和先舍一次 BF16、再做 BF16 跨卡求和。这和修法②（`embedding_grad_fp32_local_bf16_comm`：本卡 FP32 累加、舍一次、BF16 跨卡）是同一种舍入结构，修法② 在训练里实测回到噪声内（B1h，r 0.93）。

### 排序是哪个 pass 插的、什么条件

【实测】逐 pass 的 HLO（`--xla_dump_hlo_pass_re=.*`，N = 16384）里，`sort` 第一次出现在 `Pre_main_fusion` 流水线里的 `tpu_scatter_gather_optimizer` 之后，前一个 dump（`layout_fixup`）里还没有。它插的是：对 `(下标, iota)` 稳定排序（比较函数先比下标、相等再比 iota）→ 按排序后的置换 gather 更新量 → scatter 标 `indices_are_sorted=true`。N = 8192 时这个 pass 没有改动模块。

这是 TPU 后端（libtpu）自己的 pass，openxla/xla 公开代码里没有（按名字搜不到），触发条件只能从实验反推。对不同的表形状各二分出最小的插排序的 N：

| 表行数 V | 列数 D | 最小插排序的 N | V / 8 |
|---|---|---|---|
| 1000 | 1024 | 到 65536 都不插 | — |
| 30000 | 128 | 到 65536 都不插 | — |
| 30000 | 1024 | 3751 | 3750 |
| 60000 | 128 | 7501 | 7500 |
| 120832 | 128 | 15105 | 15104 |
| 120832 | 1024 | 15105 | 15104 |
| 480000 | 128 | 60001 | 60000 |

- 条件是 **N > V / 8**，和列数无关；表太小时不插（30000 × 128 的 BF16 表约 7.7 MB 不插，60000 × 128 约 15 MB 插；中间没细分，这条只是推断）。同一个 libtpu 为 v7x、v6e 编译，V = 120832 的门槛也是 15104 / 15105。
- 和切分无关：FSDP 一侧过线，是因为每张卡的那次 scatter 收的是 8 张卡全部的 token。已有训练步 HLO 全部对得上：

| 运行 | 每张卡一次 scatter 的下标数 | > 15104 | scatter 标 sorted |
|---|---|---|---|
| `ts-g-a`（golden，FSDP 8）、`ts-c-noexcess`、`ts-h-v7b`（v7x FSDP） | 32768 | 是 | 是 |
| `ts-f16`（FSDP 16） | 65536 | 是 | 是 |
| `ts-dp8`、`ts-f16-dp16`、`ts-h-v7-dp8b` | 4096 | 否 | 否 |
| `ts-dp8-ga1` | 8192 | 否 | 否 |

### 后端代码里看到的累加

【实测】N = 4096 四格各编一次，打开 TPU 后端 dump（`LIBTPU_INIT_ARGS="--xla_jf_dump_to=…"`），数 scatter 那颗 kernel 最终指令（`final_bundles`）里几类向量指令的静态条数：

| | kernel | bundle 数 | `vadd.f32` | `vpack.c.bf16` | `vpack.i.bf16` |
|---|---|---|---|---|---|
| 不排序，excess 开 | `scatter_fusion` | 4911 | 256 | 256 | 0 |
| 不排序，excess 关 | `scatter_fusion` | 4999 | 256 | 256 | 256 |
| 排序 + 标记，excess 开 | `scatter_custom_fusion` | 23660 | 4096 | 7418 | 0 |
| 排序 + 标记，excess 关 | `scatter_custom_fusion` | 25708 | 4096 | 7418 | 12288 |

- **不排序**：循环体对每个更新做「`vld` 读输出里那一行 BF16 → `vunpack` 成 FP32 → `vadd.f32` → `vpack.c.bf16` 舍回 BF16 → `vst` 写回」。值在两次更新之间住在 BF16 的输出里，excess precision 没地方省这次舍入，所以开关不影响结果。清零是另一颗 kernel 先做的。
- **排序 + 标记**：换成另一颗 kernel，清零并在里面（第四节「清零在融合内部」就是它）。累加器是片上 VMEM 里的 FP32 缓冲：`vld` 读 FP32 → `vadd.f32` → `vst` 写 FP32，中间没有舍入；BF16 只在写出时舍。
- 同一份 HLO 只关掉 excess precision，排序那颗 kernel 多出 12288 条 `vpack.i.bf16`，夹在每次 `vadd.f32` 前后（它的作用是把值舍到 BF16 精度，由指令名和数值结果逐位等于 E2 推断）。这就是「HLO 一字不差、高频行远 35 倍」在后端的样子。
- 条数是静态的，不是执行次数；结论只说「在哪里舍入」，不说耗时。

### 能不能让纯数据并行也走这条路、代价多少

- **开关**：没找到能调低门槛的开关。libtpu 里 scatter / sort 相关的开关名扫过一遍，试了 `xla_tpu_gather_scatter_flatten_min_index_count=1` 和 `xla_tpu_enable_offloading_scatter_to_sparsecore=true`，N = 4096 时都不插排序。没有逐个试完。
- **在 JAX 里显式排序**：单算子里有效，结果逐位等于 E1（上面四格的 `jaxsort` 与 8 卡的第三行）。耗时（单设备，20 次中位数）：N = 4096 时 0.80 → 0.94 ms，N = 8192 时 1.31 → 1.00 ms（排好序的写法反而更快）。估算：混元 4 mini 纯数据并行每步 4 次这样的 scatter（2 个微批 × 主输入与 MTP 输入），最多多 0.6 ms，约占 0.494 s 步时的 1e-3；修法① 实测多 0.6–0.8 个百分点（[影响逐件「纯数据并行」](impacts/01-data-parallel.md#代价)）。
- **能不能不开 `embedding_grad_fp32` 就修好**：推断能修到和修法② 一样（同一种舍入结构，修法② 训练里 r 0.93），但**没在训练里跑过**。MaxText 里要改词嵌入的反向（例如前向用排好序的 token 号取行、再按逆置换还原，让自动微分出来的 scatter 带上 `indices_are_sorted=True`；或者写 `custom_vjp`）。它还依赖两件隐含的事：excess precision 开着（默认开），以及 libtpu 排序 scatter 的这种写法不变。`embedding_grad_fp32` 不依赖这些，仍是推荐修法。
