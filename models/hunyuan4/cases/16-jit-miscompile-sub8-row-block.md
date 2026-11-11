[janus](../../../README.md) › [混元 4](../README.md) ｜ [案例](README.md) 16

# 案例 16：jit 静默算错：不满 8 行的行块

一个全 FP32、最大张量 12×6 的 JAX 函数，在 v5p 上 jit 以后有两行算错，相对误差 0.47；用 `jax.disable_jit()` 逐算子执行是对的。没有报错，结果静默出错。查下来是 libtpu 的编译器 bug，不是数值精度问题：0.0.46 到 0.0.52 都有，上游 2026-11-06 起的 nightly 已修。混元 3、混元 4 存档的 292 份训练步程序里没有这个 bug 的特征。

> **状态：机理已查明（2026-11-11）。** 正式版里还没有带修复的版本（PyPI 截至 11-11 最新是 0.0.52）。下面 4 种 workaround 都在 v5p 上验证过。

## 先看结论

- **是编译 bug，不是精度问题。** 错的两行误差 0.47，不是舍入量级。错得很有规律：等于把 `a3[:2] + a6 * a0` 这一项整个当成 0。每个编译阶段之后的程序拿到 CPU 上执行都是对的，错在程序翻译成 TPU 指令这一步。
- **触发点。** 编译器有一步叫 multi-output fusion：把几颗读同一份输入的 kernel 合成一颗，一次读入、多个输出。这里它合并了两颗 kernel：
  - 一颗用 dynamic-update-slice（DUS）原地改写 `a3` 的前 2 行，即 `a3.at[:2].set(a4)`；
  - 另一颗读 `a3` 的前 2 行，即 `a3[:2]`。

  合出来的 kernel 在 v5p 上把第二颗的输出算丢了。11-06 起的 nightly 不再把含 DUS 的 kernel 合进去，问题随之消失。
- **为什么和「8 行」有关。** TPU 把 FP32 数组按 8 行 × 128 列一块存（一块正好是一个向量寄存器）。出错的组合都是：被改写的那块缓冲按 8 行一块存，而读出来的小块只有 2 行或 4 行，按 2 行或 4 行一块存。issue 里说的「两块合起来不满 8 行」不准确：小块 2 行、下面的块 6 行，合起来 8 行，照样错。
- **版本**：v5p 上 0.0.46、0.0.48、0.0.51、0.0.52 都错；nightly 0.0.53.dev20261106 起全对。v7x 上同样的合并也会发生，但三个版本都算对。
- **怎么办**（详见 [workaround](#workaround)）：
  1. 不想动代码：`LIBTPU_INIT_ARGS=--xla_jf_enable_multi_output_fusion=false`；
  2. 只改出事的那一处：对被改写缓冲的读取加 `jax.lax.optimization_barrier`；
  3. 小块行数补到 8 的倍数；
  4. 等带修复的正式版。nightly 不建议上生产。
- **对混元 3、混元 4**：存档的 292 份训练步程序里，这个 bug 的特征出现 0 次；mini 现行比较 `tz-off-rep` 与推荐组合 `x5-prod` 没存 HLO，按它们的 `meta.json` 在 CPU 上重编训练步再扫，也是 0 次（[影响](#对混元-3混元-4-的影响)）。生产尺寸的程序没扫过，所以不能完全排除。

## 怎么查到的

| # | 问题 | 做了什么 | 结果 |
|---|---|---|---|
| 1 | 能不能复现 | issue 的脚本原样跑：v5p 单设备，镜像同 [`frameworks/maxtext/run.sh`](../frameworks/maxtext/run.sh)，jax 0.11.1，libtpu 0.0.46（`Built on Aug 12 2026`，changelist 963683785） | jit 对 CPU 0.470，逐算子 9.0e-8，和 issue 一样 |
| 2 | 是编译 bug 还是精度 | 同一函数在 TPU 上逐算子执行（[方法](../../../docs/guide/06-debugging.md#怀疑编译-bug逐算子执行对-jit)）；再看错成什么样 | 逐算子对，jit 错，错到 0.47：不是舍入。TPU 的结果等于把 `a3[:2] + a6 * a0` 这一项当成 0，差到 1e-7 |
| 3 | 从哪个版本修好 | 同一镜像里 `pip install --no-deps` 换 libtpu，146 个用例都跑一遍（[`jit_miscompile_probe.py`](../analysis/jit_miscompile_probe.py)） | 见[各版本状态](#各版本状态)：0.0.52 和 11-05 的 nightly 还错，11-06 的 nightly 起全对 |
| 4 | 哪个编译阶段改坏了 | `--xla_dump_to=<目录> --xla_dump_hlo_pass_re=.*` 导出每个 pass 之后的 HLO，逐份改写成 CPU 能编的形式，在 CPU 上执行（[`hlo_pass_cpu_replay.py`](../analysis/hlo_pass_cpu_replay.py)） | 编译前、94 个 pass 之后、最终程序共 96 份，CPU 上全部算对，和编译前逐位相同（[`jit-miscompile-pass-replay-v5p-0046.json`](../results/jit-miscompile-pass-replay-v5p-0046.json)）。HLO 一直是对的，错在 HLO 之下 |
| 5 | 哪颗 kernel 出事 | 比较相邻 pass 的程序，找 DUS 所在的 fusion 什么时候变的 | 「multi output fusion and bf16 folding」阶段里的 `tpu_multi_output_fusion` 之后，DUS 和 `a3[:2] + a6*a0`、矩阵乘 `a1·a2ᵀ` 及两个求和合成一颗 4 输出的 fusion。11-06 的 nightly 不再把 DUS 合进去 |
| 6 | 关哪个开关能修 | 逐个关候选 pass，开关名取自 dump 里的 `tpu_comp_env.txt` | 只有 `--xla_jf_enable_multi_output_fusion=false` 能修。其余 9 个都不行（见下） |
| 7 | 触发面多大 | 小块行数 k、下面的块行数 m、宽度 w 扫网格，读每格的 dump | 出错的只有 k = 2（m ≥ 3）和 k = w = 4。分块组合见[机理](#机理) |
| 8 | v7x 上呢 | 同 146 个用例；导出最终程序 | 0.0.46、0.0.52、nightly 全对。同一颗 4 输出 fusion 照样生成，区别是 v7x 上 DUS 的目标缓冲和兄弟输出都放在片上内存 |

第 6 步关了不起作用的 9 个开关：

- `--xla_jf_enable_producer_consumer_multi_output_fusion=false`
- `--xla_vf_vmem_memory_space_assignment=false`
- `--xla_vf_vmem_enable_cross_program_prefetch=false`
- `--xla_tpu_enable_aggressive_loop_fusion_layout_opt=false`
- `--xla_tpu_rwb_fusion=false`
- `--xla_tpu_enable_copy_fusion=false`
- `--xla_jf_conv_input_fusion=false`
- `--xla_jf_conv_output_fusion=false`
- `--xla_jf_enable_final_priority_fusion=false`

其中关掉片上内存分配（第 2 个）也照样错，所以和数据放不放进片上内存无关。

## 机理

1. **出错的那颗 kernel。** v5p、0.0.46 的最终程序里，这颗 fusion 有 4 个输出：
   - DUS：原地把 `a4` 写进 `a3` 的前 2 行。目标缓冲 `f32[6,2]`，按 8×128 分块，和输入 `a3` 共用一块内存；
   - `a3[:2] + a6 * a0`：读的正是 DUS 改写的那块缓冲的前 2 行。输出 `f32[2,2]`，按 2×128 分块；
   - 矩阵乘 `a1·a2ᵀ` 的两个求和（`out0` 和 `t14` 都从它来）。

   程序语义没错：CPU 上执行最终程序结果对。但 v5p 为它生成的代码把第二个输出算丢了，下游读到的这一项等于 0。
2. **上游的修法是不再这样合并。** 11-06 起的 nightly 里，DUS 单独一颗 kernel，`a3[:2] + a6*a0` 和矩阵乘合在另一颗（第 5 步两个版本最终程序的对比）。所以修复是在合并这一步挡住，不是改代码生成；v5p 的代码生成碰到同样的 kernel 还会不会错，不知道。
3. **为什么每一项都不能去掉。** issue 里去掉任何一项都对，包括数学上不影响 `out1` 的 `t14`。推断是这几项决定了合并会不会发生（逐项去掉后的程序没有逐个读）：
   - 合并要求几颗 kernel 共享输入、输出形状兼容；
   - `t14` 和 `out0` 各要矩阵乘结果的一个求和，这两个求和都成了合并后 kernel 的输出；
   - 去掉 `a3[:2]`、`a6 * a0` 或 `a5`，参与合并的 kernel 就变了。
4. **哪些分块组合出错**（v5p 0.0.46，读 dump）：

   | 用例 | DUS 目标的分块 | 兄弟输出的分块 | 合没合 | 结果 |
   |---|---|---|---|---|
   | k = 2，m = 4（issue） | 8×128 | 2×128 | 合 | 错 |
   | k = w = 4，m = 4 | 8×128 | 4×128 | 合 | 错 |
   | k = 2，m = 2 | 4×128 | 2×128 | 合 | 对 |
   | k = w = 8，m = 4 | 8×128 | 8×128 | 合 | 对 |
   | k = w = 3，m = 4 | — | — | 没合 | 对 |

   出错的两行都是「目标缓冲按整块（8 行）存，兄弟输出按小块（2 行或 4 行）存」。只有这一种分块组合，或者还要别的条件，没有查清。
5. **v7x 为什么对。** 同一颗 fusion、同样的分块，v7x 上 DUS 的目标缓冲和兄弟输出都分配在片上内存，v5p 上都在 HBM。按这个差别推断：出错的是 v5p 上「DUS 原地写 HBM 里的缓冲 + 兄弟输出」这条代码生成路径。没有读到代码生成层，是推断。

## workaround

全部在 v5p、libtpu 0.0.46 上验证（[`jit-miscompile-probe-v5p.json`](../results/jit-miscompile-probe-v5p.json)）。0.0.52 上只验证了 ①。

| # | 办法 | 怎么做 | 验证 | 代价 |
|---|---|---|---|---|
| ① | 关 multi-output fusion | `LIBTPU_INIT_ARGS=--xla_jf_enable_multi_output_fusion=false` | 0.0.46 到 0.0.52 每个版本 121 个用例全对，原来错的 14 个都在里面。比默认的 146 个少 25 个：10 个是默认那组里重复跑的用例，9 个是 ② 的屏障用例，6 个是 ③ 的 k = 8 / 16、w = 4 用例（默认就对） | 全局改融合，会改别处的数值（不再逐位）；步时随模型而变：MaxText llama3.1-8b +0.5%，deepseek3-tiny −2.4%（v5p 4 片，[`jit-miscompile-mof-off-steptime.json`](../results/jit-miscompile-mof-off-steptime.json)）。混元 3 / 混元 4 没量 |
| ② | 隔开读和写 | 对被改写那块缓冲的读取加屏障：`head = jax.lax.optimization_barrier(a3[:k])`，再用 `head` | 原来错的 14 个里测了 9 个，全对：k = 2 时 m = 3、4、6、8、16，k = 4、w = 4 时 m = 4、8、12，DUS 写法。另外 5 个没加屏障测：issue 正文里的原函数、k = 2 时 m = 5 / 12、k = 2、m = 4 时 `d` = 8 / 128 | 只挡住这一处合并，程序其余部分不变。屏障加在写入结果上（`optimization_barrier(a3.at[:2].set(a4))`）**没用**，照样错（探索时测的，不在结果文件里）；加在 `acc` 上也没用 |
| ③ | 小块行数补到 8 的倍数 | 每段区域按 8 行对齐 | k = 8、16 配各种 m 和 w，16 个用例全对；k = 1、3、5–7、9、10 也都对 | 补的行要算、要占内存；只在 toy 上验过，换了形状要重新验 |
| ④ | 换版本 | 等带修复的正式版；nightly 0.0.53.dev20261106 起已修 | 146 个用例全对 | nightly 没经过发版测试，不建议上生产；换版本会改别处数值，要重建 golden（[差异清单 F7](../../../docs/reference/tpu-vs-gpu-differences.md#f7-libtpu-版本改的不只是求和顺序)） |

推荐顺序：先用 ② 修出事的那一处，代价最小、不动别处；找不到出事位置时用 ① 先止血，再拿 ② 或 ③ 换掉；有正式版以后换 ④。

## 各版本状态

| libtpu | 构建（changelist） | v5p：146 个用例里 jit 算错 | v7x |
|---|---|---|---|
| 0.0.46（镜像自带） | 963683785 | 14 | 0 |
| 0.0.48 | 981334199 | 14 | 没测 |
| 0.0.51 | 1003875606 | 14 | 没测 |
| 0.0.52 | 1014260217 | 14 | 0 |
| nightly 0.0.52.dev20261105 | 1014260217 | 14 | 没测 |
| nightly 0.0.53.dev20261106 | 1015271470 | 0 | 没测 |
| nightly 0.0.53.dev20261111 | 1018744029 | 0 | 0 |

- 逐算子执行在每个版本、每个用例上都对，最大误差 1.4e-7。
- 0.0.50 / 0.0.50.1 没测：这两个版本 scatter 有已知回归，不用（[环境](../../../docs/reference/environment.md#tpu-侧)）。
- 0.0.47、0.0.49 没测：构建早于修复，推断同样出错。
- 0.0.45（`cl/957406610`）没测：它早于 0.0.46，bug 是哪个构建引入的没查，所以既不能说有、也不推断。下面扫描里有 5 份 0.0.45 的训练步程序（TPU 自比 F2 / F2b 两臂），都没有这个 bug 的特征。
- 第一个带修复的正式版预计是 0.0.53。这是推断：0.0.52 就是主干直接构建的，0.0.53 的 nightly 已经修了。
- 数据：v5p 见 [`jit-miscompile-probe-v5p.json`](../results/jit-miscompile-probe-v5p.json)，v7x 见 [`jit-miscompile-probe-v7x.json`](../results/jit-miscompile-probe-v7x.json)。

## 对混元 3、混元 4 的影响

**扫描**：[`hlo_mof_dus_scan.py`](../analysis/hlo_mof_dus_scan.py) 读训练步的优化后程序，找这个 bug 的特征：

- 一颗多输出 fusion 里有 DUS；
- DUS 改写的那块缓冲，在同一颗 fusion 里还被别的指令读。

扫了存档的全部 292 份训练步程序，特征出现 **0 次**（[`jit-miscompile-hlo-scan.json`](../results/jit-miscompile-hlo-scan.json)）：

| 模型 | 份数 | 平台 | libtpu | 多输出 fusion 含 DUS | 特征 |
|---|---|---|---|---|---|
| 混元 3 截 4 层 | 72 | v5p | 0.0.46 | 71 份里有（70 份各 2 处，1 份 5 处） | 0 |
| 混元 4 mini | 209 | v5p | 0.0.45–0.0.52 | 5 份里有（都是实验臂；0.0.52 的 golden `ts-g52-a` 没有，0.0.46 的 golden `ts-g-a` 另扫也没有） | 0 |
| 混元 4 mini | 11 | v7x | 0.0.46、0.0.52 | 11 份都有 | 0 |

- 混元 3：DUS 的目标是 `bf16[3,2,4096,…]` 这类缓冲，沿最外维写进一份，另一个输出是写进去的值本身，不读目标缓冲。
- 混元 4 v5p：4 个臂的目标是新分配的 `bf16[128,4096]` 空缓冲；张量并行 2 的那个臂，目标是 `f32[4]`、`f32[8]`、`f32[128]` 这类一维小缓冲。都不读目标缓冲。
- 混元 4 v7x：目标是沿最外维写的 `bf16[2,8,4096,256]`，或 `f32[2,1,8]` 这类小缓冲。同样不读目标缓冲。
- 这些组合 toy 里没复现出错，但也没有逐个证明安全。

**没存 HLO 的两次运行：在 CPU 上重编再扫**（[`jit_miscompile_offline_scan.py`](../analysis/jit_miscompile_offline_scan.py)，结果 [`jit-miscompile-offline-scan.json`](../results/jit-miscompile-offline-scan.json)）。mini 现行比较 `tz-off-rep` 与推荐组合 `x5-prod` 的运行目录里没有 `hlo/`。按各自 `meta.json` 推出程序规格（镜像 `tpu_pre_training@sha256:b152355a…`、libtpu 0.0.46 构建 `cl/963683785`、MaxText `b87a55aa`、yml 按 sha256 核对、覆盖参数与 `LIBTPU_INIT_ARGS` 照抄），走 TPU 自比离线筛选的同一条路径（`hlo_screen.compile_spec` → `frameworks/maxtext/aot_hlo.py`），在 CPU 上为 v5p 8 片（FSDP 8）编出训练步优化后 HLO，再用 `hlo_mof_dus_scan.py` 扫：

| 运行 | 相对 golden `ts-g-a` 的规格差别 | 多输出 fusion 含 DUS | 特征 |
|---|---|---|---|
| `tz-off-rep` | 覆盖 `sa_use_base2_exp=false` | 0 | 0 |
| `x5-prod` | yml 换 janus `2a1d08e7` 那版（sha256 `86beb9e5…`），另加 9 个覆盖参数（开 `splash_dsinks_fp32`、`embedding_grad_fp32`、`rmsnorm_scale_grad_fp32`，关其余 6 个） | 0 | 0 |

离线编的程序是不是上卡的那一份：这两次运行自己没存 HLO，不能直接对程序指纹，下面两条是间接证据，结论是推断。

1. 同一次扫描里把存了 HLO 的 golden `ts-g-a` 也按 `meta.json` 重编：CPU 编出的程序与 TPU 上存下的规范化后逐行相同（`calib` 一节，程序指纹两边都是 `program_fp` 里那一个）。
2. 同一条路径在 TPU 自比的离线筛选里对 97 个存了 HLO 的运行做过同样的比较，全部逐行相同（[`hlo-screen.json`](../alignments/04-tpu-self/results/hlo-screen.json) 的 `cpu_vs_tpu`）。

`tz-off-rep`、`x5-prod` 与 `ts-g-a` 用同一镜像、同一 libtpu 构建、同一 MaxText 提交，只差上表那几项输入。所以推断离线编出的就是上卡的程序。没有直接对过。

**为什么还是写「不能完全排除」**：

1. 扫的只有混元 3 截 4 层和混元 4 mini。混元 4 离生产形状最近的真实宽度运行（02 的截层 B / D1、03 的 mid、06 的 D3 16 片）的训练步程序没扫。生产尺寸的形状、切分、融合都不一样，没有生产尺寸的训练步程序可扫。mini 的现行比较 `tz-off-rep`（关 base2）与推荐组合 `x5-prod` 是离线重编后扫的（见上），离线程序和上卡程序是同一份，是推断。
2. 触发条件没有完全查清（[机理](#机理)第 4、5 条）。特征只是必要条件，v7x 有特征照样算对；会不会有不带这个特征的变种，不知道。
3. 扫到的版本有限。按构建号数（[扫描结果](../results/jit-miscompile-hlo-scan.json) 的 `summary`）：混元 3 全是 0.0.46（72 份）；混元 4 mini 主要是 0.0.46（v5p 167 份、v7x 8 份），另有 0.0.52（v5p 33 份、v7x 3 份）、0.0.45（5 份）、0.0.47（2 份）、0.0.48（2 份），后三个只在 v5p 上。没扫到的版本、换开关都会改融合，要重扫。

推断：0.47 量级的错只要落在训练用得到的张量上，第 1 步逐张量对 FP64 的定责就会看到远超舍入的偏差。

**MaxText 里哪些写法可能撞上**（路径相对 `src/maxtext/`；要同时满足：一次 jit 里，同一块缓冲既被 DUS 或 `.at[].set` 改写一小块，又被读一部分，而且行数小到会按 2 行或 4 行一块存）：

| 写法 | 位置 | 风险 |
|---|---|---|
| 推理的 KV cache 写入 | `inference/kvcache.py`：`dynamic_update_slice_in_dim` 写入新 token，同一步注意力读 cache | 解码时批小、每次写 1 个或几个 token，最像。训练不走这条路径 |
| 流水线并行的循环缓冲 | `layers/pipeline.py`：按微批 `dynamic_update_slice_in_dim` 写 state，同一步再读 | 微批行数小时要查；janus 的对齐配置没开流水线，生产配置要另查 |
| 每段长度不同的多流 / 多模态拼接 | 用户自己写的 `.at[:n].set` + 切片 + `pad` / `concatenate`（issue 的来源就是这种） | 段长 2 行或 4 行时最危险 |
| MTP 的移位 | `layers/multi_token_prediction.py`：`roll(...).at[:, shift:].set(0)` | 只写序列维上最后 1 个位置；toy 里 1 行的小块没出错，训练程序扫描也没有特征 |
| 按层堆叠的扫描 | `layers/nnx_decoders.py`：`dynamic_update_slice_in_dim` 沿层维写 | 扫描结果里出现过，但不读目标缓冲，见上 |
| 优化器里的小数组 | `optimizers/optimizers.py`：`losses.at[idx].set(loss)` | 一维小数组，没扫到出事的组合 |

## 怎么在自己的代码里自查

1. **先分清是 bug 还是精度。** TPU 上 jit 的结果和 CPU 差得远（相对误差 1e-2 以上），或者只在某些形状下错，就在 TPU 上用 `jax.disable_jit()` 逐算子跑同一个函数。逐算子对、jit 错，就是编译问题，不要再往精度开关上找（[方法](../../../docs/guide/06-debugging.md#怀疑编译-bug逐算子执行对-jit)）。
2. **扫优化后的程序。** 训练加 `XLA_DUMP=1`（运行目录 `hlo/` 下就有），或者自己加 `--xla_dump_to`，然后：
   ```bash
   python models/hunyuan4/analysis/hlo_mof_dus_scan.py <运行目录>/hlo/train_step.after_optimizations.txt
   ```
   - 第二列不为 0，会列出可疑的 fusion：DUS 目标和更新的形状、分块，以及读目标缓冲的指令。
   - 有特征时退出码为 1，可以接进检查脚本。
3. **有特征就验。** 在可疑位置加 workaround ②，或者整体加 ①，各跑一次第 1 步：
   - 结果逐位不变：这一处没有算错；
   - 结果变了：再用 FP64 参照判谁对。
4. **复现与扫描。** 换了 libtpu 或形状，跑一遍 [`jit_miscompile_probe.py`](../analysis/jit_miscompile_probe.py)，看 146 个用例里 jit 错几个、逐算子错几个。
5. **定位到 pass。** 程序算错又不是这个特征时，用 [`hlo_pass_cpu_replay.py`](../analysis/hlo_pass_cpu_replay.py) 把每个 pass 之后的程序拿到 CPU 上执行：
   - 找到第一份结果变了的，就是那个 pass 改坏了语义；
   - 全对，错就在代码生成，这时只能靠关开关二分。

## 拿不准的

- 代码生成里具体哪一步把兄弟输出丢了，没有查到指令层。「v5p 上 HBM 里原地写的路径出错」是从 v7x 对、v5p 错推的。
- 触发的分块组合只扫了 toy：k 取 1–10、16，m 取 1–6、8、12、16，w 取 1–5、8、16，矩阵乘内维取 2、8、128。BF16、更大的形状、多设备切分都没扫。
- v7x 只测了 0.0.46、0.0.52 和 11-11 的 nightly。
- 第一个带修复的正式版是哪一个，是推断。

## 这次排查用到的 janus 能力

| 能力 | 在本案例中的作用 |
|---|---|
| 逐算子执行对 jit（[第 6 章](../../../docs/guide/06-debugging.md#怀疑编译-bug逐算子执行对-jit)） | 一步分清编译 bug 和精度问题 |
| 同一镜像里只换 libtpu | 找到修复的版本窗口 |
| 逐 pass 程序在 CPU 上回放（`hlo_pass_cpu_replay.py`） | 证明 HLO 一直对，问题在代码生成 |
| 训练步优化后 HLO 的存档 | 离线扫 292 份程序判断影响，不用上卡 |
| 按 `meta.json` 在 CPU 上重编训练步（`hlo_screen.compile_spec`） | 没存 HLO 的运行（`tz-off-rep`、`x5-prod`）也能扫 |
