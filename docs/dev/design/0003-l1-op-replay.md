# L1 单算子：真实输入回放与合成用例

**目的**：在算子这一级回答三个问题——

1. 回放写法和训练是不是同一个东西（逐位）；
2. GPU 自己离 FP64 真值有多远（被测换成 TPU 时的尺子）；
3. GPU 的舍入方式能不能在 CPU 上猜中（猜中了，TPU 侧就能拿同一套模拟做「降精度复现」）。

**检查类型**：`op_replay`（规格 [`reference/checks/op_replay.md`](../../reference/checks/op_replay.md)）。

- CPU 参照（FP64 正确舍入、R-ratio、GEMM 舍入模拟、打平规则、解析 VJP）在 `janus/analysis/ops.py`。
- 纯 numpy、有单测。

**脚本**：`models/hunyuan3/frameworks/megatron/l1_ops.py`。

- GPU 回放要 torch、Transformer Engine 与 Megatron 的路由函数，所以不走 `janus run` 的目录。
- 而是独立脚本直接调 `op_replay`，写同一格式的 `results.json` + `details/` + `report.md`。
  - `catalog` 字段填脚本路径与 sha256；
  - 另加 `environment` 记进程内的 TF32 开关与版本。
- 只用一张卡、小规模，和训练任务同时跑不冲突。

```bash
source /data/env.sh; export PYTHONPATH=<janus 检出>:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python models/hunyuan3/frameworks/megatron/l1_ops.py --run /data/janus-runs2/d-a --out /data/janus-l1/d-a
```

输入：

- 真实输入是确定性运行 `d-a` 第 1 步抓下的张量（rank 0、第一个 microbatch）。
- 合成输入用固定种子。

## 检查一览

门槛的来源写在每条检查的 `source` 里，下表只摘要。「推断」的门槛是起点，看实测再收紧。

| 检查 | 做什么 | 判据 |
|---|---|---|
| `L1.gemm.router.replay` | 路由 GEMM：抓下的路由输入与路由权重，调 Megatron 的 `router_gating_linear` 重算 logits（见注 1） | 与抓下的 logits 逐位（exact）；另报对 FP64 的 rel_L2、逐位相同率、ULP |
| `L1.gemm.router.upcast_fp32` | 同一输入「先升 FP32 再 `torch.mm`」，TF32 关 / 开各一次 | 诊断 |
| `L1.gemm.attn_proj.replay` | 注意力输出投影：`attn.core.out × linear_proj.weightᵀ`，TE `general_gemm` 与 `F.linear` 两种写法 | 与 `attn.proj.out` 逐位 |
| `L1.gemm.qkv.replay` | qkv：TE `LayerNormLinear`（RMSNorm 融合 GEMM），输入 `layers.<i>.in` | 与 `attn.qkv.out` 逐位；FP64 真值取「RMSNorm 的 BF16 输出 · Wᵀ」 |
| `L1.norm.replay` | TE `RMSNorm`：q / k 归一化、MLP 前归一化（见注 2） | 与抓下的输出逐位 |
| `L1.elem.{exp,sigmoid,silu,rsqrt}.fp32` | FP32 初等函数，输入用真实激活与合成区间（见注 3） | `ulp_max`：exp、rsqrt ≤ 2（CUDA 手册 expf / rsqrtf）；sigmoid、silu ≤ 4（推断） |
| `L1.elem.tail` | sigmoid / silu 在 [−110, 110]，看 x < −88.7 的尾部 | 诊断 |
| `L1.elem.bf16_vs_fp32` | 全部有限 BF16 值穷举：BF16 直接算 vs 先升 FP32 算再舍到 BF16 | 诊断 |
| `L1.elem.basic` | FP32 / BF16 加、乘与几种类型转换；torch GPU 对 numpy CPU（见注 4） | 逐位 100%、对正确舍入 100%（IEEE 754 / CUDA 手册 0 ULP） |
| `L1.rratio.clean` | R-ratio 三方比：torch FP32 公式的 RMSNorm 对 TE RMSNorm；FP64 算 sigmoid 再舍 | `r_ratio ≤ 1.2`（推断：起点） |
| `L1.rratio.drill` | 同上，被测换成「多舍一次」：`x·r` 先舍到 BF16 再乘权重；logits 先舍到 BF16 再 sigmoid | 演练：门槛必须报警 |
| `L1.emulate.gemm` | CPU 上用张量核累加模型（见下）复现路由 logits、注意力投影与 qkv 的 GEMM 部分（见注 5） | 逐位命中 100% |
| `L1.emulate.other_guesses` | 朴素猜法与改一个参数的模型；qkv 连 RMSNorm 一起在 CPU 上模拟 | 诊断 |
| `L1.extreme.rmsnorm` | 全零行、1e-20 行、1e-40 次正规行、正常行；四种实现对 FP64 公式（见注 6） | `ulp_max ≤ 2`、非有限值位置一致（推断） |
| `L1.extreme.rmsnorm.overflow` | 平方和超过 FP32 最大值的行 | 诊断 |
| `L1.extreme.sigmoid` / `softmax` | sigmoid、softmax 的极端输入（见注 7） | sigmoid `ulp_max ≤ 4`；softmax `rel_l2 ≤ 1e-5`（推断）；非有限值位置一致 |
| `L1.ties.synthetic` | 人造打平的分数，GPU `torch.topk` 对 CPU 稳定排序（见注 8） | 逐行集合一致 |
| `L1.ties.cpu_torch` | CPU `torch.topk` 对同一规则 | 诊断 |
| `L1.ties.real` | 真实路由分数（sigmoid(logits) + expert_bias）：回放 top-k 对抓下的 `topk_idx`、对稳定排序；数第 8 / 9 名逐位打平的行 | 逐行集合一致 |
| `L1.vjp.fp32` / `bf16` | 随机 cotangent：GPU autograd 对 FP64 解析梯度（见注 9） | `rel_l2`：FP32 ≤ 1e-5，BF16 ≤ 1e-2（推断） |

- 注 1：路由输入是 `mlp_norm.out`（BF16）；调用时 `moe_router_dtype=fp32`。
- 注 2：q / k 从抓下的 qkv 按 Megatron 分组布局切出；MLP 前归一化的输入 = 层输入 + 注意力输出。
- 注 3：
  - 真实激活：路由 logits、softmax 输入、稠密 MLP 的 gate、RMSNorm 的均方。
  - 合成区间：rsqrt 穷举 [1, 4) 全部 FP32。
- 注 4：类型转换包括 FP32→BF16、FP32→FP16、BF16→FP32、FP64→FP32。
- 注 5：路由 logits 是 FP32 出、全部行；注意力投影与 qkv 是 BF16 出、每层抽 64 行。
- 注 6：
  - 1e-20 行的 x² 是次正规数。
  - 四种实现：torch 公式、`F.rms_norm`、TE FP32、TE BF16。
- 注 7：
  - sigmoid：±Inf、NaN、±0、次正规、±1e30。
  - softmax：全 −Inf、含 +Inf、全 1e30、一个 3e38 等 9 种行。
- 注 8：
  - 分数：4 档、32 档、全相等。
  - GPU `torch.topk`：k=8、192 列、sorted。
  - CPU 稳定排序：打平时下标小者优先。
- 注 9：
  - 被测：RMSNorm（torch FP32、TE BF16）、路由分数函数（FP32）与整条路由（含门控 GEMM，梯度 BF16）。
  - 解析公式先对 FP64 中心差分自检。

## 结论（`models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-l1.*`，25 条全 pass）

数字都来自结果文件；「推断」是没进结果文件的解释。

### 回放、GEMM 与张量核累加

1. **回放写法与训练逐位一致**。
   - 全部 100% 逐位：路由 logits（3 层）、注意力输出投影（4 层 × 2 种写法）、qkv（4 层）、q / k / MLP 前归一化（11 个）。
   - 回放的 top-k 与抓下的 `topk_idx` 100% 一致。
2. **`moe_router_dtype=fp32` 的路由 GEMM 不是「先升 FP32 再 matmul」**。
   - 镜像里的 Megatron 在 TE 可用时走 TE `general_gemm`（BF16 入、FP32 出）。
   - 先升 FP32 的写法与训练逐位相同率只有 0.002%–0.05%（`L1.gemm.router.upcast_fp32`）。
   - TPU 侧照「升 FP32」去对，GPU 这边的参照本身就不是那个写法。
3. **路由 logits 离 FP64 比想象的远**：
   - rel_L2 3.76e-6–3.96e-6，对正确舍入逐位相同率 0.002%–0.04%，ULP p99.9 80–173。
   - 同一输入先升 FP32、关 TF32 的 `torch.mm` 是 7.5e-7–7.6e-7（准约 5 倍）。
   - 开 TF32 是 7.4e-6–7.7e-6。
   - 原因见第 4 条。
4. **猜中了 B200 张量核的累加方式**：
   - 模型：
     - 每 16 个 BF16 乘积连同累加器一起对齐到最大指数 emax（乘积的指数取两个输入指数之和，不按乘积规格化）；
     - 低于 2^(emax−25) 的位向零截掉，精确相加；
     - 写回 FP32 时向零舍入。
   - 命中：
     - 逐位复现路由 logits 三层全部 4096 × 192 个元素（100%）；
     - 以及注意力投影、qkv GEMM 部分抽样的全部输出（BF16 出 = 该模型的 FP32 结果再 RNE 舍到 BF16，100%）。
   - 朴素猜法几乎全不中：FP64 点积舍到 FP32 0.04%、FP32 顺序累加 0.05%。
   - 模型里只改一个参数（16 → 8 个一组、25 → 24 位、向零 → 就近）都掉到 1% 以下。
   - 模型逐位命中，说明第 3 条的误差完全由这两处截断决定。
   - qkv 连 RMSNorm 一起在 CPU 上模拟时，4 层分别 99.94% / 100% / 99.92% / 100%。
     - 差在 RMSNorm 一步（CPU 模拟对 TE RMSNorm 99.9989%–99.9996%）。
     - 推断是内核里 rsqrt 的近似与均方归约顺序，没有继续追。
5. **逐位相同率（对 FP64 正确舍入）**：
   - BF16 出的 GEMM 里，FP32 累加误差多半被最后一次舍入吞掉：
     - 注意力投影 99.48%–99.61%、qkv 99.78%–99.81%（ULP p99.9 都是 1）。
   - TE RMSNorm 99.9988%–99.9993%（最大 1 ULP）。
   - FP32 出的路由 logits 只有 0.002%–0.04%。

### 初等函数与逐元素运算

6. **初等函数（FP32）**：
   - exp 最大 2 ULP（对正确舍入逐位 69%–70%）。
   - rsqrt 最大 2 ULP（74%–78%，含 [1, 4) 全部 1678 万个 FP32）。
   - sigmoid 最大 3 ULP（62%–66%），silu 最大 3 ULP（61%–71%）。
   - **尾部**：x < −88.72 时 `1/(1+exp(−x))` 里 exp 溢出，sigmoid 输出 0 而真值是次正规数。
     - 超 4 ULP 的输入落在 [−101.8, −88.72]（silu [−106.4, −88.72]）。
     - 绝对误差小于 FP32 最小正规数，对训练无影响，但 ULP 型门槛要避开这一段。
   - **BF16 输入**：
     - 四个函数「BF16 直接算」与「先升 FP32 再舍到 BF16」100% 逐位相同（推断：torch 内部按 FP32 算再舍）。
     - 对 BF16 正确舍入：exp、rsqrt 100%，sigmoid 99.988%、silu 99.974%。
     - 超线的全在同一尾部（x ∈ [−95, −89]）。
7. **逐元素加、乘、类型转换**：
   - 每类 400 万–800 万个元素（含随机位模式里的 NaN、Inf、次正规数）。
   - GPU 对 CPU 100% 逐位、对正确舍入 100%。

### R-ratio、极端输入与打平

8. **R-ratio 看得出 cosine 看不出的差**：
   - 干净实现：R = 1.0000（torch FP32 公式对 TE RMSNorm）、0.49（FP64 算 sigmoid 再舍，比参照更准）。
   - 多舍一次的 RMSNorm：R = 1.41（≈ √2：多一次同量级的独立舍入）。
     - 而它对 FP64 的 cosine 是 0.9999973、参照 0.9999986，0.9999 的 cosine 门槛两个都过。
   - logits 先舍 BF16 再 sigmoid：R = 13158，cosine 仍有 0.99999982。
   - 门槛 1.2 两边都分得开。
9. **极端输入**：
   - 全零行四种 RMSNorm 都出 0（没有 NaN）；1e-20、1e-40 行最大 1 ULP。
   - **平方和溢出**（RMS 超过 √(3.4e38 / 4096) ≈ 2.9e17）时，四种实现全部静默输出全 0（`out_zero_frac` = 1），不报 NaN / Inf。
     - 真实激活离这个量级很远。
     - 但这是「数值炸了却不红」的一种，L2 / L4 的健康检查不能指望 NaN 来报。
   - sigmoid 17 个极端输入最大 1 ULP。
   - softmax 9 种行都在门槛内，全 −Inf、含 +Inf 两边都是 NaN。
     - 正常行里小概率项的 ULP 随 |x − max| 线性变大（本次最大 15 ULP）。
     - 推断是 x − max 在 FP32 里舍入一次、exp 的条件数正好是 |x − max|，不是 exp 本身的误差（exp ≤ 2 ULP）。
     - 所以 softmax 用 rel_L2 判、不用 ULP 判。
10. **打平**：
    - GPU `torch.topk`（k = 8、192 列、sorted）在三组人造打平分数上（打平行 4096 / 3401 / 256）与「下标小者优先」100% 一致。
      - 和 `jax.lax.top_k` 默认（`is_stable=True`）的规则相同。
    - CPU `torch.topk` 不一致（4096 / 3014 / 256 行不同），**CPU 上的 torch 不能当打平规则的参照**。
    - 真实数据：3 个 MoE 层共 12288 个 token，路由分数第 8 / 9 名逐位打平的行数为 0。

### 反向与 TF32

11. **随机 cotangent VJP**：
    - FP32：RMSNorm dx 7.5e-8、dw 1.9e-7，路由分数函数 d logits 8.4e-8。
    - BF16：TE RMSNorm dx 1.66e-3、dw 1.72e-3，整条路由 dx、dW 都是 1.68e-3（BF16 一次舍入的量级）。
    - 解析梯度对 FP64 中心差分的相对差 7.7e-11 / 1.4e-10。
12. **TF32**：
    - 本镜像设了 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`。
      - 进程里 `torch.backends.cuda.matmul.allow_tf32` 与 `cudnn.allow_tf32` 默认都是 True、`float32_matmul_precision` 是 high。
    - 训练进程里的值没记进 meta。
    - 路由与各投影都是 BF16 入的 GEMM，不受这个开关影响。
    - FP32 入的 matmul 会受影响（第 3 条：开 TF32 误差大 10 倍）。
    - 后续（`l1-kernels.md`「一」）：
      - torch 开关管不到 TE 的 FP32 GEMM，只有 `NVIDIA_TF32_OVERRIDE=0` 关得干净；
      - 训练里 FP32 输入的 GEMM 只有路由反向的两个，它们一直是 TF32。

## 范围与局限

- 只回放第 1 步、rank 0 第一个 microbatch。
  - 以下见 `l1-kernels.md`（靠新加的可选观测点抓专家与注意力的输入）：专家分组 GEMM、核心注意力的 kernel 回放，MoE 层内分阶段，合成 poison 与因果性，特殊值存 / 恢复。
- 张量核累加模型只在 B200、cuBLAS 13.8、BF16 入的 GEMM 上验证过。
  - 换卡、换 cuBLAS、换 FP8 都要重新试参数。
  - 它是「对这些形状逐位命中」的经验模型，不是 NVIDIA 公开的规格。
- 打平规则只测了 k = 8、192 列、`sorted=True`（Megatron 训练时的调用）。
  - `torch.topk` 的实现随形状换算法，别的形状要重测。
