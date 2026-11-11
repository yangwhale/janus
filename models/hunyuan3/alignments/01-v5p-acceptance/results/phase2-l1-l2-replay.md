# 第 1 步 L1 单算子回放与 L2 逐层隔离回放（v5p 对 golden `g1k-a`）

> 换代前的读数：maxtext `4d4ffac2`，打开 `dense_fp32_accumulation` 之前。现行判定按 FP64 R 比，见 [`phase2-l2-replay-rratio-on`](phase2-l2-replay-rratio-on.report.md)（四层前向 R 1.06 / 1.01 / 1.01 / 1.09，都过）与 `phase2-l1-ops-own-on`。

## 怎么做的

- **脚本**：`../../../frameworks/maxtext/layer_replay.py`（L2）与 `l1_probes.py`（L1）。
- **L2 逐层隔离回放**：
  - 每层的输入用 golden（`g1k-a`，第一阶段 GPU golden 的第 1 份）的 `layers.k.in`（rank 0 第一个 microbatch，4096 个 token）；
  - 在 TPU 上只编译这一层，前向和观测点与 golden 同名比；
  - 反向用 golden 的 `layers.k.out.grad` 作余切，比 `layers.k.in.grad`。
  - 层与层之间的误差不会串起来。
  - 回放程序本身就是测量对象，所以没有「钩子改了生产图」的问题。
- **L1 单算子回放**：把算子的输入换成 golden 的，喂给同一个 MaxText 子模块。
- **代码**：
  - `maxtext` hunyuan@`4d4ffac2`（含 `routed_gate_bf16_rounding`、`rmsnorm_single_rounding`、`mlp_activation_round_per_op` 三个开关）；
  - janus `67208dd`；
  - v5p 2x2x2、8 设备，XLA 默认设置（excess precision 开）。
- **数据**：
  - `phase2-replay-<运行>.json`：每层每个观测点的 rel_l2、ULP 分布、路由一致比例与分差；
  - L1 在 `r-l2-base`（三个开关都不开的逐层回放，见下）那份的 `l1` 字段；
  - 张量在桶里，`tpu/<运行>/`。

### 四个运行

只差 MaxText 开关：

- `r-l2-base`：生产配置，三个开关都不开，另带 `--l1`；
- `r-l2-norm`：`rmsnorm_single_rounding=true`（RMSNorm 在 FP32 里算完只舍一次）；
- `r-l2-norm-perop`：再加 `mlp_activation_round_per_op=true`（门控激活按 GPU 的两个算子分别舍入）；
- `r-l2-norm-f32`：改加 `activations_in_float32=true`（门控激活在 FP32 里算完舍一次）。

`routed_gate_bf16_rounding`（路由核先舍 BF16 再乘）不影响第 1 步：核本来就是 BF16 值，L1 里开 / 关逐位相同。

## L1 结论

| 算子 | GPU 怎么算 | v5p 上 MaxText 实际怎么算 | 证据（`r-l2-base`（三个开关都不开）的 `l1` 字段） |
|---|---|---|---|
| 路由 GEMM（FP32 操作数，`matmul_precision: default`） | BF16 × BF16，FP32 累加（张量核截断累加） | **DEFAULT = 两个操作数先舍 BF16，单遍乘，FP32 累加**（见注 1） | `router_precision_probe` 三行 |
| 路由 GEMM（输入用 golden 的 `mlp_norm.out`） | — | 按 golden 偏置重选 top-8，**3 层翻转都是 0**（见注 2） | `router_gemm` 六行 |
| RMSNorm（MoE 层的 MLP 前归一化，输入 = bf16(golden `in` + `attn.out`)） | TE RMSNorm，FP32 算完舍一次 | **默认写法已经只舍一次**（见注 3） | `rmsnorm` 六行 |
| 门控激活 silu(gate)·up（第 0 层，输入用 golden `fc1.out`） | `silu` 舍一次、乘积再舍一次（两个 torch 算子） | **默认写法编出来是「FP32 算完舍一次」**（见注 4） | `silu_mul` 三行 |
| silu 单独（BF16 输入） | torch 内部按 FP32 算再舍 | 内部按 FP32 算；v5p 和 v7x 在这一点上不同（见注 5） | `silu` 一行 |

1. **路由 GEMM 的精度探针**：
   - 探针：一份不能用 BF16 表示的核（BF16 核 × (1 + 2⁻¹⁰·N(0,1))）。
   - DEFAULT 的结果与「显式把两个操作数舍成 BF16 再点积」3 层全部逐位相同。
   - 与开 `routed_gate_bf16_rounding`（路由核先舍 BF16 再乘）也全部逐位相同。
   - 对 FP64 的 rel_l2：DEFAULT 是 1.5e-4–3.1e-4，HIGHEST 是 8.0e-8–8.3e-8。
2. **路由 GEMM 用 golden 输入**：
   - TPU logits 对 GPU rel_l2 3.8e-6–4.0e-6；
   - 对 FP64：TPU 4.8e-8–5.7e-8、GPU 3.8e-6–4.0e-6。
3. **RMSNorm**：
   - 对 FP64 正确舍入 99.999%，与 golden 逐位 99.9987%；
   - `single_rounding` 开 / 关逐位相同；
   - 代码上是舍两次，XLA 的 excess precision 把中间那次省掉了。
4. **门控激活**：
   - 默认写法对 FP64 正确舍入 99.98%（`activations_in_float32`（FP32 算完舍一次）99.99%）；
   - 与 GPU 写法只有 73.7% 逐位、rel_l2 2.8e-3；
   - 按算子舍入（`reduce_precision`）与 GPU 写法 100% 逐位。
5. **silu 单独**：
   - v5p 上 BF16 的 `jax.nn.silu` 对 FP64 正确舍入 99.98%，即内部按 FP32 算。
   - 推断：v5p 没有 BF16 的向量单元，XLA 升精度计算。
   - 另一次独立测量在 v7x 上测到 34%–42% 元素不是正确舍入。

### 由此定下的开关

`hy3-4layers.yml` 的三个开关都打开：

- `routed_gate_bf16_rounding`：v5p 上是无操作，打开是为了不依赖后端；
- `rmsnorm_single_rounding`：前向无操作，反向更接近 GPU（见下表 `in.grad`）；
- `mlp_activation_round_per_op`：与 GPU 写法逐位相同。

`activations_in_float32` 在 v5p 上和默认一样只舍一次，不用。

## L2 结论

- **第一个超过 BF16 末位量级的张量**（rel_l2 > 2⁻⁸，判据见 `step1_compare.py`）：
  - 第 0 层是 `mlp.fc2.out`（4.17e-3，开按算子舍入后 4.08e-3，降到判据以下）；
  - 第 1–3 层都是 `attn.core.out`（4.3e-3 / 5.3e-3 / 6.2e-3）。
  - 第 0 层的 `attn.core.out` 只有 1.6e-3：同样的 kernel，越往后的层差越大。
  - 推断和这些层的注意力分布有关，未查。这里是对 golden 的差；现行按 FP64 R 判，四层都过（1.06 / 1.01 / 1.01 / 1.09，`phase2-l2-replay-rratio-on`）。
- **路由翻转来自上游，不是路由 GEMM 本身**：
  - L1 用 golden 的路由输入时，翻转为 0；
  - L2 用 TPU 自己算出的 `mlp_norm.out`（对 golden 3.4e-3–3.9e-3）时，每层翻 3.5%–5.4%（142–220 个 token）。
  - 翻的主要是近乎打平的 token：翻转 token 的分差中位数 2.7e-5–6.3e-5，全部 token 的中位数是 3.1e-4–8.0e-4。
  - 这个 `mlp_norm.out` 的差几乎全部来自注意力输出：RMSNorm 本身与 GPU 逐位。
- **三个开关的作用**：
  - 注意力与路由不受影响（所有开关下逐位相同）；
  - 按算子舍入让 `fc2.out`、`shared.out`、`out` 都变小，`shared.out` 降得最多：6.67e-3 / 5.22e-3 / 5.36e-3 → 6.51e-3 / 4.93e-3 / 5.01e-3；
  - RMSNorm 单次舍入只改反向，各层 `in.grad` 降 0.4%–1.5%；
  - `activations_in_float32`（门控激活 FP32 算完舍一次）前向与只开 RMSNorm 开关几乎一样（差在 1e-7 量级）。

## 逐层数字（各张量对 golden 的 rel_l2；`topk_idx` 是 top-8 集合一致比例）

代号：`r-l2-base` 三个开关都不开；`r-l2-norm` 开 RMSNorm 单次舍入；`r-l2-norm-perop` 再加门控激活按算子舍入；`r-l2-norm-f32` 改加门控激活 FP32 算完舍一次。

**第 0 层**

| 张量 | `r-l2-base` | `r-l2-norm` | `r-l2-norm-perop` | `r-l2-norm-f32` |
|---|---|---|---|---|
| `attn.qkv.out` | 2.357e-03 | 2.357e-03 | 2.357e-03 | 2.357e-03 |
| `attn.q_norm.out` | 2.578e-03 | 2.578e-03 | 2.578e-03 | 2.578e-03 |
| `attn.k_norm.out` | 2.830e-03 | 2.830e-03 | 2.830e-03 | 2.830e-03 |
| `attn.core.out` | 1.620e-03 | 1.620e-03 | 1.620e-03 | 1.620e-03 |
| `attn.proj.out` | 1.110e-03 | 1.110e-03 | 1.110e-03 | 1.110e-03 |
| `mlp_norm.out` | 9.843e-04 | 9.843e-04 | 9.843e-04 | 9.843e-04 |
| `mlp.fc1.out` | 3.225e-03 | 3.225e-03 | 3.225e-03 | 3.225e-03 |
| `mlp.fc2.out` | 4.174e-03 | 4.174e-03 | 4.078e-03 | 4.174e-03 |
| `mlp.out` | 4.174e-03 | 4.174e-03 | 4.078e-03 | 4.174e-03 |
| `out` | 1.679e-03 | 1.679e-03 | 1.654e-03 | 1.679e-03 |
| `in.grad` | 2.090e-03 | 2.059e-03 | 2.020e-03 | 2.036e-03 |

**第 1 层**

| 张量 | `r-l2-base` | `r-l2-norm` | `r-l2-norm-perop` | `r-l2-norm-f32` |
|---|---|---|---|---|
| `attn.qkv.out` | 2.359e-03 | 2.359e-03 | 2.359e-03 | 2.359e-03 |
| `attn.q_norm.out` | 2.790e-03 | 2.790e-03 | 2.790e-03 | 2.790e-03 |
| `attn.k_norm.out` | 2.615e-03 | 2.615e-03 | 2.615e-03 | 2.615e-03 |
| `attn.core.out` | 4.313e-03 | 4.313e-03 | 4.313e-03 | 4.313e-03 |
| `attn.proj.out` | 4.580e-03 | 4.580e-03 | 4.580e-03 | 4.580e-03 |
| `mlp_norm.out` | 3.857e-03 | 3.857e-03 | 3.857e-03 | 3.857e-03 |
| `mlp.router.logits` | 1.680e-03 | 1.680e-03 | 1.680e-03 | 1.680e-03 |
| `mlp.router.topk_idx` | 94.63%（翻 220） | 94.63%（翻 220） | 94.63%（翻 220） | 94.63%（翻 220） |
| `mlp.shared.out` | 6.671e-03 | 6.671e-03 | 6.514e-03 | 6.671e-03 |
| `mlp.out` | 1.689e-02 | 1.689e-02 | 1.684e-02 | 1.689e-02 |
| `out` | 2.260e-03 | 2.260e-03 | 2.253e-03 | 2.260e-03 |
| `in.grad` | 4.822e-03 | 4.790e-03 | 4.771e-03 | 4.773e-03 |

**第 2 层**

| 张量 | `r-l2-base` | `r-l2-norm` | `r-l2-norm-perop` | `r-l2-norm-f32` |
|---|---|---|---|---|
| `attn.qkv.out` | 2.441e-03 | 2.441e-03 | 2.441e-03 | 2.441e-03 |
| `attn.q_norm.out` | 2.882e-03 | 2.882e-03 | 2.882e-03 | 2.882e-03 |
| `attn.k_norm.out` | 2.731e-03 | 2.731e-03 | 2.731e-03 | 2.731e-03 |
| `attn.core.out` | 5.275e-03 | 5.275e-03 | 5.275e-03 | 5.275e-03 |
| `attn.proj.out` | 5.414e-03 | 5.414e-03 | 5.414e-03 | 5.414e-03 |
| `mlp_norm.out` | 3.429e-03 | 3.429e-03 | 3.429e-03 | 3.429e-03 |
| `mlp.router.logits` | 1.238e-03 | 1.238e-03 | 1.238e-03 | 1.238e-03 |
| `mlp.router.topk_idx` | 96.53%（翻 142） | 96.53%（翻 142） | 96.53%（翻 142） | 96.53%（翻 142） |
| `mlp.shared.out` | 5.216e-03 | 5.216e-03 | 4.934e-03 | 5.216e-03 |
| `mlp.out` | 2.037e-02 | 2.037e-02 | 2.031e-02 | 2.037e-02 |
| `out` | 2.574e-03 | 2.574e-03 | 2.560e-03 | 2.574e-03 |
| `in.grad` | 5.775e-03 | 5.729e-03 | 5.723e-03 | 5.730e-03 |

**第 3 层**

| 张量 | `r-l2-base` | `r-l2-norm` | `r-l2-norm-perop` | `r-l2-norm-f32` |
|---|---|---|---|---|
| `attn.qkv.out` | 2.400e-03 | 2.400e-03 | 2.400e-03 | 2.400e-03 |
| `attn.q_norm.out` | 2.856e-03 | 2.856e-03 | 2.856e-03 | 2.856e-03 |
| `attn.k_norm.out` | 2.591e-03 | 2.591e-03 | 2.591e-03 | 2.591e-03 |
| `attn.core.out` | 6.168e-03 | 6.168e-03 | 6.168e-03 | 6.168e-03 |
| `attn.proj.out` | 6.260e-03 | 6.260e-03 | 6.260e-03 | 6.260e-03 |
| `mlp_norm.out` | 3.597e-03 | 3.597e-03 | 3.597e-03 | 3.597e-03 |
| `mlp.router.logits` | 1.171e-03 | 1.171e-03 | 1.171e-03 | 1.171e-03 |
| `mlp.router.topk_idx` | 96.41%（翻 147） | 96.41%（翻 147） | 96.41%（翻 147） | 96.41%（翻 147） |
| `mlp.shared.out` | 5.365e-03 | 5.365e-03 | 5.005e-03 | 5.365e-03 |
| `mlp.out` | 2.761e-02 | 2.761e-02 | 2.754e-02 | 2.761e-02 |
| `out` | 3.216e-03 | 3.216e-03 | 3.211e-03 | 3.216e-03 |
| `in.grad` | 6.921e-03 | 6.891e-03 | 6.880e-03 | 6.885e-03 |

## 关 excess precision（`r-l2-noexcess`，F1 诊断臂）

- **改了什么**：同 `r-l2-base`（三个开关都不开的逐层回放），只加 `XLA_FLAGS=--xla_allow_excess_precision=false`。
- **数据**：`phase2-replay-r-l2-noexcess.json`。

### L1：v5p 上的 BF16 非线性真的按 BF16 算了

- BF16 `silu` 对 FP64 正确舍入 99.98% → 57.9%；
- RMSNorm 默认写法与 golden 逐位 99.9987% → 74.1%；
  - 开 `single_rounding`（显式 FP32、`reduce_precision`）仍是 99.999%；
- 门控激活按算子舍入仍与 GPU 写法 100% 逐位；
- 路由 GEMM、DEFAULT 精度探针不变。

### L2：每个张量对 golden 的 rel_l2 都变大

例如：

- `qkv.out` 2.36e-3 → 3.49e-3；
- `q_norm.out` 2.58e-3 → 4.15e-3；
- `attn.core.out` 第 1–3 层 4.3e-3 / 5.3e-3 / 6.2e-3 → 5.3e-3 / 6.4e-3 / 7.6e-3；
- `mlp.out` 第 3 层 2.76e-2 → 3.13e-2；
- 路由翻转 3.5%–5.4% → 4.2%–6.9%。

> **结论**：v5p 上默认开着 excess precision 更接近 GPU，不关。
