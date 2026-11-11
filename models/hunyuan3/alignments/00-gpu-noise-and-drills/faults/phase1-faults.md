# 第一阶段 · 故障注入矩阵

## 怎么做的

- **基线**：确定性基线 `d-a` 上逐个埋错。
  - 检查项目录：`models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/phase1-faults.yaml`；
  - 跑法：`janus run --test <运行>`；
  - 本页的表由 `models/hunyuan3/analysis/fault_matrix.py` 生成。
- **训练参数**：每个故障运行的参数见 `../../../frameworks/megatron/README.md`「第一阶段每次运行的参数」。
- **已从零复现**：`f-weight2`（第 2 层权重 ×(1+1e-2)）、`f-lr`（学习率差 1 个 FP32 ULP）两列逐项相同（`../results/repro-audit.md`）。

## 读表时注意

- **`f-gscale` 那列的 E 和 F 是步数不够，不是检出**：
  - `f-gscale`（第 1 层 qkv 梯度在抓数之前 ×1/8）；
  - 它只跑了 3 步，而目录的窗口是第 1–30 步；
  - `L4.*` 报 error（窗口内缺步）；
  - `L0.data.fingerprint` 报 fail，消息是「token 不同」，实际是第 4–30 步的记录缺失；
  - 现行代码对缺整步报 error 并列出缺的步，表里是当时的结果；
  - 它要验的 `L3.step1.grads.scale`、`L2.layer1.grads` 都在第 1 步。
- **`L0.data.fingerprint` 除了 token 指纹还比首步 loss**：
  - `f-weight2`（权重 ×(1+1e-2)）、`f-router`（路由改在 BF16 算）、`f-drop`（开 dropout 0.1）的输入 token 没变，它报警是因为首步 loss 不同（消息是「首步 loss 不同」）；
  - 真正的数据错位（`f-data`）是 token 指纹不同。

## 每种故障埋了什么

| 故障 | 埋了什么 | 报警的检查数 | 期望抓到而没抓到 |
|---|---|---|---|
| `f-weight` | 第 2 层 linear_proj 的 FP32 主权重 ×(1+1e-3)（小于 BF16 半格，BF16 权重逐位不变） | 6 / 32 | 无 |
| `f-weight2` | 第 2 层 linear_proj 权重 ×(1+1e-2)（大于 BF16 一格，模拟权重转换错） | 25 / 32 | 无 |
| `f-data` | rank 0 输入 token 循环错位一格（模拟数据错位） | 25 / 32 | 无 |
| `f-router` | 路由改在 BF16 算（模拟路由精度设置错） | 25 / 32 | 无 |
| `f-lr` | 学习率差 1 个 FP32 ULP | 7 / 32 | 无 |
| `f-drop` | 开 dropout 0.1（模拟非数值配置没对齐） | 27 / 32 | 无 |
| `f-grad` | 第 1 层 linear_qkv 梯度在抓数之后、优化器之前 ×1/8（抓到的梯度没被缩放） | 6 / 32 | 无 |
| `f-gscale` | 第 1 层 linear_qkv 梯度在抓数之前 ×1/8（抓到的就是缩放后的梯度） | 5 / 32 | 无 |
| `f-bias` | expert_bias 更新率置 0（模拟路由偏置不更新） | 6 / 32 | 无 |

## 检查 × 故障矩阵

代号（每列一种埋错，详见上一节）：`f-weight` 主权重 ×(1+1e-3)；`f-weight2` 权重 ×(1+1e-2)；`f-data` 输入 token 错位一格；`f-router` 路由改在 BF16 算；`f-lr` 学习率差 1 个 FP32 ULP；`f-drop` 开 dropout 0.1；`f-grad` 梯度在抓数之后缩小 8 倍；`f-gscale` 梯度在抓数之前缩小 8 倍；`f-bias` 路由偏置不更新。

| 检查 | `f-weight` | `f-weight2` | `f-data` | `f-router` | `f-lr` | `f-drop` | `f-grad` | `f-gscale` | `f-bias` |
|---|---|---|---|---|---|---|---|---|---|
| `L0.config.diff` | · | · | · | **F** | **F** | **F** | · | · | **F** |
| `L0.data.fingerprint` | · | **F** | **F** | **F** | · | **F** | · | **F** | · |
| `L0.weights.init` | · | **F** | · | · | · | · | · | · | · |
| `L0.weights.loaded` | · | **F** | · | · | · | · | · | · | · |
| `L0.main_params.loaded` | **F** | **F** | · | · | · | · | · | · | · |
| `L1.optim.adamw.replay` | · | · | · | · | · | · | **F** | · | · |
| `L2.layer0.fwd` | · | · | **F** | · | · | **F** | · | · | · |
| `L2.layer1.fwd` | · | · | **F** | **F** | · | **F** | · | · | · |
| `L2.layer2.fwd` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L2.layer3.fwd` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L2.layer1.router` | · | · | **F** | **F** | · | **F** | · | · | · |
| `L2.layer2.router` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L2.layer3.router` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L2.layer0.bwd` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L2.layer1.bwd` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L2.layer2.bwd` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L2.layer3.bwd` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L2.layer0.grads` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L2.layer1.grads` | · | **F** | **F** | **F** | · | **F** | · | **F** | · |
| `L2.layer2.grads` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L2.layer3.grads` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L3.step1.grads.scale` | · | **F** | **F** | **F** | · | **F** | · | **F** | · |
| `L3.step1.embed` | · | · | **F** | · | · | **F** | · | · | · |
| `L3.step1.loss.per_token` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L3.step1.logits` | · | **F** | **F** | **F** | · | **F** | · | · | · |
| `L3.step1.updated` | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | · |
| `L3.step1.bias_after` | · | **F** | · | **F** | · | **F** | · | · | **F** |
| `L4.loss` | **F** | **F** | **F** | **F** | **F** | **F** | **F** | E | **F** |
| `L4.lr` | · | · | · | · | **F** | · | · | E | · |
| `L4.grad_norm` | **F** | **F** | **F** | **F** | **F** | **F** | **F** | E | **F** |
| `L4.router.bias_std` | **F** | **F** | **F** | **F** | **F** | **F** | **F** | E | **F** |
| `L4.determinism` | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** |

**F** = fail（报警），· = pass，E = error，✗ = 期望报警但没报。
