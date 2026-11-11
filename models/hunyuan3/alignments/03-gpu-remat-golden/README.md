[janus](../../../../README.md) › [用户指南](../../../../docs/guide/README.md) › [混元 3](../../README.md) ｜ 对齐实践 03；怎么读见指南[第 2 章 制作 GPU golden](../../../../docs/guide/02-gpu-golden.md)

# 混元 3 · 对齐实践 03：B200×8 自比，整层重算 + 路由回放（截 4 层）

**这个实践干什么**：做 GPU 生产 golden `rc`（整层重算 + 路由回放，即原版生产程序，5 份）。最终 golden `rcc` 就是在它上面只改输出头 dgrad 一处；读 01 之前先看这里。

- **GPU 自比**：和 `00` 的 `g1k-*`（第一阶段 GPU golden，不开重算，1000 步，3 份）同权重、同数据、同顺序、同超参，**只多两项：整层激活重算、路由回放**。
  - 路由回放：反向重算沿用首次前向选中的专家。
- 跑出 5 份 1000 步的 `rc-a`..`rc-e`，即 **GPU 生产程序 `rc`**。
- `00` 的 `g1k-a/b/c`（不重算）保留为**归因参照**。

**本页结构**：现状 → 结论（6 条）→ 1000 步逐步量对比 → 用法 → 证据 → 逐级大表（生成）。

> **现状（2026-11-06）**
>
> - **本页的 `rc`** 就是 GPU 生产训练程序（整层重算 + 路由回放）。本页的结论仍成立：「重算 + 路由回放」这一项本身不改 1000 步统计。
> - **迁移验收没有直接拿 `rc` 当参照**，而是用最终 golden [`rcc`](../../../../docs/glossary.md#rcc)。
>   - `rcc` = `rc` 其余不变，只把输出头 dgrad 改成分段 FP32 累加，修掉 B200 上这次长求和的系统偏差。
>   - TPU 只和事先指定的一份 [`rcc-a`](../../../../docs/glossary.md#rcc-a)（最终 golden 的第 1 份）比，噪声取 `rcc-a..j` 10 份重跑两两配对得到的 45 对比较。
> - **TPU 对 `rc` 本身的差**是已解释差异：
>   - param_norm +2.09e-2；
>   - loss 第 301–1000 步 v5p −1.37e-4、v7x 约 −1.08e-4（v7x 是推断值，见对齐报告第 3 节末）；
>   - 见[对齐报告](../../ALIGNMENT-REPORT.md)第 3 节与 [`goldens.md`](../../goldens.md)。
> - 为什么这样分工、重算设置取的什么，见 [`setup.md`](setup.md)；运行清单见 [`runs.yaml`](runs.yaml)。

> **重算设置是假设的（仍未关闭）。**
>
> - 真实的生产重算配置还没拿到，这里取 Megatron 最常见的整层、逐层重算（`full` / `uniform` / `1`）。
> - 关闭条件：拿到生产重算配置。
> - 如果不同，按它重跑 `rc-*`（GPU 生产 golden）与 `rcc-*`（最终 golden），旧的改名保留，并重判 L4。

## 结论

1. **回放生效，重算与首次前向逐位相同。**
   - 确定性的 `rv-det`（2 步）第 1 步：重算那次前向的 57 个张量全部与首次前向逐位相同（[`remat-replay`](results/remat-replay.report.md) 的 `DET.replay.vs_first`）。
     - 包括 3 个 MoE 层各自的路由 logits、选中的专家（`topk_idx`）、路由概率（`probs`）。
   - 默认设置的 `rc-a`（GPU 生产 golden 的第 1 份）同样 57 / 57 逐位（[`remat-noise`](results/remat-noise.report.md) 的 `DET.replay.vs_first`）。
   - 自检：每个 microbatch 开始、每步结束时回放队列都已取空（`rv-det` 2 步 4 个 microbatch，每个 microbatch 3 层各回放一次）。
   - 回放不改首次前向：`rv-det` 首次前向与 `00` 的 `r-remat`（重算、不回放）57 个张量逐位、第 1 步 loss 相同（`DET.replay.fwd_vs_remat`、`DET.replay.loss_vs_remat`）。
2. **回放只管下标，概率重算。**
   - Megatron 的回放只替换 top-k 下标。
   - sigmoid + expert_bias 分支里 top-k 的返回值只取下标；概率由重算出的 logits 重新取 sigmoid、在回放的下标上取值、归一化、乘 2.826。
   - 负载均衡偏置的计数只在建图的前向里累加。整层重算下就是重算那一次，回放后数的正是首次前向用的专家。
   - TPU 的 `moe_routing_maps=device` 也只存下标与排序置换（[`config-alignment.md`](../../config-alignment.md)「重算」两行）。
3. **第一阶段「重算不逐位」的原因查清了：top-k 不排序。**
   - Megatron 调 `torch.topk(..., sorted=torch.is_grad_enabled())`。
     - 整层重算下首次前向在 no_grad 里跑、不排序；
     - 重算与不重算的前向建图、排序。
   - 选中集合相同，顺序不同，top-k 权重归一化时求和顺序跟着变。
   - 实测：第一个 MoE 层输入与 logits 和不重算的基线逐位相同，`probs` 有 0.84% 的元素不同（rel_L2 4.8e-8）。
   - 复现：用基线抓下的 logits 与 expert_bias 重算 `probs`，`sorted=True` 与基线逐位、`sorted=False` 与首次前向逐位。
     - 证据：[`remat-router.rv-det-vs-d-a.json`](results/remat-router.rv-det-vs-d-a.json) 的 `topk_order_repro`；
     - 脚本：[`remat_router_check.py`](../../analysis/remat_router_check.py)。
   - 不回放的 `r-remat`：每层重算的选中集合与首次前向相同（重算从首次前向存下的层输入开始），差在顺序，所以重算的专家输出与首次前向不逐位（`00` 的 `phase1-remat`）。
   - 回放把首次前向的下标连同顺序交给重算，这一点就消失了。
4. **「重算 + 回放」改首次前向，不改 1000 步的统计。**
   - 首次前向本身和不重算不同（top-k 不排序），往后各层连带：
     - 第 2、3 层分别有 20 / 98 个 token（共 4096）选中集合与基线不同；
     - 第 1 步 loss `11.465764999389648`，不重算是 `11.4657564163208`（相对差 7.5e-7）。
   - 1000 步上 `rc-a` 对 `g1k-a/b/c`（第一阶段 GPU golden，3 份；[`remat-vs-g1k`](results/remat-vs-g1k.report.md)，噪声参照只有 `g1k-a/b/c` 3 份两两 3 对）：
     - loss 在噪声参照内；
     - `param_norm`、`grad_norm` 介于 1–2 倍噪声或参照不足。
   - 看 5 份 `rc` 各自对 3 份 `g1k` 的 15 对（下表）：
     - 后段（301–1000 步）loss 带符号均值 −4.2e-5 ～ +6.2e-5，落在 `rc` 自己两两 10 对的 −6.6e-5 ～ +6.0e-5 里；
     - 第 1000 步 `param_norm` 差 −4.9e-4 ～ +9.9e-4，`rc` 两两是 −4.9e-4 ～ +6.6e-4、`g1k` 两两 −8.2e-4 ～ −3.3e-4。
   - 重算 + 回放在这些逐步量上看不出系统偏移。
5. **`rc` 的噪声底和 `g1k` 同量级。**
   - 5 份重跑两两配对得到 10 对比较：后段 loss 带符号均值 |≤ 6.6e-5|，逐步标准差 1.29e-4 ～ 1.41e-4。
     - 对照 `g1k` 3 份两两 3 对：|≤ 3.8e-5|，1.33e-4 ～ 1.34e-4。
   - 第 1 步 46 个激活、首步 loss 逐位相同（默认设置下前向确定）。
   - `rc-a` 对 `rc-b..e` 的 L4 在噪声内（参照是 `rc-b..e` 4 份两两的 6 对）；只有路由偏置标准差的最大单步是 1.04 倍、待定（下面大表）。
   - `rc` 的 |带符号均值| 上限比 `g1k` 大，推断主要是参照对数多了（3 对 → 10 对；上限取各对最大值，对数越多越宽），不是重算更抖。
6. **TPU `t-a` 对 `rc` 的偏移没有变小。**
   - 历史：`t-a`（TPU v5p 早期 1000 步运行，不开精度开关，第 1 步全量抓数）是当时的 TPU 配置，比的是 `rc`。换成最终 golden `rcc` 后，TPU 当前对齐配置的 L4 全部在噪声内，最差 r 0.85，见[对齐报告](../../ALIGNMENT-REPORT.md)第 3 节。
   - 后段 loss 带符号均值：对 5 份 `rc` 是 −1.24e-4 ～ −1.90e-4（均值 −1.45e-4），对 3 份 `g1k` 是 −1.28e-4 ～ −1.66e-4（均值 −1.53e-4）。
   - 第 1000 步 `param_norm` 多涨：对 `rc` +2.01e-2 ～ +2.08e-2，对 `g1k` +2.03e-2 ～ +2.11e-2。
   - 第 3 层路由偏置标准差：对 `rc` −1.1e-2 ～ −1.4e-2，对 `g1k` −1.3e-2 ～ −1.4e-2。
   - **偏移和 `param_norm` 漂移都不来自「GPU 不重算、TPU 重算」这一差别。**
   - 后来案例 01 把来源定到 GPU 生产路径上输出头 dgrad 的长求和（[`../01-v5p-acceptance/cases/01-loss-offset-param-norm.md`](../01-v5p-acceptance/cases/01-loss-offset-param-norm.md) 第 0 节）：
     - 张量核累加截断，加上 cuBLAS FP32 一次长累加的偏差；
     - 偏差台账 #1、#2。

### 1000 步逐步量对比（后段 = 第 301–1000 步，相对差的分母是后一个）

代号：`rc` 是 GPU 生产 golden（整层重算加路由回放，5 份）；`g1k` 是第一阶段 GPU golden（不开重算，3 份）；`t-a` 是 TPU v5p 早期 1000 步运行。

| 对比 | 对数 | loss 后段带符号均值 | `param_norm` 后段相对差 | 第 1000 步 `param_norm` 差 | 第 3 层偏置标准差后段相对差 |
|---|---|---|---|---|---|
| `rc` 两两（噪声） | 10 | −6.6e-5 ～ +6.0e-5 | −1.4e-7 ～ +1.3e-7 | −4.9e-4 ～ +6.6e-4 | −1.8e-3 ～ +2.3e-3 |
| `g1k` 两两（噪声） | 3 | −3.8e-5 ～ +3.8e-5 | −1.2e-7 ～ −5.4e-8 | −8.2e-4 ～ −3.3e-4 | −1.3e-4 ～ +4.1e-4 |
| `rc` − `g1k` | 15 | −4.2e-5 ～ +6.2e-5 | −8.0e-8 ～ +1.8e-7 | −4.9e-4 ～ +9.9e-4 | −2.4e-3 ～ +3.2e-4 |
| `t-a` − `g1k` | 3 | −1.66e-4 ～ −1.28e-4 | +3.9e-6 ～ +4.0e-6 | +2.03e-2 ～ +2.11e-2 | −1.36e-2 ～ −1.32e-2 |
| `t-a` − `rc` | 5 | −1.90e-4 ～ −1.24e-4 | +3.8e-6 ～ +4.0e-6 | +2.01e-2 ～ +2.08e-2 | −1.36e-2 ～ −1.13e-2 |

**数据来源**

- 由 [`pair_summary.py`](../../analysis/pair_summary.py) 从 `results/series/` 与 `00`、`01` 的逐步原值算出。
  - 原始输出（含每一对的值）在 [`results/remat-pair-summary.json`](results/remat-pair-summary.json)；
  - 复现命令写在脚本的文件头。
- 分段（2–100 / 101–300 / 301–600 / 601–1000 步）与分布（标准差、被测更低的步数占比、自相关）见 [`offset_attribution.py`](../../analysis/offset_attribution.py) 的输出：
  - `results/remat-offset.rc-noise.json`（`rc-a` 对 `rc-b..e`，含 b..e 两两）；
  - `results/remat-offset.rc-{a..e}-vs-g1k.json`；
  - `results/remat-offset.t-a-vs-rc.json`。

**图**（`janus curves compare --prefix <前缀> --fields lm_loss param_norm` 画，被测 / golden 见文件名）

- `rc-a` 对 `g1k`：[loss](figures/remat-vs-g1k.rc-a-lm_loss-pairdiff.png)、[`param_norm`](figures/remat-vs-g1k.rc-a-param_norm-pairdiff.png)
- `t-a` 对 5 份 `rc`：[loss](figures/case01-t-a-vs-rc-lm_loss-pairdiff.png)、[`param_norm`](figures/case01-t-a-vs-rc-param_norm-pairdiff.png)

**一处弱信号，记下不下结论**：第 2 层路由偏置均值。

- `rc` − `g1k`（`rc-a..e` 5 份 × `g1k-a/b/c` 3 份交叉配对得 15 对）后段相对差全为负（−1.5e-2 ～ −2.8e-2）。
- `rc` 两两 −7.2e-3 ～ +5.6e-3、`g1k` 两两 +1.1e-3 ～ +5.9e-3。
- 偏置均值本身接近 0，相对差放大得厉害；loss、`param_norm` 都看不出对应的偏移。

## 用法

```bash
run.sh rc-x 1000 --load /data/hy3-4l-mcore --data /data/ds/fw_hy3_text_document \
    --recompute --routing-replay --dump-steps 1 --full --janus-run <根>/rc-x --side golden-rc-x
```

- `--routing-replay` 要求 `--recompute`。
- 开了以后 `meta.json` 的 `recompute` 记下四个值：`recompute_granularity`、`recompute_method`、`recompute_num_layers`、`moe_enable_routing_replay`。
- 训练结束 rank 0 打印 `[janus] routing replay {...}`（步数、microbatch 数、回放次数）。
- 做法见 [`../../frameworks/megatron/routing_replay.py`](../../frameworks/megatron/routing_replay.py)，不改 Megatron 源码：
  - 模型前向开始设 RECORD；
  - 结束（loss 已出、反向未开始）把记下的下标交给各层并设 REPLAY_BACKWARD；
  - Megatron 的训练循环不会自己切这两个动作（只有推理引擎会设 RECORD）。

## 证据

代号：`rv-det` 是整层重算 + 路由回放的确定性 2 步运行（回放验证）；`r-remat` 是 `00` 的整层重算、不回放的确定性 3 步运行；`d-a` 是第一阶段确定性设置、不重算的 100 步运行；`rc-a..e` 是 GPU 生产 golden 的 5 份；`g1k-a/b/c` 是第一阶段 GPU golden 的 3 份。

| 结果 | 被测 / golden | 看什么 |
|---|---|---|
| [`remat-replay`](results/remat-replay.report.md) | `rv-det` / `r-remat` | 重算 = 首次前向；回放不改首次前向 |
| [`remat-replay-base`](results/remat-replay-base.report.md) | `rv-det` / `d-a` | 相对不重算改了多少（只报：34 / 57 个张量、50 / 50 个参数梯度不同） |
| [`remat-router.*.json`](results/remat-router.rv-det-vs-d-a.json) | `rv-det`、`r-remat`、`rc-a` 各对不重算的基线 | 逐层路由：重算 vs 首次前向、首次前向 vs 基线、top-k 顺序复现 |
| [`remat-noise`](results/remat-noise.report.md) | `rc-a` / `rc-b..e` | 噪声底（下面大表） |
| [`remat-vs-g1k`](results/remat-vs-g1k.report.md) | `rc-a` / `g1k-a/b/c` | 重算 + 回放改不改 1000 步轨迹 |

<!-- janus:table:begin（以下由脚本生成，不要手改） -->

## 逐级大表

### 来源

- 由 `models/<模型>/dashboard/build.py` 调 `janus/report/tables.py` 生成。
- 数据来自本文件夹 `results/` 下的 `results.json`，数字不手写。

### 两档门槛

- 含义见 `docs/reference/judging.md`「两档门槛」。
- **严格档**是验收判定：
  - calibrated 检查对噪声上限 N 判：≤ N 过，> 2N 不过，中间待定；
  - 参照不足 3 对最多待定；
  - 逐位类必须逐位相同；
  - 数值门槛（如 FP64 R 比）照目录里写的线。
  - 符号：✅ 过 ｜ 🟡 待定 ｜ ❌ 不过或出错 ｜ ⬜ 还没跑 ｜ — 不适用。
- **参考档**是同类 GPU→TPU 迁移项目的实践门槛：
  - 只并列报告、不决定验收。
  - 符号：✅ 过 ｜ ☑️ 记录了的都过，但有门槛项没记录 ｜ ❌ 不过 ｜ ❔ 所需指标未记录 ｜ — 无对应门槛。

### 说明栏怎么读

- 格式一律是「实测（指标）｜严格档 门槛 → 离门槛多远，判定｜参考档 门槛：实测（实测 / 门槛的倍数），判定」。
- calibrated 检查先写主判指标 rel_L2：
  - 没有 rel_L2 就写最差的指标；
  - 最差的不是它时，补一句最差的那个。
- L4 逐步量三个指标各占一行：
  - 各自按同一规则判；
  - 整条检查取最差，见报告。
- 首步差：严格档只报不判，参考档要判（MARE＝平均绝对相对差）。
- 进行中的项在证据栏写证据位置。
- 严格档里「N=…（k 对）」的 k 是算 N 用的噪声参照对数：
  - 一对是两份运行的一次比较，k 对不是 k 份运行；
  - 用了哪几份运行、怎么配对见该检查项目录的 `runs.noise`。

### 逐行结果

- 共 18 行。
- 严格档：✅ 14　🟡 1　— 3
- 参考档（有对应门槛的 4 行）：✅ 4

| 级 | 检查项 | 严格档 | 参考档 | 说明 | 证据 |
|---|---|---|---|---|---|
| L0 | 每 rank 每步输入指纹（回放数据）<br>`L0.data.fingerprint` | ✅ | — | 实测 全部 65,000 个单元一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [remat-noise](results/remat-noise.report.md) |
| L2 | 激活重算首次前向 = 重算<br>`DET.replay.vs_first` | ✅ | — | 实测 全部 57 个张量一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [remat-noise](results/remat-noise.report.md) |
| L3 | 全部激活逐位<br>`L3.step1.act.bitwise` | ✅ | — | 实测 全部 46 个张量一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [remat-noise](results/remat-noise.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | — | ✅ | 实测 0（首步差）｜严格档 只报，不判｜参考档 首步差 ≤ 2e-4：0（0×），过 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-lm_loss-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-lm_loss-pairdiff.png)<br>[图·mean](figures/L4.rc-a-lm_loss-mean.png) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | ✅ | ✅ | 实测 4.94e-05（带符号均值）｜严格档 N=6.83e-05（6 对）→ 0.723×N，过｜参考档 带符号均值 ≤ 2e-4：4.94e-05（0.247×），过 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-lm_loss-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-lm_loss-pairdiff.png)<br>[图·mean](figures/L4.rc-a-lm_loss-mean.png) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | ✅ | — | 实测 1.03e-04（MARE）｜严格档 N=1.18e-04（6 对）→ 0.868×N，过｜参考档 这个量无对应门槛 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-lm_loss-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-lm_loss-pairdiff.png)<br>[图·mean](figures/L4.rc-a-lm_loss-mean.png) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | ✅ | — | 实测 4.88e-04（最大单步）｜严格档 N=5.92e-04（6 对）→ 0.824×N，过｜参考档 这个量无对应门槛 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-lm_loss-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-lm_loss-pairdiff.png)<br>[图·mean](figures/L4.rc-a-lm_loss-mean.png) |
| L4 | grad_norm<br>`L4.grad_norm` | — | — | 实测 -6.09e-06（首步差）｜严格档 只报，不判｜参考档 无对应门槛 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-grad_norm-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-grad_norm-pairdiff.png)<br>[图·mean](figures/L4.rc-a-grad_norm-mean.png) |
| L4 | grad_norm<br>`L4.grad_norm` | ✅ | — | 实测 0.00147（带符号均值）｜严格档 N=0.00339（6 对）→ 0.435×N，过｜参考档 无对应门槛 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-grad_norm-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-grad_norm-pairdiff.png)<br>[图·mean](figures/L4.rc-a-grad_norm-mean.png) |
| L4 | grad_norm<br>`L4.grad_norm` | ✅ | — | 实测 0.0249（MARE）｜严格档 N=0.0319（6 对）→ 0.782×N，过｜参考档 无对应门槛 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-grad_norm-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-grad_norm-pairdiff.png)<br>[图·mean](figures/L4.rc-a-grad_norm-mean.png) |
| L4 | grad_norm<br>`L4.grad_norm` | ✅ | — | 实测 0.155（最大单步）｜严格档 N=0.334（6 对）→ 0.464×N，过｜参考档 无对应门槛 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-grad_norm-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-grad_norm-pairdiff.png)<br>[图·mean](figures/L4.rc-a-grad_norm-mean.png) |
| L4 | param_norm<br>`L4.param_norm` | — | — | 实测 0（首步差）｜严格档 只报，不判｜参考档 这个量无对应门槛 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-param_norm-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-param_norm-pairdiff.png)<br>[图·mean](figures/L4.rc-a-param_norm-mean.png) |
| L4 | param_norm<br>`L4.param_norm` | ✅ | ✅ | 实测 5.57e-08（带符号均值）｜严格档 N=1.23e-07（6 对）→ 0.454×N，过｜参考档 带符号均值 ≤ 5e-6：5.57e-08（0.0111×），过 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-param_norm-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-param_norm-pairdiff.png)<br>[图·mean](figures/L4.rc-a-param_norm-mean.png) |
| L4 | param_norm<br>`L4.param_norm` | ✅ | — | 实测 7.15e-08（MARE）｜严格档 N=1.24e-07（6 对）→ 0.578×N，过｜参考档 这个量无对应门槛 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-param_norm-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-param_norm-pairdiff.png)<br>[图·mean](figures/L4.rc-a-param_norm-mean.png) |
| L4 | param_norm<br>`L4.param_norm` | ✅ | ✅ | 实测 2.16e-07（最大单步）｜严格档 N=3.24e-07（6 对）→ 0.667×N，过｜参考档 最大单步 ≤ 1.25e-5：2.16e-07（0.0173×），过 | [remat-noise](results/remat-noise.report.md)<br>[图·overlay](figures/L4.rc-a-param_norm-overlay.png)<br>[图·pairdiff](figures/L4.rc-a-param_norm-pairdiff.png)<br>[图·mean](figures/L4.rc-a-param_norm-mean.png) |
| L4 | 路由偏置 / 负载曲线<br>`L4.router.bias_std` | ✅ | — | 实测 0.00151（带符号均值）｜严格档 N=0.00244（6 对）→ 0.62×N，过｜参考档 无对应门槛 | [remat-noise](results/remat-noise.report.md) |
| L4 | 路由偏置 / 负载曲线<br>`L4.router.bias_std` | ✅ | — | 实测 0.00265（MARE）｜严格档 N=0.00311（6 对）→ 0.851×N，过｜参考档 无对应门槛 | [remat-noise](results/remat-noise.report.md) |
| L4 | 路由偏置 / 负载曲线<br>`L4.router.bias_std` | 🟡 | — | 实测 0.0139（最大单步）｜严格档 N=0.0133（6 对）→ 1.044×N，待定｜参考档 无对应门槛 | [remat-noise](results/remat-noise.report.md) |

<!-- janus:table:end -->
