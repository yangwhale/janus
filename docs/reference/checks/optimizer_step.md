[janus](../../../README.md) › [用户指南](../../guide/README.md) › [检查类型规格](README.md) › `optimizer_step`

# optimizer_step

**用途**：优化器语义与状态精度。

- 用抓下的 FP32 主权重、进优化器前的梯度、**实际生效**的超参，在 FP64 里重放一步 AdamW（[`janus/replay.py`](../../../janus/replay.py)）。
- 与运行真实的更新比。

**输入**：

- 同一步的 `main_param.*`、`grad.*`（来自 `replay_from` 指定的运行，默认被测自己）。
- 被测的 `main_param_after.*`、`exp_avg.*`、`exp_avg_sq.*`。
- 学习率取 `metrics.jsonl` 的 `lr`。
- 权重衰减按优先级取：每步记录的 `wd` > 目录 `params.weight_decay` > `meta.scheduler_config.start_weight_decay` > `meta.optimizer_config.weight_decay`。
  - 一维参数（norm 权重、bias）不做权重衰减。
- 梯度裁剪系数由该步 `grad_norm` 与 `meta.optimizer_config.clip_grad` 算出：min(1, clip_grad ÷ (grad_norm + 1e-6))。`grad_norm` 必须是裁剪之前的全局范数（[`dump-format.md`](../dump-format.md#metricsjsonl)）；缺 `grad_norm` 或缺 `clip_grad`（或 `clip_grad` ≤ 0）时按不裁剪算，不报错。

**适用范围与限定**（照 [`janus/run.py`](../../../janus/run.py) 的 `_optimizer_step` 与 `janus/replay.py`）：

- 只支持 AdamW（Megatron 默认语义）。Muon 等别的优化器用不了，要写模型层脚本：混元 4（Muon）的优化器一步用 [`l3_optim_step1.py`](../../../models/hunyuan4/analysis/l3_optim_step1.py)。
- 步号与动量：
  - `select.step: k` 时动量按 0 起算，偏差修正却用步号 k，所以只对第 1 步成立；k > 1 结果必错。
  - `select.steps: "a:b"` 连续区间里，非首步用上一步抓下的 `exp_avg.*`、`exp_avg_sq.*` 接着算；区间第一步同样按 0 起算，所以区间要从第 1 步开始。
- β1、β2、eps 从 `meta.optimizer_config` 的 `adam_beta1`、`adam_beta2`、`adam_eps` 读，缺键时静默用 0.9、0.999、1e-8，不报错。接新框架时先核这三个键写进了 meta。
- 不做权重衰减的参数写死为「一维或名字以 `.bias` 结尾」（Megatron 的规则）。别的模型衰减分组不同，会被判成优化器错，要先改这条规则。

**比什么**：**更新量**（更新后 − 更新前）而不是更新后的参数——参数比更新量大三个数量级，比参数会把更新错藏进舍入里；另比两个动量。

**判据**：

- 学习率不大于 0 → `error`（预热第一步参数不动，验证不了）。
- 其余按 `tensor_tolerance` 判（通常是数值门槛，`source` 写明来自 FP32 存储舍入地板的实测）。

**能抓到什么**：

- 超参与配置不符（例：配置写 0.1、调度器实际用 0.033 的权重衰减）；
- 进优化器的梯度与抓下的梯度不一致（梯度在别处被缩放）；
- 动量 dtype 被降。
