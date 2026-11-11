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
- 梯度裁剪系数由该步 `grad_norm` 与 `clip_grad` 算出。

**比什么**：**更新量**（更新后 − 更新前）而不是更新后的参数——参数比更新量大三个数量级，比参数会把更新错藏进舍入里；另比两个动量。

**判据**：

- 学习率不大于 0 → `error`（预热第一步参数不动，验证不了）。
- 其余按 `tensor_tolerance` 判（通常是数值门槛，`source` 写明来自 FP32 存储舍入地板的实测）。

**能抓到什么**：

- 超参与配置不符（例：配置写 0.1、调度器实际用 0.033 的权重衰减）；
- 进优化器的梯度与抓下的梯度不一致（梯度在别处被缩放）；
- 动量 dtype 被降。
