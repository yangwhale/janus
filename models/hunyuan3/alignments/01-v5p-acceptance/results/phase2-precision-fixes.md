# GA = 2 与精度开关：30 步对比（v5p 对 golden `g1k-a/b/c`）

## 怎么跑的

- **只差列出的开关**：每个运行其余同 `hy3-4layers.yml`，全部 `--no-hooks --dump-steps none`。
- **数据在哪**：
  - 逐步数字：`series/<运行>.metrics.jsonl`；
  - janus 检查结果：`phase2-<运行>.{results.json,report.md}`；
  - 检查项目录：`../catalogs/phase2-<运行>.yaml`，和 `t-loss30`（TPU 30 步冒烟，只看 loss）同一组检查。
- **表里的「对 g1k-a」**：相对差 (TPU − g1k-a) / g1k-a。带符号均值和 MARE 取第 2–30 步。
  - `g1k-a` 是第一阶段 GPU golden 的第 1 份（不开重算，1000 步）。

## 结果

| 运行 | 和上一行比改了什么 | 第 1 步 loss | loss 带符号均值 | loss MARE | grad_norm 第 1 步 | grad_norm MARE | janus `L4.loss` |
|---|---|---|---|---|---|---|---|
| `t-loss30` | 基线：`per_device_batch_size=2`、GA 1，三个开关都不开 | +2.50e-6 | +7.56e-6 | 1.89e-5 | +3.09e-4 | 4.79e-4 | fail |
| `t-ga2-30` | `per_device_batch_size=1` + GA 2 | +2.50e-6 | +1.23e-5 | 2.07e-5 | +1.94e-4 | 4.78e-4 | fail |
| `t-fix-perop-30` | + 三个开关（见注 1） | +7.49e-6 | +1.58e-5 | 2.12e-5 | +1.55e-4 | 3.56e-4 | fail |
| `t-fix-f32-30` | 同上，门控激活改用 `activations_in_float32`（FP32 算完舍一次） | +4.08e-6 | +1.12e-5 | 1.85e-5 | +1.21e-4 | 3.77e-4 | fail |
| `t-ga2-noexcess-30` | `t-ga2-30` + 关 excess precision（见注 2） | −1.16e-5 | +3.78e-5 | 4.46e-5 | −8.08e-4 | 6.57e-4 | fail |

1. 三个开关：`routed_gate_bf16_rounding`（路由核先舍 BF16 再乘）、`rmsnorm_single_rounding`（RMSNorm 在 FP32 里算完只舍一次）、`mlp_activation_round_per_op`（门控激活按 GPU 的两个算子分别舍入）。
2. 即 `XLA_FLAGS=--xla_allow_excess_precision=false`。

**GPU 自己的噪声参照**（三份 golden 两两，`phase2-t-loss30`）：

- loss 带符号均值上限 5.72e-6；
- MARE 8.16e-6。

## 判读

- **只改梯度累积没有变好**：
  - `t-ga2-30` 的输入指纹逐个 microbatch 逐位相同（`L0.data.fingerprint` pass）；
  - 但 loss MARE 2.07e-5，和基线 1.89e-5 是同一量级。
- **三个精度开关在 30 步上也看不出明显变好**：
  - loss 的 MARE 落在 1.85e-5 – 2.12e-5，和基线同量级；
  - grad_norm 的 MARE 小了约 25%（4.8e-4 → 3.6e-4）；
  - 第 1 步 grad_norm 的差也变小（3.1e-4 → 1.2e-4–1.6e-4）。
  - 按约定，「明显变好」才跑 1000 步，所以这一组没有单独跑 1000 步。
  - 开关已写进生产 yml，后面的 1000 步运行（`t-rmap-1000`，TPU 生产配置 1000 步）就是带着它们跑的。
- **关 excess precision 变差**：
  - loss MARE 翻倍（4.46e-5）；
  - 逐层隔离回放里每个张量对 GPU 都变差 10%–60%（`phase2-l1-l2-replay.md`「关 excess precision」）；
  - v5p 默认开着的结果更接近 GPU，不用这个开关。
- **30 步看不出系统偏差**：
  - 30 步的 loss 差由早期的抖动主导；
  - 1000 步上的系统偏差 30 步看不出来，定责见 `../cases/01-loss-offset-param-norm.md`。
