# janus 报告：hunyuan3-4l / phase2-v7x-obs-gate

- 被测：`obs-v7x-t1`；LKG：`reg-v7x-t1`；golden：`reg-v7x-t1`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-b200x8-to-v7x4-4layers/catalogs/phase2-v7x-obs-gate.yaml`（sha256 188a25049d65…）
- 结果：pass 3、fail 1、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `OBS.gate3.loss` | loss_series | L4 | pass | d0=3.327e-06, signed_mean=3.130e-05, mare=1.023e-04, max_step=5.142e-04 | 在噪声参照内 |
| `OBS.gate3.grad_norm` | loss_series | L4 | pass | d0=4.055e-05, signed_mean=2.802e-03, mare=2.613e-02, max_step=4.746e-01 | 在噪声参照内 |
| `OBS.gate3.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-3.864e-08, mare=4.683e-08, max_step=2.407e-07 | 在噪声参照内 |
| `DET.hooks.noop` | determinism | L4 | fail | fields=6, first_differ=['lm_loss@1', 'grad_norm@1', 'param_norm@12', 'router.bias_std.1@1', 'router.bias_std.2@2'], tensors=0, tensors_differ=0 | 不可复现：['lm_loss@1', 'grad_norm@1', 'param_norm@12']，张量不同 0 |

数字全部来自 `results.json`，本报告不手写数字。
