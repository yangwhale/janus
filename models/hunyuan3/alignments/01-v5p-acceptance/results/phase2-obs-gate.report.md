# janus 报告：hunyuan3-4l / phase2-obs-gate

- 被测：`obs-t1`；LKG：`reg-t1-lkg2-a`；golden：`reg-t1-lkg2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-b200x8-to-v5p8-4layers/catalogs/phase2-obs-gate.yaml`（sha256 d4bda0c60af9…）
- 结果：pass 3、fail 1、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `OBS.gate3.loss` | loss_series | L4 | pass | d0=1.580e-06, signed_mean=2.795e-05, mare=1.025e-04, max_step=4.567e-04 | 在噪声参照内 |
| `OBS.gate3.grad_norm` | loss_series | L4 | pass | d0=2.447e-05, signed_mean=2.991e-03, mare=2.638e-02, max_step=2.508e-01 | 在噪声参照内 |
| `OBS.gate3.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-3.598e-08, mare=4.948e-08, max_step=3.210e-07 | 在噪声参照内 |
| `DET.hooks.noop` | determinism | L4 | fail | fields=6, first_differ=['lm_loss@1', 'grad_norm@1', 'param_norm@2', 'router.bias_std.1@1', 'router.bias_std.2@5'], tensors=0, tensors_differ=0 | 不可复现：['lm_loss@1', 'grad_norm@1', 'param_norm@2']，张量不同 0 |

数字全部来自 `results.json`，本报告不手写数字。
