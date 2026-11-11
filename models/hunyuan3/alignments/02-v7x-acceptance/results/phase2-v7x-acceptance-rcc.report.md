# janus 报告：hunyuan3-4l / phase2-v7x-acceptance-rcc

- 被测：`t-v7x-on-1000`；golden：`rcc-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-b200x8-to-v7x4-4layers/catalogs/phase2-v7x-acceptance-rcc.yaml`（sha256 d86687dab4e5…）
- 结果：pass 7、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.747e-05, signed_mean_max=5.116e-05, mare_max=1.126e-04, max_step_max=4.596e-04 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.747e-05, signed_mean_max=6.322e-05, mare_max=1.244e-04, max_step_max=4.596e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-2.371e-04, signed_mean=1.500e-03, mare=2.812e-02, max_step=2.655e-01 | 在噪声参照内 |
| `L4.grad_norm.golden_den` | loss_series | L4 | pass | d0=-2.370e-04, signed_mean=2.506e-03, mare=2.831e-02, max_step=3.061e-01 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.260e-07, signed_mean=-9.015e-08, mare=9.480e-08, max_step=2.583e-07 | 在噪声参照内 |
| `L4.param_norm.raw` | loss_series | L4 | pass | d0=-1.043e-07, signed_mean=-5.539e-08, mare=7.451e-08, max_step=2.790e-07 | 在噪声参照内 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |

数字全部来自 `results.json`，本报告不手写数字。
