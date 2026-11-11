# janus 报告：hunyuan3-4l / phase2-v7x-acceptance-rcc-ep8

- 被测：`t-v7x-ep8-1000`；golden：`rcc-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-b200x8-to-v7x4-4layers/catalogs/phase2-v7x-acceptance-rcc-ep8.yaml`（sha256 764bc7abd48f…）
- 结果：pass 7、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.730e-05, signed_mean_max=4.441e-05, mare_max=1.057e-04, max_step_max=4.781e-04 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.730e-05, signed_mean_max=3.658e-05, mare_max=1.128e-04, max_step_max=4.781e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-2.385e-04, signed_mean=7.792e-04, mare=2.262e-02, max_step=1.722e-01 | 在噪声参照内 |
| `L4.grad_norm.golden_den` | loss_series | L4 | pass | d0=-2.384e-04, signed_mean=1.272e-03, mare=2.265e-02, max_step=1.885e-01 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.294e-07, signed_mean=-2.346e-08, mare=4.946e-08, max_step=1.718e-07 | 在噪声参照内 |
| `L4.param_norm.raw` | loss_series | L4 | pass | d0=-1.043e-07, signed_mean=-9.327e-10, mare=5.747e-08, max_step=2.227e-07 | 在噪声参照内 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |

数字全部来自 `results.json`，本报告不手写数字。
