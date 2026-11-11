# janus 报告：hunyuan3-4l / phase2-acceptance-rcc-ep8

- 被测：`t-ep8-1000`；golden：`rcc-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-b200x8-to-v5p8-4layers/catalogs/phase2-acceptance-rcc-ep8.yaml`（sha256 6b6f2d67b503…）
- 结果：pass 7、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.198e-05, signed_mean_max=1.097e-05, mare_max=1.049e-04, max_step_max=4.607e-04 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.198e-05, signed_mean_max=2.028e-05, mare_max=1.163e-04, max_step_max=4.607e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-2.248e-04, signed_mean=-2.337e-04, mare=2.458e-02, max_step=1.511e-01 | 在噪声参照内 |
| `L4.grad_norm.golden_den` | loss_series | L4 | pass | d0=-2.247e-04, signed_mean=3.486e-04, mare=2.459e-02, max_step=1.461e-01 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.241e-07, signed_mean=-5.079e-08, mare=6.330e-08, max_step=2.330e-07 | 在噪声参照内 |
| `L4.param_norm.raw` | loss_series | L4 | pass | d0=5.615e-08, signed_mean=-2.278e-08, mare=6.344e-08, max_step=2.710e-07 | 在噪声参照内 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |

数字全部来自 `results.json`，本报告不手写数字。
