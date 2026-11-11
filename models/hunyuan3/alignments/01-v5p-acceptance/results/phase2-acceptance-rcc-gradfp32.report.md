# janus 报告：hunyuan3-4l / phase2-acceptance-rcc-gradfp32

- 被测：`t-rmap-on-g32-1000`；golden：`rcc-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-acceptance-rcc-gradfp32.yaml`（内容指纹 39cc642eb722…）
- 结果：pass 7、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.031e-05, signed_mean_max=8.036e-05, mare_max=1.205e-04, max_step_max=5.315e-04 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.031e-05, signed_mean_max=9.763e-05, mare_max=1.357e-04, max_step_max=5.315e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-3.599e-05, signed_mean=2.829e-03, mare=2.597e-02, max_step=1.874e-01 | 在噪声参照内 |
| `L4.grad_norm.golden_den` | loss_series | L4 | pass | d0=-3.599e-05, signed_mean=3.523e-03, mare=2.611e-02, max_step=2.067e-01 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.309e-07, signed_mean=-9.422e-08, mare=9.778e-08, max_step=2.907e-07 | 在噪声参照内 |
| `L4.param_norm.raw` | loss_series | L4 | pass | d0=5.615e-08, signed_mean=-5.716e-08, mare=8.299e-08, max_step=3.048e-07 | 在噪声参照内 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |

数字全部来自 `results.json`，本报告不手写数字。
