# janus 报告：hunyuan3-4l / phase2-acceptance-rcb-on

- 被测：`t-rmap-on-1000`；golden：`rcb-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-acceptance-rcb-on.yaml`（内容指纹 6f3232f124c9…）
- 结果：pass 6、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.031e-05, signed_mean_max=3.655e-07, mare_max=1.089e-04, max_step_max=6.993e-04 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.031e-05, signed_mean_max=1.550e-05, mare_max=1.144e-04, max_step_max=5.147e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-2.328e-04, signed_mean=-7.872e-04, mare=2.844e-02, max_step=3.524e-01 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.309e-07, signed_mean=1.034e-07, mare=1.227e-07, max_step=3.500e-07 | 在噪声参照内 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.120e-03, mare_max=2.760e-03, max_step_max=1.323e-02, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
