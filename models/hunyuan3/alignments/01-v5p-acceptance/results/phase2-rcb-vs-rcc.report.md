# janus 报告：hunyuan3-4l / phase2-rcb-vs-rcc

- 被测：`rcb-a`；golden：`rcc-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-rcb-vs-rcc.yaml`（内容指纹 b2fabb649a08…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=0.000e+00, signed_mean_max=3.813e-05, mare_max=1.063e-04, max_step_max=5.740e-04 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=0.000e+00, signed_mean_max=4.892e-05, mare_max=1.146e-04, max_step_max=4.525e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=9.025e-06, signed_mean=1.910e-03, mare=2.770e-02, max_step=3.346e-01 | 在噪声参照内 |
| `L4.grad_norm.golden_den` | loss_series | L4 | pass | d0=9.025e-06, signed_mean=2.905e-03, mare=2.784e-02, max_step=3.240e-01 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-8.743e-08, mare=9.889e-08, max_step=3.241e-07 | 在噪声参照内 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.263e-03, mare_max=3.078e-03, max_step_max=1.453e-02, stats=3 | 在噪声参照内 |
| `L4.health` | health_signals | L4 | pass | fields=3, problems=0 | 健康 |

数字全部来自 `results.json`，本报告不手写数字。
