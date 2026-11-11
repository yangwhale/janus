# janus 报告：hunyuan3-4l / phase2-v7x-acceptance-rcb

- 被测：`t-v7x-on-1000`；golden：`rcb-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-v7x-acceptance/catalogs/phase2-v7x-acceptance-rcb.yaml`（内容指纹 879be794c6f1…）
- 结果：pass 5、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.747e-05, signed_mean_max=1.304e-05, mare_max=1.164e-04, max_step_max=6.596e-04 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.747e-05, signed_mean_max=1.431e-05, mare_max=1.240e-04, max_step_max=5.168e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-2.461e-04, signed_mean=-4.045e-04, mare=3.587e-02, max_step=3.960e-01 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.260e-07, signed_mean=-2.718e-09, mare=6.570e-08, max_step=2.522e-07 | 在噪声参照内 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.345e-03, mare_max=2.625e-03, max_step_max=1.111e-02, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
