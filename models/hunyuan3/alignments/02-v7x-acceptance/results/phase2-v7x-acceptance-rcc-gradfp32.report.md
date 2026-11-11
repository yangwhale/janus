# janus 报告：hunyuan3-4l / phase2-v7x-acceptance-rcc-gradfp32

- 被测：`t-v7x-on-g32-1000`；golden：`rcc-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-v7x-acceptance/catalogs/phase2-v7x-acceptance-rcc-gradfp32.yaml`（内容指纹 0378f41e0522…）
- 结果：pass 5、fail 0、inconclusive 2、error 0

**证据不足（inconclusive）**：`L4.grad_norm`、`L4.grad_norm.golden_den`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.747e-05, signed_mean_max=1.692e-05, mare_max=1.071e-04, max_step_max=7.516e-04 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.747e-05, signed_mean_max=8.258e-06, mare_max=1.093e-04, max_step_max=3.923e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=1.598e-05, signed_mean=-4.731e-04, mare=3.219e-02, max_step=4.817e-01 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm.golden_den` | loss_series | L4 | inconclusive | d0=1.598e-05, signed_mean=1.393e-03, mare=3.246e-02, max_step=6.346e-01 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.260e-07, signed_mean=-3.198e-08, mare=4.763e-08, max_step=2.571e-07 | 在噪声参照内 |
| `L4.param_norm.raw` | loss_series | L4 | pass | d0=-1.043e-07, signed_mean=3.726e-09, mare=5.327e-08, max_step=2.444e-07 | 在噪声参照内 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |

数字全部来自 `results.json`，本报告不手写数字。
