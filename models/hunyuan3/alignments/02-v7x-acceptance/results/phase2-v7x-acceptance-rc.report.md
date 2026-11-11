# janus 报告：hunyuan3-4l / phase2-v7x-acceptance-rc

- 被测：`t-v7x-rmap-1000-a`；golden：`rc-a`, `rc-b`, `rc-c`, `rc-d`, `rc-e`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-b200x8-to-v7x4-4layers/catalogs/phase2-v7x-acceptance-rc.yaml`（sha256 82205ed632e0…）
- 结果：pass 1、fail 2、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=1.447e-05, signed_mean_max=1.200e-04, mare_max=1.510e-04, max_step_max=6.212e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | fail | d0_max=1.447e-05, signed_mean_max=1.600e-04, mare_max=1.774e-04, max_step_max=6.212e-04 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=1.686e-04, signed_mean=-4.000e-03, mare=2.256e-02, max_step=1.484e-01 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | fail | d0=-1.043e-07, signed_mean=2.933e-06, mare=2.936e-06, max_step=6.902e-06 | 超过噪声参照两倍 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |

数字全部来自 `results.json`，本报告不手写数字。
