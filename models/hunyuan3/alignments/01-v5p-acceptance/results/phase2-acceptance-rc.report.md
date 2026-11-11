# janus 报告：hunyuan3-4l / phase2-acceptance-rc

- 被测：`t-rmap-1000`；golden：`rc-a`, `rc-b`, `rc-c`, `rc-d`, `rc-e`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-b200x8-to-v5p8-4layers/catalogs/phase2-acceptance-rc.yaml`（sha256 27f67fda39d2…）
- 结果：pass 1、fail 1、inconclusive 3、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=8.733e-06, signed_mean_max=9.135e-05, mare_max=1.391e-04, max_step_max=7.035e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=8.733e-06, signed_mean_max=1.285e-04, mare_max=1.600e-04, max_step_max=7.035e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=1.361e-04, signed_mean=-4.079e-03, mare=2.336e-02, max_step=2.509e-01 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | fail | d0=5.615e-08, signed_mean=2.933e-06, mare=2.935e-06, max_step=7.008e-06 | 超过噪声参照两倍 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |

数字全部来自 `results.json`，本报告不手写数字。
