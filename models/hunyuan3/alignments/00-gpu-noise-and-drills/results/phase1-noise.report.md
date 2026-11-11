# janus 报告：hunyuan3-4l / phase1-noise

- 被测：`rd-golden-3`；golden：`rd-golden-1`, `rd-golden-2`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-noise.yaml`（sha256 28696f4f1397…）
- 结果：pass 5、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.step1000.act`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=0.000e+00, signed_mean_max=4.990e-05, mare_max=1.050e-04, max_step_max=5.481e-04 | 在噪声参照内 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |
| `L3.step1.act` | tensor_tolerance | L3 | pass | rel_l2_max=0.000e+00, cosine_min=1.000e+00, max_abs_max=0.000e+00, tensors=14 | 在噪声参照内 |
| `L3.step1.ulp` | ulp_distribution | L3 | pass | zero_frac_min=1.000e+00, p999_max=0.000e+00, max_max=0, cross_zero=0 | 诊断：只报分布，不判 |
| `L4.step1000.act` | tensor_tolerance | L4 | inconclusive | rel_l2_max=1.500e-01, cosine_min=9.888e-01, max_abs_max=1.479e+00, tensors=14 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.step1000.router` | discrete_agreement | L4 | pass | set_frac_min=3.413e-01, order_frac_min=3.413e-01 | set_frac 最低 0.341309 |

数字全部来自 `results.json`，本报告不手写数字。
