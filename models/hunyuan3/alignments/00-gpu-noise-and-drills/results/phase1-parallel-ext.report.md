# janus 报告：hunyuan3-4l / phase1-parallel-ext

- 被测：`p-ep4f`；golden：`d-a`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-parallel-ext.yaml`（sha256 90bad43d15e6…）
- 结果：pass 1、fail 2、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L3.parallel.updated`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.parallel.grads.bitwise` | bitwise_equal | L2 | fail | tensors=32, differ=30 | 30 / 32 个张量不同 |
| `L2.parallel.grads.ulp` | ulp_distribution | L2 | pass | zero_frac_min=9.467e-05, p999_max=7.602e+06, max_max=821035008, cross_zero=35964 | 诊断：只报分布，不判 |
| `L2.parallel.grads.tol` | tensor_tolerance | L2 | fail | rel_l2_max=1.033e-03, cosine_min=1.000e+00, norm_ratio_dev_max=3.069e-04, tensors=32 | 超过噪声参照两倍 |
| `L3.parallel.updated` | tensor_tolerance | L3 | inconclusive | rel_l2_max=6.403e-05, max_abs_max=1.909e-05, tensors=32 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
