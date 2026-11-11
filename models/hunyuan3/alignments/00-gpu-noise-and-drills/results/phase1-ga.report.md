# janus 报告：hunyuan3-4l / phase1-ga

- 被测：`p-mbs2`；golden：`d-a`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-ga.yaml`（sha256 07f3cd7dff1c…）
- 结果：pass 1、fail 1、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L3.ga.updated`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L3.ga.loss` | loss_series | L3 | pass | d0=-8.318e-08, signed_mean=-4.131e-06, mare=4.131e-06, max_step=6.395e-06 | 在门槛内 |
| `L3.ga.grads` | tensor_tolerance | L3 | fail | rel_l2_max=2.065e-02, cosine_min=9.998e-01, norm_ratio_dev_max=7.256e-03, tensors=50 | 超过噪声参照两倍 |
| `L3.ga.updated` | tensor_tolerance | L3 | inconclusive | rel_l2_max=3.675e-05, max_abs_max=1.976e-05, tensors=50 | 介于噪声参照一到两倍之间，或参照不足 |

数字全部来自 `results.json`，本报告不手写数字。
