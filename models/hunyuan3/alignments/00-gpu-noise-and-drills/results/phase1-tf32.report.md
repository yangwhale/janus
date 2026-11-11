# janus 报告：hunyuan3-4l / phase1-tf32

- 被测：`g61-notf32`；golden：`d-a`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-tf32.yaml`（sha256 5020c4662c48…）
- 结果：pass 4、fail 3、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`TF32.step1.bwd.tol`、`TF32.step1.updated`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `TF32.step1.bitwise` | bitwise_equal | L3 | fail | tensors=425, differ=234 | 234 / 425 个张量不同 |
| `TF32.step1.coverage` | coverage | L3 | pass |  | 全部比到 |
| `TF32.step1.fwd.bitwise` | bitwise_equal | L3 | pass | tensors=46, differ=0 | 全部逐位相同 |
| `TF32.step1.bwd.tol` | tensor_tolerance | L3 | inconclusive | rel_l2_max=5.657e-03, cosine_min=1.000e+00, tensors=51 | 介于噪声参照一到两倍之间，或参照不足 |
| `TF32.step1.grads.tol` | tensor_tolerance | L3 | fail | rel_l2_max=1.135e-03, cosine_min=1.000e+00, norm_ratio_dev_max=2.656e-04, tensors=50 | 超过噪声参照两倍 |
| `TF32.step1.updated` | tensor_tolerance | L3 | inconclusive | rel_l2_max=4.203e-05, max_abs_max=1.957e-05, tensors=50 | 介于噪声参照一到两倍之间，或参照不足 |
| `TF32.loss.exact` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.819e-06, mare=8.426e-06, max_step=1.876e-05 |  |
| `TF32.loss` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=-1.819e-06, mare=8.426e-06, max_step=1.876e-05 | 在噪声参照内 |
| `TF32.grad_norm` | loss_series | L4 | pass | d0=4.068e-05, signed_mean=1.600e-05, mare=2.454e-04, max_step=1.017e-03 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
