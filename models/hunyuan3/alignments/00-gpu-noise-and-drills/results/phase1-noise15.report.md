# janus 报告：hunyuan3-4l / phase1-noise15

- 被测：`s-a`；golden：`s-b`, `s-c`, `s-d`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-noise15.yaml`（sha256 0784068f8a09…）
- 结果：pass 6、fail 0、inconclusive 3、error 0

**证据不足（inconclusive）**：`L2.layer3.bwd`、`L4.loss`、`L4.router.bias_std`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.layer0.bwd` | tensor_tolerance | L2 | pass | rel_l2_max=4.285e-03, cosine_min=1.000e+00, norm_ratio_dev_max=3.913e-04, tensors=12 | 在噪声参照内 |
| `L2.layer3.bwd` | tensor_tolerance | L2 | inconclusive | rel_l2_max=1.042e-05, cosine_min=1.000e+00, norm_ratio_dev_max=6.567e-09, tensors=12 | 介于噪声参照一到两倍之间，或参照不足 |
| `L2.layer0.grads` | tensor_tolerance | L2 | pass | rel_l2_max=4.129e-04, cosine_min=1.000e+00, norm_ratio_dev_max=2.481e-04, tensors=8 | 在噪声参照内 |
| `L2.layer3.grads` | tensor_tolerance | L2 | pass | rel_l2_max=4.349e-06, cosine_min=1.000e+00, norm_ratio_dev_max=4.669e-08, tensors=13 | 在噪声参照内 |
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=0.000e+00, signed_mean_max=2.873e-05, mare_max=3.939e-05, max_step_max=2.406e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-1.251e-05, signed_mean=-1.779e-03, mare=7.662e-03, max_step=6.417e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=5.457e-10, mare=1.364e-08, max_step=1.080e-07 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=7.110e-04, mare_max=1.256e-03, max_step_max=4.975e-03, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.load_cv` | router_stats | L4 | pass | signed_mean_max=1.304e-03, mare_max=8.842e-03, max_step_max=6.118e-02, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
