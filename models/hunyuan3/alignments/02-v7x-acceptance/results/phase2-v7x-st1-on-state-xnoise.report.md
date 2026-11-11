# janus 报告：hunyuan3-4l / phase2-v7x-st1-on-state-xnoise

- 被测：`t-v7x-st1n-base-gm`；golden：`g1k-a`, `g1k-b`, `g1k-c`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-b200x8-to-v7x4-4layers/catalogs/phase2-v7x-st1-on-state-xnoise.yaml`（sha256 055537cb1ee0…）
- 结果：pass 4、fail 2、inconclusive 6、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L2.layer0.grads`、`L2.layer2.grads`、`L2.layer3.grads`、`L3.step1.grads.all`、`L3.step1.grads.dist`、`L3.step1.optim_state`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L1.optim.adamw.replay` | optimizer_step | L1 | pass | rel_l2_max=2.127e-03, max_abs_max=1.162e-07, cosine_min=1.000e+00, tensors=144 | 全部在门槛内 |
| `L2.layer0.grads` | tensor_tolerance | L2 | inconclusive | rel_l2_max=2.765e-03, cosine_min=1.000e+00, norm_ratio_dev_max=1.881e-03, tensors=8 | 介于噪声参照一到两倍之间，或参照不足 |
| `L2.layer1.grads` | tensor_tolerance | L2 | fail | rel_l2_max=3.859e-02, cosine_min=9.993e-01, norm_ratio_dev_max=4.302e-03, tensors=13 | 超过噪声参照两倍 |
| `L2.layer2.grads` | tensor_tolerance | L2 | inconclusive | rel_l2_max=6.774e-02, cosine_min=9.977e-01, norm_ratio_dev_max=4.239e-03, tensors=13 | 介于噪声参照一到两倍之间，或参照不足 |
| `L2.layer3.grads` | tensor_tolerance | L2 | inconclusive | rel_l2_max=1.004e-01, cosine_min=9.950e-01, norm_ratio_dev_max=6.697e-03, tensors=13 | 介于噪声参照一到两倍之间，或参照不足 |
| `L3.step1.grads.all` | tensor_tolerance | L3 | inconclusive | rel_l2_max=1.004e-01, cosine_min=9.950e-01, norm_ratio_dev_max=6.697e-03, tensors=50 | 介于噪声参照一到两倍之间，或参照不足 |
| `L3.step1.grads.dist` | tensor_tolerance | L3 | inconclusive | std_ratio_dev_max=6.697e-03, zero_frac_diff_max=7.812e-03, rel_err_p99_max=1.677e+00, tensors=50 | 介于噪声参照一到两倍之间，或参照不足 |
| `L3.step1.updated` | tensor_tolerance | L3 | pass | rel_l2_max=5.239e-05, max_abs_max=1.957e-05, cosine_min=1.000e+00, tensors=50 | 在噪声参照内 |
| `L3.step1.optim_state` | tensor_tolerance | L3 | inconclusive | rel_l2_max=1.004e-01, cosine_min=9.950e-01, tensors=100, oot_frac_max=7.825e-01 | 介于噪声参照一到两倍之间，或参照不足 |
| `L3.step1.fake_fp32.main_param` | fake_fp32 | L3 | pass | low16_zero_frac_max=7.812e-03, tensors=50, fake=0, skipped_non_fp32=0 | 50 个 float32 张量低 16 位全 0 的比例最高 7.81e-03 |
| `L3.step1.fake_fp32.optim_state` | fake_fp32 | L3 | pass | low16_zero_frac_max=1.844e-05, tensors=100, fake=0, skipped_non_fp32=0 | 100 个 float32 张量低 16 位全 0 的比例最高 1.84e-05 |
| `L3.step1.bias_after` | tensor_tolerance | L3 | fail | rel_l2_max=1.268e-02, max_abs_max=2.000e-03, tensors=3, oot_frac_max=5.208e-03 | 超过噪声参照两倍 |

数字全部来自 `results.json`，本报告不手写数字。
