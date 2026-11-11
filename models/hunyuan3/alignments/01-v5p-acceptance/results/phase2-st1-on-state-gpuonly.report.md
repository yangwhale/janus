# janus 报告：hunyuan3-4l / phase2-st1-on-state-gpuonly

- 被测：`t-st1n-base-gm`；golden：`g1k-a`, `g1k-b`, `g1k-c`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-st1-on-state-gpuonly.yaml`（内容指纹 2af837123801…）
- 结果：pass 4、fail 7、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L3.step1.updated`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L1.optim.adamw.replay` | optimizer_step | L1 | pass | rel_l2_max=2.127e-03, max_abs_max=1.148e-07, cosine_min=1.000e+00, tensors=144 | 全部在门槛内 |
| `L2.layer0.grads` | tensor_tolerance | L2 | fail | rel_l2_max=2.163e-03, cosine_min=1.000e+00, norm_ratio_dev_max=1.550e-03, tensors=8 | 超过噪声参照两倍 |
| `L2.layer1.grads` | tensor_tolerance | L2 | fail | rel_l2_max=3.839e-02, cosine_min=9.993e-01, norm_ratio_dev_max=2.210e-03, tensors=13 | 超过噪声参照两倍 |
| `L2.layer2.grads` | tensor_tolerance | L2 | fail | rel_l2_max=7.216e-02, cosine_min=9.974e-01, norm_ratio_dev_max=3.160e-03, tensors=13 | 超过噪声参照两倍 |
| `L2.layer3.grads` | tensor_tolerance | L2 | fail | rel_l2_max=1.047e-01, cosine_min=9.946e-01, norm_ratio_dev_max=8.150e-03, tensors=13 | 超过噪声参照两倍 |
| `L3.step1.grads.all` | tensor_tolerance | L3 | fail | rel_l2_max=1.047e-01, cosine_min=9.946e-01, norm_ratio_dev_max=8.150e-03, tensors=50 | 超过噪声参照两倍 |
| `L3.step1.grads.dist` | tensor_tolerance | L3 | fail | std_ratio_dev_max=8.150e-03, zero_frac_diff_max=7.255e-04, rel_err_p99_max=1.685e+00, tensors=50 | 超过噪声参照两倍 |
| `L3.step1.updated` | tensor_tolerance | L3 | inconclusive | rel_l2_max=5.280e-05, max_abs_max=1.957e-05, cosine_min=1.000e+00, tensors=50 | 介于噪声参照一到两倍之间，或参照不足 |
| `L3.step1.optim_state` | tensor_tolerance | L3 | fail | rel_l2_max=1.047e-01, cosine_min=9.946e-01, tensors=100, oot_frac_max=7.895e-01 | 超过噪声参照两倍 |
| `L3.step1.fake_fp32.main_param` | fake_fp32 | L3 | pass | low16_zero_frac_max=2.756e-03, tensors=50, fake=0, skipped_non_fp32=0 | 50 个 float32 张量低 16 位全 0 的比例最高 2.76e-03 |
| `L3.step1.fake_fp32.optim_state` | fake_fp32 | L3 | pass | low16_zero_frac_max=2.035e-05, tensors=100, fake=0, skipped_non_fp32=0 | 100 个 float32 张量低 16 位全 0 的比例最高 2.03e-05 |
| `L3.step1.bias_after` | tensor_tolerance | L3 | pass | rel_l2_max=0.000e+00, max_abs_max=0.000e+00, tensors=3, oot_frac_max=0.000e+00 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
