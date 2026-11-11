# janus 报告：hunyuan3-4l / phase2-v7x-l2-replay-bwd-on-xnoise

- 被测：`r-v7x-l2-on`；golden：`g1k-a`, `g1k-b`, `g1k-c`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/02-b200x8-to-v7x4-4layers/catalogs/phase2-v7x-l2-replay-bwd-on-xnoise.yaml`（sha256 b724874a2bc3…）
- 结果：pass 1、fail 3、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.layer0.bwd` | tensor_tolerance | L2 | pass | rel_l2_max=2.096e-03, cosine_min=1.000e+00, norm_ratio_dev_max=3.116e-06, tensors=1 | 在噪声参照内 |
| `L2.layer1.bwd` | tensor_tolerance | L2 | fail | rel_l2_max=4.547e-03, cosine_min=1.000e+00, norm_ratio_dev_max=1.004e-05, tensors=1 | 超过噪声参照两倍 |
| `L2.layer2.bwd` | tensor_tolerance | L2 | fail | rel_l2_max=5.329e-03, cosine_min=1.000e+00, norm_ratio_dev_max=7.449e-06, tensors=1 | 超过噪声参照两倍 |
| `L2.layer3.bwd` | tensor_tolerance | L2 | fail | rel_l2_max=6.281e-03, cosine_min=1.000e+00, norm_ratio_dev_max=5.155e-06, tensors=1 | 超过噪声参照两倍 |

数字全部来自 `results.json`，本报告不手写数字。
