# janus 报告：hunyuan3-4l / phase2-l2-replay-bwd-on

- 被测：`r-l2-on`；golden：`g1k-a`, `g1k-b`, `g1k-c`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-b200x8-to-v5p8-4layers/catalogs/phase2-l2-replay-bwd-on.yaml`（sha256 abb6bc85096c…）
- 结果：pass 1、fail 3、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.layer0.bwd` | tensor_tolerance | L2 | pass | rel_l2_max=2.116e-03, cosine_min=1.000e+00, norm_ratio_dev_max=3.317e-06, tensors=1 | 在噪声参照内 |
| `L2.layer1.bwd` | tensor_tolerance | L2 | fail | rel_l2_max=5.732e-03, cosine_min=1.000e+00, norm_ratio_dev_max=4.306e-05, tensors=1 | 超过噪声参照两倍 |
| `L2.layer2.bwd` | tensor_tolerance | L2 | fail | rel_l2_max=5.414e-03, cosine_min=1.000e+00, norm_ratio_dev_max=1.489e-05, tensors=1 | 超过噪声参照两倍 |
| `L2.layer3.bwd` | tensor_tolerance | L2 | fail | rel_l2_max=6.540e-03, cosine_min=1.000e+00, norm_ratio_dev_max=1.239e-05, tensors=1 | 超过噪声参照两倍 |

数字全部来自 `results.json`，本报告不手写数字。
