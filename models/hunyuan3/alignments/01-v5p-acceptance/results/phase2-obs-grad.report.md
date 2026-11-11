# janus 报告：hunyuan3-4l / phase2-obs-grad

- 被测：`obs-t1`；golden：`t-st1-t1-gm`, `g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-b200x8-to-v5p8-4layers/catalogs/phase2-obs-grad.yaml`（sha256 7d29d83e400b…）
- 结果：pass 0、fail 2、inconclusive 1、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`OBS.updated`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `OBS.gate3.grads` | tensor_tolerance | L3 | fail | rel_l2_max=8.951e-02, cosine_min=9.960e-01, norm_ratio_dev_max=7.358e-03, tensors=50 | 超过噪声参照两倍 |
| `OBS.updated` | tensor_tolerance | L3 | inconclusive | rel_l2_max=3.398e-05, max_abs_max=1.962e-05, tensors=50 | 介于噪声参照一到两倍之间，或参照不足 |
| `OBS.optim_state` | tensor_tolerance | L3 | fail | rel_l2_max=8.951e-02, cosine_min=9.960e-01, tensors=100 | 超过噪声参照两倍 |

数字全部来自 `results.json`，本报告不手写数字。
