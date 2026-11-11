# janus 报告：hunyuan3-4l / phase1-remat

- 被测：`r-remat`；golden：`d-a`
- janus 0.0.1；目录 `/tmp/jc/catalogs/hunyuan3-4l/phase1-remat.yaml`（sha256 b6ff333fe378…）
- 结果：pass 0、fail 4、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `DET.remat.vs_first` | bitwise_equal | L2 | fail | tensors=51, differ=9 | 9 / 51 个张量不同 |
| `DET.remat.recompute_vs_baseline` | bitwise_equal | L2 | fail | tensors=51, differ=26 | 26 / 51 个张量不同 |
| `DET.remat.fwd_vs_baseline` | bitwise_equal | L2 | fail | tensors=46, differ=26 | 26 / 46 个张量不同 |
| `DET.remat.loss` | loss_series | L4 | fail | d0=7.486e-07, signed_mean=-6.303e-06, mare=6.303e-06, max_step=8.107e-06 |  |

数字全部来自 `results.json`，本报告不手写数字。
