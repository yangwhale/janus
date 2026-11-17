# janus 报告：hunyuan4-rw3 / rw3-l4

- 被测：`rfc-a`；golden：`rw1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/06-rw3/catalogs/rw3-l4.yaml`（内容指纹 90a594773a3d…）
- 结果：pass 8、fail 0、inconclusive 1、error 0

**证据不足（inconclusive）**：`L4.loss`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=6.143e-05, signed_mean_max=1.584e-06, mare_max=2.374e-05, max_step_max=1.076e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=6.143e-05, signed_mean_max=4.337e-06, mare_max=2.026e-05, max_step_max=9.338e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-5.070e-05, signed_mean=-4.750e-06, mare=5.791e-05, max_step=2.701e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-5.070e-05, signed_mean=-4.548e-06, mare=5.481e-05, max_step=2.701e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-6.687e-04, signed_mean=-4.460e-05, mare=5.275e-04, max_step=2.723e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=1.170e-03, signed_mean=3.318e-04, mare=7.573e-03, max_step=7.192e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-8.790e-10, signed_mean=-1.740e-09, mare=5.073e-09, max_step=1.840e-08 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.012e-02, mare_max=4.141e-02, max_step_max=1.831e-01, stats=3 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=4.424e-04, mare_max=1.398e-03, max_step_max=5.945e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
