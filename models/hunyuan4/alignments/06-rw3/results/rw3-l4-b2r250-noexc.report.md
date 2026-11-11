# janus 报告：hunyuan4-rw3 / rw3-l4-b2r250

- 被测：`b2r-noexc`；golden：`rw1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/06-rw3/catalogs/rw3-l4-b2r250.yaml`（内容指纹 38ac5f2ef9ef…）
- 结果：pass 1、fail 5、inconclusive 3、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.grad_norm`、`L4.param_norm`、`L4.router.bias_mean`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=3.357e-05, signed_mean_max=1.581e-04, mare_max=1.581e-04, max_step_max=4.622e-04 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | fail | d0_max=3.357e-05, signed_mean_max=1.218e-04, mare_max=1.218e-04, max_step_max=3.384e-04 | 超过噪声参照两倍 |
| `L4.mtp_loss` | loss_series | L4 | fail | d0=-1.496e-04, signed_mean=1.491e-04, mare=2.628e-04, max_step=6.740e-04 | 超过噪声参照两倍 |
| `L4.mtp_loss.tail` | loss_series | L4 | fail | d0=-1.496e-04, signed_mean=8.940e-06, mare=1.976e-04, max_step=6.505e-04 | 超过噪声参照两倍 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-3.465e-04, signed_mean=4.629e-04, mare=5.997e-04, max_step=2.401e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=1.006e-03, signed_mean=2.316e-04, mare=5.303e-03, max_step=3.313e-02 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=-9.062e-10, signed_mean=1.590e-09, mare=2.710e-09, max_step=1.487e-08 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_mean` | router_stats | L4 | inconclusive | signed_mean_max=7.792e-02, mare_max=8.596e-02, max_step_max=2.192e-01, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.136e-03, mare_max=2.023e-03, max_step_max=7.375e-03, stats=3 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
