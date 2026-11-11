# janus 报告：hunyuan4-mini / g3x16-noise

- 被测：`g3-i`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/gpu-noise/catalogs/g3x16-noise.yaml`（内容指纹 8cf93311667c…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.790e-06, signed_mean_max=1.967e-06, mare_max=7.243e-06, max_step_max=3.749e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.790e-06, signed_mean_max=2.118e-06, mare_max=7.674e-06, max_step_max=3.749e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=-2.235e-06, mare=3.270e-05, max_step=2.254e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=6.107e-07, signed_mean=-2.750e-06, mare=3.587e-05, max_step=2.254e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-2.832e-05, signed_mean=5.553e-05, mare=2.791e-04, max_step=1.268e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-2.162e-05, signed_mean=-7.843e-05, mare=2.304e-03, max_step=3.944e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.036e-09, signed_mean=-9.625e-07, mare=1.337e-06, max_step=5.111e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=9.418e-01, mare_max=9.784e-01, max_step_max=7.009e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=7.836e-04, mare_max=4.524e-03, max_step_max=2.302e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
