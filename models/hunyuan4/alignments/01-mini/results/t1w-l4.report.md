# janus 报告：hunyuan4-mini / t1w-l4

- 被测：`t1w-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/t1w-l4.yaml`（内容指纹 3e582f1106b5…）
- 结果：pass 6、fail 1、inconclusive 2、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.loss`、`L4.loss.tail`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=2.930e-06, signed_mean_max=7.083e-06, mare_max=1.105e-05, max_step_max=6.283e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.loss.tail` | multi_golden | L4 | inconclusive | d0_max=2.930e-06, signed_mean_max=7.575e-06, mare_max=1.166e-05, max_step_max=6.283e-05 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=-1.120e-06, signed_mean=-1.254e-05, mare=3.543e-05, max_step=1.998e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=-1.120e-06, signed_mean=-1.409e-05, mare=3.892e-05, max_step=1.998e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-4.112e-04, signed_mean=4.338e-04, mare=7.470e-04, max_step=2.253e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-3.187e-04, signed_mean=4.091e-05, mare=2.528e-03, max_step=4.254e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=1.279e-08, signed_mean=-1.395e-06, mare=1.741e-06, max_step=8.340e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=2.518e-01, mare_max=2.943e-01, max_step_max=7.009e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.263e-03, mare_max=4.542e-03, max_step_max=2.141e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
