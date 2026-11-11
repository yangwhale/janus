# janus 报告：hunyuan4-mini / tq-alloff-l4

- 被测：`tq-alloff`；golden：`g2-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/catalogs/tq-alloff-l4.yaml`（内容指纹 568331987d4b…）
- 结果：pass 2、fail 4、inconclusive 3、error 0；**有硬检查失败**

**证据不足（inconclusive）**：`L4.mtp_loss`、`L4.mtp_loss.tail`、`L4.grad_norm`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | fail | d0_max=1.221e-06, signed_mean_max=1.059e-05, mare_max=2.537e-05, max_step_max=2.676e-04 | 超过噪声参照两倍 |
| `L4.loss.tail` | multi_golden | L4 | fail | d0_max=1.221e-06, signed_mean_max=1.044e-05, mare_max=1.325e-05, max_step_max=4.048e-05 | 超过噪声参照两倍 |
| `L4.mtp_loss` | loss_series | L4 | inconclusive | d0=1.425e-06, signed_mean=-1.603e-06, mare=1.595e-05, max_step=1.312e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.mtp_loss.tail` | loss_series | L4 | inconclusive | d0=1.425e-06, signed_mean=-2.882e-06, mare=2.074e-05, max_step=1.312e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.indexer_loss` | loss_series | L4 | fail | d0=-9.563e-04, signed_mean=-5.192e-04, mare=5.848e-04, max_step=1.274e-03 | 超过噪声参照两倍 |
| `L4.grad_norm` | loss_series | L4 | inconclusive | d0=-2.685e-04, signed_mean=1.204e-04, mare=6.231e-04, max_step=2.637e-03 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.param_norm` | loss_series | L4 | fail | d0=-1.845e-10, signed_mean=-7.039e-07, mare=7.059e-07, max_step=1.512e-06 | 超过噪声参照两倍 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=7.692e-01, mare_max=7.774e-01, max_step_max=3.663e+01, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.706e-03, mare_max=5.065e-03, max_step_max=2.277e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
