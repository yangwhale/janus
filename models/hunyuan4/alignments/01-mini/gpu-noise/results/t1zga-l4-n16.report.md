# janus 报告：hunyuan4-mini / t1zga-l4-n16

- 被测：`t1zg-a`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/01-mini/gpu-noise/catalogs/t1zga-l4-n16.yaml`（内容指纹 1b44c7f3b21b…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=2.604e-06, signed_mean_max=1.359e-06, mare_max=8.049e-06, max_step_max=3.447e-05 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=2.604e-06, signed_mean_max=1.172e-06, mare_max=8.360e-06, max_step_max=3.447e-05 | 在噪声参照内 |
| `L4.mtp_loss` | loss_series | L4 | pass | d0=2.239e-06, signed_mean=-8.103e-06, mare=3.160e-05, max_step=1.633e-04 | 在噪声参照内 |
| `L4.mtp_loss.tail` | loss_series | L4 | pass | d0=2.239e-06, signed_mean=-9.263e-06, mare=3.467e-05, max_step=1.633e-04 | 在噪声参照内 |
| `L4.indexer_loss` | loss_series | L4 | pass | d0=-6.736e-05, signed_mean=-1.265e-04, mare=2.961e-04, max_step=1.617e-03 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-5.223e-04, signed_mean=-4.138e-04, mare=2.910e-03, max_step=5.872e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=2.227e-10, signed_mean=4.830e-07, mare=1.572e-06, max_step=9.367e-06 | 在噪声参照内 |
| `L4.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.044e-01, mare_max=5.041e-01, max_step_max=8.411e+00, stats=7 | 在噪声参照内 |
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.821e-03, mare_max=4.916e-03, max_step_max=2.451e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
