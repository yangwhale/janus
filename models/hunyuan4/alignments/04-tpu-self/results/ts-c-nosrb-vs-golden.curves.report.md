# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-c-nosrb`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=-4.868e-07, mare=5.529e-06, max_step=2.157e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-8.139e-08, signed_mean=-3.363e-07, mare=6.337e-06, max_step=2.157e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=-3.053e-07, signed_mean=-3.522e-06, mare=1.198e-05, max_step=5.938e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-1.116e-05, signed_mean=7.617e-05, mare=2.296e-04, max_step=1.191e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=2.024e-06, signed_mean=-3.866e-05, mare=3.631e-04, max_step=1.601e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=3.412e-08, mare=4.960e-08, max_step=1.558e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.469e-01, mare_max=4.485e-01, max_step_max=2.017e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=2.302e-03, mare_max=5.118e-03, max_step_max=2.188e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
