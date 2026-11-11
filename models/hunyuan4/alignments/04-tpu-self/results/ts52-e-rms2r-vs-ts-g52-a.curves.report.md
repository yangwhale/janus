# janus 报告：hunyuan4-mini / tpu-self-vs-ts-g52-a

- 被测：`ts52-e-rms2r`；golden：`ts-g52-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-ts-g52-a.yaml`（内容指纹 cd3ad30aee1b…）
- 结果：pass 6、fail 2、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=4.069e-07, signed_mean=1.655e-05, mare=1.675e-05, max_step=5.002e-05 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=4.069e-07, signed_mean=2.246e-05, mare=2.246e-05, max_step=5.002e-05 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=3.053e-07, signed_mean=3.420e-07, mare=9.986e-06, max_step=5.833e-05 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=-3.218e-06, signed_mean=4.467e-06, mare=1.916e-04, max_step=1.157e-03 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-2.189e-05, signed_mean=2.531e-05, mare=2.881e-04, max_step=1.416e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=0.000e+00, signed_mean=4.813e-08, mare=5.294e-08, max_step=1.620e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.925e-01, mare_max=2.092e-01, max_step_max=4.034e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=9.489e-04, mare_max=3.737e-03, max_step_max=2.270e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
