# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-dp8-embf32`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 1、fail 7、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=7.325e-07, signed_mean=1.839e-03, mare=1.839e-03, max_step=2.628e-03 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=7.325e-07, signed_mean=2.221e-03, mare=2.221e-03, max_step=2.628e-03 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | fail | d0=6.107e-07, signed_mean=2.112e-04, mare=2.112e-04, max_step=5.009e-04 | 超过噪声参照两倍 |
| `C.indexer_loss` | loss_series | L4 | fail | d0=1.727e-05, signed_mean=1.159e-02, mare=1.159e-02, max_step=2.139e-02 | 超过噪声参照两倍 |
| `C.grad_norm` | loss_series | L4 | fail | d0=-2.646e-02, signed_mean=1.813e-02, mare=2.073e-02, max_step=3.386e-02 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | fail | d0=-1.131e-07, signed_mean=-2.049e-05, mare=2.049e-05, max_step=3.937e-05 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=4.141e-01, mare_max=5.583e-01, max_step_max=1.210e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | fail | signed_mean_max=1.603e-02, mare_max=1.662e-02, max_step_max=4.187e-02, stats=7 | 超过噪声参照两倍 |

数字全部来自 `results.json`，本报告不手写数字。
