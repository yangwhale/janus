# janus 报告：hunyuan4-mini / tpu-self-vs-golden

- 被测：`ts-s-lr1e-1`；golden：`ts-g-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml`（内容指纹 433f6f355299…）
- 结果：pass 1、fail 7、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.356e-02, mare=1.356e-02, max_step=2.309e-02 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.781e-02, mare=1.781e-02, max_step=2.309e-02 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-3.005e-03, mare=3.005e-03, max_step=9.361e-03 | 超过噪声参照两倍 |
| `C.indexer_loss` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.876e-02, mare=1.876e-02, max_step=3.321e-02 | 超过噪声参照两倍 |
| `C.grad_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=-1.155e-02, mare=1.503e-02, max_step=5.700e-02 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | fail | d0=0.000e+00, signed_mean=4.951e-04, mare=4.952e-04, max_step=1.467e-03 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=6.461e-01, mare_max=6.718e-01, max_step_max=1.324e+01, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | fail | signed_mean_max=1.956e-02, mare_max=2.005e-02, max_step_max=4.963e-02, stats=7 | 超过噪声参照两倍 |

数字全部来自 `results.json`，本报告不手写数字。
