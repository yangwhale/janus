# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`mn-lr`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 1、fail 7、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | fail | d0=-1.953e-06, signed_mean=-1.356e-02, mare=1.356e-02, max_step=2.309e-02 | 超过噪声参照两倍 |
| `C.loss.tail` | loss_series | L4 | fail | d0=-1.953e-06, signed_mean=-1.781e-02, mare=1.781e-02, max_step=2.309e-02 | 超过噪声参照两倍 |
| `C.mtp_loss` | loss_series | L4 | fail | d0=-3.053e-07, signed_mean=-3.003e-03, mare=3.003e-03, max_step=9.324e-03 | 超过噪声参照两倍 |
| `C.indexer_loss` | loss_series | L4 | fail | d0=-1.652e-05, signed_mean=-1.866e-02, mare=1.866e-02, max_step=3.288e-02 | 超过噪声参照两倍 |
| `C.grad_norm` | loss_series | L4 | fail | d0=1.743e-06, signed_mean=-1.141e-02, mare=1.496e-02, max_step=5.564e-02 | 超过噪声参照两倍 |
| `C.param_norm` | loss_series | L4 | fail | d0=-9.204e-08, signed_mean=5.003e-04, mare=5.003e-04, max_step=1.477e-03 | 超过噪声参照两倍 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=5.724e-01, mare_max=5.798e-01, max_step_max=8.411e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | fail | signed_mean_max=1.782e-02, mare_max=1.851e-02, max_step_max=4.904e-02, stats=7 | 超过噪声参照两倍 |

数字全部来自 `results.json`，本报告不手写数字。
